"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Sentence attention: gist ("end-of-sentence") token ids. Non-empty => sentence attention
    # is active (block-causal + global-gist mask, confined per-document). Empty => standard
    # causal attention (the model is bit-for-bit unchanged on this path).
    end_of_sentence_token_ids: tuple[int, ...] = ()
    # Layers that bypass the sentence mask and use plain full-causal attention. Empty here
    # (pure sentence attention); kept for completeness / future use.
    full_attention_layers: tuple[int, ...] = ()
    # BOS token id: marked always-visible and used as the per-document segment delimiter for
    # the sentence mask (segment id = cumulative count of BOS tokens). -1 disables BOS handling.
    bos_token_id: int = -1
    # Auxiliary gist-reconstruction head (experiments/gist-token-reconstruction-1.md).
    # TRAINING-ONLY: the head is allocated ONLY when gist_recon_enabled is True, is never used
    # by the next-token path (val loss / bpb / CORE / generate), and is discarded when a
    # checkpoint is loaded for eval/inference (see checkpoint_manager.build_model).
    # False => not a single extra parameter or FLOP, so arms trained before this existed stay
    # bit-for-bit comparable.
    gist_recon_enabled: bool = False
    # Tokens further than this from the start of their sentence get no reconstruction target
    # (bounds the reconstruction span; also the size of the within-sentence position table).
    gist_recon_max_sentence_len: int = 64
    # Reconstruct only every Nth token position (a static-shape subsample). The auxiliary loss
    # is a mean over supervised positions, so the stride trades gradient variance for the
    # memory/compute of the head's extra (B, T/stride, vocab) logits and does not rescale it.
    gist_recon_stride: int = 8


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def _closest_boundary_idx(gist_mask):
    """For each query position, the index of the most recent sentence boundary strictly
    before it. A boundary is the LAST token of each contiguous run of gist tokens.
    Vectorized cummax trick (ported from the reference). gist_mask: (B, T) bool over gist ids.
    Returns (B, T) long."""
    B, T = gist_mask.shape
    nxt = torch.zeros_like(gist_mask)
    nxt[:, :-1] = gist_mask[:, 1:]
    at_boundary = gist_mask & ~nxt                                  # last token of each gist run
    pos = torch.arange(T, device=gist_mask.device).unsqueeze(0).expand(B, -1)
    bidx = torch.where(at_boundary, pos, torch.zeros_like(pos))
    last_incl, _ = torch.cummax(bidx, dim=1)                        # most recent boundary <= q
    last_prev = torch.roll(last_incl, 1, dims=1)                    # shift to "strictly before q"
    last_prev[:, 0] = 0
    return last_prev.long()


def _next_boundary_idx(gist_mask):
    """Mirror image of _closest_boundary_idx: for each position, the index of the closest
    sentence boundary at-or-after it, or T if there is none. Same boundary definition (the LAST
    token of each contiguous run of gist tokens), computed with a reverse cummin instead of a
    forward cummax so it also stays a single vectorized on-GPU pass.
    gist_mask: (B, T) bool. Returns (B, T) long with values in [0, T]."""
    B, T = gist_mask.shape
    nxt = torch.zeros_like(gist_mask)
    nxt[:, :-1] = gist_mask[:, 1:]
    at_boundary = gist_mask & ~nxt                                  # last token of each gist run
    pos = torch.arange(T, device=gist_mask.device).unsqueeze(0).expand(B, -1)
    bidx = torch.where(at_boundary, pos, torch.full_like(pos, T))   # T = "not a boundary"
    flipped, _ = torch.cummin(torch.flip(bidx, [1]), dim=1)         # running min from the right
    return torch.flip(flipped, [1]).long()


def gist_reconstruction_targets(idx, targets, end_of_sentence_token_ids, bos_token_id,
                                max_sentence_len, stride):
    """Build the supervision for the auxiliary gist-reconstruction loss, fully vectorized and
    on-device (no Python loops over the batch), reusing the same cummax/cummin boundary
    machinery as the sentence mask.

    Every real token t belongs to the sentence that is closed by the FIRST gist boundary at or
    after t; that boundary's K gist tokens are the ones asked to reconstruct t. Positions are
    subsampled with a fixed `stride` so all shapes are static (torch.compile friendly).

    A position is EXCLUDED (target -1) when it is:
      - a gist token (the gists are the encoder, never a reconstruction target),
      - the BOS token (a document delimiter, not sentence content),
      - an ignore/padding position (its next-token target is -1),
      - in the LAST sentence of its document: that sentence emits no gist boundary at all (the
        dataloader inserts gists *between* sentences only), so there is nothing to reconstruct
        it from. We SKIP those tokens rather than attaching them to the next document's
        boundary, which would leak across documents.
      - further than `max_sentence_len` tokens from the start of its sentence (bounds the
        reconstruction span; keeps the within-sentence position table small).

    Args:
        idx:      (B, T) input token ids
        targets:  (B, T) next-token targets, only used for its -1 ignore positions
        end_of_sentence_token_ids: the K gist ids
        bos_token_id: BOS id, or -1 to disable document segmentation
        max_sentence_len: reconstruction span bound (also the position-embedding table range)
        stride:   subsample every `stride`-th position

    Returns a tuple of:
        sel:      (N,)   long, the selected (strided) query positions
        boundary: (B, N) long, index of the gist boundary closing each selected token's
                  sentence (clamped into range; meaningless where the target is -1)
        rel:      (B, N) long, within-sentence offset of the selected token, in
                  [0, max_sentence_len)
        target:   (B, N) long, the token id to reconstruct, or -1 where excluded
    """
    B, T = idx.shape
    device = idx.device
    eos_ids = torch.tensor(end_of_sentence_token_ids, device=device)
    gist_mask = (idx.unsqueeze(-1) == eos_ids).any(-1)              # (B, T) gist positions
    pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

    # The boundary that closes this token's sentence (T where there is none to the right).
    nxt_b = _next_boundary_idx(gist_mask)                           # (B, T)
    has_b = nxt_b < T
    nxt_b_safe = nxt_b.clamp(max=T - 1)

    # Document segmentation + the position of the BOS that opens each token's document.
    if bos_token_id >= 0:
        is_bos = idx == bos_token_id
        seg = is_bos.cumsum(dim=1)                                  # document id within the row
        same_doc = seg.gather(1, nxt_b_safe) == seg                 # boundary in the same doc?
        doc_start, _ = torch.cummax(torch.where(is_bos, pos, torch.zeros_like(pos)), dim=1)
    else:
        is_bos = torch.zeros_like(gist_mask)
        same_doc = torch.ones_like(has_b)
        doc_start = torch.zeros_like(pos)

    # Sentence start = just after the previous boundary, clipped to the document start (so the
    # first sentence of a packed document never inherits the previous document's boundary).
    prev_b = _closest_boundary_idx(gist_mask)                       # (B, T)
    sent_start = torch.maximum(prev_b + 1, doc_start + 1)
    rel = pos - sent_start

    valid = (~gist_mask) & (~is_bos) & has_b & same_doc & (rel >= 0) & (rel < max_sentence_len)
    valid = valid & (targets >= 0)

    # Static-shape subsample of query positions.
    sel = torch.arange(0, T, stride, device=device)
    boundary = nxt_b_safe.index_select(1, sel)
    rel_sel = rel.index_select(1, sel).clamp_(0, max_sentence_len - 1)
    valid_sel = valid.index_select(1, sel)
    target = torch.where(valid_sel, idx.index_select(1, sel), torch.full_like(boundary, -1))
    return sel, boundary, rel_sel, target


class GistReconstructionHead(nn.Module):
    """Auxiliary, TRAINING-ONLY head: reconstruct the token ids of a sentence from the K gist
    hidden states emitted at that sentence's boundary.

    Design (see experiments/gist-token-reconstruction-1.md):
    - Parallel / non-autoregressive. Each real token of the sentence is predicted independently
      from (its boundary's gist states, its within-sentence position), so the whole auxiliary
      objective is one extra matmul + cross-entropy, with no sequential decode.
    - The K gist states are CONCATENATED rather than pooled: the gist run always has exactly K
      tokens (the dataloader inserts K), and concatenation lets the head exploit their ordering
      (later gists in a run see more of the sentence than earlier ones).
    - Scored against its OWN output projection, deliberately NOT the model's lm_head. The plan
      requires the auxiliary objective to never touch the next-token path: sharing lm_head
      weights would add an extra gradient path into the next-token predictor and change its
      effective capacity, which would be a different (confounded) experiment.
    - Deliberately shallow (one projection + one MLP block). A deep decoder could solve the
      reconstruction task by itself; keeping it shallow pushes the pressure onto the gist
      representations, which is the thing the hypothesis is about.
    """

    def __init__(self, config, padded_vocab_size):
        super().__init__()
        n_embd = config.n_embd
        self.num_gist = len(config.end_of_sentence_token_ids)
        assert self.num_gist > 0, "gist reconstruction requires gist tokens (sentence attention)"
        # Pad the position table's leading dim to a multiple of 64: DistMuonAdamW reduce_scatters
        # any param with numel >= 1024 along dim 0 and requires shape[0] % world_size == 0.
        self.pos_table_size = ((config.gist_recon_max_sentence_len + 63) // 64) * 64
        self.gist_proj = Linear(self.num_gist * n_embd, n_embd, bias=False)
        self.pos_emb = nn.Embedding(self.pos_table_size, n_embd)
        self.c_fc = Linear(n_embd, 2 * n_embd, bias=False)
        self.c_proj = Linear(2 * n_embd, n_embd, bias=False)
        self.out = Linear(n_embd, padded_vocab_size, bias=False)

    @torch.no_grad()
    def init_weights(self):
        n_embd = self.gist_proj.out_features
        fan_in = self.gist_proj.in_features
        torch.nn.init.uniform_(self.gist_proj.weight, -(3**0.5) * fan_in**-0.5, (3**0.5) * fan_in**-0.5)
        torch.nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.8)  # like wte, it feeds a norm
        s = 3**0.5 * n_embd**-0.5
        torch.nn.init.uniform_(self.c_fc.weight, -s * 0.4, s * 0.4)
        torch.nn.init.zeros_(self.c_proj.weight)   # zero-init residual branch, as in the trunk
        torch.nn.init.normal_(self.out.weight, mean=0.0, std=0.001)  # like lm_head

    def forward(self, gists, rel_pos):
        """gists: (B, N, num_gist * n_embd), rel_pos: (B, N) long -> logits (B, N, padded_vocab)."""
        h = self.gist_proj(gists) + self.pos_emb(rel_pos).to(gists.dtype)
        h = norm(h)
        # The nonlinearity is what makes the prediction depend on (content x position) jointly;
        # a purely additive content+position mix would give every position of a sentence the
        # same distribution up to a position bias.
        h = h + self.c_proj(F.relu(self.c_fc(h)).square())
        return self.out(norm(h))


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache, attn_mask=None):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            if attn_mask is None:
                # Training: causal attention with optional sliding window (unchanged baseline path)
                y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
            else:
                # Sentence attention: a custom boolean [B,1,T,T] mask (True = attend). FA3 cannot
                # consume an arbitrary mask, so route through SDPA explicitly. q,k are already
                # RoPE'd, QK-normed and pre-scaled by 1.2, so SDPA's default 1/sqrt(head_dim)
                # softmax scale matches the FA3 path exactly.
                qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, H, T, D)
                enable_gqa = qh.size(1) != kh.size(1)
                y = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_mask, enable_gqa=enable_gqa)
                y = y.transpose(1, 2)  # back to (B, T, H, D)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, attn_mask=None):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache, attn_mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # Auxiliary gist-reconstruction head (training-only, opt-in). Constructed LAST so that
        # with the same seed the trunk/lm_head/value-embed init is bit-identical whether or not
        # the head exists (see also init_weights, which inits it last for the same reason).
        self.gist_recon = GistReconstructionHead(config, padded_vocab_size) if config.gist_recon_enabled else None
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @property
    def max_seq_len(self):
        # Hard limit on the sequence length the model can forward: the rotary cache size.
        # Exposed so evaluation code (e.g. core_eval) can truncate over-long sequences
        # (such as long few-shot CORE prompts) instead of tripping the rotary assert in forward().
        return self.cos.size(1)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Auxiliary gist-reconstruction head, initialized LAST so that everything above draws
        # exactly the same random numbers with or without the head (the lambda=0 control and a
        # lambda>0 arm therefore start from identical trunk weights for a given seed).
        if self.gist_recon is not None:
            self.gist_recon.init_weights()

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def _build_sentence_mask(self, idx):
        """Build the boolean [B, 1, T, T] sentence-attention mask (True = attend) for a full
        sequence (training / no-kv-cache forward). Only called when sentence attention is active.

        allowed = (block_causal | special_visible) & same_doc, with the diagonal forced True.
        - block_causal:    causal AND k >= closest_boundary(q)  -> the query's own sentence block
        - special_visible: causal AND key is a gist or BOS      -> earlier gists/BOS always visible
        - same_doc:        query and key share a document (segment id = cumulative BOS count),
                           so packed multi-doc rows never leak attention across document boundaries
        """
        B, T = idx.shape
        device = idx.device
        eos_ids = torch.tensor(self.config.end_of_sentence_token_ids, device=device)
        gist_mask = (idx.unsqueeze(-1) == eos_ids).any(-1)              # (B, T) gist ids only
        bos = self.config.bos_token_id
        special_mask = gist_mask.clone()
        if bos >= 0:
            special_mask |= (idx == bos)                                # BOS always-visible
        eos_idx = _closest_boundary_idx(gist_mask).unsqueeze(-1)        # (B, T, 1)
        q = torch.arange(T, device=device).view(1, T, 1)
        k = torch.arange(T, device=device).view(1, 1, T)
        causal = k <= q                                                 # (1, T, T)
        block_causal = causal & (k >= eos_idx)                          # (B, T, T)
        special_visible = causal & special_mask.view(B, 1, T)          # (B, T, T)
        allowed = block_causal | special_visible
        if bos >= 0:
            seg = (idx == bos).cumsum(dim=1)                            # (B, T) document id within row
            same_doc = seg.unsqueeze(2) == seg.unsqueeze(1)           # (B, T, T)
            allowed = allowed & same_doc                               # confine attention per-document
        allowed = allowed | torch.eye(T, dtype=torch.bool, device=device).view(1, T, T)  # self always
        return allowed.unsqueeze(1)                                     # (B, 1, T, T) bool

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        # Also exclude the auxiliary gist-reconstruction head: it is training-only and discarded
        # at eval, so this metric keeps describing the model that is actually deployed/scored
        # and stays directly comparable to the lambda=0 control.
        gist_recon_numel = sum(p.numel() for p in self.gist_recon.parameters()) if self.gist_recon is not None else 0
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel() +
                          gist_recon_numel)
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        # Reported in its own bucket and NOT folded into any of the buckets used for scaling
        # (base_train uses transformer_matrices + lm_head), so enabling the training-only
        # auxiliary head cannot shift the auto-computed batch size / training horizon.
        gist_recon = sum(p.numel() for p in self.gist_recon.parameters()) if self.gist_recon is not None else 0
        total = wte + value_embeds + lm_head + transformer_matrices + scalars + gist_recon
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'gist_recon': gist_recon,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        # Auxiliary gist-reconstruction head: kept in its own AdamW group(s) so the trunk's Muon
        # groups (and hence the lambda=0 control's optimizer state layout) are untouched.
        recon_out_params = [self.gist_recon.out.weight] if self.gist_recon is not None else []
        recon_body_params = [p for n, p in self.gist_recon.named_parameters() if n != "out.weight"] if self.gist_recon is not None else []
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params) + len(recon_out_params) + len(recon_body_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Auxiliary head groups (only present when the head is allocated): the vocab projection
        # is treated like lm_head, the rest like ordinary AdamW matrices.
        if recon_out_params:
            param_groups.append(dict(kind='adamw', params=recon_out_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01))
            param_groups.append(dict(kind='adamw', params=recon_body_params, lr=matrix_lr * dmodel_lr_scale, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.01))
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _gist_recon_loss(self, x, idx, targets):
        """Auxiliary loss: reconstruct each sentence's token ids from the K gist hidden states
        emitted at that sentence's boundary. Training-only; see GistReconstructionHead.

        x: (B, T, C) FINAL hidden states (the same tensor lm_head consumes). Only the gist
        positions of x are read, so the auxiliary gradient enters the trunk exclusively through
        the gist representations. Returns a scalar mean CE over the supervised positions.
        """
        head = self.gist_recon
        cfg = self.config
        B, T = idx.shape
        C = x.size(-1)
        K = head.num_gist
        sel, boundary, rel, target = gist_reconstruction_targets(
            idx, targets, cfg.end_of_sentence_token_ids, cfg.bos_token_id,
            cfg.gist_recon_max_sentence_len, cfg.gist_recon_stride,
        )
        N = sel.numel()
        # The K gist tokens of a boundary occupy [boundary-K+1, boundary]; gather their states.
        offsets = torch.arange(K - 1, -1, -1, device=idx.device)         # K-1, ..., 1, 0
        gist_idx = (boundary.unsqueeze(-1) - offsets).clamp(min=0)       # (B, N, K)
        gists = x.gather(1, gist_idx.reshape(B, N * K, 1).expand(-1, -1, C)).reshape(B, N, K * C)
        # NOTE: no slicing to vocab_size here (unlike lm_head). The padded columns are never
        # targets, they just learn a low logit, and skipping the slice avoids materializing a
        # second (B, N, padded_vocab) tensor.
        logits = head(gists, rel).float()
        # reduction='sum' / count instead of reduction='mean' so an all-masked micro-batch
        # yields 0 instead of NaN, with no data-dependent control flow (torch.compile safe).
        loss_sum = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1, reduction='sum')
        num_supervised = (target >= 0).sum()
        return loss_sum / num_supervised.clamp(min=1)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', return_aux=False):
        """return_aux=True additionally computes the auxiliary gist-reconstruction loss and
        returns (next_token_loss, aux_loss). It is opt-in and used ONLY by the training loop:
        every reported metric (val loss, bpb, CORE) goes through the default return_aux=False
        path and therefore stays a pure next-token metric that never touches the aux head."""
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Sentence attention: build the block-causal + global-gist mask from the input ids.
        # Only for full-sequence (no kv-cache) forward and only when gist tokens are configured;
        # otherwise attn_mask stays None and the standard causal flash path is used unchanged.
        attn_mask = None
        if kv_cache is None and self.config.end_of_sentence_token_ids:
            attn_mask = self._build_sentence_mask(idx)

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, attn_mask)
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            if return_aux:
                # Auxiliary reconstruction is training-only: it needs the full sequence (no
                # kv-cache) and the input ids, and it is skipped entirely when the head is
                # absent (gist_recon_weight == 0) — in which case the loss above is bit-for-bit
                # the pre-existing computation and a zero constant is returned for logging.
                if self.gist_recon is not None and kv_cache is None:
                    aux_loss = self._gist_recon_loss(x, idx, targets)
                else:
                    aux_loss = torch.zeros((), dtype=torch.float32, device=idx.device)
                return loss, aux_loss
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
