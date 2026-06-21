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

# Two distinct large odd primes used to hash the ordered (prev, cur) token pair into a bucket id for the
# hashed bigram-identity input embedding (mechanism B, see GPT.forward). Both are genuine large primes
# (hence odd), so the multiplicative hash mixes the two token-ids well even under a power-of-two bucket
# modulus.
_BIGRAM_HASH_PRIME_PREV = 1_000_000_007
_BIGRAM_HASH_PRIME_CUR = 998_244_353
# Small NON-zero init for the learned bigram-hash residual gate (mechanism B), so the joint pair path is
# active but gentle from step 0 (paired with the small non-zero projection init embed_bigram_hash_init_std).
# This is deliberately NOT zero-init: the request is for a path that contributes from the very first step.
_BIGRAM_HASH_GATE_INIT = 0.1
# Small NON-zero init std for the two projections of the gated MULTIPLICATIVE joint-bigram path
# (mechanism A, embed_ctx_mode='mult'). The term is gate * (embed_proj(low_dim_embed) ⊙ ctx_embed_proj(ctx_low_dim_embed)).
# The gate is zero-init so the term is an exact no-op at init. But if the two projections were ALSO zero-init
# (as in the plain additive path), then cur=ctx_prev=0 and the gate's gradient (∝ cur ⊙ ctx_prev) would be
# zero too — every gradient into the path would vanish and the all-zero point is a frozen fixed point. Init
# the projections small-NONZERO so cur, ctx_prev are nonzero and give the zero-init gate a live gradient;
# with gate=0 the term is still exactly zero at init. (See memory: multiplicative-gated-path-dead-init.)
_MULT_PROJ_INIT_STD = 0.02

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
    # Linear projection embedding: low-rank correction added to wte.
    # 0 = disabled. When > 0, adds low_dim_embed (vocab × embed_proj_dim) + Linear (embed_proj_dim → n_embd).
    embed_proj_dim: int = 0
    # Mechanism A — gated MULTIPLICATIVE joint-bigram input path.
    # 'none' = disabled (default). 'mult' = the previous-token low-dim vector MODULATES the current-token
    # low-dim embedding element-wise: x = x + embed_ctx_gate * (embed_proj(low_dim_embed(token_t)) ⊙
    # ctx_embed_proj(ctx_low_dim_embed(token_{t-1}))). This is a genuine (token_t, token_{t-1}) interaction
    # — non-absorbable into wte and non-redundant with the separable smear / attention. Requires
    # embed_proj_dim>0 (the current-token low-dim path); the previous-token path (ctx_low_dim_embed /
    # ctx_embed_proj) is sized to embed_proj_dim. The scalar gate is ZERO-init (exact no-op at init), but
    # both projections are small-NONZERO so the product has a live gradient at step 0 (see
    # _MULT_PROJ_INIT_STD). In 'mult' mode the current-token low-dim vector enters ONLY through the product
    # (no standalone additive term).
    embed_ctx_mode: str = "none"
    # Mechanism B — hashed (prev, cur) bigram-identity low-dim INPUT embedding.
    # 0 = disabled (e.g. 64 when enabled). When > 0, the ordered pair (token_{t-1}, token_t) is hashed into
    # embed_bigram_hash_buckets buckets, a low-dim vector of width embed_bigram_hash_dim is looked up
    # (bigram_hash_embed) and bias-free up-projected to n_embd (bigram_hash_proj), scaled by a learned gate,
    # and ADDED to wte at the input. A pair-identity table captures the NON-additive bigram interaction: it
    # is neither absorbable into the per-token wte nor decomposable into per-token sums.
    embed_bigram_hash_dim: int = 0
    # Number of hash buckets for the ordered (prev, cur) pair (default 2^18 = 262144).
    embed_bigram_hash_buckets: int = 262144
    # Init std for the bigram hash projection (bigram_hash_proj). Small NON-zero (NOT zero-init), so the
    # path contributes from step 0; paired with the small non-zero learned gate, the joint term is active
    # but gentle from the first step rather than inert until it moves off zero.
    embed_bigram_hash_init_std: float = 0.005


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

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
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
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
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

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
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
        # Linear projection embedding: low-rank learnable correction summed with wte
        # inputs_embeds = Linear(low_dim_embed(token_ids)) + wte(token_ids)
        if config.embed_proj_dim > 0:
            self.low_dim_embed = nn.Embedding(padded_vocab_size, config.embed_proj_dim)
            self.embed_proj = Linear(config.embed_proj_dim, config.n_embd, bias=False)
        else:
            self.low_dim_embed = None
            self.embed_proj = None
        # Mechanism A: gated MULTIPLICATIVE (joint) bigram input path (embed_ctx_mode='mult'). The
        # previous-token low-dim vector modulates the CURRENT-token low-dim embedding element-wise behind a
        # learned scalar gate. 'mult' requires the current-token low-dim path (embed_proj_dim>0) so the two
        # projections (each up to n_embd) multiply in the same space — misconfigured arms fail fast here. The
        # previous-token path (ctx_low_dim_embed / ctx_embed_proj) is sized to embed_proj_dim. The gate is
        # ZERO-INIT (in init_weights) so the whole multiplicative term is an exact no-op at init: the run
        # starts identical to the dense baseline and the model opts the path in only if it helps. 'none' (the
        # default) leaves the gate / ctx modules as None and is unchanged.
        assert config.embed_ctx_mode in ("none", "mult"), \
            f"embed_ctx_mode must be 'none' or 'mult', got {config.embed_ctx_mode!r}"
        if config.embed_ctx_mode == "mult":
            assert config.embed_proj_dim > 0, (
                "embed_ctx_mode='mult' requires embed_proj_dim>0 (the current-token low-dim path) so the "
                "joint product is well-defined"
            )
            self.ctx_low_dim_embed = nn.Embedding(padded_vocab_size, config.embed_proj_dim)
            self.ctx_embed_proj = Linear(config.embed_proj_dim, config.n_embd, bias=False)
            self.embed_ctx_gate = nn.Parameter(torch.zeros(1))  # fake init (meta); real zero-init in init_weights()
        else:
            self.ctx_low_dim_embed = None
            self.ctx_embed_proj = None
            self.embed_ctx_gate = None
        # Mechanism B: hashed bigram-identity low-dim INPUT embedding. A JOINT (prev, cur) pair-identity
        # table. The ordered pair (token_{t-1}, token_t) is hashed into embed_bigram_hash_buckets buckets
        # (forward()), a low-dim vector is looked up and bias-free up-projected to n_embd, scaled by a learned
        # gate, and added to the wte input sum. Both bigram_hash_proj and bigram_hash_gate use a small NON-zero
        # init (in init_weights), so the path is active (not a no-op) from step 0.
        if config.embed_bigram_hash_dim > 0:
            self.bigram_hash_embed = nn.Embedding(config.embed_bigram_hash_buckets, config.embed_bigram_hash_dim)
            self.bigram_hash_proj = Linear(config.embed_bigram_hash_dim, config.n_embd, bias=False)
            self.bigram_hash_gate = nn.Parameter(torch.zeros(1))  # fake init (meta); real small non-zero init in init_weights()
        else:
            self.bigram_hash_embed = None
            self.bigram_hash_proj = None
            self.bigram_hash_gate = None
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
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

        # Linear projection embedding init
        if self.low_dim_embed is not None:
            torch.nn.init.normal_(self.low_dim_embed.weight, mean=0.0, std=0.8)
            if self.config.embed_ctx_mode == "mult":
                # Mechanism A reuses embed_proj as a FACTOR of the gated product. It must be small-NONZERO
                # (not zero) so the zero-init gate receives a live gradient (∝ cur ⊙ ctx_prev); a zero-init
                # projection here would freeze the whole multiplicative path at the all-zero fixed point.
                torch.nn.init.normal_(self.embed_proj.weight, mean=0.0, std=_MULT_PROJ_INIT_STD)
            else:
                # Additive path: zero-init the projection so the correction starts at zero (no change to
                # wte at init); the additive term is linear in embed_proj so low_dim_embed still trains it.
                torch.nn.init.zeros_(self.embed_proj.weight)

        # Mechanism A (embed_ctx_mode='mult'): ctx_low_dim_embed like wte (normal, std 0.8); ctx_embed_proj
        # small-NONZERO (the previous-token FACTOR of the product, same dead-init reasoning as embed_proj);
        # the scalar gate is ZERO-init so the joint product term gate * (cur ⊙ ctx_prev) is an exact no-op at
        # init — the run starts identical to dense.
        if self.ctx_low_dim_embed is not None:
            torch.nn.init.normal_(self.ctx_low_dim_embed.weight, mean=0.0, std=0.8)
            torch.nn.init.normal_(self.ctx_embed_proj.weight, mean=0.0, std=_MULT_PROJ_INIT_STD)
            torch.nn.init.zeros_(self.embed_ctx_gate)

        # Mechanism B (hashed bigram-identity input embedding) init: bigram_hash_embed like wte / the other
        # low-dim tables (normal, std 0.8); bigram_hash_proj at a small NON-zero std (embed_bigram_hash_init_std)
        # and the gate at a small non-zero value, so the joint pair path is active (NOT a no-op) from step 0.
        if self.bigram_hash_embed is not None:
            torch.nn.init.normal_(self.bigram_hash_embed.weight, mean=0.0, std=0.8)
            torch.nn.init.normal_(self.bigram_hash_proj.weight, mean=0.0, std=self.config.embed_bigram_hash_init_std)
            torch.nn.init.constant_(self.bigram_hash_gate, _BIGRAM_HASH_GATE_INIT)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            if self.low_dim_embed is not None:
                self.low_dim_embed.to(dtype=COMPUTE_DTYPE)
            if self.ctx_low_dim_embed is not None:
                self.ctx_low_dim_embed.to(dtype=COMPUTE_DTYPE)
            if self.bigram_hash_embed is not None:
                self.bigram_hash_embed.to(dtype=COMPUTE_DTYPE)
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
        low_dim_embed_numel = self.low_dim_embed.weight.numel() if self.low_dim_embed is not None else 0
        # ctx_low_dim_embed (mechanism A) and bigram_hash_embed (mechanism B) are embedding lookups (not
        # matmuls) => excluded like low_dim_embed. Their projections (ctx_embed_proj / bigram_hash_proj) ARE
        # real matmuls and stay counted in the 6*(nparams - exclude) term. The two scalar gates are
        # non-matmul scalars => excluded like the other scalars.
        ctx_low_dim_embed_numel = self.ctx_low_dim_embed.weight.numel() if self.ctx_low_dim_embed is not None else 0
        embed_ctx_gate_numel = self.embed_ctx_gate.numel() if self.embed_ctx_gate is not None else 0
        bigram_hash_embed_numel = self.bigram_hash_embed.weight.numel() if self.bigram_hash_embed is not None else 0
        bigram_hash_gate_numel = self.bigram_hash_gate.numel() if self.bigram_hash_gate is not None else 0
        nparams_exclude = (self.transformer.wte.weight.numel() + low_dim_embed_numel + ctx_low_dim_embed_numel + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel() +
                          embed_ctx_gate_numel + bigram_hash_embed_numel + bigram_hash_gate_numel)
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
        low_dim_embed = self.low_dim_embed.weight.numel() if self.low_dim_embed is not None else 0
        embed_proj = self.embed_proj.weight.numel() if self.embed_proj is not None else 0
        # Mechanism A (mult): the previous-token low-dim table is an embedding lookup; its projection is a
        # real matmul. Reported separately (mirroring low_dim_embed / embed_proj). The 1-element gate folds
        # into scalars below.
        ctx_low_dim_embed = self.ctx_low_dim_embed.weight.numel() if self.ctx_low_dim_embed is not None else 0
        ctx_embed_proj = self.ctx_embed_proj.weight.numel() if self.ctx_embed_proj is not None else 0
        embed_ctx_gate = self.embed_ctx_gate.numel() if self.embed_ctx_gate is not None else 0
        # Mechanism B (hashed bigram-identity): the bucket table is an embedding lookup, the projection is a
        # real matmul; the 1-element gate folds into scalars below.
        bigram_hash_embed = self.bigram_hash_embed.weight.numel() if self.bigram_hash_embed is not None else 0
        bigram_hash_proj = self.bigram_hash_proj.weight.numel() if self.bigram_hash_proj is not None else 0
        bigram_hash_gate = self.bigram_hash_gate.numel() if self.bigram_hash_gate is not None else 0
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel() + embed_ctx_gate + bigram_hash_gate
        total = wte + low_dim_embed + embed_proj + ctx_low_dim_embed + ctx_embed_proj + bigram_hash_embed + bigram_hash_proj + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'low_dim_embed': low_dim_embed,
            'embed_proj': embed_proj,
            'ctx_low_dim_embed': ctx_low_dim_embed,
            'ctx_embed_proj': ctx_embed_proj,
            'bigram_hash_embed': bigram_hash_embed,
            'bigram_hash_proj': bigram_hash_proj,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        low_dim_embed_params = list(self.low_dim_embed.parameters()) if self.low_dim_embed is not None else []
        embed_proj_params = list(self.embed_proj.parameters()) if self.embed_proj is not None else []
        # Mechanism A (mult): dedicated AdamW groups mirroring low_dim_embed / embed_proj (same LRs). The
        # previous-token table rides the embedding group; its projection rides the projection group.
        ctx_low_dim_embed_params = list(self.ctx_low_dim_embed.parameters()) if self.ctx_low_dim_embed is not None else []
        ctx_embed_proj_params = list(self.ctx_embed_proj.parameters()) if self.ctx_embed_proj is not None else []
        # Mechanism B (hashed bigram-identity): dedicated AdamW groups mirroring low_dim_embed / embed_proj.
        # The learned gate rides the projection group (bigram_hash_proj + gate), like a small matmul-side scalar.
        bigram_hash_embed_params = list(self.bigram_hash_embed.parameters()) if self.bigram_hash_embed is not None else []
        bigram_hash_proj_params = (list(self.bigram_hash_proj.parameters()) + [self.bigram_hash_gate]) if self.bigram_hash_proj is not None else []
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        # Mechanism A multiplicative gate: a learned scalar; ride the scalar-style smear AdamW group
        # (lr=0.2, no weight decay) when present. Owned by exactly one group.
        if self.embed_ctx_gate is not None:
            smear_params.append(self.embed_ctx_gate)
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(low_dim_embed_params) + len(embed_proj_params) + len(ctx_low_dim_embed_params) + len(ctx_embed_proj_params) + len(bigram_hash_embed_params) + len(bigram_hash_proj_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            *(
                [dict(kind='adamw', params=low_dim_embed_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001)]
                if low_dim_embed_params else []
            ),
            *(
                [dict(kind='adamw', params=embed_proj_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01)]
                if embed_proj_params else []
            ),
            *(
                [dict(kind='adamw', params=ctx_low_dim_embed_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001)]
                if ctx_low_dim_embed_params else []
            ),
            *(
                [dict(kind='adamw', params=ctx_embed_proj_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01)]
                if ctx_embed_proj_params else []
            ),
            *(
                [dict(kind='adamw', params=bigram_hash_embed_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001)]
                if bigram_hash_embed_params else []
            ),
            *(
                [dict(kind='adamw', params=bigram_hash_proj_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01)]
                if bigram_hash_proj_params else []
            ),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
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

    def _shift_ctx_prev(self, ctx, kv_cache):
        """Causally right-shift the per-current-token context vector so position t reads token t-1.

        Position 0 has no predecessor => zeros (training / full-sequence eval; no future-token leakage).
        Under a KV cache the previous token's ctx is carried across forward calls (mirrors the smear
        `prev_embedding` plumbing) so the joint product is well-defined under single-token decode.
        """
        if kv_cache is None:
            # Training / full-sequence eval: shift right by one so position t reads token t-1;
            # position 0 has no predecessor and gets zeros (causal — no future-token leakage).
            return F.pad(ctx, (0, 0, 1, 0))[:, :-1, :]
        # KV-cache inference: carry the previous token's ctx across forward calls (mirrors smear).
        prev_ctx = kv_cache.prev_ctx
        kv_cache.prev_ctx = ctx[:, -1:, :]  # stash current last-token ctx for the next step
        T = ctx.size(1)
        if T > 1:
            # Prefill: shift within the chunk; first position uses the carried prev (zeros if fresh cache).
            head = prev_ctx if prev_ctx is not None else torch.zeros_like(ctx[:, :1, :])
            return torch.cat([head, ctx[:, :-1, :]], dim=1)
        elif prev_ctx is not None:
            # Decode: single token reads the cached previous token's ctx.
            return prev_ctx
        else:
            return torch.zeros_like(ctx)

    def _shift_prev_token(self, idx, kv_cache):
        """Causally right-shift the token-id sequence so position t sees token t-1.

        Position 0 has no predecessor => sentinel id 0 (no future-token leakage). Under a KV cache the
        previous token id is carried across forward calls (mirrors the smear / ctx plumbing) so the ordered
        (prev, cur) pair is well-defined under single-token decode.
        """
        if kv_cache is None:
            # Training / full-sequence eval: shift right by one; position 0 uses sentinel 0 (causal).
            return F.pad(idx, (1, 0))[:, :-1]
        # KV-cache inference: carry the previous token id across forward calls (mirrors prev_ctx / smear).
        prev_tok = kv_cache.prev_token_id
        kv_cache.prev_token_id = idx[:, -1:]  # stash current last token id for the next step
        T = idx.size(1)
        if T > 1:
            # Prefill: shift within the chunk; first position uses the carried prev (sentinel 0 if fresh).
            head = prev_tok if prev_tok is not None else torch.zeros_like(idx[:, :1])
            return torch.cat([head, idx[:, :-1]], dim=1)
        elif prev_tok is not None:
            # Decode: single token reads the cached previous token id.
            return prev_tok
        else:
            return torch.zeros_like(idx)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
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
        # Linear projection: add low-rank correction from projected low-dim embedding
        if self.low_dim_embed is not None:
            cur_proj = self.embed_proj(self.low_dim_embed(idx)) # (B, T, n_embd): per-current-token low-rank vector
            if self.config.embed_ctx_mode == "mult":
                # Mechanism A: gated MULTIPLICATIVE (joint) bigram input path. The previous-token low-dim
                # vector modulates the CURRENT-token low-dim embedding element-wise. In this mode the
                # current-token low-dim vector enters ONLY through the product (NOT also added standalone).
                # The gate is zero-init, so the whole term is an exact no-op at init.
                ctx = self.ctx_embed_proj(self.ctx_low_dim_embed(idx)) # (B, T, n_embd): per-current-token low-rank vector
                ctx_prev = self._shift_ctx_prev(ctx, kv_cache)         # causally shifted/carried to token t-1
                x = x + self.embed_ctx_gate.to(x.dtype) * (cur_proj * ctx_prev)
            else:
                # Additive per-token projection (default): a low-rank correction summed into wte.
                x = x + cur_proj
        # Mechanism B: hashed (prev, cur) bigram-identity INPUT embedding. The ordered pair
        # (token_{t-1}, token_t) is hashed (two distinct large odd primes; deterministic, on-device) into a
        # fixed bucket table, looked up, bias-free up-projected, gated, and ADDED to the wte input sum. The
        # previous token is the causal right-shift of idx (position 0 -> sentinel 0, no future-token leakage),
        # so the term depends jointly on two token-ids and is neither absorbable into the per-token wte nor
        # decomposable into per-token sums. Independent of embed_ctx_mode.
        if self.bigram_hash_embed is not None:
            prev = self._shift_prev_token(idx, kv_cache)
            buckets = self.config.embed_bigram_hash_buckets
            bucket = (prev.to(torch.long) * _BIGRAM_HASH_PRIME_PREV + idx.to(torch.long) * _BIGRAM_HASH_PRIME_CUR) % buckets
            bigram = self.bigram_hash_proj(self.bigram_hash_embed(bucket)) # (B, T, n_embd)
            x = x + self.bigram_hash_gate.to(x.dtype) * bigram
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

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
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
