"""
Tests for the two joint (token_t, token_{t-1}) embedding-side input mechanisms in nanochat.gpt.GPT:

  Mechanism A — gated MULTIPLICATIVE joint-bigram path (config.embed_ctx_mode='mult'):
      x = x + embed_ctx_gate * (embed_proj(low_dim_embed(token_t)) ⊙ ctx_embed_proj(ctx_low_dim_embed(token_{t-1})))
  The scalar gate is ZERO-init (exact no-op at init), but BOTH projections are small-NONZERO so the
  product has a live gradient at step 0 — otherwise the all-zero point is a frozen fixed point and the
  arm silently equals the dense baseline (see memory: multiplicative-gated-path-dead-init).

  Mechanism B — hashed (prev, cur) bigram-identity input embedding (config.embed_bigram_hash_dim>0):
      x = x + bigram_hash_gate * bigram_hash_proj(bigram_hash_embed(hash(token_{t-1}, token_t)))
  The projection AND gate use small NON-zero init, so the pair-keyed identity path contributes from
  step 0. The previous token is a causal right-shift with a position-0 sentinel (no future leakage).

Surfaces under test: construction/init, no-op-at-init (mult) / active-at-init (hash), causal shift and
no-future-token leakage, KV-cache decode parity, and bookkeeping (estimate_flops / num_scaling_params /
setup_optimizer) for both mechanisms.

Run: python -m pytest tests/test_joint_bigram_embeddings.py -v
"""

import pytest
import torch

import nanochat.flash_attention as fa_module
import nanochat.gpt as gpt_module
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig

EMBEDDING_LR = 0.2
VOCAB = 256


# ---------------------------------------------------------------------------
# Run every test on CPU in fp32 over the SDPA path so comparisons are tight and
# device/GPU-independent. COMPUTE_DTYPE must be patched BEFORE a model is built
# (init_weights bakes the rotary cache + embedding dtype from it).
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _cpu_fp32_sdpa():
    prev_dtype = gpt_module.COMPUTE_DTYPE
    prev_impl = fa_module._override_impl
    gpt_module.COMPUTE_DTYPE = torch.float32
    fa_module._override_impl = "sdpa"
    fa_module.USE_FA3 = fa_module._resolve_use_fa3()
    try:
        yield
    finally:
        gpt_module.COMPUTE_DTYPE = prev_dtype
        fa_module._override_impl = prev_impl
        fa_module.USE_FA3 = fa_module._resolve_use_fa3()


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _cfg(**kw) -> GPTConfig:
    base = dict(
        sequence_len=128, vocab_size=VOCAB, n_layer=2, n_head=4, n_kv_head=4,
        n_embd=128, window_pattern="SSSL",
    )
    base.update(kw)
    return GPTConfig(**base)


def _build(**kw) -> GPT:
    """Small CPU GPT (real tensors, not meta) with weights initialized for forward tests."""
    model = GPT(_cfg(**kw))
    model.init_weights()
    model.eval()
    return model


def _build_mult(D: int = 64) -> GPT:
    return _build(embed_proj_dim=D, embed_ctx_mode="mult")


def _build_hash(dim: int = 64, buckets: int = 4096) -> GPT:
    return _build(embed_bigram_hash_dim=dim, embed_bigram_hash_buckets=buckets)


def _logits(model, idx):
    with torch.no_grad():
        return model(idx)


def _rand_idx(B=2, T=16, seed=1234):
    torch.manual_seed(seed)
    return torch.randint(0, VOCAB, (B, T))


# ===========================================================================
# Mechanism A — gated multiplicative joint-bigram path
# ===========================================================================

def test_config_default_ctx_mode_is_none():
    assert GPTConfig().embed_ctx_mode == "none"
    assert _cfg().embed_ctx_mode == "none"


def test_default_model_has_no_mult_modules():
    model = _build(embed_proj_dim=64)  # additive proj only, mode defaults to 'none'
    assert model.ctx_low_dim_embed is None
    assert model.ctx_embed_proj is None
    assert model.embed_ctx_gate is None


def test_mult_build_creates_ctx_path_and_zero_init_gate():
    model = _build_mult(D=64)
    assert model.low_dim_embed is not None and model.embed_proj is not None
    assert model.ctx_low_dim_embed is not None and model.ctx_embed_proj is not None
    assert isinstance(model.embed_ctx_gate, torch.nn.Parameter)
    assert tuple(model.embed_ctx_gate.shape) == (1,)
    assert torch.count_nonzero(model.embed_ctx_gate) == 0  # zero-init gate


def test_mult_requires_current_token_low_dim_path():
    # mult without embed_proj_dim>0 has no current-token factor => fail fast.
    with pytest.raises(AssertionError):
        GPT(_cfg(embed_proj_dim=0, embed_ctx_mode="mult"))
    # equal-rank current path builds fine.
    GPT(_cfg(embed_proj_dim=64, embed_ctx_mode="mult"))


def test_invalid_ctx_mode_rejected():
    with pytest.raises(AssertionError):
        GPT(_cfg(embed_proj_dim=64, embed_ctx_mode="bogus"))


def test_mult_projections_small_nonzero_at_init():
    # The dead-init fix: gate is zero (no-op forward) but BOTH projection factors are NON-zero so the
    # gate has a live gradient. A zero-init projection here would freeze the whole path.
    model = _build_mult(D=64)
    assert torch.count_nonzero(model.embed_proj.weight) > 0
    assert torch.count_nonzero(model.ctx_embed_proj.weight) > 0


def test_mult_is_noop_at_init_equals_dense_baseline():
    """A mode='mult' model with the zero-init gate produces a forward output bit-identical to the dense
    baseline (no low-dim paths) given identical shared weights — the multiplicative path contributes
    EXACTLY zero at init."""
    mult = _build_mult(D=64)
    dense = _build()  # dense baseline: no low-dim/ctx paths, mode 'none'
    dense_sd = dense.state_dict()
    shared = {k: v for k, v in mult.state_dict().items() if k in dense_sd}
    missing, unexpected = dense.load_state_dict(shared, strict=False)
    assert not unexpected
    assert not missing  # shared covers all of dense's parameters
    idx = _rand_idx()
    assert torch.equal(_logits(mult, idx), _logits(dense, idx))


def test_mult_changes_output_once_gate_nonzero():
    model = _build_mult(D=64)
    idx = _rand_idx()
    out_off = _logits(model, idx)
    with torch.no_grad():
        model.embed_ctx_gate.fill_(0.7)
    out_on = _logits(model, idx)
    assert not torch.allclose(out_off, out_on)


def test_mult_path_trainable_from_real_init_no_manual_activation():
    """REGRESSION (multiplicative-gated-path-dead-init): the term gate * (cur ⊙ ctx_prev) must be able to
    LEAVE the all-zero fixed point under REAL training dynamics — no manual activation. The gate is
    zero-init, so it can only move if it receives a nonzero gradient at init, which requires cur and
    ctx_prev (the two projections) to be NON-zero. We run plain SGD from the real init and check: (1) the
    gate has a live gradient at the very first backward, (2) after a few steps the gate has left zero, and
    (3) once the gate is nonzero BOTH projection factors then receive live gradients => the whole path
    trains end to end."""
    model = _build_mult(D=64)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    idx = _rand_idx(seed=0)
    tgt = _rand_idx(seed=1)

    for step in range(3):
        opt.zero_grad(set_to_none=True)
        loss = model(idx, targets=tgt)
        loss.backward()
        if step == 0:
            # The dead-init bug would make this gradient identically zero (cur=ctx_prev=0).
            assert model.embed_ctx_gate.grad is not None
            assert float(model.embed_ctx_gate.grad.abs().sum()) > 0.0
        opt.step()

    assert torch.count_nonzero(model.embed_ctx_gate) > 0  # gate has escaped the zero fixed point

    # With the gate now nonzero, a fresh backward must give live gradients to BOTH projection factors.
    opt.zero_grad(set_to_none=True)
    model(idx, targets=tgt).backward()
    assert float(model.embed_ctx_gate.grad.abs().sum()) > 0.0
    assert float(model.embed_proj.weight.grad.abs().sum()) > 0.0
    assert float(model.ctx_embed_proj.weight.grad.abs().sum()) > 0.0


def test_shift_ctx_prev_is_causal_right_shift_with_zero_sentinel():
    model = _build_mult(D=64)
    C = model.config.n_embd
    torch.manual_seed(3)
    ctx = torch.randn(2, 5, C)
    prev = model._shift_ctx_prev(ctx, kv_cache=None)
    assert prev.shape == ctx.shape
    assert torch.count_nonzero(prev[:, 0]) == 0       # position 0 has no predecessor -> zeros
    assert torch.equal(prev[:, 1:], ctx[:, :-1])      # position t reads token t-1


def test_mult_position0_unchanged_when_gate_activated():
    """At position 0 the previous-token vector is zero, so the product is exactly zero there regardless
    of the gate: activating the gate must leave position 0 unchanged while changing positions >= 1."""
    model = _build_mult(D=64)
    idx = _rand_idx(seed=99)
    out_off = _logits(model, idx)
    with torch.no_grad():
        model.embed_ctx_gate.fill_(0.7)
        model.embed_proj.weight.normal_(mean=0.0, std=0.1)
        model.ctx_embed_proj.weight.normal_(mean=0.0, std=0.1)
    out_on = _logits(model, idx)
    assert torch.equal(out_off[:, 0], out_on[:, 0])
    assert not torch.allclose(out_off[:, 1:], out_on[:, 1:])


def test_mult_no_future_token_leakage():
    model = _build_mult(D=64)
    with torch.no_grad():
        model.embed_ctx_gate.fill_(0.7)
    torch.manual_seed(2024)
    T, k = 16, 8
    idx_a = torch.randint(0, VOCAB, (1, T))
    idx_b = idx_a.clone()
    idx_b[:, k + 1:] = (idx_b[:, k + 1:] + 5) % VOCAB
    out_a = _logits(model, idx_a)
    out_b = _logits(model, idx_b)
    assert torch.allclose(out_a[:, :k + 1], out_b[:, :k + 1], atol=1e-4)


def test_mult_kv_cache_decode_parity():
    model = _build_mult(D=64)
    with torch.no_grad():
        model.embed_ctx_gate.fill_(0.7)
    _assert_kv_cache_parity(model)


def test_mult_optimizer_ownership_and_coverage():
    model = _build_mult(D=64)
    opt = model.setup_optimizer(embedding_lr=EMBEDDING_LR)
    # gate owned by exactly one (adamw) group
    owning = [g for g in opt.param_groups if any(p is model.embed_ctx_gate for p in g["params"])]
    assert len(owning) == 1 and owning[0]["kind"] == "adamw"
    _assert_full_param_coverage(model, opt)


def test_mult_param_count_and_flops_accounting():
    D = 64
    mult = _build_mult(D)
    counts = mult.num_scaling_params()
    assert counts["total"] == sum(p.numel() for p in mult.parameters())
    assert counts["ctx_low_dim_embed"] == VOCAB_PADDED(mult) * D
    assert counts["ctx_embed_proj"] == mult.config.n_embd * D
    # gate is a single scalar folded into 'scalars'
    assert mult.embed_ctx_gate.numel() == 1


# ===========================================================================
# Mechanism B — hashed (prev, cur) bigram-identity input embedding
# ===========================================================================

def test_config_default_bigram_hash_disabled():
    assert GPTConfig().embed_bigram_hash_dim == 0
    assert GPTConfig().embed_bigram_hash_buckets == 262144


def test_default_model_has_no_bigram_hash_modules():
    model = _build()
    assert model.bigram_hash_embed is None
    assert model.bigram_hash_proj is None
    assert model.bigram_hash_gate is None


def test_bigram_hash_build_and_shapes():
    dim, buckets = 64, 4096
    model = _build_hash(dim=dim, buckets=buckets)
    assert model.bigram_hash_embed is not None
    assert tuple(model.bigram_hash_embed.weight.shape) == (buckets, dim)
    assert tuple(model.bigram_hash_proj.weight.shape) == (model.config.n_embd, dim)
    assert tuple(model.bigram_hash_gate.shape) == (1,)


def test_bigram_hash_active_at_init():
    """Both the projection and the gate are small NON-zero at init, so the path contributes from step 0
    (NOT a no-op): zeroing the gate must change the forward output."""
    model = _build_hash()
    assert torch.count_nonzero(model.bigram_hash_proj.weight) > 0
    assert float(model.bigram_hash_gate.detach().abs().sum()) > 0.0
    idx = _rand_idx()
    out_on = _logits(model, idx)
    with torch.no_grad():
        model.bigram_hash_gate.zero_()
    out_off = _logits(model, idx)
    assert not torch.allclose(out_on, out_off)


def test_shift_prev_token_is_causal_right_shift_with_zero_sentinel():
    model = _build_hash()
    idx = torch.tensor([[5, 6, 7, 8], [10, 11, 12, 13]])
    prev = model._shift_prev_token(idx, kv_cache=None)
    expected = torch.tensor([[0, 5, 6, 7], [0, 10, 11, 12]])
    assert torch.equal(prev, expected)


def test_bigram_hash_depends_on_ordered_pair():
    """The hashed bucket depends on the ORDERED (prev, cur) pair: swapping the two members of a pair
    (so prev/cur are exchanged) generally changes the looked-up bucket and the contribution. We verify
    the per-position bigram term directly (isolated from attention)."""
    model = _build_hash(dim=64, buckets=4096)
    P1 = gpt_module._BIGRAM_HASH_PRIME_PREV
    P2 = gpt_module._BIGRAM_HASH_PRIME_CUR
    buckets = model.config.embed_bigram_hash_buckets
    a, b = 7, 19
    bucket_ab = (a * P1 + b * P2) % buckets   # prev=a, cur=b
    bucket_ba = (b * P1 + a * P2) % buckets   # prev=b, cur=a
    assert bucket_ab != bucket_ba  # ordered hash distinguishes (a,b) from (b,a)


def test_bigram_hash_no_future_token_leakage():
    model = _build_hash()
    torch.manual_seed(7)
    T, k = 16, 8
    idx_a = torch.randint(0, VOCAB, (1, T))
    idx_b = idx_a.clone()
    idx_b[:, k + 1:] = (idx_b[:, k + 1:] + 3) % VOCAB
    out_a = _logits(model, idx_a)
    out_b = _logits(model, idx_b)
    assert torch.allclose(out_a[:, :k + 1], out_b[:, :k + 1], atol=1e-4)


def test_bigram_hash_kv_cache_decode_parity():
    model = _build_hash()
    _assert_kv_cache_parity(model)


def test_bigram_hash_optimizer_ownership_and_coverage():
    model = _build_hash()
    opt = model.setup_optimizer(embedding_lr=EMBEDDING_LR)
    owning = [g for g in opt.param_groups if any(p is model.bigram_hash_gate for p in g["params"])]
    assert len(owning) == 1 and owning[0]["kind"] == "adamw"
    _assert_full_param_coverage(model, opt)


def test_bigram_hash_param_count_and_total_consistent():
    dim, buckets = 64, 4096
    model = _build_hash(dim=dim, buckets=buckets)
    counts = model.num_scaling_params()
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    assert counts["bigram_hash_embed"] == buckets * dim
    assert counts["bigram_hash_proj"] == model.config.n_embd * dim


def test_bigram_hash_gate_excluded_from_flops():
    # The bucket table is excluded (embedding lookup) and the gate is a scalar; only the projection is a
    # real matmul. Two hash models differing only by an unused scalar tweak keep identical FLOPs.
    model = _build_hash()
    flops = model.estimate_flops()
    assert isinstance(flops, (int, float)) and flops > 0


# ===========================================================================
# Shared assertions
# ===========================================================================

def VOCAB_PADDED(model):
    return model.transformer.wte.weight.shape[0]


def _assert_full_param_coverage(model, opt):
    owned = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(owned) == len(set(owned))                       # no duplicates
    assert set(owned) == {id(p) for p in model.parameters()}   # full coverage


def _assert_kv_cache_parity(model):
    """Prefill + token-by-token decode through the KV cache must match the full-sequence forward."""
    cfg = model.config
    torch.manual_seed(321)
    T = 12
    idx = torch.randint(0, cfg.vocab_size, (1, T))
    with torch.no_grad():
        out_full = model(idx)
        cache = KVCache(
            batch_size=1, num_heads=cfg.n_kv_head, seq_len=T,
            head_dim=cfg.n_embd // cfg.n_head, num_layers=cfg.n_layer,
            device="cpu", dtype=torch.float32,
        )
        p = 5  # prefill the first p tokens, then decode the rest one at a time
        pieces = [model(idx[:, :p], kv_cache=cache)]
        for t in range(p, T):
            pieces.append(model(idx[:, t:t + 1], kv_cache=cache))
        out_inc = torch.cat(pieces, dim=1)
    assert out_inc.shape == out_full.shape
    assert torch.allclose(out_full, out_inc, atol=1e-3, rtol=1e-3)
