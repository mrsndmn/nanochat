"""
Tests for the linear-projection-embeddings MULTI-SEED validation phase.

Two pieces are exercised:

1. scripts.jobs.run_training.linear_projection_embedding_experiments: it must emit one
   training config per (variant, training_seed) pair for exactly the two decisive variants
   (baseline embed_proj_dim=0, proj512 embed_proj_dim=512), with unique model_tags that
   encode both the variant and the seed and a --seed CLI arg threaded to base_train.py.

2. The seeding mechanism --seed relies on: seeding the global torch RNG before GPT.init_weights
   makes the low-rank embedding projection init (low_dim_embed) reproducible per seed and
   distinct across seeds, while the projection (embed_proj) stays zero-initialized so the
   correction starts at zero. This is what lets distinct --seed values produce genuinely
   different training runs whose run-to-run variance can be measured.

These intentionally avoid running base_train.py / base_eval.py end-to-end.

Run: python -m pytest tests/test_multiseed_projection.py -v
"""
import re

import torch

from scripts.jobs.run_training import linear_projection_embedding_experiments


# =============================================================================
# run_training: per-(variant, seed) config expansion
# =============================================================================
def test_only_two_variants_emitted():
    configs = linear_projection_embedding_experiments()
    # Variant is the token between depth and seed in the model_tag: d12_<variant>_s<seed>.
    variants = {re.match(r"d12_(.+)_s\d+$", c["model_tag"]).group(1) for c in configs}
    assert variants == {"baseline", "proj512"}


def test_at_least_three_seeds_per_variant():
    configs = linear_projection_embedding_experiments()
    seeds_by_variant = {}
    for c in configs:
        m = re.match(r"d12_(.+)_s(\d+)$", c["model_tag"])
        seeds_by_variant.setdefault(m.group(1), set()).add(int(m.group(2)))
    for variant, seeds in seeds_by_variant.items():
        assert len(seeds) >= 3, f"{variant} has only {len(seeds)} seeds (need >=3)"
    # Both variants must use the SAME set of seeds for a paired comparison.
    assert len(set(map(frozenset, seeds_by_variant.values()))) == 1


def test_model_tags_unique():
    configs = linear_projection_embedding_experiments()
    tags = [c["model_tag"] for c in configs]
    assert len(tags) == len(set(tags)), "model_tags must be unique so checkpoints never collide"


def test_total_config_count_matches_variants_times_seeds():
    configs = linear_projection_embedding_experiments()
    seeds = {int(re.match(r"d12_.+_s(\d+)$", c["model_tag"]).group(1)) for c in configs}
    assert len(configs) == 2 * len(seeds)


def test_seed_passed_to_base_train():
    configs = linear_projection_embedding_experiments()
    for c in configs:
        seed = int(re.match(r"d12_.+_s(\d+)$", c["model_tag"]).group(1))
        assert f"--seed {seed}" in c["args"], f"{c['model_tag']} args missing matching --seed"


def test_proj512_uses_embed_proj_dim_512_and_baseline_none():
    configs = linear_projection_embedding_experiments()
    for c in configs:
        variant = re.match(r"d12_(.+)_s\d+$", c["model_tag"]).group(1)
        if variant == "proj512":
            assert "--embed-proj-dim 512" in c["args"]
        else:  # baseline
            assert "--embed-proj-dim" not in c["args"]


def test_shared_hparams_identical_across_seeds():
    """Every non-seed training arg must be identical across seeds within a variant, so the only
    thing varying is the seed (the run-to-run noise source we are trying to measure)."""
    configs = linear_projection_embedding_experiments()
    def strip_seed(args):
        return re.sub(r"\s*--seed \d+", "", args).strip()
    by_variant = {}
    for c in configs:
        variant = re.match(r"d12_(.+)_s\d+$", c["model_tag"]).group(1)
        by_variant.setdefault(variant, set()).add(strip_seed(c["args"]))
    for variant, arg_sets in by_variant.items():
        assert len(arg_sets) == 1, f"{variant} has differing non-seed args across seeds"


def test_required_config_keys_present():
    configs = linear_projection_embedding_experiments()
    required = {"args", "model_tag", "description", "cmd_hash", "instance_type",
               "experiment_slug", "num_gpus"}
    for c in configs:
        assert required <= set(c), f"missing keys: {required - set(c)}"


# =============================================================================
# Seeding mechanism: projection init reproducible per seed, distinct across seeds
# =============================================================================
def _build_tiny_gpt(embed_proj_dim, seed):
    """Build a tiny GPT on CPU with a fixed global-RNG seed, mirroring base_train's flow
    (meta build -> to_empty -> seed -> init_weights)."""
    from nanochat.gpt import GPT, GPTConfig

    config = GPTConfig(
        sequence_len=64, vocab_size=128,
        n_layer=2, n_head=2, n_kv_head=2, n_embd=64,
        window_pattern="L",
        embed_proj_dim=embed_proj_dim,
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    torch.manual_seed(seed)
    model.init_weights()
    return model


def test_projection_init_reproducible_same_seed():
    a = _build_tiny_gpt(embed_proj_dim=32, seed=0)
    b = _build_tiny_gpt(embed_proj_dim=32, seed=0)
    assert torch.equal(a.low_dim_embed.weight, b.low_dim_embed.weight)


def test_projection_init_differs_across_seeds():
    a = _build_tiny_gpt(embed_proj_dim=32, seed=0)
    b = _build_tiny_gpt(embed_proj_dim=32, seed=1)
    assert not torch.equal(a.low_dim_embed.weight, b.low_dim_embed.weight)


def test_projection_correction_starts_at_zero():
    """embed_proj is zero-initialized so the low-rank correction is a no-op at init,
    regardless of seed (the design invariant of the experiment)."""
    for seed in (0, 3):
        m = _build_tiny_gpt(embed_proj_dim=32, seed=seed)
        assert torch.count_nonzero(m.embed_proj.weight) == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
