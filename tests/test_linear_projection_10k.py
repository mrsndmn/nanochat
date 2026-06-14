"""
Tests for the linear-projection-embeddings 10k-step dimension ablation.

Exercises scripts.jobs.run_training.linear_projection_embeddings_10k_experiments: it must
emit a clean sweep over --embed-proj-dim (a dense baseline + a set of low projection dims),
exactly one run per arm, all at a single seed and a 10k-step horizon, with model tags / slug
that do not collide with each other or with the prior short-horizon multi-seed study.

Run: python -m pytest tests/test_linear_projection_10k.py -v
"""

import re

from scripts.jobs.run_training import (
    linear_projection_embedding_experiments,
    linear_projection_embeddings_10k_experiments,
)

REQUIRED_KEYS = {
    "args", "model_tag", "description", "cmd_hash", "instance_type",
    "experiment_slug", "num_gpus",
}

# The low-dim projection sweep (must match the training function).
EXPECTED_PROJ_DIMS = [64, 128, 256, 512]


def test_one_run_per_arm_baseline_plus_sweep():
    configs = linear_projection_embeddings_10k_experiments()
    # Dense baseline + one arm per swept projection dim.
    assert len(configs) == 1 + len(EXPECTED_PROJ_DIMS)
    tags = [c["model_tag"] for c in configs]
    # Exactly one run per arm — no duplicate tags.
    assert len(set(tags)) == len(tags)
    expected_tags = ["d12_baseline_10k"] + [f"d12_proj{d:03d}_10k" for d in EXPECTED_PROJ_DIMS]
    assert tags == expected_tags


def test_single_seed_only():
    configs = linear_projection_embeddings_10k_experiments()
    seeds = set()
    for c in configs:
        m = re.search(r"--seed (\d+)", c["args"])
        assert m, f"no --seed in args: {c['args']}"
        seeds.add(m.group(1))
    # Every run uses the same single seed (no multi-seed fan-out).
    assert seeds == {"0"}


def test_ten_thousand_steps():
    configs = linear_projection_embeddings_10k_experiments()
    for c in configs:
        assert "--num-iterations 10000" in c["args"]


def test_d12_and_no_d20_or_d6():
    configs = linear_projection_embeddings_10k_experiments()
    for c in configs:
        assert "--depth 12" in c["args"]
        assert "--depth 20" not in c["args"]
        assert "--depth 6" not in c["args"]
        assert "d20" not in c["model_tag"]
        assert "d6" not in c["model_tag"]


def test_baseline_has_no_projection_flag():
    configs = linear_projection_embeddings_10k_experiments()
    by_tag = {c["model_tag"]: c for c in configs}
    # dense baseline arm omits the projection flag entirely (embed_proj_dim=0).
    assert "--embed-proj-dim" not in by_tag["d12_baseline_10k"]["args"]


def test_each_projection_dim_emits_correct_flag():
    configs = linear_projection_embeddings_10k_experiments()
    by_tag = {c["model_tag"]: c for c in configs}
    for dim in EXPECTED_PROJ_DIMS:
        tag = f"d12_proj{dim:03d}_10k"
        assert tag in by_tag, f"missing arm for proj dim {dim}"
        assert f"--embed-proj-dim {dim}" in by_tag[tag]["args"]


def test_projection_dims_are_low_and_below_dense_width():
    # d12 dense embedding width is depth * aspect_ratio = 12 * 64 = 768; every swept
    # projection dim must be a low dim strictly below that to be parameter-efficient.
    dense_width = 12 * 64
    for dim in EXPECTED_PROJ_DIMS:
        assert 0 < dim < dense_width


def test_distinct_slug():
    configs = linear_projection_embeddings_10k_experiments()
    for c in configs:
        assert c["experiment_slug"] == "linear-projection-embeddings-10k"


def test_tags_do_not_collide_with_short_horizon_study():
    tenk_tags = {c["model_tag"] for c in linear_projection_embeddings_10k_experiments()}
    short_tags = {c["model_tag"] for c in linear_projection_embedding_experiments()}
    assert tenk_tags.isdisjoint(short_tags)


def test_required_keys_and_node_settings():
    configs = linear_projection_embeddings_10k_experiments()
    for c in configs:
        assert REQUIRED_KEYS.issubset(c.keys())
        assert c["instance_type"] == "a100.4gpu"
        assert c["num_gpus"] == 4


# --- data budget: <= 1 epoch over the expanded shards ------------------------

# Global batch is 524,288 tokens/step (device_batch 32 * grad_accum 2 * gpus 4 * seq 2048),
# so 10k steps train 5.243e9 tokens. With ~44.7e6 trained tokens/shard that needs >=118 shards
# for a single epoch with no wrap-around.
TOKENS_PER_STEP = 32 * 2 * 4 * 2048
MIN_SHARDS_FOR_ONE_EPOCH = 118


def test_num_train_shards_pinned_and_sufficient():
    configs = linear_projection_embeddings_10k_experiments()
    for c in configs:
        m = re.search(r"--num-train-shards (\d+)", c["args"])
        assert m, f"no --num-train-shards in args: {c['args']}"
        n = int(m.group(1))
        # Must provide at least one full epoch of data so the loader never wraps in 10k steps.
        assert n >= MIN_SHARDS_FOR_ONE_EPOCH, (
            f"{n} shards < {MIN_SHARDS_FOR_ONE_EPOCH} needed for 1 epoch over 10k steps"
        )


def test_one_epoch_budget_holds():
    # The pinned shard count must cover the full 10k-step token budget at <=1 epoch.
    configs = linear_projection_embeddings_10k_experiments()
    trained_tokens_per_shard = 44.7e6  # measured (bestfit, ~17% crop at T=2048)
    total_trained_tokens = TOKENS_PER_STEP * 10000
    for c in configs:
        n = int(re.search(r"--num-train-shards (\d+)", c["args"]).group(1))
        epoch_tokens = n * trained_tokens_per_shard
        assert total_trained_tokens <= epoch_tokens, (
            f"10k steps train {total_trained_tokens:.3e} tokens > {epoch_tokens:.3e} "
            f"available in {n} shards (would wrap past 1 epoch)"
        )


def test_global_batch_size_unchanged():
    # The data-budget change must NOT touch the global/effective batch size: per-device batch,
    # grad-accum, seq-len and total-batch-size are all left at their previous (default) values.
    configs = linear_projection_embeddings_10k_experiments()
    for c in configs:
        assert "--device-batch-size" not in c["args"]
        assert "--max-seq-len" not in c["args"]
        assert "--total-batch-size" not in c["args"]
