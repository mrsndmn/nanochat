"""
Tests for the full-dataset single-epoch 10k-step linear-projection comparison.

Exercises scripts.jobs.run_training.linear_projection_embedding_experiments: it must emit
exactly one training run per arm (baseline + proj_512), at a single seed and the 10k-step
horizon, capped to <=1 epoch over the expanded shards with the global batch size unchanged.

Run: python -m pytest tests/test_linear_projection_embeddings.py -v
"""

import re

from scripts.jobs.run_training import linear_projection_embedding_experiments

REQUIRED_KEYS = {
    "args", "model_tag", "description", "cmd_hash", "instance_type",
    "experiment_slug", "num_gpus",
}


def test_exactly_two_single_run_configs():
    configs = linear_projection_embedding_experiments()
    assert len(configs) == 2
    tags = [c["model_tag"] for c in configs]
    assert tags == ["d12_baseline_10k_1ep", "d12_proj512_10k_1ep"]
    assert len(set(tags)) == 2


def test_single_seed_only():
    configs = linear_projection_embedding_experiments()
    seeds = set()
    for c in configs:
        m = re.search(r"--seed (\d+)", c["args"])
        assert m, f"no --seed in args: {c['args']}"
        seeds.add(m.group(1))
    assert seeds == {"0"}


def test_ten_thousand_steps():
    configs = linear_projection_embedding_experiments()
    for c in configs:
        assert "--num-iterations 10000" in c["args"]


def test_d12_and_no_d20_or_d6():
    configs = linear_projection_embedding_experiments()
    for c in configs:
        assert "--depth 12" in c["args"]
        assert "--depth 20" not in c["args"]
        assert "--depth 6" not in c["args"]
        assert "d20" not in c["model_tag"]
        assert "d6" not in c["model_tag"]


def test_arms_are_baseline_and_proj512():
    configs = linear_projection_embedding_experiments()
    by_tag = {c["model_tag"]: c for c in configs}
    assert "--embed-proj-dim" not in by_tag["d12_baseline_10k_1ep"]["args"]
    assert "--embed-proj-dim 512" in by_tag["d12_proj512_10k_1ep"]["args"]


def test_slug():
    configs = linear_projection_embedding_experiments()
    for c in configs:
        assert c["experiment_slug"] == "linear-projection-embeddings-10k-1ep"


def test_required_keys_and_node_settings():
    configs = linear_projection_embedding_experiments()
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
    configs = linear_projection_embedding_experiments()
    for c in configs:
        m = re.search(r"--num-train-shards (\d+)", c["args"])
        assert m, f"no --num-train-shards in args: {c['args']}"
        n = int(m.group(1))
        assert n >= MIN_SHARDS_FOR_ONE_EPOCH, (
            f"{n} shards < {MIN_SHARDS_FOR_ONE_EPOCH} needed for 1 epoch over 10k steps"
        )


def test_one_epoch_budget_holds():
    configs = linear_projection_embedding_experiments()
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
    # The run must NOT touch the global/effective batch size: per-device batch, grad-accum,
    # seq-len and total-batch-size are all left at their previous (default) values.
    configs = linear_projection_embedding_experiments()
    for c in configs:
        assert "--device-batch-size" not in c["args"]
        assert "--max-seq-len" not in c["args"]
        assert "--total-batch-size" not in c["args"]
