"""
Tests for the fresh-tag, full-dataset single-epoch 10k-step phase.

Exercises scripts.jobs.run_training.linear_projection_embeddings_10k_full_experiments: it must
emit exactly one training run per arm (baseline + proj_512), at a single seed and the 10k-step
horizon, with DISTINCT `_1ep` model tags / slug so the run launches as fresh jobs instead of
colliding with the stale `d12_*_10k` multi-epoch checkpoints. The run must otherwise be
byte-for-byte identical (args) to linear_projection_embeddings_10k_experiments apart from the
model_tag.

Run: python -m pytest tests/test_linear_projection_10k_full.py -v
"""

import re

from scripts.jobs.run_training import (
    linear_projection_embedding_experiments,
    linear_projection_embeddings_10k_experiments,
    linear_projection_embeddings_10k_full_experiments,
)

REQUIRED_KEYS = {
    "args", "model_tag", "description", "cmd_hash", "instance_type",
    "experiment_slug", "num_gpus",
}


def test_exactly_two_single_run_configs():
    configs = linear_projection_embeddings_10k_full_experiments()
    assert len(configs) == 2
    tags = [c["model_tag"] for c in configs]
    assert tags == ["d12_baseline_10k_1ep", "d12_proj512_10k_1ep"]
    assert len(set(tags)) == 2


def test_single_seed_only():
    configs = linear_projection_embeddings_10k_full_experiments()
    seeds = set()
    for c in configs:
        m = re.search(r"--seed (\d+)", c["args"])
        assert m, f"no --seed in args: {c['args']}"
        seeds.add(m.group(1))
    assert seeds == {"0"}


def test_ten_thousand_steps():
    configs = linear_projection_embeddings_10k_full_experiments()
    for c in configs:
        assert "--num-iterations 10000" in c["args"]


def test_d12_and_no_d20_or_d6():
    configs = linear_projection_embeddings_10k_full_experiments()
    for c in configs:
        assert "--depth 12" in c["args"]
        assert "--depth 20" not in c["args"]
        assert "--depth 6" not in c["args"]
        assert "d20" not in c["model_tag"]
        assert "d6" not in c["model_tag"]


def test_arms_are_baseline_and_proj512():
    configs = linear_projection_embeddings_10k_full_experiments()
    by_tag = {c["model_tag"]: c for c in configs}
    assert "--embed-proj-dim" not in by_tag["d12_baseline_10k_1ep"]["args"]
    assert "--embed-proj-dim 512" in by_tag["d12_proj512_10k_1ep"]["args"]


def test_distinct_slug():
    configs = linear_projection_embeddings_10k_full_experiments()
    for c in configs:
        assert c["experiment_slug"] == "linear-projection-embeddings-10k-1ep"


def test_fresh_tags_do_not_collide_with_any_prior_study():
    full_tags = {c["model_tag"] for c in linear_projection_embeddings_10k_full_experiments()}
    short_tags = {c["model_tag"] for c in linear_projection_embedding_experiments()}
    tenk_tags = {c["model_tag"] for c in linear_projection_embeddings_10k_experiments()}
    # Must not collide with the short-horizon multi-seed study...
    assert full_tags.isdisjoint(short_tags)
    # ...nor with the stale-tag 10k multi-epoch run whose checkpoints caused the skip.
    assert full_tags.isdisjoint(tenk_tags)


def test_required_keys_and_node_settings():
    configs = linear_projection_embeddings_10k_full_experiments()
    for c in configs:
        assert REQUIRED_KEYS.issubset(c.keys())
        assert c["instance_type"] == "a100.4gpu"
        assert c["num_gpus"] == 4


def test_args_identical_to_prior_10k_run_except_tag():
    # The ONLY allowed differences from the prior 10k configs are the model_tag (and slug);
    # the training args (which fully define the run) must be byte-for-byte identical.
    full = {c["model_tag"]: c for c in linear_projection_embeddings_10k_full_experiments()}
    prior = {c["model_tag"]: c for c in linear_projection_embeddings_10k_experiments()}
    for full_tag, prior_tag in [
        ("d12_baseline_10k_1ep", "d12_baseline_10k"),
        ("d12_proj512_10k_1ep", "d12_proj512_10k"),
    ]:
        assert full[full_tag]["args"] == prior[prior_tag]["args"]


# --- data budget: <= 1 epoch over the expanded shards ------------------------

# Global batch is 524,288 tokens/step (device_batch 32 * grad_accum 2 * gpus 4 * seq 2048),
# so 10k steps train 5.243e9 tokens. With ~44.7e6 trained tokens/shard that needs >=118 shards
# for a single epoch with no wrap-around.
TOKENS_PER_STEP = 32 * 2 * 4 * 2048
MIN_SHARDS_FOR_ONE_EPOCH = 118


def test_num_train_shards_pinned_and_sufficient():
    configs = linear_projection_embeddings_10k_full_experiments()
    for c in configs:
        m = re.search(r"--num-train-shards (\d+)", c["args"])
        assert m, f"no --num-train-shards in args: {c['args']}"
        n = int(m.group(1))
        assert n >= MIN_SHARDS_FOR_ONE_EPOCH, (
            f"{n} shards < {MIN_SHARDS_FOR_ONE_EPOCH} needed for 1 epoch over 10k steps"
        )


def test_one_epoch_budget_holds():
    configs = linear_projection_embeddings_10k_full_experiments()
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
    # The fresh-tag run must NOT touch the global/effective batch size: per-device batch,
    # grad-accum, seq-len and total-batch-size are all left at their previous (default) values.
    configs = linear_projection_embeddings_10k_full_experiments()
    for c in configs:
        assert "--device-batch-size" not in c["args"]
        assert "--max-seq-len" not in c["args"]
        assert "--total-batch-size" not in c["args"]
