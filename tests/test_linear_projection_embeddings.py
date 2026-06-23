"""
Tests for the joint (token_t, token_{t-1}) embedding-side 10k-step comparison.

Exercises scripts.jobs.run_training.linear_projection_embedding_experiments: it must emit exactly one
training run per arm — the REUSED dense baseline (d12_baseline_10k_bb2, an existing checkpoint the
launcher skips), the two original joint-bigram arms (d12_multbigram512_10k_bb2, d12_bigramhash512_10k_bb2),
plus the 6 NEW bigram-hash SCALING arms: a HASH-DIM sweep (dim 32/128/256/512 at fixed 2^18 buckets) and a
BUCKET sweep (2^16/2^20 buckets at fixed dim 64) around the winning bigramhash512 center — at a single seed
and the 10k-step horizon, capped to <=1 epoch over the pinned shards with the global batch size unchanged.

Run: python -m pytest tests/test_linear_projection_embeddings.py -v
"""

import re

from scripts.jobs.run_training import linear_projection_embedding_experiments

REQUIRED_KEYS = {
    "args", "model_tag", "description", "cmd_hash", "instance_type",
    "experiment_slug", "num_gpus",
}

EXPECTED_TAGS = [
    "d12_baseline_10k_bb2",
    "d12_multbigram512_10k_bb2",
    "d12_bigramhash512_10k_bb2",
    # HASH-DIM sweep (buckets fixed 2^18, init-std 0.005)
    "d12_bigramhash_d32_10k_bb2",
    "d12_bigramhash_d128_10k_bb2",
    "d12_bigramhash_d256_10k_bb2",
    "d12_bigramhash_d512_10k_bb2",
    # BUCKET sweep (dim fixed 64, init-std 0.005)
    "d12_bigramhash_b16_10k_bb2",
    "d12_bigramhash_b20_10k_bb2",
]


def test_exactly_nine_single_run_configs():
    configs = linear_projection_embedding_experiments()
    assert len(configs) == 9
    tags = [c["model_tag"] for c in configs]
    assert tags == EXPECTED_TAGS
    assert len(set(tags)) == 9  # all distinct => no checkpoint collisions


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


def test_baseline_reuses_bb2_checkpoint_and_has_no_embed_flags():
    by_tag = {c["model_tag"]: c for c in linear_projection_embedding_experiments()}
    assert "d12_baseline_10k_bb2" in by_tag  # reuse => launcher skips retraining the dense baseline
    base = by_tag["d12_baseline_10k_bb2"]["args"]
    for flag in ("--embed-proj-dim", "--embed-ctx-mode", "--embed-bigram-hash-dim",
                 "--embed-bigram-hash-buckets", "--embed-bigram-hash-init-std"):
        assert flag not in base


def test_multbigram_arm_carries_mult_flags_at_proj512_rank():
    by_tag = {c["model_tag"]: c for c in linear_projection_embedding_experiments()}
    args = by_tag["d12_multbigram512_10k_bb2"]["args"]
    # The joint multiplicative path needs the current-token low-dim path at the proj512-equivalent rank
    # plus mult mode (the previous-token path is sized to embed_proj_dim by the model).
    assert "--embed-proj-dim 512" in args
    assert "--embed-ctx-mode mult" in args
    # It is NOT the hashed-bigram mechanism.
    assert "--embed-bigram-hash-dim" not in args


def test_bigramhash_arm_carries_hash_flags():
    by_tag = {c["model_tag"]: c for c in linear_projection_embedding_experiments()}
    args = by_tag["d12_bigramhash512_10k_bb2"]["args"]
    assert "--embed-bigram-hash-dim 64" in args
    assert "--embed-bigram-hash-buckets 262144" in args
    assert "--embed-bigram-hash-init-std 0.005" in args
    # It is NOT the multiplicative mechanism.
    assert "--embed-ctx-mode" not in args
    assert "--embed-proj-dim" not in args


# --- bigram-hash scaling sweeps around the bigramhash512 center --------------

# HASH-DIM sweep: buckets fixed at 2^18=262144, init-std 0.005, vary the hash dim.
HASH_DIM_SWEEP = {
    "d12_bigramhash_d32_10k_bb2": 32,
    "d12_bigramhash_d128_10k_bb2": 128,
    "d12_bigramhash_d256_10k_bb2": 256,
    "d12_bigramhash_d512_10k_bb2": 512,
}

# BUCKET sweep: dim fixed at 64, init-std 0.005, vary the bucket count.
BUCKET_SWEEP = {
    "d12_bigramhash_b16_10k_bb2": 65536,
    "d12_bigramhash_b20_10k_bb2": 1048576,
}


def test_hash_dim_sweep_arms_carry_expected_flags():
    by_tag = {c["model_tag"]: c for c in linear_projection_embedding_experiments()}
    for tag, dim in HASH_DIM_SWEEP.items():
        assert tag in by_tag, f"missing hash-dim sweep arm {tag}"
        args = by_tag[tag]["args"]
        assert f"--embed-bigram-hash-dim {dim}" in args
        assert "--embed-bigram-hash-buckets 262144" in args  # buckets fixed at the center
        assert "--embed-bigram-hash-init-std 0.005" in args
        # Pure hashed-bigram mechanism — not multiplicative / additive projection.
        assert "--embed-ctx-mode" not in args
        assert "--embed-proj-dim" not in args


def test_bucket_sweep_arms_carry_expected_flags():
    by_tag = {c["model_tag"]: c for c in linear_projection_embedding_experiments()}
    for tag, buckets in BUCKET_SWEEP.items():
        assert tag in by_tag, f"missing bucket sweep arm {tag}"
        args = by_tag[tag]["args"]
        assert "--embed-bigram-hash-dim 64" in args  # dim fixed at the center
        assert f"--embed-bigram-hash-buckets {buckets}" in args
        assert "--embed-bigram-hash-init-std 0.005" in args
        # Pure hashed-bigram mechanism — not multiplicative / additive projection.
        assert "--embed-ctx-mode" not in args
        assert "--embed-proj-dim" not in args


def test_sweeps_do_not_duplicate_the_center():
    # The bigramhash512 center (dim=64, buckets=262144) must appear exactly once and not be
    # re-added under a sweep tag.
    by_tag = {c["model_tag"]: c for c in linear_projection_embedding_experiments()}
    center_args = by_tag["d12_bigramhash512_10k_bb2"]["args"]
    for tag in list(HASH_DIM_SWEEP) + list(BUCKET_SWEEP):
        assert by_tag[tag]["args"] != center_args, f"{tag} duplicates the bigramhash512 center"


def test_slug():
    configs = linear_projection_embedding_experiments()
    for c in configs:
        assert c["experiment_slug"] == "linear-projection-embeddings-10k"


def test_required_keys_and_node_settings():
    configs = linear_projection_embedding_experiments()
    for c in configs:
        assert REQUIRED_KEYS.issubset(c.keys())
        assert c["instance_type"] == "a100.4gpu"
        assert c["num_gpus"] == 4


# --- data budget: <= 1 epoch over the pinned shards --------------------------

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
