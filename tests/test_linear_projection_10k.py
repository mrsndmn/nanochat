"""
Tests for the linear-projection-embeddings 10k-step projection-dimension ablation.

Exercises scripts.jobs.run_training.linear_projection_embeddings_10k_experiments: it must
emit exactly one training run per arm (baseline + one per embed-proj-dim in
{128, 256, 512, 1024}), all at a single seed and a 10k-step horizon, with model tags / slug
that encode the projection dim and do not collide with the prior short-horizon multi-seed
study.

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

PROJ_DIMS = [128, 256, 512, 1024]


def test_five_single_run_configs():
    configs = linear_projection_embeddings_10k_experiments()
    # Baseline + one arm per projection dim, one run each — no multi-seed fan-out.
    assert len(configs) == 1 + len(PROJ_DIMS)
    tags = [c["model_tag"] for c in configs]
    expected = ["d12_baseline_10k"] + [f"d12_proj{d}_10k" for d in PROJ_DIMS]
    assert tags == expected
    # One run per config (no duplicate tags).
    assert len(set(tags)) == len(tags)


def test_single_seed_only():
    configs = linear_projection_embeddings_10k_experiments()
    seeds = set()
    for c in configs:
        m = re.search(r"--seed (\d+)", c["args"])
        assert m, f"no --seed in args: {c['args']}"
        seeds.add(m.group(1))
    # Every run uses the same single seed.
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
    by_tag = {c["model_tag"]: c for c in linear_projection_embeddings_10k_experiments()}
    # baseline arm omits the projection flag entirely.
    assert "--embed-proj-dim" not in by_tag["d12_baseline_10k"]["args"]


def test_each_proj_dim_arm_sets_its_dim():
    by_tag = {c["model_tag"]: c for c in linear_projection_embeddings_10k_experiments()}
    for d in PROJ_DIMS:
        args = by_tag[f"d12_proj{d}_10k"]["args"]
        assert f"--embed-proj-dim {d}" in args


def test_only_varying_factor_is_projection_dim():
    # Strip the proj flag and seed; every arm must share identical remaining args.
    configs = linear_projection_embeddings_10k_experiments()
    stripped = set()
    for c in configs:
        s = re.sub(r"--embed-proj-dim \d+\s*", "", c["args"]).strip()
        stripped.add(s)
    assert len(stripped) == 1, f"arms differ beyond projection dim: {stripped}"


def test_intermediate_evaluation_disabled():
    # Every arm disables in-training-loop eval so evaluation runs ONLY at the end of the run
    # (final CORE + BPB are produced by the separate run_evaluation.py / base_eval.py stage).
    # base_train.py treats -1 as "disable" for each periodic hook.
    configs = linear_projection_embeddings_10k_experiments()
    for c in configs:
        assert "--eval-every -1" in c["args"], f"val-bpb eval not disabled: {c['args']}"
        assert "--core-metric-every -1" in c["args"], f"CORE eval not disabled: {c['args']}"
        assert "--sample-every -1" in c["args"], f"in-loop sampling not disabled: {c['args']}"


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
