"""
Tests for the linear-projection-embeddings 10k-step single-seed phase.

Exercises scripts.jobs.run_training.linear_projection_embeddings_10k_experiments: it must
emit exactly one training run per arm (baseline + proj_512), all at a single seed and a
10k-step horizon, with model tags / slug that do not collide with the prior short-horizon
multi-seed study.

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


def test_exactly_two_single_run_configs():
    configs = linear_projection_embeddings_10k_experiments()
    # Exactly two arms, one run each — no multi-seed fan-out.
    assert len(configs) == 2
    tags = [c["model_tag"] for c in configs]
    assert tags == ["d12_baseline_10k", "d12_proj512_10k"]
    # One run per config (no duplicate tags).
    assert len(set(tags)) == 2


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


def test_arms_are_baseline_and_proj512():
    configs = linear_projection_embeddings_10k_experiments()
    by_tag = {c["model_tag"]: c for c in configs}
    # baseline arm omits the projection flag entirely.
    assert "--embed-proj-dim" not in by_tag["d12_baseline_10k"]["args"]
    # proj512 arm sets embed_proj_dim=512.
    assert "--embed-proj-dim 512" in by_tag["d12_proj512_10k"]["args"]


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
