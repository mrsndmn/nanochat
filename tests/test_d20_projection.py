"""
Tests for the linear-projection-embeddings d20 DEPTH-SCALING phase.

Exercises scripts.jobs.run_training.linear_projection_embedding_d20_experiments: it must
emit one training config per (variant, training_seed) pair for exactly the two decisive
arms (baseline embed_proj_dim=0, proj512 embed_proj_dim=512) at depth 20, with unique
model_tags that encode depth, proj dim AND seed, a matching --seed CLI arg threaded to
base_train.py, --depth 20 on every run, and node settings consistent with the rest of the
file (a100.4gpu / 4 GPUs). Tags must not collide with the d12 multi-seed runs.

These intentionally avoid running base_train.py / base_eval.py end-to-end.

Run: python -m pytest tests/test_d20_projection.py -v
"""
import re

from scripts.jobs.run_training import (
    linear_projection_embedding_d20_experiments,
    linear_projection_embedding_experiments,
)


TAG_RE = re.compile(r"d20_(.+)_s(\d+)$")


def test_only_two_variants_emitted():
    configs = linear_projection_embedding_d20_experiments()
    variants = {TAG_RE.match(c["model_tag"]).group(1) for c in configs}
    assert variants == {"proj0", "proj512"}


def test_three_seeds_per_variant():
    configs = linear_projection_embedding_d20_experiments()
    seeds_by_variant = {}
    for c in configs:
        m = TAG_RE.match(c["model_tag"])
        seeds_by_variant.setdefault(m.group(1), set()).add(int(m.group(2)))
    for variant, seeds in seeds_by_variant.items():
        assert len(seeds) >= 3, f"{variant} has only {len(seeds)} seeds (need >=3)"
    # Both arms must use the SAME set of seeds for a paired comparison.
    assert len(set(map(frozenset, seeds_by_variant.values()))) == 1


def test_six_configs_total():
    configs = linear_projection_embedding_d20_experiments()
    assert len(configs) == 6


def test_model_tags_unique():
    configs = linear_projection_embedding_d20_experiments()
    tags = [c["model_tag"] for c in configs]
    assert len(tags) == len(set(tags)), "model_tags must be unique so checkpoints never collide"


def test_d20_tags_do_not_collide_with_d12():
    d20_tags = {c["model_tag"] for c in linear_projection_embedding_d20_experiments()}
    d12_tags = {c["model_tag"] for c in linear_projection_embedding_experiments()}
    assert d20_tags.isdisjoint(d12_tags)


def test_depth_20_on_every_run():
    configs = linear_projection_embedding_d20_experiments()
    for c in configs:
        assert "--depth 20" in c["args"], f"{c['model_tag']} missing --depth 20"


def test_seed_passed_to_base_train():
    configs = linear_projection_embedding_d20_experiments()
    for c in configs:
        seed = int(TAG_RE.match(c["model_tag"]).group(2))
        assert f"--seed {seed}" in c["args"], f"{c['model_tag']} args missing matching --seed"


def test_proj512_uses_embed_proj_dim_512_and_baseline_none():
    configs = linear_projection_embedding_d20_experiments()
    for c in configs:
        variant = TAG_RE.match(c["model_tag"]).group(1)
        if variant == "proj512":
            assert "--embed-proj-dim 512" in c["args"]
        else:  # proj0 baseline
            assert "--embed-proj-dim" not in c["args"]


def test_node_settings_consistent_4gpu():
    configs = linear_projection_embedding_d20_experiments()
    for c in configs:
        assert c["instance_type"] == "a100.4gpu"
        assert c["num_gpus"] == 4


def test_shared_hparams_identical_across_seeds():
    """Every non-seed training arg must be identical across seeds within an arm, so the only
    thing varying is the seed (the run-to-run noise source we are measuring)."""
    configs = linear_projection_embedding_d20_experiments()
    def strip_seed(args):
        return re.sub(r"\s*--seed \d+", "", args).strip()
    by_variant = {}
    for c in configs:
        variant = TAG_RE.match(c["model_tag"]).group(1)
        by_variant.setdefault(variant, set()).add(strip_seed(c["args"]))
    for variant, arg_sets in by_variant.items():
        assert len(arg_sets) == 1, f"{variant} has differing non-seed args across seeds"


def test_required_config_keys_present():
    configs = linear_projection_embedding_d20_experiments()
    required = {"args", "model_tag", "description", "cmd_hash", "instance_type",
               "experiment_slug", "num_gpus"}
    for c in configs:
        assert required <= set(c), f"missing keys: {required - set(c)}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
