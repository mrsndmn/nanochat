"""
Tests for the CORE-reliability improvements (linear-projection-embeddings investigation).

The root cause of the flat-CORE artifact was that CORE was read from a step-only results
file shared across variants. These tests cover the pieces that fix and guard that:

- nanochat.loss_eval.evaluate_bpb(..., return_stats=True): val_bpb AND raw loss are reported.
- nanochat.core_eval: the few-shot seed is threaded through task_meta.
- scripts.results: CORE is read per-(model_tag, step) with mean/std, so two variants that
  finish at the same step get DISTINCT numbers (the regression this whole stage is about).
- scripts.jobs.run_evaluation._has_eval_results: seed-aware skip logic.

These intentionally avoid running the training/eval scripts end-to-end; they exercise the
library helpers and the results-aggregation logic with synthetic artifacts on disk.

Run: python -m pytest tests/test_eval_reliability.py -v
"""
import json
import math

import torch

from nanochat.loss_eval import evaluate_bpb
from nanochat.core_eval import evaluate_example
import scripts.results as results
import scripts.jobs.run_evaluation as run_eval


# =============================================================================
# loss_eval.evaluate_bpb return_stats -> continuous (bpb, loss)
# =============================================================================
class _ConstLossModel:
    """Fake model whose per-token loss is always 1.0 nat, for a deterministic bpb/loss."""

    def __init__(self, device="cpu"):
        self._device = torch.device(device)

    def get_device(self):
        return self._device

    def __call__(self, x, y, loss_reduction="none"):
        return torch.ones_like(y, dtype=torch.float32)


def test_evaluate_bpb_return_stats_basic():
    # token_bytes: token 0 is special (0 bytes), tokens 1..3 are 2 bytes each.
    token_bytes = torch.tensor([0, 2, 2, 2], dtype=torch.int64)
    y = torch.tensor([[1, 2, 3]])
    x = torch.zeros_like(y)
    stats = evaluate_bpb(_ConstLossModel(), [(x, y)], steps=1, token_bytes=token_bytes, return_stats=True)
    # 3 counted tokens, 1 nat each -> mean loss 1.0; 6 bytes -> bpb = 3 / (ln2 * 6).
    assert stats["total_tokens"] == 3
    assert stats["total_bytes"] == 6
    assert math.isclose(stats["loss"], 1.0, rel_tol=1e-6)
    assert math.isclose(stats["bpb"], 3.0 / (math.log(2) * 6), rel_tol=1e-6)


def test_evaluate_bpb_ignores_masked_targets():
    token_bytes = torch.tensor([0, 2, 2, 2], dtype=torch.int64)
    y = torch.tensor([[1, -1, 3]])  # middle target is ignore_index
    x = torch.zeros_like(y)
    stats = evaluate_bpb(_ConstLossModel(), [(x, y)], steps=1, token_bytes=token_bytes, return_stats=True)
    # Only 2 tokens counted (the -1 is masked out).
    assert stats["total_tokens"] == 2
    assert stats["total_bytes"] == 4
    assert math.isclose(stats["loss"], 1.0, rel_tol=1e-6)


def test_evaluate_bpb_scalar_backward_compatible():
    """Without return_stats the function still returns just the scalar bpb (unchanged API)."""
    token_bytes = torch.tensor([0, 2, 2, 2], dtype=torch.int64)
    y = torch.tensor([[1, 2, 3]])
    x = torch.zeros_like(y)
    bpb = evaluate_bpb(_ConstLossModel(), [(x, y)], steps=1, token_bytes=token_bytes)
    assert isinstance(bpb, float)
    assert math.isclose(bpb, 3.0 / (math.log(2) * 6), rel_tol=1e-6)


# =============================================================================
# core_eval: few-shot seed is threaded through task_meta
# =============================================================================
class _RecordingTokenizer:
    """Minimal tokenizer that records which prompts get rendered (never actually forwards)."""

    def get_bos_token_id(self):
        return 0


def test_core_eval_fewshot_seed_changes_selection():
    """Different fewshot_seed -> different few-shot draw; same seed -> identical draw.

    We stop before the model forward by raising from a sentinel, capturing the rendered
    few-shot indices via monkeypatched render. This verifies the seed actually drives
    selection (the basis for multi-seed CORE).
    """
    import nanochat.core_eval as ce

    captured = {}

    def fake_render_mc(item, delim, fewshot_examples):
        captured["fewshot"] = [ex["_id"] for ex in fewshot_examples]
        raise StopIteration  # bail out before tokenization/model forward

    orig = ce.render_prompts_mc
    ce.render_prompts_mc = fake_render_mc
    try:
        data = [{"_id": i, "gold": 0} for i in range(20)]
        task_meta_a = {"task_type": "multiple_choice", "num_fewshot": 3,
                       "continuation_delimiter": " ", "fewshot_seed": 1234 + 1337}
        task_meta_b = {"task_type": "multiple_choice", "num_fewshot": 3,
                       "continuation_delimiter": " ", "fewshot_seed": 1234 + 2024}

        def draw(task_meta):
            try:
                evaluate_example(0, model=None, tokenizer=_RecordingTokenizer(),
                                 data=data, device="cpu", task_meta=task_meta)
            except StopIteration:
                pass
            return list(captured["fewshot"])

        a1 = draw(task_meta_a)
        a2 = draw(task_meta_a)
        b1 = draw(task_meta_b)
        assert a1 == a2, "same seed must give identical few-shot selection"
        assert a1 != b1, "different seed must change few-shot selection"
    finally:
        ce.render_prompts_mc = orig


# =============================================================================
# results.py: per-(model_tag, step) CORE with mean/std; no cross-variant sharing
# =============================================================================
def _write_checkpoint(root, tag, step, val_bpb, core_dict=None, n_layer=12, n_embd=768):
    ckpt = root / "base_checkpoints" / tag
    ckpt.mkdir(parents=True, exist_ok=True)
    meta = {
        "step": step,
        "val_bpb": val_bpb,
        "model_config": {"n_layer": n_layer, "n_embd": n_embd},
        "user_config": {"depth": n_layer},
    }
    (ckpt / f"meta_{step:06d}.json").write_text(json.dumps(meta))
    if core_dict is not None:
        evald = ckpt / "evaluation"
        evald.mkdir(exist_ok=True)
        record = {"step": step, "bpb": {"val": val_bpb}, "core": core_dict}
        (evald / f"eval_{step:06d}.json").write_text(json.dumps(record))
    return ckpt


def test_read_core_from_json_mean_std(tmp_path):
    ckpt = _write_checkpoint(
        tmp_path, "d12_proj_512", 2520, 1.7289,
        core_dict={"core_metric_mean": 0.0623, "core_metric_std": 0.0011, "num_seeds": 3,
                   "per_seed": {"1337": 0.061, "2024": 0.063, "7": 0.0629}},
    )
    mean, std, n = results._read_core_from_json(ckpt, 2520)
    assert math.isclose(mean, 0.0623, rel_tol=1e-9)
    assert math.isclose(std, 0.0011, rel_tol=1e-9)
    assert n == 3


def test_distinct_variants_get_distinct_core(tmp_path):
    """Regression test for the flat-CORE artifact: two variants finishing at the SAME step
    must read DISTINCT CORE from their own per-tag JSON files."""
    _write_checkpoint(tmp_path, "d12_baseline_aaaa", 2520, 1.7877,
                      core_dict={"core_metric_mean": 0.0601, "core_metric_std": 0.0008, "num_seeds": 3})
    _write_checkpoint(tmp_path, "d12_proj_512_bbbb", 2520, 1.7289,
                      core_dict={"core_metric_mean": 0.0644, "core_metric_std": 0.0009, "num_seeds": 3})

    rows = results.collect_rows(tmp_path, model_filter=None)
    by_tag = {r[0]: r for r in rows}
    # headers: model, step, val_bpb, CORE, CORE_std, n_layer, n_embd
    assert by_tag["d12_baseline_aaaa"][3] == "0.0601"
    assert by_tag["d12_proj_512_bbbb"][3] == "0.0644"
    assert by_tag["d12_baseline_aaaa"][3] != by_tag["d12_proj_512_bbbb"][3]
    # std column populated for multi-seed runs
    assert by_tag["d12_proj_512_bbbb"][4] == "0.0009"


def test_core_csv_fallback_is_per_tag(tmp_path):
    """When no JSON exists, CORE falls back to the per-(model_tag, step) CSV, not a shared one."""
    _write_checkpoint(tmp_path, "d12_proj_256_cccc", 2520, 1.7401, core_dict=None)
    base_eval_dir = tmp_path / "base_eval"
    base_eval_dir.mkdir(parents=True, exist_ok=True)
    (base_eval_dir / "d12_proj_256_cccc_002520.csv").write_text(
        "Task                               , Accuracy  , Centered  \n"
        "CORE                               ,           , 0.058000  \n"
    )
    stats = results._read_core_from_csv(tmp_path, "d12_proj_256_cccc", 2520)
    assert stats is not None
    assert math.isclose(stats[0], 0.058, rel_tol=1e-6)


# =============================================================================
# run_evaluation._has_eval_results: seed-aware skip logic
# =============================================================================
def _write_eval_json(tmp_path, step, modes, seeds):
    ckpt = tmp_path / "d12_x"
    evald = ckpt / "evaluation"
    evald.mkdir(parents=True, exist_ok=True)
    data = {"seeds": seeds}
    for m in modes:
        data[m] = {} if m == "core" else {"val": 1.0}
    (evald / f"eval_{step:06d}.json").write_text(json.dumps(data))
    return ckpt


def test_has_eval_results_requires_all_seeds(tmp_path):
    ckpt = _write_eval_json(tmp_path, 2520, modes=["core", "bpb"], seeds=[1337])
    # Existing single-seed result satisfies a single-seed request...
    assert run_eval._has_eval_results(ckpt, 2520, {"core", "bpb"}, [1337]) is True
    # ...but NOT a request for additional seeds (must re-run, not skip stale).
    assert run_eval._has_eval_results(ckpt, 2520, {"core", "bpb"}, [1337, 2024]) is False


def test_has_eval_results_requires_all_modes(tmp_path):
    ckpt = _write_eval_json(tmp_path, 2520, modes=["bpb"], seeds=[1337])
    assert run_eval._has_eval_results(ckpt, 2520, {"core", "bpb"}, [1337]) is False
    assert run_eval._has_eval_results(ckpt, 2520, {"bpb"}, [1337]) is True


def test_has_eval_results_missing_file(tmp_path):
    ckpt = tmp_path / "no_evaluation"
    ckpt.mkdir()
    assert run_eval._has_eval_results(ckpt, 2520, {"core", "bpb"}, [1337]) is False


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
