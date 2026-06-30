"""
Tests for the component-3 analysis (nanochat/polysemy_analysis.py) and the probe helpers.

Covers gap(L) computation, the decision rule (resolved vs decaying vs growing), the
lexical-vs-total H_m decomposition, BPC-vs-floor, the probe summary, and the probe's pure
fitting/bucketing helpers.

Run: PYTHONPATH=. pytest -q tests/test_polysemy_analysis.py
"""

import math

import numpy as np
import pytest

from nanochat.polysemy_analysis import (
    bits_per_token, bpc_vs_floor, decide, decide_all, gap_curve,
    lexical_hm_decomposition, perplexity, probe_resolution_summary,
)

LN2 = math.log(2)


def test_perplexity_and_bpc():
    assert perplexity(0.0) == pytest.approx(1.0)
    assert perplexity(math.log(3)) == pytest.approx(3.0)
    assert bits_per_token(LN2) == pytest.approx(1.0)


def test_gap_curve_against_mono():
    cells = {"mono": {8: 2.0, 128: 2.0}, "poly": {8: 2.5, 128: 2.1}}
    gc = gap_curve(cells)
    assert set(gc) == {"poly"}
    assert gc["poly"][8]["gap_ppl"] == pytest.approx(math.exp(2.5) - math.exp(2.0))
    assert gc["poly"][128]["gap_bpc"] == pytest.approx(bits_per_token(2.1) - bits_per_token(2.0))


def test_gap_curve_requires_mono():
    with pytest.raises(AssertionError):
        gap_curve({"poly": {8: 2.5}}, mono_slug="mono")


def test_gap_curve_intersects_L():
    # only L present in BOTH condition and baseline is reported
    cells = {"mono": {8: 2.0}, "poly": {8: 2.5, 128: 2.1}}
    gc = gap_curve(cells)
    assert set(gc["poly"]) == {8}


def test_decision_rule_resolved_decaying_growing():
    # resolved: large decay to ~0
    resolved = {8: {"gap_ppl": 4.0}, 32: {"gap_ppl": 1.5}, 128: {"gap_ppl": 0.5}, 512: {"gap_ppl": 0.05}}
    assert decide(resolved)["verdict"] == "resolved"
    # decaying: shrinks but plateaus well above 0
    decaying = {8: {"gap_ppl": 4.0}, 32: {"gap_ppl": 3.2}, 128: {"gap_ppl": 2.8}, 512: {"gap_ppl": 2.6}}
    assert decide(decaying)["verdict"] == "decaying"
    # growing: increases with L
    growing = {8: {"gap_ppl": 1.0}, 128: {"gap_ppl": 2.0}, 512: {"gap_ppl": 3.0}}
    assert decide(growing)["verdict"] == "growing"
    # insufficient
    assert decide({8: {"gap_ppl": 1.0}})["verdict"] == "insufficient_L"


def test_decide_all_keys():
    gc = {"a": {8: {"gap_ppl": 4.0}, 512: {"gap_ppl": 0.05}}}
    out = decide_all(gc)
    assert set(out) == {"a"} and out["a"]["verdict"] in {"resolved", "decaying", "flat", "growing"}


def test_lexical_hm_decomposition():
    md = {"h_m_bits": {"forms_total": {"1": 5.0, "2": 3.0}, "senses_syntactic": {"1": 4.0, "2": 2.6}}}
    d = lexical_hm_decomposition(md)
    assert d[1]["lexical"] == pytest.approx(1.0)
    assert d[2]["lexical"] == pytest.approx(0.4)


def test_lexical_decomposition_missing_syntactic():
    md = {"h_m_bits": {"forms_total": {"1": 5.0}}}  # no syntactic block
    d = lexical_hm_decomposition(md)
    assert d[1]["syntactic"] is None and d[1]["lexical"] is None


def test_bpc_vs_floor():
    md = {"analytic_pcfg_entropy": {"bits_per_sense": 2.0},
          "h_m_bits": {"forms_total": {"1": 4.0, "2": 2.5}}}
    cells_bpc = {"mono": {8: 2.4, 128: 2.1}}
    out = bpc_vs_floor(cells_bpc, {"mono": md})
    assert out["mono"][8]["analytic_min_bpc"] == pytest.approx(2.0)
    assert out["mono"][8]["excess_over_source_floor"] == pytest.approx(0.4)
    # empirical form entropy rate = forms_total at max m = 2.5
    assert out["mono"][128]["empirical_form_entropy_rate"] == pytest.approx(2.5)


def test_probe_resolution_summary():
    s = probe_resolution_summary({0: 0.3, 1: 0.5, 7: 0.8})
    assert s["resolves_with_context"] is True
    assert s["acc_gain"] == pytest.approx(0.5)
    flat = probe_resolution_summary({0: 0.5, 7: 0.5})
    assert flat["resolves_with_context"] is False


# --- probe pure helpers (nanochat/probe_utils.py: numpy + torch only) ---

def test_ctx_bucket():
    from nanochat.probe_utils import ctx_bucket
    assert [ctx_bucket(c) for c in (0, 1, 3, 5, 9, 600)] == [0, 1, 2, 4, 8, 512]


def test_fit_linear_probe_separable():
    from nanochat.probe_utils import fit_linear_probe
    rng = np.random.default_rng(0)
    C, D = 4, 12
    centers = rng.normal(size=(C, D)) * 3
    y = rng.integers(0, C, size=500)
    X = (centers[y] + rng.normal(size=(500, D))).astype(np.float32)
    preds = fit_linear_probe(X[:350], y[:350], X[350:], C, steps=150)
    assert (preds == y[350:]).mean() > 0.9


def test_bucket_accuracy():
    from nanochat.probe_utils import bucket_accuracy
    preds = np.array([1, 1, 0, 0]); labels = np.array([1, 0, 0, 1]); ctx = np.array([0, 0, 5, 5])
    acc, overall = bucket_accuracy(preds, labels, ctx)
    assert overall == pytest.approx(0.5)
    assert acc[0] == (0.5, 2) and acc[4] == (0.5, 2)
