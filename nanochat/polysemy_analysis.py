"""
Metrics & analysis for the Polysemy × Context experiment (component 3).

Pure, dependency-light functions (numpy only) that turn trained-checkpoint numbers and the
generator's metadata sidecar into the readouts the hypothesis needs:

  * PPL(L) / BPC(L) per (condition, context-length L);
  * **gap(L) = PPL_poly(L) - PPL_mono(L)** — the polysemy perplexity penalty vs the
    monosemous baseline, which the hypothesis says -> 0 as L grows (context resolves the
    latent sense, so the next form becomes as predictable as in the monosemous language);
  * **BPC vs analytic minimum** — achieved bits-per-form vs the PCFG source entropy rate
    (the shared, held-constant syntactic floor) and vs the empirical form m-local entropy;
  * **lexical-vs-total H_m decomposition** — splits the form stream's m-local entropy into
    the held-constant syntactic baseline (sense stream) and the lexical contribution the
    sense->form layer adds, isolating polysemy's effect on local entropy at each order m;
  * **decision rules** — a per-condition verdict on whether gap(L) decays / resolves.

All entropies are in bits. PPL is unitless. The I/O (reading checkpoints + metadata) lives
in scripts/analyze_polysemy.py; everything here operates on plain dicts/numbers so it is
trivially unit-testable.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

LN2 = math.log(2.0)


# -----------------------------------------------------------------------------
# Per-cell scalar conversions


def perplexity(loss_nats: float) -> float:
    """Perplexity = exp(mean cross-entropy in nats/token)."""
    return float(math.exp(loss_nats))


def bits_per_token(loss_nats: float) -> float:
    """Cross-entropy in bits/token. Under the identity tokenizer (1 form = 1 token = 1
    'byte') this equals bits-per-form == the trainer's bpb, i.e. the experiment's BPC."""
    return float(loss_nats / LN2)


def loss_from_bpc(bpc_bits: float) -> float:
    """Inverse of bits_per_token: nats/token from bits/form."""
    return float(bpc_bits * LN2)


# -----------------------------------------------------------------------------
# gap(L): the polysemy perplexity penalty vs the monosemous baseline


def gap_curve(cells: Dict[str, Dict[int, float]], mono_slug: str = "mono") -> Dict[str, Dict[int, dict]]:
    """Compute gap(L) for every polysemous condition against the monosemous baseline.

    ``cells[cond_slug][L] = loss_nats`` (per-token cross-entropy of the trained model on the
    val split). Returns ``{poly_slug: {L: {ppl_poly, ppl_mono, gap_ppl, bpc_poly, bpc_mono,
    gap_bpc}}}`` for each L present in BOTH the polysemous condition and the baseline.

    gap_ppl = PPL_poly(L) - PPL_mono(L) is the headline; gap_bpc is the same penalty in
    bits/form (additive, often easier to reason about than the multiplicative PPL).
    """
    assert mono_slug in cells, f"baseline condition {mono_slug!r} not present in cells: {sorted(cells)}"
    mono = cells[mono_slug]
    out: Dict[str, Dict[int, dict]] = {}
    for slug, by_L in cells.items():
        if slug == mono_slug:
            continue
        per_L: Dict[int, dict] = {}
        for L, loss in sorted(by_L.items()):
            if L not in mono:
                continue
            ppl_poly, ppl_mono = perplexity(loss), perplexity(mono[L])
            bpc_poly, bpc_mono = bits_per_token(loss), bits_per_token(mono[L])
            per_L[L] = {
                "ppl_poly": ppl_poly,
                "ppl_mono": ppl_mono,
                "gap_ppl": ppl_poly - ppl_mono,
                "bpc_poly": bpc_poly,
                "bpc_mono": bpc_mono,
                "gap_bpc": bpc_poly - bpc_mono,
            }
        if per_L:
            out[slug] = per_L
    return out


# -----------------------------------------------------------------------------
# BPC vs the analytic minimum (the source entropy floor)


def analytic_min_bpc(metadata: dict) -> Optional[float]:
    """The PCFG source entropy rate (bits/sense) — the shared, held-constant syntactic floor.

    Because the sense stream is identical across conditions, this is the absolute
    perfect-context floor for predicting one symbol of the underlying process; the
    monosemous form stream (bijective sense<->form, no synonymy) attains it exactly. For
    polysemous conditions it is an approximate floor (merges coarsen the form, lowering it;
    the |V|-restoring synonym splits add a small irreducible term, raising it). Returns None
    if the analytic entropy was not computed (e.g. supercritical grammar).
    """
    ape = metadata.get("analytic_pcfg_entropy")
    if not ape:
        return None
    return float(ape["bits_per_sense"])


def empirical_form_entropy_rate(metadata: dict) -> Optional[float]:
    """Upper-bound estimate of the form stream's entropy rate: the m-local form entropy at
    the largest available m. Block-entropy increments H_block(m)-H_block(m-1) decrease toward
    the true entropy rate as m grows, so the largest-m value is the tightest (still upper)
    estimate the metadata offers. Per-condition (unlike analytic_min_bpc)."""
    hm = _forms_total(metadata)
    if not hm:
        return None
    max_m = max(hm)
    return float(hm[max_m])


def bpc_vs_floor(cells_bpc: Dict[str, Dict[int, float]], metadata_by_slug: Dict[str, dict],
                 mono_slug: str = "mono") -> Dict[str, Dict[int, dict]]:
    """Per (condition, L): achieved BPC, the shared analytic source floor, and the excess.

    ``cells_bpc[slug][L]`` = achieved bits/form (== val bpb under the identity tokenizer).
    Excess over floor = achieved - source_rate; it should shrink as L grows (more context
    lets the model approach the conditional entropy). The source floor is taken from the
    monosemous metadata (the sense stream is shared, so it is the same for all conditions)."""
    floor = None
    if mono_slug in metadata_by_slug:
        floor = analytic_min_bpc(metadata_by_slug[mono_slug])
    out: Dict[str, Dict[int, dict]] = {}
    for slug, by_L in cells_bpc.items():
        emp_rate = empirical_form_entropy_rate(metadata_by_slug.get(slug, {}))
        per_L = {}
        for L, bpc in sorted(by_L.items()):
            per_L[L] = {
                "bpc": float(bpc),
                "analytic_min_bpc": floor,
                "excess_over_source_floor": (float(bpc) - floor) if floor is not None else None,
                "empirical_form_entropy_rate": emp_rate,
                "excess_over_empirical_rate": (float(bpc) - emp_rate) if emp_rate is not None else None,
            }
        out[slug] = per_L
    return out


# -----------------------------------------------------------------------------
# lexical-vs-total H_m decomposition (from the generator metadata)


def _hm_block(metadata: dict, key: str) -> Dict[int, float]:
    """Pull metadata['h_m_bits'][key] as {int m: bits}. JSON stringifies int keys, so coerce."""
    hm = metadata.get("h_m_bits", {})
    block = hm.get(key, {})
    return {int(m): float(v) for m, v in block.items()}


def _forms_total(metadata: dict) -> Dict[int, float]:
    return _hm_block(metadata, "forms_total")


def lexical_hm_decomposition(metadata: dict) -> Dict[int, dict]:
    """Decompose the form stream's m-local entropy into syntactic + lexical parts.

    For each order m:
        total     = H_m of the FORM stream            (what the LM sees)
        syntactic = H_m of the SENSE stream           (held constant across conditions)
        lexical   = total - syntactic                 (the sense->form layer's contribution)

    The lexical term is the polysemy/synonymy fingerprint on local entropy. The hypothesis
    predicts it is largest at m=1 (a form alone is sense-ambiguous) and shrinks at higher m
    (neighbours resolve the sense) — the decomposition of Someya et al.'s m-local entropy.
    """
    total = _forms_total(metadata)
    syn = _hm_block(metadata, "senses_syntactic")
    out: Dict[int, dict] = {}
    for m in sorted(total):
        s = syn.get(m)
        out[m] = {
            "total": total[m],
            "syntactic": s,
            "lexical": (total[m] - s) if s is not None else None,
        }
    return out


# -----------------------------------------------------------------------------
# Decision rules


def _loglog_slope(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Least-squares slope of y vs log2(x). None if <2 points or degenerate."""
    xs = [x for x in xs if x > 0]
    if len(xs) < 2:
        return None
    lx = np.log2(np.asarray(xs, dtype=np.float64))
    yv = np.asarray(ys[: len(xs)], dtype=np.float64)
    if np.allclose(lx, lx[0]):
        return None
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, _ = np.linalg.lstsq(A, yv, rcond=None)[0]
    return float(slope)


def decide(gap_by_L: Dict[int, dict], *, resolve_frac: float = 0.5,
           plateau_frac: float = 0.25, metric: str = "gap_ppl") -> dict:
    """Classify one polysemous condition's gap(L) trajectory.

    gap_by_L is one entry of ``gap_curve`` output ({L: {gap_ppl, gap_bpc, ...}}). Verdicts:
      * ``resolved``  — gap shrinks by >= resolve_frac of its L_min value AND ends near 0
        (|gap(L_max)| <= plateau_frac * gap(L_min)). Homonymy (disjoint contexts) should
        land here: context fully removes the penalty.
      * ``decaying``  — gap shrinks with L (negative log-L slope, gap(L_max) < gap(L_min))
        but does not reach ~0. Overlapping polysemy is expected here: context helps but a
        residual H(S|W,C) remains.
      * ``flat``      — no meaningful change with L.
      * ``growing``   — gap increases with L (unexpected; flags a confound or under-training).
    Returns the verdict plus the raw slope and endpoint gaps for the report.
    """
    Ls = sorted(gap_by_L)
    if len(Ls) < 2:
        return {"verdict": "insufficient_L", "n_points": len(Ls)}
    vals = [gap_by_L[L][metric] for L in Ls]
    g_min_L, g_max_L = vals[0], vals[-1]  # at smallest L and largest L
    slope = _loglog_slope(Ls, vals)
    denom = abs(g_min_L) if abs(g_min_L) > 1e-9 else 1e-9
    decay_frac = 1.0 - (g_max_L / g_min_L) if abs(g_min_L) > 1e-9 else 0.0

    if slope is not None and slope > 0 and g_max_L > g_min_L:
        verdict = "growing"
    elif decay_frac >= resolve_frac and abs(g_max_L) <= plateau_frac * denom:
        verdict = "resolved"
    elif (slope is not None and slope < 0) and g_max_L < g_min_L:
        verdict = "decaying"
    else:
        verdict = "flat"
    return {
        "verdict": verdict,
        "metric": metric,
        "L_min": Ls[0], "L_max": Ls[-1],
        "gap_at_Lmin": g_min_L, "gap_at_Lmax": g_max_L,
        "decay_fraction": decay_frac,        # 1.0 = fully resolved, 0 = unchanged, <0 = grew
        "loglog_slope": slope,               # bits or PPL per doubling of L; negative = decay
    }


def decide_all(gap_curves: Dict[str, Dict[int, dict]], **kwargs) -> Dict[str, dict]:
    """Apply ``decide`` to every polysemous condition in a gap_curve output."""
    return {slug: decide(by_L, **kwargs) for slug, by_L in gap_curves.items()}


# -----------------------------------------------------------------------------
# Representation-probe summary (consumes scripts/probe_polysemy.py output)


def probe_resolution_summary(acc_by_bucket: Dict[int, float]) -> dict:
    """Summarize a probe's sense-decoding accuracy vs left-context length.

    ``acc_by_bucket[c]`` = accuracy of decoding the latent sense from the model's hidden
    state for tokens having c preceding tokens of context (a context bucket). Rising accuracy
    with context is the direct signal that the model uses context to resolve the sense.
    Returns first/last bucket accuracy, the gain, and the log-context slope.
    """
    buckets = sorted(acc_by_bucket)
    if not buckets:
        return {"verdict": "no_data"}
    accs = [acc_by_bucket[b] for b in buckets]
    slope = _loglog_slope([b + 1 for b in buckets], accs)  # +1 so context==0 is loggable
    return {
        "min_ctx": buckets[0], "max_ctx": buckets[-1],
        "acc_at_min_ctx": accs[0], "acc_at_max_ctx": accs[-1],
        "acc_gain": accs[-1] - accs[0],
        "logctx_slope": slope,
        "resolves_with_context": bool(slope is not None and slope > 0 and accs[-1] > accs[0]),
    }
