"""
Analyze the Polysemy × Context experiment (component 3).

Discovers the trained polysemy checkpoints (``base_checkpoints/poly_<slug>_L<seqlen>``),
reads each one's validation cross-entropy and the generator's per-condition metadata, and
emits the hypothesis readouts:

  * PPL(L) and BPC(L) per (condition, context-length L);
  * gap(L) = PPL_poly(L) - PPL_mono(L) per polysemous condition (the headline);
  * BPC vs the analytic source-entropy floor;
  * the lexical-vs-total m-local entropy decomposition (from the metadata sidecar);
  * a per-condition decision-rule verdict on whether gap(L) decays / resolves.

The per-cell validation loss is taken from ``evaluation/bpb_<step>.json`` if present (a
controlled-budget eval written by base_eval.py), else from the checkpoint's training
``meta_<step>.json`` (the last in-training val eval). Both store nats/token (``val_loss``)
and bits/form (``val_bpb``); under the identity tokenizer these are exactly proportional.

Outputs go to ``<base_dir>/polysemy_analysis/`` (override with --out-dir): markdown report
+ CSVs (rows = conditions, columns = L, per the project's table convention). No GPU needed.

Usage:
    python -m scripts.analyze_polysemy                 # analyze all poly_* checkpoints
    python -m scripts.analyze_polysemy --out-dir runs/poly_analysis
    python -m scripts.analyze_polysemy --metric gap_bpc
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, Optional

from nanochat.common import get_base_dir
from nanochat.polysemy_analysis import (
    bits_per_token, bpc_vs_floor, decide_all, gap_curve, lexical_hm_decomposition,
    perplexity,
)

# Model-tag scheme from run_training.polysemy_context_experiments: poly_<slug>_L<seqlen>.
TAG_RE = re.compile(r"^poly_(?P<slug>.+)_L(?P<L>\d+)$")
MONO_SLUG = "mono"


def _last_step(checkpoint_dir: str) -> Optional[int]:
    steps = []
    for f in glob.glob(os.path.join(checkpoint_dir, "model_*.pt")):
        m = re.search(r"model_(\d+)\.pt", os.path.basename(f))
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else None


def _read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _cell_loss(checkpoint_dir: str, step: int) -> Optional[float]:
    """Per-token val cross-entropy (nats) for a checkpoint: prefer the base_eval bpb JSON,
    else the training meta's last val_loss. Returns None if neither has a usable value."""
    bpb = _read_json(os.path.join(checkpoint_dir, "evaluation", f"bpb_{step:06d}.json"))
    if bpb is not None:
        v = bpb.get("val_loss")
        if v is None:
            v = (bpb.get("loss") or {}).get("val")
        if v is not None:
            return float(v)
    meta = _read_json(os.path.join(checkpoint_dir, f"meta_{step:06d}.json"))
    if meta is not None and meta.get("val_loss") is not None:
        return float(meta["val_loss"])
    return None


def _condition_metadata(checkpoint_dir: str, step: int) -> Optional[dict]:
    """Load the generator metadata.json for this checkpoint's training data dir (from meta)."""
    meta = _read_json(os.path.join(checkpoint_dir, f"meta_{step:06d}.json"))
    if not meta:
        return None
    data_dir = (meta.get("user_config") or {}).get("data_dir")
    if not data_dir:
        return None
    return _read_json(os.path.join(data_dir, "metadata.json"))


def discover(checkpoints_root: str):
    """Walk poly_* checkpoints. Returns (loss_cells, bpc_cells, metadata_by_slug, rows).

    loss_cells[slug][L] = val nats/token; bpc_cells[slug][L] = bits/form; rows is a flat
    list of per-cell dicts for the CSV. Cells without a usable loss are skipped (warned)."""
    loss_cells: Dict[str, Dict[int, float]] = {}
    bpc_cells: Dict[str, Dict[int, float]] = {}
    metadata_by_slug: Dict[str, dict] = {}
    rows = []
    if not os.path.isdir(checkpoints_root):
        return loss_cells, bpc_cells, metadata_by_slug, rows
    for name in sorted(os.listdir(checkpoints_root)):
        m = TAG_RE.match(name)
        if not m:
            continue
        slug, L = m.group("slug"), int(m.group("L"))
        cdir = os.path.join(checkpoints_root, name)
        step = _last_step(cdir)
        if step is None:
            print(f"  skip {name}: no model_*.pt checkpoint")
            continue
        loss = _cell_loss(cdir, step)
        if loss is None:
            print(f"  skip {name}: no val_loss in eval/bpb JSON or training meta "
                  f"(was --eval-every disabled and no base_eval run?)")
            continue
        loss_cells.setdefault(slug, {})[L] = loss
        bpc_cells.setdefault(slug, {})[L] = bits_per_token(loss)
        if slug not in metadata_by_slug:
            md = _condition_metadata(cdir, step)
            if md:
                metadata_by_slug[slug] = md
        rows.append({"condition": slug, "L": L, "step": step,
                     "val_loss_nats": loss, "ppl": perplexity(loss), "bpc_bits": bits_per_token(loss)})
    return loss_cells, bpc_cells, metadata_by_slug, rows


# -----------------------------------------------------------------------------
# Report rendering (markdown tables: rows = conditions, columns = L)


def _matrix_table(cells: Dict[str, Dict[int, float]], title: str, fmt="{:.4f}") -> str:
    Ls = sorted({L for by_L in cells.values() for L in by_L})
    header = "| condition | " + " | ".join(f"L={L}" for L in Ls) + " |"
    sep = "|" + "---|" * (len(Ls) + 1)
    lines = [f"### {title}", "", header, sep]
    for slug in sorted(cells):
        cs = cells[slug]
        cellstrs = [fmt.format(cs[L]) if L in cs else "—" for L in Ls]
        lines.append(f"| {slug} | " + " | ".join(cellstrs) + " |")
    return "\n".join(lines) + "\n"


def _gap_table(gap_curves: Dict[str, Dict[int, dict]], key: str, title: str) -> str:
    Ls = sorted({L for by_L in gap_curves.values() for L in by_L})
    header = "| polysemous condition | " + " | ".join(f"L={L}" for L in Ls) + " |"
    sep = "|" + "---|" * (len(Ls) + 1)
    lines = [f"### {title}", "", header, sep]
    for slug in sorted(gap_curves):
        by_L = gap_curves[slug]
        cellstrs = [f"{by_L[L][key]:+.4f}" if L in by_L else "—" for L in Ls]
        lines.append(f"| {slug} | " + " | ".join(cellstrs) + " |")
    return "\n".join(lines) + "\n"


def _decomp_table(metadata_by_slug: Dict[str, dict]) -> str:
    lines = ["### Lexical-vs-total m-local entropy decomposition (bits)", "",
             "total = H_m(forms), syntactic = H_m(senses, held constant), lexical = total − syntactic", ""]
    for slug in sorted(metadata_by_slug):
        decomp = lexical_hm_decomposition(metadata_by_slug[slug])
        if not decomp:
            continue
        ms = sorted(decomp)
        lines.append(f"**{slug}**")
        lines.append("")
        lines.append("| component | " + " | ".join(f"m={m}" for m in ms) + " |")
        lines.append("|" + "---|" * (len(ms) + 1))
        for comp in ("total", "syntactic", "lexical"):
            vals = [decomp[m][comp] for m in ms]
            cellstrs = ["—" if v is None else f"{v:.4f}" for v in vals]
            lines.append(f"| {comp} | " + " | ".join(cellstrs) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def _verdict_table(verdicts: Dict[str, dict], metric: str) -> str:
    lines = [f"### Decision rule (metric = {metric})", "",
             "| polysemous condition | verdict | gap(Lmin) | gap(Lmax) | decay frac | log₂L slope |",
             "|---|---|---|---|---|---|"]
    for slug in sorted(verdicts):
        v = verdicts[slug]
        if "gap_at_Lmin" not in v:
            lines.append(f"| {slug} | {v.get('verdict','?')} | — | — | — | — |")
            continue
        slope = v.get("loglog_slope")
        lines.append(
            f"| {slug} | **{v['verdict']}** | {v['gap_at_Lmin']:+.4f} | {v['gap_at_Lmax']:+.4f} | "
            f"{v['decay_fraction']:+.3f} | {('—' if slope is None else f'{slope:+.4f}')} |")
    return "\n".join(lines) + "\n"


def _maybe_plot(gap_curves, out_dir, metric):
    """Plot gap(L) curves if matplotlib is available; return the image path or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if not gap_curves:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for slug in sorted(gap_curves):
        by_L = gap_curves[slug]
        Ls = sorted(by_L)
        ax.plot(Ls, [by_L[L][metric] for L in Ls], marker="o", label=slug)
    ax.set_xscale("log", base=2)
    ax.axhline(0.0, color="k", lw=0.7, ls="--")
    ax.set_xlabel("context length L"); ax.set_ylabel(metric)
    ax.set_title("Polysemy gap vs context length"); ax.legend(fontsize=8)
    path = os.path.join(out_dir, "gap_curves.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def write_csv(rows, path):
    cols = ["condition", "L", "step", "val_loss_nats", "ppl", "bpc_bits"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in sorted(rows, key=lambda r: (r["condition"], r["L"])):
            f.write(",".join(str(r[c]) for c in cols) + "\n")


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze the polysemy×context experiment.")
    p.add_argument("--checkpoints-root", default=None, help="base_checkpoints dir (default: <base_dir>/base_checkpoints)")
    p.add_argument("--out-dir", default=None, help="output dir (default: <base_dir>/polysemy_analysis)")
    p.add_argument("--metric", default="gap_ppl", choices=["gap_ppl", "gap_bpc"], help="gap metric for the decision rule + plot")
    p.add_argument("--mono-slug", default=MONO_SLUG, help="slug of the monosemous baseline condition")
    return p.parse_args()


def main() -> int:
    args = build_args()
    base_dir = get_base_dir()
    checkpoints_root = args.checkpoints_root or os.path.join(base_dir, "base_checkpoints")
    out_dir = args.out_dir or os.path.join(base_dir, "polysemy_analysis")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Scanning {checkpoints_root} for poly_<slug>_L<seqlen> checkpoints...")
    loss_cells, bpc_cells, metadata_by_slug, rows = discover(checkpoints_root)
    if not rows:
        print("No usable polysemy checkpoints found. Train them first "
              "(run_training.polysemy_context_experiments) and/or run base_eval --eval bpb.")
        return 1

    conds = sorted(loss_cells)
    print(f"Found {len(rows)} cells across conditions={conds}")
    has_mono = args.mono_slug in loss_cells

    ppl_cells = {s: {L: perplexity(v) for L, v in by_L.items()} for s, by_L in loss_cells.items()}
    gap_curves = gap_curve(loss_cells, mono_slug=args.mono_slug) if has_mono else {}
    verdicts = decide_all(gap_curves, metric=args.metric) if gap_curves else {}
    floor = bpc_vs_floor(bpc_cells, metadata_by_slug, mono_slug=args.mono_slug)

    # ---- assemble markdown report ----
    md = ["# Polysemy × Context — analysis (component 3)", "",
          f"Conditions: {conds}", f"Baseline (monosemous): `{args.mono_slug}`"
          + ("" if has_mono else "  ⚠️ MISSING — gap(L) not computable"), ""]
    md.append(_matrix_table(ppl_cells, "Perplexity PPL(condition, L) = exp(val loss)"))
    md.append(_matrix_table(bpc_cells, "BPC(condition, L) = bits/form (= val bpb)"))
    if gap_curves:
        md.append(_gap_table(gap_curves, "gap_ppl", "gap(L) = PPL_poly(L) − PPL_mono(L)"))
        md.append(_gap_table(gap_curves, "gap_bpc", "gap(L) in bits/form = BPC_poly(L) − BPC_mono(L)"))
        md.append(_verdict_table(verdicts, args.metric))
    floor_val = next((v[L]["analytic_min_bpc"] for v in floor.values() for L in v), None)
    md.append(f"### BPC vs analytic minimum\n\nAnalytic source-entropy floor (bits/sense, shared "
              f"across conditions): **{('n/a' if floor_val is None else f'{floor_val:.4f}')}**\n")
    md.append(_matrix_table({s: {L: d["excess_over_source_floor"] for L, d in by_L.items()
                                 if d["excess_over_source_floor"] is not None}
                             for s, by_L in floor.items() if any(d["excess_over_source_floor"] is not None for d in by_L.values())},
                            "Excess over source floor = BPC(L) − analytic_min (→ 0 with capacity+context)"))
    md.append(_decomp_table(metadata_by_slug))

    plot_path = _maybe_plot(gap_curves, out_dir, args.metric)
    if plot_path:
        md.append(f"\n![gap(L) curves]({os.path.basename(plot_path)})\n")
    else:
        md.append("\n_(matplotlib not available — see gap_curves.csv for the data.)_\n")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    write_csv(rows, os.path.join(out_dir, "cells.csv"))

    # gap CSV (rows = poly condition, columns = L)
    if gap_curves:
        gap_csv = os.path.join(out_dir, "gap_curves.csv")
        Ls = sorted({L for by_L in gap_curves.values() for L in by_L})
        with open(gap_csv, "w", encoding="utf-8") as f:
            f.write("condition," + ",".join(f"gap_ppl_L{L}" for L in Ls) + "," + ",".join(f"gap_bpc_L{L}" for L in Ls) + "\n")
            for slug in sorted(gap_curves):
                by_L = gap_curves[slug]
                gp = [f"{by_L[L]['gap_ppl']:.6f}" if L in by_L else "" for L in Ls]
                gb = [f"{by_L[L]['gap_bpc']:.6f}" if L in by_L else "" for L in Ls]
                f.write(slug + "," + ",".join(gp) + "," + ",".join(gb) + "\n")

    # ---- console summary ----
    print("\n" + "\n".join(md[: ]))
    print(f"\nReport: {report_path}")
    print(f"CSV:    {os.path.join(out_dir, 'cells.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
