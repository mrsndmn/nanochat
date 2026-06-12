"""Dump evaluation results from artifacts: one row per trained checkpoint.

For each model under ``artifacts/base_checkpoints/<model_tag>/`` this picks the
latest training step and reports:

- ``val_bpb``     — validation bits-per-byte (from ``meta_<step>.json``)
- ``CORE``        — CORE metric (mean centered accuracy over ICL tasks)
- ``depth``/``n_embd`` — model size, for context

CORE is read from the canonical eval JSON
(``artifacts/base_checkpoints/<tag>/evaluation/eval_<step>.json``, written by the
eval pipeline) when present, otherwise from the per-step CSV that ``base_eval.py``
writes to ``artifacts/base_eval/base_model_<step>.csv``.

Prints a GitHub-flavoured table to stdout (consumed by the research-loop results
stage). No arguments are required; ``--artifacts`` / ``--tablefmt`` are optional.
"""

import argparse
import csv
import json
from pathlib import Path

from tabulate import tabulate


def _read_core_from_json(checkpoint_dir: Path, step: int) -> float | None:
    """Read the CORE metric from the canonical evaluation JSON, if it exists."""
    eval_file = checkpoint_dir / "evaluation" / f"eval_{step:06d}.json"
    if not eval_file.exists():
        return None
    try:
        with open(eval_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    core = data.get("core")
    if isinstance(core, dict):
        core = core.get("core_metric", core.get("metric"))
    return core if isinstance(core, (int, float)) else None


def _read_core_from_csv(artifacts_root: Path, step: int) -> float | None:
    """Read the CORE metric from base_eval's per-step CSV (the 'CORE' summary row)."""
    csv_path = artifacts_root / "base_eval" / f"base_model_{step:06d}.csv"
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, newline="") as f:
            for row in csv.reader(f):
                cells = [c.strip() for c in row]
                if cells and cells[0] == "CORE":
                    # Row layout: Task, Accuracy, Centered -> CORE value is the last cell.
                    value = cells[-1]
                    return float(value) if value else None
    except (OSError, ValueError):
        return None
    return None


def _latest_checkpoint_meta(model_dir: Path) -> tuple[int, dict] | None:
    """Return (step, meta_dict) for the highest-step meta_<step>.json in a model dir."""
    metas: list[tuple[int, Path]] = []
    for meta_path in model_dir.glob("meta_*.json"):
        try:
            step = int(meta_path.stem.split("_")[-1])
        except ValueError:
            continue
        metas.append((step, meta_path))
    if not metas:
        return None
    step, meta_path = max(metas, key=lambda t: t[0])
    try:
        with open(meta_path) as f:
            return step, json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _fmt(value: float | None, precision: int = 4) -> str:
    return f"{value:.{precision}f}" if isinstance(value, (int, float)) else ""


def collect_rows(artifacts_root: Path, model_filter: str | None) -> list[list[str]]:
    checkpoints_dir = artifacts_root / "base_checkpoints"
    if not checkpoints_dir.is_dir():
        return []

    rows: list[list[str]] = []
    for model_dir in sorted(checkpoints_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_tag = model_dir.name
        if model_filter is not None and model_filter.lower() not in model_tag.lower():
            continue

        latest = _latest_checkpoint_meta(model_dir)
        if latest is None:
            continue
        step, meta = latest

        cfg = meta.get("model_config", {})
        user_cfg = meta.get("user_config", {})
        core = _read_core_from_json(model_dir, step)
        if core is None:
            core = _read_core_from_csv(artifacts_root, step)

        rows.append([
            model_tag,
            str(step),
            _fmt(meta.get("val_bpb")),
            _fmt(core),
            str(cfg.get("n_layer", user_cfg.get("depth", ""))),
            str(cfg.get("n_embd", "")),
        ])

    return sorted(rows, key=lambda r: r[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump evaluation results for the latest checkpoint of each nanochat model.",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Path to artifacts root (default: artifacts).",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="SUBSTR",
        help="Only include models whose tag contains this substring.",
    )
    parser.add_argument(
        "--tablefmt",
        choices=("simple", "latex", "github"),
        default="github",
        help="Output table format.",
    )
    args = parser.parse_args()

    rows = collect_rows(args.artifacts.resolve(), model_filter=args.model)
    if not rows:
        print("No checkpoints found.", flush=True)
        return

    headers = ["model", "step", "val_bpb", "CORE", "n_layer", "n_embd"]
    print(
        tabulate(rows, headers=headers, tablefmt=args.tablefmt, disable_numparse=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
