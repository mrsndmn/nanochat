"""
Launch evaluation jobs for nanochat trained checkpoints.

Discovers model checkpoints under $NANOCHAT_BASE_DIR/base_checkpoints/
and submits evaluation jobs (CORE metric + BPB) via the MLSpace job API.

Usage:
    python scripts/jobs/run_evaluation.py --dry              # preview commands
    python scripts/jobs/run_evaluation.py                    # submit jobs
    python scripts/jobs/run_evaluation.py --eval core        # only CORE metric
    python scripts/jobs/run_evaluation.py --model-filter d12 # only d12 models
    python scripts/jobs/run_evaluation.py --force            # re-run even if results exist
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List


EVAL_MODES = ("core", "bpb", "sample")


def _find_model_tags(base_checkpoints_dir: Path, model_filter: str = None) -> List[str]:
    """Discover model tags (directory names) under base_checkpoints/."""
    if not base_checkpoints_dir.is_dir():
        return []
    tags = []
    for entry in sorted(base_checkpoints_dir.iterdir()):
        if not entry.is_dir():
            continue
        # Must contain at least one model_*.pt file
        if not list(entry.glob("model_*.pt")):
            continue
        if model_filter and model_filter not in entry.name:
            continue
        tags.append(entry.name)
    return tags


def _find_last_step(checkpoint_dir: Path) -> int:
    """Find the highest step number from model_*.pt files."""
    steps = []
    for f in checkpoint_dir.glob("model_*.pt"):
        match = re.search(r"model_(\d+)\.pt", f.name)
        if match:
            steps.append(int(match.group(1)))
    return max(steps) if steps else -1


def _has_eval_results(checkpoint_dir: Path, step: int, eval_modes: set, seeds: List[int]) -> bool:
    """Check if evaluation results already exist for the given step and seeds.

    The canonical per-checkpoint file is evaluation/eval_{step:06d}.json (written by
    base_eval.py). To avoid skipping on stale/partial results we require: the file exists,
    it covers every requested eval mode, and (when CORE is requested) it covers every
    requested seed. This is also what prevents a shared/step-only artifact from being
    mistaken for a per-variant result.
    """
    eval_dir = checkpoint_dir / "evaluation"
    if not eval_dir.is_dir():
        return False
    results_file = eval_dir / f"eval_{step:06d}.json"
    if not results_file.exists():
        return False
    try:
        with open(results_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if not all(mode in data for mode in eval_modes):
        return False
    # When CORE is requested, require all requested seeds to be present.
    if "core" in eval_modes:
        done_seeds = set(str(s) for s in data.get("seeds", []))
        if not all(str(s) in done_seeds for s in seeds):
            return False
    return True


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch nanochat evaluation jobs.")

    # Output format
    parser.add_argument("--output", choices=("text", "json"), default="text",
                        help="Output format: text (default) or json.")

    # General execution/runtime configuration
    parser.add_argument("--profile", default="default", help="Profile name for training_job_api_from_profile.")
    parser.add_argument("--base_image", default="cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41")

    # Eval configuration
    parser.add_argument("--eval", type=str, default="core,bpb",
                        help=f"Comma-separated eval modes: {', '.join(EVAL_MODES)} (default: core,bpb)")
    parser.add_argument("--model-filter", default=None,
                        help="Only evaluate checkpoints whose model_tag contains this substring.")
    parser.add_argument("--max-per-task", type=int, default=-1,
                        help="Max examples per CORE task (-1 = all, passed to base_eval.py).")
    parser.add_argument("--seeds", type=str, default="1337,1338,1339,1340,1341",
                        help="Comma-separated eval seeds passed to base_eval.py. Default is 5 "
                             "distinct seeds so CORE gets a mean +/- std per checkpoint (eval "
                             "seeds vary only CORE few-shot sampling; val_bpb is deterministic).")

    # Job description
    parser.add_argument("--author_name", default="ARKHIP (d.tarasov)", help="Author name tag for job description.")
    parser.add_argument("--telegram_nick", default="mrsndmn", help="Telegram nick for job notifications.")

    # Behavior
    parser.add_argument("--dry", action="store_true", help="Only print generated scripts, do not launch jobs.")
    parser.add_argument("--force", action="store_true", help="Run jobs even if evaluation results already exist.")

    return parser.parse_args()


def _log(msg: str, output_json: bool) -> None:
    if output_json:
        print(msg, file=sys.stderr)
    else:
        print(msg)


if __name__ == "__main__":
    from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

    args = build_args()
    output_json = args.output == "json"

    eval_modes = set(m.strip() for m in args.eval.split(","))
    invalid = eval_modes - set(EVAL_MODES)
    if invalid:
        print(f"Invalid eval modes: {invalid}. Valid: {set(EVAL_MODES)}", file=sys.stderr)
        sys.exit(1)

    try:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    except ValueError:
        print(f"Invalid --seeds: {args.seeds!r} (must be comma-separated integers)", file=sys.stderr)
        sys.exit(1)
    if not seeds:
        print("No valid --seeds provided", file=sys.stderr)
        sys.exit(1)

    workdir = os.getcwd()
    workdir = workdir.replace('/mnt/virtual_ai0001053-00054_SR004-nfs2/', '/workspace-SR004.nfs2/')

    # Persistent base dir holding checkpoints + eval bundle. Worker containers
    # resolve ~ to /home/user (no prepared data), so point the job at the
    # workspace-mounted artifacts/ directory.
    base_dir_job = f"{workdir}/artifacts"
    base_dir_local = os.path.join(os.getcwd(), "artifacts")

    python_path = sys.executable
    env_prefix = python_path.removesuffix("/python").replace('/home/jovyan/.mlspace/envs/', '/workspace-SR004.nfs2/d.tarasov/envs/')
    _log(f"env_prefix={env_prefix}", output_json)
    _log(f"workdir={workdir}", output_json)

    client, extra_options = training_job_api_from_profile(args.profile)

    author_name = args.author_name
    telegram_nick = args.telegram_nick
    experiment_slug = "nanochat_eval"
    num_gpus = 4
    instance_type = f"a100.{num_gpus}gpu"

    in_progress_jobs = get_in_progress_jobs()
    in_progress_job_descs = {job.get("job_desc", "") for job in in_progress_jobs}

    # Discover checkpoints (in the persistent artifacts base dir)
    base_checkpoints_dir = Path(base_dir_local) / "base_checkpoints"
    model_tags = _find_model_tags(base_checkpoints_dir, model_filter=args.model_filter)

    if not model_tags:
        _log(f"No model checkpoints found under {base_checkpoints_dir}", output_json)
        if output_json:
            print(json.dumps({"jobs": [], "launched": 0}))
        sys.exit(0)

    _log(f"Found {len(model_tags)} model(s) to evaluate: {', '.join(model_tags)}", output_json)
    _log(f"Eval seeds: {seeds}", output_json)

    # Sanity check: each model_tag must resolve to a DISTINCT checkpoint dir + latest step.
    # Identical (dir, step) across tags would mean variants share weights/results — the exact
    # failure mode behind the flat-CORE artifact. Log the resolved triples and warn on any
    # collision so it is visible rather than silent.
    resolved = {}  # (resolved_dir, step) -> model_tag
    for model_tag in model_tags:
        checkpoint_dir = base_checkpoints_dir / model_tag
        step = _find_last_step(checkpoint_dir)
        key = (str(checkpoint_dir.resolve()), step)
        _log(f"  resolved: model_tag={model_tag} step={step} path={checkpoint_dir.resolve()}", output_json)
        if key in resolved:
            _log(f"\033[31mWARNING: {model_tag} resolves to the SAME (path, step) as "
                 f"{resolved[key]} — they would evaluate identical weights!\033[0m", output_json)
        else:
            resolved[key] = model_tag

    launched_jobs: List[dict] = []

    for model_tag in model_tags:
        checkpoint_dir = base_checkpoints_dir / model_tag
        step = _find_last_step(checkpoint_dir)
        if step < 0:
            _log(f"\033[33mSkipping {model_tag}: no model checkpoints found\033[0m", output_json)
            continue

        if _has_eval_results(checkpoint_dir, step, eval_modes, seeds) and not args.force:
            _log(f"\033[33mSkipping {model_tag} step {step}: eval results already exist\033[0m", output_json)
            continue

        eval_str = ",".join(sorted(eval_modes))
        extra_args = [f"--seeds {','.join(str(s) for s in seeds)}"]
        if args.max_per_task > 0:
            extra_args.append(f"--max-per-task {args.max_per_task}")
        extra_str = " ".join(extra_args)

        base_cmd = (
            f"cd {workdir} && ./scripts/jobs/prepare_torchrun.sh "
            f"-m scripts.base_eval --model-tag {model_tag} --step {step} --eval {eval_str}"
        )
        if extra_str:
            base_cmd += f" {extra_str}"

        seeds_str = ",".join(str(s) for s in seeds)
        job_desc = (
            f"[nanochat/{experiment_slug}]: Eval {eval_str} model={model_tag} step={step} seeds={seeds_str} "
            f"#{author_name} #rnd #multimodal #notify_completed @{telegram_nick}"
        )

        if job_desc in in_progress_job_descs and not args.force:
            _log(f"\033[33mSkipping: job already in queue:\033[0m {job_desc}", output_json)
            continue

        payload = {
            "script": base_cmd,
            "job_desc": job_desc,
            "env_variables": {
                "ENV_PREFIX": env_prefix,
                "WORKDIR": workdir,
                "NANOCHAT_BASE_DIR": base_dir_job,
            },
            "instance_type": instance_type,
            "region": extra_options["region"],
            "type": "binary_exp",
            "shm_size_class": "medium",
            "base_image": args.base_image,
            "n_workers": 1,
            "processes_per_worker": 1,
        }

        _log(f"\033[32mWould launch:\033[0m {job_desc}", output_json)
        _log(f"\033[90m    Command: {base_cmd}\033[0m", output_json)

        if args.dry:
            continue
        result = client.run_job(payload=payload)
        job_name = result.get("job_name") if isinstance(result, dict) else None
        launched_jobs.append({
            "job_name": job_name,
            "job_desc": job_desc,
            "model_tag": model_tag,
            "result": result,
        })
        in_progress_job_descs.add(job_desc)
        _log(f"Job launched. {result}", output_json)

    if args.dry:
        _log(f"\n[DRY] {len(model_tags)} model(s) discovered, jobs previewed above.", output_json)

    if output_json:
        jobs = [{"job_name": e.get("job_name"), "job_desc": e.get("job_desc")} for e in launched_jobs]
        print(json.dumps({"jobs": jobs, "launched": len(launched_jobs)}, ensure_ascii=False, default=str))
