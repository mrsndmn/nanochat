"""
Launch training jobs for nanochat experiments.

Experiments are defined as Python functions that return lists of config dicts.
Each config dict maps to a single `base_train.py` invocation.

Usage:
    python scripts/jobs/run_training.py --dry          # preview commands
    python scripts/jobs/run_training.py                # submit jobs
    python scripts/jobs/run_training.py --force        # re-run even if checkpoint exists
"""

import argparse
import hashlib
import json
import os
import sys
from typing import List


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------


def linear_projection_embedding_experiments() -> list[dict]:
    """Full-dataset single-epoch (<=1 epoch) 10k-step baseline-vs-proj_512 comparison.

    Tests the linear-projection embed mechanism (`--embed-proj-dim 512` on the proj arm) at
    d12 over a 10k-step horizon, with the data capped to <=1 epoch (no repetition). Two arms:
    baseline (embed_proj_dim=0, the default) and proj512. Single seed only — one config per arm
    (no multi-seed fan-out, per project convention).

    Data budget: pin all 150 train shards so 10k steps at the (unchanged) 524,288 tok/step
    global batch consume ~0.78 epoch and the loader never wraps. We deliberately do NOT pass
    --device-batch-size / --max-seq-len / --total-batch-size, so the global/effective batch
    size stays exactly as the prior d12 runs.
    """
    experiment_slug = "linear-projection-embeddings-10k-1ep"
    num_gpus = 4
    instance_type = "a100.4gpu"
    depth = 12

    # Single seed only — no multi-seed fan-out this phase.
    seed = 0

    num_train_shards = 150  # >= 118 required for >=1 epoch over 10k steps

    shared_args = [
        f"--depth {depth}",
        "--window-pattern SSSL",
        "--num-iterations 10000",
        f"--num-train-shards {num_train_shards}",
    ]

    # Two arms; the tag encodes proj dim explicitly (baseline = no projection).
    variants = [
        ("baseline", "d12 baseline (no embed projection), 10k steps, full-dataset single-epoch", []),
        ("proj512", "d12 embed_proj_dim=512, 10k steps, full-dataset single-epoch", ["--embed-proj-dim 512"]),
    ]

    configs = []
    for tag, base_description, extra_args in variants:
        args_parts = shared_args + extra_args + [f"--seed {seed}"]
        args_str = " ".join(args_parts).strip()
        cmd_hash = hashlib.sha1(args_str.encode("utf-8")).hexdigest()[:8]
        model_tag = f"d{depth}_{tag}_10k_1ep"
        configs.append({
            "args": args_str,
            "model_tag": model_tag,
            "description": base_description,
            "cmd_hash": cmd_hash,
            "instance_type": instance_type,
            "experiment_slug": experiment_slug,
            "num_gpus": num_gpus,
        })
    return configs


def sentence_attention_experiments() -> list[dict]:
    """Sentence attention: 1 full-causal baseline + 4 sentence-attention arms.

    Sentence attention replaces the causal mask with a block-causal + global-gist mask
    (a token sees its own current sentence block plus all earlier gist tokens), confined
    per-document. Gist/end-of-sentence tokens are inserted at NLTK-Punkt sentence boundaries.
    This group sweeps the number of gist tokens per boundary K in {1,4,8,16} against a
    full-causal baseline, at d12 / 10k steps / single seed.

    All arms use --window-pattern L so the comparison isolates the sentence mechanism rather
    than confounding it with nanochat's default sliding-window pattern.

    Evaluation protocol (reviewer-mandated): NO intermediate evaluation runs during training.
    --eval-every / --core-metric-every / --sample-every are all set to -1, so the 10k steps
    run uninterrupted and the model is scored ONLY at the end, by the separate post-training
    evaluation stage (run_evaluation.py -> base_eval.py) on CORE + BPB of the final checkpoint.
    Gists are excluded from bpb/nats; CORE is reference-only (CORE prompts carry no gists). See
    experiments/sentence-attention.md for the hypothesis, decision rule, and known threats.
    """
    experiment_slug = "sentence-attention"
    num_gpus = 4
    instance_type = "a100.4gpu"
    depth = 12
    seed = 0

    shared_args = [
        f"--depth {depth}",
        "--window-pattern L",
        "--num-iterations 10000",
        # Reviewer-mandated: disable ALL in-training evaluation/sampling. Evaluation is deferred
        # entirely to the post-training stage so the run is never slowed or interrupted mid-train.
        "--eval-every -1",
        "--core-metric-every -1",
        "--sample-every -1",
    ]

    # (tag, description, extra args). Baseline = full causal, no gists.
    arms = [
        ("baseline", "d12 full-causal baseline (no gists), 10k steps", []),
        ("nltk_k1", "d12 sentence-attn NLTK K=1, 10k steps", ["--gist-placement sentence_nltk", "--num-gist-tokens 1"]),
        ("nltk_k4", "d12 sentence-attn NLTK K=4, 10k steps", ["--gist-placement sentence_nltk", "--num-gist-tokens 4"]),
        ("nltk_k8", "d12 sentence-attn NLTK K=8, 10k steps", ["--gist-placement sentence_nltk", "--num-gist-tokens 8"]),
        ("nltk_k16", "d12 sentence-attn NLTK K=16, 10k steps", ["--gist-placement sentence_nltk", "--num-gist-tokens 16"]),
    ]

    configs = []
    for tag, description, extra_args in arms:
        args_parts = shared_args + extra_args + [f"--seed {seed}"]
        args_str = " ".join(args_parts).strip()
        cmd_hash = hashlib.sha1(args_str.encode("utf-8")).hexdigest()[:8]
        model_tag = f"d{depth}_sa_{tag}"
        configs.append({
            "args": args_str,
            "model_tag": model_tag,
            "description": description,
            "cmd_hash": cmd_hash,
            "instance_type": instance_type,
            "experiment_slug": experiment_slug,
            "num_gpus": num_gpus,
        })
    return configs


# ---------------------------------------------------------------------------
# CLI and job submission
# ---------------------------------------------------------------------------


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch nanochat training jobs.")

    # General execution/runtime configuration
    parser.add_argument("--profile", default="default", help="Profile name for training_job_api_from_profile.")
    parser.add_argument("--base_image", default="cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41")

    # Job description
    parser.add_argument("--author_name", default="ARKHIP (d.tarasov)", help="Author name tag for job description.")
    parser.add_argument("--telegram_nick", default="mrsndmn", help="Telegram nick for job notifications.")

    # Behavior
    parser.add_argument("--dry", action="store_true", help="Only print generated scripts, do not launch jobs.")
    parser.add_argument("--force", action="store_true", help="Run jobs even if checkpoint directory already exists.")

    return parser.parse_args()


if __name__ == "__main__":
    from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile

    args = build_args()

    workdir = os.getcwd()
    workdir = workdir.replace('/mnt/virtual_ai0001053-00054_SR004-nfs2/', '/workspace-SR004.nfs2/')

    # Persistent base dir holding the prepared tokenizer, training data and checkpoints.
    # The job's base dir is the worktree's `artifacts` symlink: nanochat.common
    # (_ensure_worktree_artifacts_symlink) points it at the absolute shared store
    # (SHARED_ARTIFACTS_DIR = /workspace-SR004.nfs2/d.tarasov/nanochat-artifacts), so the
    # symlink is auto-created for new worktrees and resolves inside worker containers
    # (which mount /workspace-SR004.nfs2, not /mnt/virtual_*). The local checkpoint-exists
    # check reads the shared store directly (absolute) so it is correct even before the
    # worktree symlink exists.
    base_dir_job = f"{workdir}/artifacts"
    base_dir_local = "/workspace-SR004.nfs2/d.tarasov/nanochat-artifacts"

    python_path = sys.executable
    env_prefix = python_path.removesuffix("/python").replace('/home/jovyan/.mlspace/envs/', '/workspace-SR004.nfs2/d.tarasov/envs/')
    print(f"env_prefix={env_prefix}")
    print(f"workdir={workdir}")
    print(f"base_dir_job={base_dir_job}")

    client, extra_options = training_job_api_from_profile(args.profile)

    author_name = args.author_name
    telegram_nick = args.telegram_nick

    in_progress_jobs = get_in_progress_jobs()
    in_progress_job_descs = {job.get("job_desc", "") for job in in_progress_jobs}

    jobs_planned = 0
    jobs_launched = 0
    jobs_dry = 0
    launched_jobs: List[dict] = []

    # -----------------------------------------------------------------------
    # Aggregate all experiment configs
    # -----------------------------------------------------------------------
    experiment_configs = [
        *linear_projection_embedding_experiments(),
        *sentence_attention_experiments(),
    ]

    for experiment_config in experiment_configs:
        jobs_planned += 1

        training_args = experiment_config["args"]
        model_tag = experiment_config["model_tag"]
        description = experiment_config["description"]
        instance_type = experiment_config["instance_type"]
        cmd_hash = experiment_config["cmd_hash"]
        experiment_slug = experiment_config["experiment_slug"]

        # Check if checkpoint already exists (in the persistent artifacts base dir)
        checkpoint_dir = os.path.join(base_dir_local, "base_checkpoints", model_tag)
        if os.path.isdir(checkpoint_dir) and not args.force:
            print(f"\033[33mSkipping: checkpoint already exists at:\033[0m {checkpoint_dir}")
            continue

        base_cmd = (
            f"cd {workdir} && ./scripts/jobs/prepare_torchrun.sh "
            f"-m scripts.base_train {training_args} --model-tag {model_tag}"
        )

        job_desc = (
            f"[nanochat/{experiment_slug}]: {description} {cmd_hash} "
            f"#{author_name} #rnd #multimodal #notify_completed @{telegram_nick}"
        )

        if job_desc in in_progress_job_descs:
            print(f"\033[33mSkipping: job already in queue:\033[0m {job_desc}")
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

        print(f"\033[32m Would launch:\033[0m {job_desc}")
        print(f"\033[90m     Command: {base_cmd}\033[0m")
        jobs_dry += 1
        if args.dry:
            continue

        result = client.run_job(payload=payload)
        jobs_launched += 1
        job_name = result.get("job_name") if isinstance(result, dict) else None
        if job_name:
            launched_jobs.append({
                "job_name": job_name,
                "job_desc": job_desc,
                "model_tag": model_tag,
            })
        print("result", result)

    if args.dry:
        print(f"\n[DRY] Total jobs planned: {jobs_planned}")
        print(f"[DRY] Jobs printed (dry): {jobs_dry}")
    else:
        print(f"\nTotal jobs planned: {jobs_planned}")
        print(f"Jobs launched: {jobs_launched}")

    out = {"jobs": launched_jobs, "launched": len(launched_jobs)}
    print("__TRAINING_JOBS_JSON__")
    print(json.dumps(out))
