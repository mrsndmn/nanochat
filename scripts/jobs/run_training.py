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
    """Multi-seed validation of the two decisive embed-projection variants.

    Narrowed (per the experiment plan's multi-seed validation phase) to the only two
    variants that drive the d12 default decision: baseline (embed_proj_dim=0) and proj_512
    (embed_proj_dim=512). The intermediate dims (128/256/1024/2048) from the original sweep
    are intentionally dropped — the single-run sweep already located 512 as the val_bpb sweet
    spot, so this phase spends its budget *confirming* that one comparison across training
    seeds rather than re-sweeping.

    Each variant is trained with multiple independent training seeds (--seed) to bound
    run-to-run (training) variance — the noise source the single-run point estimate could not
    see. One config is emitted per (variant, seed) pair, each with a unique model_tag that
    encodes both the variant and the seed (e.g. d12_baseline_s0, d12_proj512_s3) so
    checkpoints never collide and the results stage can group by variant. All other d12
    hyperparameters are identical to the prior baseline run.
    """
    experiment_group = "linear-projection-embeddings"
    experiment_slug = "linear_proj_emb"
    num_gpus = 4
    instance_type = "a100.4gpu"
    depth = 12

    # >=3 (5 preferred) independent training seeds to bound run-to-run variance.
    training_seeds = [0, 1, 2, 3, 4]

    # Shared training args (identical to the prior baseline run).
    shared_args = [
        f"--depth {depth}",
        "--window-pattern SSSL",
    ]

    # Only the two decisive variants. Baseline uses embed_proj_dim=0 (the default, omitted).
    variants = [
        ("baseline", "d12 baseline (no embed projection)", []),
        ("proj512", "d12 embed_proj_dim=512", ["--embed-proj-dim 512"]),
    ]

    configs = []
    for tag, base_description, extra_args in variants:
        for seed in training_seeds:
            args_parts = shared_args + extra_args + [f"--seed {seed}"]
            args_str = " ".join(args_parts).strip()
            cmd_hash = hashlib.sha1(args_str.encode("utf-8")).hexdigest()[:8]
            # model_tag encodes variant AND seed so checkpoints never collide and the
            # results stage can group runs by variant.
            model_tag = f"d{depth}_{tag}_s{seed}"
            configs.append({
                "args": args_str,
                "model_tag": model_tag,
                "description": f"{base_description} (seed {seed})",
                "cmd_hash": cmd_hash,
                "instance_type": instance_type,
                "experiment_slug": experiment_slug,
                "num_gpus": num_gpus,
            })
    return configs


def linear_projection_embedding_d20_experiments() -> list[dict]:
    """Depth-scaling (d20) of the decisive embed-projection comparison.

    Mirrors `linear_projection_embedding_experiments` (the d12 multi-seed function) at greater
    depth to answer the depth-scaling question from the plan: does the input-projection val_bpb
    advantage persist (or grow) at d20, or wash out as the model gains capacity? Two arms only —
    baseline (embed_proj_dim=0) and proj_512 (embed_proj_dim=512) — each trained with 3
    independent training seeds (--seed) to bound run-to-run variance → 6 runs total. val_bpb
    (mean ± std per arm) is the primary metric; CORE is reference-only this phase.

    One config is emitted per (variant, seed) pair, each with a unique model_tag encoding depth,
    proj dim, and seed (e.g. d20_proj0_s0, d20_proj512_s2) so checkpoints never collide with each
    other or with the d12 runs, and the results stage can group by variant. Node settings match
    every other job in this file (a100.4gpu / 4 GPUs — all jobs were standardized onto 4-GPU
    nodes); larger depth does not use a different node here.
    """
    experiment_group = "linear-projection-embeddings"
    experiment_slug = "linear_proj_emb"
    num_gpus = 4
    instance_type = "a100.4gpu"
    depth = 20

    # 3 independent training seeds to bound run-to-run (training) variance per arm.
    training_seeds = [0, 1, 2]

    # Shared training args (same as the d12 comparison, only depth differs).
    shared_args = [
        f"--depth {depth}",
        "--window-pattern SSSL",
    ]

    # Two arms; the tag encodes proj dim explicitly (proj0 = baseline, no projection).
    variants = [
        ("proj0", "d20 baseline (no embed projection)", []),
        ("proj512", "d20 embed_proj_dim=512", ["--embed-proj-dim 512"]),
    ]

    configs = []
    for tag, base_description, extra_args in variants:
        for seed in training_seeds:
            args_parts = shared_args + extra_args + [f"--seed {seed}"]
            args_str = " ".join(args_parts).strip()
            cmd_hash = hashlib.sha1(args_str.encode("utf-8")).hexdigest()[:8]
            # model_tag encodes depth, proj dim AND seed so checkpoints never collide and the
            # results stage can group runs by variant.
            model_tag = f"d{depth}_{tag}_s{seed}"
            configs.append({
                "args": args_str,
                "model_tag": model_tag,
                "description": f"{base_description} (seed {seed})",
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

    # Persistent base dir holding the prepared tokenizer, training data and
    # checkpoints. Worker containers resolve ~ to /home/user (no prepared data),
    # so the job must be pointed at the workspace-mounted artifacts/ directory.
    base_dir_job = f"{workdir}/artifacts"
    base_dir_local = os.path.join(os.getcwd(), "artifacts")

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
        *linear_projection_embedding_d20_experiments(),
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
