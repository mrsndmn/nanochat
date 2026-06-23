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
    """Joint (token_t, token_{t-1}) embedding-side arms vs the reused dense baseline (d12 / 10k / <=1 epoch).

    Prior phases established that the additive per-token projection TIES the dense baseline (baseline
    val_bpb 0.8058 vs proj512 0.8066 — absorbable into wte) and that separable context terms regress
    (prevtok512 0.8099, adapter512 0.8075 — redundant with the smear gate + attention). This phase tests
    two genuinely JOINT (token_t, token_{t-1}) input-side terms — both NON-absorbable into wte and
    NON-redundant with the separable smear/attention — against the reused dense baseline:

      - Arm A — multbigram512: gated MULTIPLICATIVE joint-bigram path (`--embed-ctx-mode mult` over the
        proj512-equivalent low-dim path). The previous-token low-dim vector modulates the current-token
        low-dim embedding element-wise behind a learned scalar gate that is ZERO-init (no-op at start) with
        small-NONZERO projections (so the product trains from step 0).
      - Arm B — bigramhash512: hashed (prev, cur) bigram-identity input embedding (`--embed-bigram-hash-dim
        64`, 2^18 buckets), a pair-keyed identity term added to wte behind a small-nonzero gate.

    Arm B won the prior phase (bigramhash512 val_bpb 0.8037, -0.0021 vs the 0.8058 baseline — the best of the
    whole linear-projection line). This phase SCALES the winning hashed pair-identity path with two 1-D
    ablation sweeps around its operating point (dim=64, buckets=2^18=262144, init-std 0.005), reusing the
    existing bigramhash512 tuple as the shared center for BOTH sweeps (the center is NOT re-added as a
    duplicate, and we do NOT build the full dim x bucket grid):

      - HASH-DIM sweep (buckets fixed at 262144, init-std 0.005): dim in {32, 128, 256, 512}
        (tags bigramhash_d32 / d128 / d256 / d512) — find the best joint-embedding width for the pair term.
      - BUCKET sweep (dim fixed at 64, init-std 0.005): buckets in {2^16=65536, 2^20=1048576}
        (tags bigramhash_b16 / b20) — find the best hash bucket count (collision rate) for the pair term.

    Intent: locate the best hash-dim and bucket count for the joint (prev,cur) pair-identity term, i.e. whether
    a wider low-dim lookup or more/fewer buckets widens the -0.002 win. Arm A (multbigram512) is kept unchanged
    as a reference; it regressed (+0.0014) and is not swept.

    Arms (single seed, no multi-seed fan-out per project convention):
      - baseline: reuses the existing d12_baseline_10k_bb2 checkpoint (the launcher skips it).
      - multbigram512 / bigramhash512 + the 6 NEW sweep arms: distinct tags so they cannot collide with any
        existing checkpoint.

    NAMING NOTE: the '512' suffix on multbigram512 / bigramhash512 is a SERIES label for width-comparability
    with the earlier proj512 arm — it is NOT the literal low-dim width of the hashed path (64) nor the hash
    bucket count (262144). The NEW sweep tags instead encode the swept value directly: bigramhash_d{N} is the
    hash-dim (N in {32,128,256,512}) and bigramhash_b{K} is log2 of the bucket count (b16=2^16, b20=2^20).

    Data budget: pin 150 train shards so 10k steps at the (unchanged) 524,288 tok/step global batch consume
    <=1 epoch and the loader never wraps. We deliberately do NOT pass --device-batch-size / --max-seq-len /
    --total-batch-size, so the global/effective batch size stays exactly as the prior d12 runs.
    """
    experiment_slug = "linear-projection-embeddings-10k"
    num_gpus = 4
    instance_type = "a100.4gpu"
    depth = 12

    # Single seed only — no multi-seed fan-out this phase.
    seed = 0

    num_train_shards = 150  # >= 118 required for <=1 epoch (no wrap) over 10k steps

    shared_args = [
        f"--depth {depth}",
        "--window-pattern SSSL",
        "--num-iterations 10000",
        f"--num-train-shards {num_train_shards}",
    ]

    # Reused dense baseline (existing _bb2 checkpoint) + two NEW joint-bigram arms with distinct tags.
    variants = [
        ("baseline", "d12 baseline (no embed projection), 10k steps, full-dataset single-epoch", []),
        ("multbigram512", "d12 gated MULTIPLICATIVE joint-bigram input path embed_proj_dim=512 (mult mode), zero-init gate (no-op start) + small-nonzero projections, 10k steps, full-dataset single-epoch", ["--embed-proj-dim 512", "--embed-ctx-mode mult"]),
        ("bigramhash512", "d12 hashed bigram-identity low-dim input embedding: 2^18 buckets, 64-d, small non-zero proj/gate init (active from step 0), 10k steps, full-dataset single-epoch", ["--embed-bigram-hash-dim 64", "--embed-bigram-hash-buckets 262144", "--embed-bigram-hash-init-std 0.005"]),
        # HASH-DIM sweep around the bigramhash512 center (buckets fixed 2^18=262144, init-std 0.005): vary the low-dim joint-embedding width.
        ("bigramhash_d32", "d12 hashed bigram-identity input embedding HASH-DIM sweep: dim=32, 2^18 buckets, init-std 0.005, 10k steps, full-dataset single-epoch", ["--embed-bigram-hash-dim 32", "--embed-bigram-hash-buckets 262144", "--embed-bigram-hash-init-std 0.005"]),
        ("bigramhash_d128", "d12 hashed bigram-identity input embedding HASH-DIM sweep: dim=128, 2^18 buckets, init-std 0.005, 10k steps, full-dataset single-epoch", ["--embed-bigram-hash-dim 128", "--embed-bigram-hash-buckets 262144", "--embed-bigram-hash-init-std 0.005"]),
        ("bigramhash_d256", "d12 hashed bigram-identity input embedding HASH-DIM sweep: dim=256, 2^18 buckets, init-std 0.005, 10k steps, full-dataset single-epoch", ["--embed-bigram-hash-dim 256", "--embed-bigram-hash-buckets 262144", "--embed-bigram-hash-init-std 0.005"]),
        ("bigramhash_d512", "d12 hashed bigram-identity input embedding HASH-DIM sweep: dim=512, 2^18 buckets, init-std 0.005, 10k steps, full-dataset single-epoch", ["--embed-bigram-hash-dim 512", "--embed-bigram-hash-buckets 262144", "--embed-bigram-hash-init-std 0.005"]),
        # BUCKET sweep around the bigramhash512 center (dim fixed 64, init-std 0.005): vary the hash bucket count (collision rate).
        ("bigramhash_b16", "d12 hashed bigram-identity input embedding BUCKET sweep: 2^16=65536 buckets, 64-d, init-std 0.005, 10k steps, full-dataset single-epoch", ["--embed-bigram-hash-dim 64", "--embed-bigram-hash-buckets 65536", "--embed-bigram-hash-init-std 0.005"]),
        ("bigramhash_b20", "d12 hashed bigram-identity input embedding BUCKET sweep: 2^20=1048576 buckets, 64-d, init-std 0.005, 10k steps, full-dataset single-epoch", ["--embed-bigram-hash-dim 64", "--embed-bigram-hash-buckets 1048576", "--embed-bigram-hash-init-std 0.005"]),
    ]

    configs = []
    for tag, base_description, extra_args in variants:
        args_parts = shared_args + extra_args + [f"--seed {seed}"]
        args_str = " ".join(args_parts).strip()
        cmd_hash = hashlib.sha1(args_str.encode("utf-8")).hexdigest()[:8]
        model_tag = f"d{depth}_{tag}_10k_bb2"
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
    # (SHARED_ARTIFACTS_DIR = /workspace-SR004.nfs2/d.tarasov/nanochat-artifacts-low-dim-projection), so the
    # symlink is auto-created for new worktrees and resolves inside worker containers
    # (which mount /workspace-SR004.nfs2, not /mnt/virtual_*). The local checkpoint-exists
    # check reads the shared store directly (absolute) so it is correct even before the
    # worktree symlink exists.
    base_dir_job = f"{workdir}/artifacts"
    base_dir_local = "/workspace-SR004.nfs2/d.tarasov/nanochat-artifacts-low-dim-projection"

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
