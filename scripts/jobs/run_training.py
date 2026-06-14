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


def linear_projection_embeddings_10k_experiments() -> list[dict]:
    """10k-step, single-seed *dimension ablation* over the embed-projection width.

    Reframes the longer-horizon embed-projection study (continuation of
    `linear_projection_embedding_experiments`) from a single baseline-vs-proj_512 comparison
    into a clean **sweep of `--embed-proj-dim`** at d12 / 10k steps. The goal is to bracket the
    crossover with the dense baseline: find the smallest low-dim linear projection whose
    CORE/BPB is at least as good as the dense embedding while using fewer embedding parameters.

    Ablation grid. d12 has model_dim = depth * aspect_ratio = 12 * 64 = 768, so the dense
    embedding (wte) is vocab x 768. We sweep a small -> moderate set of projection widths that
    sit below the dense width and bracket the previously-observed proj_512 sweet spot:

        embed_proj_dim in {64, 128, 256, 512}   (small / medium / large low-dim projections)

    alongside the **dense baseline** (`embed_proj_dim=0`, no projection) as the reference arm.
    One config is emitted per dimension — exactly one run per arm.

    Single seed only: this phase deliberately drops the multi-seed fan-out (project convention:
    one training run per config). The d12 multi-seed phase already established that the
    proj_512 val_bpb advantage clears 2σ of training-seed variance, so a single seed suffices
    to read the dimension trend; no per-arm std is computed here.

    The d20 depth-scaling and d6 configs have been cancelled and are not part of this group.

    Each config carries a unique model_tag that encodes the projection dim (e.g.
    d12_baseline_10k, d12_proj064_10k, d12_proj512_10k) so the 10k checkpoints and eval
    results never collide with each other or with the prior short-horizon runs. Node settings
    match every other job in this file (a100.4gpu / 4 GPUs).
    """
    experiment_group = "linear-projection-embeddings-10k"
    # Distinct slug so this longer-horizon group never collides (in job_desc / grouping)
    # with the original short-horizon multi-seed study.
    experiment_slug = "linear-projection-embeddings-10k"
    num_gpus = 4
    instance_type = "a100.4gpu"
    depth = 12

    # Single seed only — no multi-seed fan-out this phase (see docstring).
    seed = 0

    # ------------------------------------------------------------------
    # Data budget: keep 10k steps within AT MOST one epoch (no wrap / no
    # data repetition) over the now-expanded ClimbMix shards.
    #
    # The global (effective) batch size is UNCHANGED from the prior 10k run:
    #   device_batch_size(32) * grad_accum(2) * num_gpus(4) * seq_len(2048)
    #     = total_batch_size 524,288 tokens/step  (auto-computed at d12, verified
    #       from the prior checkpoint meta). We deliberately do NOT pass
    #       --device-batch-size / --max-seq-len / --total-batch-size so all of
    #       these stay exactly as before.
    #
    # Tokens trained over the run:
    #   524,288 tokens/step * 10,000 steps = 5.243e9 trained tokens.
    # Measured trained tokens per shard (bestfit packer, ~17% crop at T=2048):
    #   ~44.7e6 tokens/shard.
    # Shards needed for >= 1 epoch:
    #   ceil(5.243e9 / 44.7e6) = 118 shards.
    # We pin all 150 available train shards (~6.7e9 trained tokens/epoch), so
    # 10k steps consume ~0.78 epoch and the loader never wraps within the run
    # (the prior run looped a tiny shard set 58x — pure memorization).
    # ------------------------------------------------------------------
    num_train_shards = 150  # >= 118 required; provides ~0.78-epoch headroom

    # Longer training horizon: 10k explicit optimization steps, capped to <=1 epoch of data.
    shared_args = [
        f"--depth {depth}",
        "--window-pattern SSSL",
        "--num-iterations 10000",
        f"--num-train-shards {num_train_shards}",
    ]

    # Low-dim projection widths to sweep (all < dense model_dim=768), bracketing the
    # crossover with the dense baseline. The baseline arm (embed_proj_dim=0) omits the flag.
    proj_dims = [64, 128, 256, 512]

    # Reference arm first, then one arm per projection dim.
    variants = [
        ("baseline", "d12 baseline (no embed projection), 10k steps", []),
    ]
    variants += [
        (
            f"proj{dim:03d}",
            f"d12 embed_proj_dim={dim}, 10k steps",
            [f"--embed-proj-dim {dim}"],
        )
        for dim in proj_dims
    ]

    configs = []
    for tag, base_description, extra_args in variants:
        args_parts = shared_args + extra_args + [f"--seed {seed}"]
        args_str = " ".join(args_parts).strip()
        cmd_hash = hashlib.sha1(args_str.encode("utf-8")).hexdigest()[:8]
        # model_tag encodes the projection dim AND the 10k horizon so checkpoints never
        # collide with each other or with the short runs.
        model_tag = f"d{depth}_{tag}_10k"
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


def linear_projection_embeddings_10k_full_experiments() -> list[dict]:
    """Fresh-tag, full-dataset single-epoch (<=1 epoch) re-run of the 10k-step comparison.

    Why this exists: the prior `linear_projection_embeddings_10k_experiments` arms reuse the
    tags `d12_baseline_10k` / `d12_proj512_10k`. Checkpoints already exist at those tags from
    the original *multi-epoch* (58-epoch) run, so the launcher skipped re-training and the
    intended single-epoch regime was never actually exercised — eval just re-read stale
    weights (see experiments/linear-projection-embeddings-10k.md, 2026-06-13 anomalies).

    This function emits the SAME experiment with DISTINCT model tags (suffix `_1ep`) so the
    full-dataset single-epoch run launches as FRESH jobs instead of colliding with the old
    checkpoints. Everything that defines the run — architecture, the linear-projection embed
    mechanism (`--embed-proj-dim 512` on the proj arm), 10k steps (`--num-iterations 10000`),
    the LR schedule, per-device batch size, gradient accumulation, sequence length, num_gpus,
    and the global/effective batch size (524,288 tok/step, auto-computed at d12) — is BYTE-FOR-
    BYTE identical to `linear_projection_embeddings_10k_experiments`. The only differences are
    (a) the distinct `_1ep` model tags and slug, and (b) the already-committed ≤1-epoch data
    extent (`--num-train-shards 150`).

    Single seed only — no multi-seed fan-out (one config per arm), consistent with the rest of
    this file and the 10k phase plan. d20/d6 remain cancelled and are not part of this group.
    """
    # Distinct slug so the fresh full-dataset run groups separately in job_desc / results from
    # the stale-tag multi-epoch run that shared the `linear-projection-embeddings-10k` slug.
    experiment_slug = "linear-projection-embeddings-10k-1ep"
    num_gpus = 4
    instance_type = "a100.4gpu"
    depth = 12

    # Single seed only — no multi-seed fan-out this phase.
    seed = 0

    # ------------------------------------------------------------------
    # Data budget: identical to the prior 10k config — pin all 150 train shards so 10k steps
    # at the (unchanged) 524,288 tok/step global batch consume ~0.78 epoch and the loader never
    # wraps. 10k * 524,288 = 5.243e9 trained tokens; ~44.7e6 trained tokens/shard => >=118
    # shards needed for one epoch; 150 provides headroom. We deliberately do NOT pass
    # --device-batch-size / --max-seq-len / --total-batch-size, so the global/effective batch
    # size stays EXACTLY as before.
    # ------------------------------------------------------------------
    num_train_shards = 150  # >= 118 required for >=1 epoch over 10k steps

    shared_args = [
        f"--depth {depth}",
        "--window-pattern SSSL",
        "--num-iterations 10000",
        f"--num-train-shards {num_train_shards}",
    ]

    # Two arms; the tag encodes proj dim explicitly (baseline = no projection).
    variants = [
        ("baseline", "d12 baseline (no embed projection), 10k steps, full-dataset single-epoch (fresh tag)", []),
        ("proj512", "d12 embed_proj_dim=512, 10k steps, full-dataset single-epoch (fresh tag)", ["--embed-proj-dim 512"]),
    ]

    configs = []
    for tag, base_description, extra_args in variants:
        args_parts = shared_args + extra_args + [f"--seed {seed}"]
        args_str = " ".join(args_parts).strip()
        cmd_hash = hashlib.sha1(args_str.encode("utf-8")).hexdigest()[:8]
        # `_1ep` suffix => distinct tag => fresh checkpoints, never colliding with the stale
        # multi-epoch `d12_*_10k` checkpoints that caused the previous run to be skipped.
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
    # Use the fresh-tag full-dataset single-epoch configs so the run launches as NEW jobs
    # instead of colliding with the stale `d12_*_10k` multi-epoch checkpoints (which made the
    # previous launch skip every job). The old function is kept for reference but no longer
    # registered — its tags already have checkpoints and would just be skipped again.
    experiment_configs = [
        *linear_projection_embeddings_10k_full_experiments(),
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
