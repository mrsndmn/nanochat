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

# Absolute polysemy data root on the shared artifacts store (mirrors base_dir_local below,
# so the path resolves identically in this session and inside worker containers). The
# generator (scripts.gen_polysemy_data) writes one <condition_slug>/ dir of parquet shards +
# vocab.json + metadata.json under here. Run it before launching these jobs.
POLYSEMY_DATA_ROOT = "/workspace-SR004.nfs2/d.tarasov/nanochat-artifacts/base_data_polysemy"

# The context-length sweep L (= model sequence length / dataloader row capacity-1). The
# experiment hypothesis is gap(L) = PPL_poly(L) - PPL_mono(L) -> 0 as L grows. The corpus is
# generated with long center-embedding documents (median ~3-4k tokens), so every L here
# truncates the typical document -> more context = more of the same derivation's prefix.
POLYSEMY_SEQ_LENS = (512, 1024, 2048)

# Per-L device (micro) batch size chosen so grad-accum == 1 at the fixed 32768-token global
# batch on a 4-GPU node: device_batch_size * L * 4 == 32768. Keeps the global batch (hence
# the optimization trajectory and step count) identical across L while maximizing throughput.
POLYSEMY_DEVICE_BATCH_BY_L = {512: 16, 1024: 8, 2048: 4}


def polysemy_context_experiments(data_root: str = POLYSEMY_DATA_ROOT) -> list[dict]:
    """Polysemy × Context: train one tiny LM per (condition, context-length L) cell.

    Component 2 of the synthetic-language experiment (see
    run/deep-interview/deep-interview-polysemy-context.md and experiments/polysemy_context.md).
    Each arm trains on a single synthetic condition corpus via the **identity tokenizer**
    (1 form = 1 token id; no BPE) at one context length L. Holding model + data + horizon
    fixed and sweeping only L lets component 3 read off gap(L): the polysemy perplexity
    penalty should shrink as context grows (more left-context resolves the latent sense).

    Grid = default_conditions() {mono, hsw0.5×{homonymy,overlap}, hsw1.5×{homonymy,overlap}}
    × L ∈ {8,32,128,512}. mono is the monosemous baseline (H(S|W)=0) that every gap(L) is
    measured against; |V| is held equal across conditions by the generator.

    Fixed knobs (source of truth — do not duplicate in the .md):
    - depth 6 (a small model; |V| ~ hundreds, so capacity is never the bottleneck and any
      gap reflects context-resolvability, not under-parameterization);
    - --window-pattern L: FULL attention. A sliding window would cap the usable context and
      confound the very axis we sweep, so it is disabled;
    - --total-batch-size 32768 held constant across L; per-L --device-batch-size (16/8/4 for
      512/1024/2048) gives grad-accum == 1 (optimized throughput) while keeping the global
      batch — and hence the optimization trajectory and the 10k optimization steps — identical
      across all arms;
    - 10k steps, single seed (project convention: one run per config, no multi-seed fan-out);
    - --eval-every 2500 (5 in-training val evals) so the checkpoint meta records val loss / PPL
      for the analysis; --eval-tokens left at the base default (no override); a final controlled
      BPB also comes from run_evaluation.py --eval bpb;
    - CORE + sampling disabled (English ICL / prompts are meaningless for a synthetic vocab).
    """
    experiment_slug = "polysemy-context"
    num_gpus = 4
    instance_type = "a100.4gpu"
    depth = 6
    seed = 0

    # nanochat.polysemy is the source of truth for the condition grid (slug + target H(S|W)).
    from nanochat.polysemy import default_conditions
    conditions = default_conditions()

    shared_args = [
        f"--depth {depth}",
        "--window-pattern L",            # full attention: the context-length sweep must not be capped by a window
        "--tokenizer identity",
        "--num-iterations 10000",
        "--total-batch-size 32768",      # held constant across L (global batch / optimization steps identical)
        "--eval-every 2500",             # 5 in-training val evals -> checkpoint meta carries val loss / PPL
        "--core-metric-every -1",        # CORE = English ICL: meaningless for the synthetic vocab
        "--sample-every -1",             # sample prompts are English: meaningless here
        f"--seed {seed}",
    ]

    configs = []
    for cond in conditions:
        data_dir = f"{data_root}/{cond.slug}"
        for seq_len in POLYSEMY_SEQ_LENS:
            device_bs = POLYSEMY_DEVICE_BATCH_BY_L[seq_len]  # grad-accum == 1 at the fixed global batch
            args_parts = shared_args + [
                f"--max-seq-len {seq_len}", f"--device-batch-size {device_bs}", f"--data-dir {data_dir}",
            ]
            args_str = " ".join(args_parts).strip()
            cmd_hash = hashlib.sha1(args_str.encode("utf-8")).hexdigest()[:8]
            model_tag = f"poly_{cond.slug}_L{seq_len}"
            description = (
                f"polysemy×context: cond={cond.slug} (H(S|W) target={cond.target_hsw}, "
                f"overlap={cond.overlap}) L={seq_len}, d{depth} identity-tok 10k"
            )
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
    args = build_args()

    # The MLS SDK is only needed to actually submit; --dry previews offline (and stays usable
    # in envs where the SDK's deps are unavailable). Defer the import + the client / in-progress
    # query to real launches.
    client = None
    extra_options = {"region": "<dry>"}
    in_progress_job_descs = set()
    if not args.dry:
        from mls.manager.job.utils import get_in_progress_jobs, training_job_api_from_profile
        client, extra_options = training_job_api_from_profile(args.profile)
        in_progress_job_descs = {job.get("job_desc", "") for job in get_in_progress_jobs()}

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

    author_name = args.author_name
    telegram_nick = args.telegram_nick

    jobs_planned = 0
    jobs_launched = 0
    jobs_dry = 0
    launched_jobs: List[dict] = []

    # -----------------------------------------------------------------------
    # Aggregate all experiment configs
    # -----------------------------------------------------------------------
    experiment_configs = [
        *polysemy_context_experiments(),
    ]

    # Warn (don't fail) if a condition's data dir is missing — the generator must be run
    # first (python -m scripts.gen_polysemy_data). Check the absolute shared store directly.
    data_dirs = {
        c["args"].split("--data-dir ", 1)[1].split()[0]
        for c in experiment_configs if "--data-dir " in c["args"]
    }
    for d in sorted(d for d in data_dirs if not os.path.isdir(d)):
        print(f"\033[33mWARNING: polysemy data dir not found (run scripts.gen_polysemy_data first):\033[0m {d}")

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
