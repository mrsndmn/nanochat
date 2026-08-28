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


def gist_hypernetwork_experiments() -> list[dict]:
    """Gist hypernetwork: 2 arms against the existing fixed-gist control (d12_sa_nltk_k8).

    Tests whether CONTENT-conditioned gist input embeddings beat the fixed learned gist rows
    in the strict block-causal regime, where the gist channel is the only cross-sentence path
    (ADR 0001). The hypernetwork is one masked cross-attention pass: K=8 per-slot learned
    queries attend over the completed sentence's wte embeddings (nanochat.gpt.GistHypernet).

    Two arms so a null is interpretable (ADR 0002):
      - gated:  gist embedding = fixed row + alpha_k * h(sentence), per-slot alpha zero-init
                (bit-exact equal to the control at init; alpha readout diagnoses "gate shut")
      - forced: gist embedding = h(sentence) outright (no fallback; the model MUST use content)

    Training protocol matches sentence_attention_experiments exactly (d12 / 10k steps /
    single seed / --window-pattern L / no in-training evaluation) so the only difference vs
    the d12_sa_nltk_k8 control is the --gist-hypernet flag. Decision rule and win margin:
    experiments/gist-hypernetwork.md.
    """
    experiment_slug = "gist-hypernetwork"
    num_gpus = 4
    instance_type = "a100.4gpu"
    depth = 12
    seed = 0

    shared_args = [
        f"--depth {depth}",
        "--window-pattern L",
        "--num-iterations 10000",
        # Reviewer-mandated (inherited from the sentence-attention group): no in-training eval.
        "--eval-every -1",
        "--core-metric-every -1",
        "--sample-every -1",
        "--gist-placement sentence_nltk",
        "--num-gist-tokens 8",
    ]

    arms = [
        ("hnet_gated", "d12 sentence-attn NLTK K=8 + gist hypernet GATED (fixed row + alpha*h, alpha=0 init), 10k steps", ["--gist-hypernet gated"]),
        ("hnet_forced", "d12 sentence-attn NLTK K=8 + gist hypernet FORCED (pure h(sentence), no fallback), 10k steps", ["--gist-hypernet forced"]),
    ]

    configs = []
    for tag, description, extra_args in arms:
        args_parts = shared_args + extra_args + [f"--seed {seed}"]
        args_str = " ".join(args_parts).strip()
        cmd_hash = hashlib.sha1(args_str.encode("utf-8")).hexdigest()[:8]
        model_tag = f"d{depth}_sa_nltk_k8_{tag}"
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


def gist_hypernetwork_engram_experiments() -> list[dict]:
    """Gist hypernetwork iteration 2: Engram-style sparse bigram memory in the KV stream.

    Iteration 1 found the content channel USED (final |alpha| ~ 1.15) but REDUNDANT — the
    trunk already computes sentence content into gist positions, so re-encoding context
    bought ~nothing (gated 0.81826 vs control 0.81906). This iteration injects what the
    trunk CANNOT compute from context: stored n-gram associations. A zero-init hashed
    bigram table (engram-lite recipe from dev/LOG 2026-01-27) feeds retrieved memories
    into the hypernet's key/value inputs; the K slot queries select over them.

    Four arms sweep table capacity (2**17..2**20 rows x 128 dim) on top of the GATED
    hypernet. Zero-init table => each arm is bit-exact equal to the plain gated arm at
    init. Protocol identical to iteration 1 (d12 / 10k / single seed / no in-training
    eval); the table is excluded from the scaling-params horizon computation. Decision
    rule unchanged: win = val BPB <= 0.8161 (experiments/gist-hypernetwork.md). Post-hoc
    readouts: alpha gates + engram table row-norm stats (used capacity).
    """
    experiment_slug = "gist-hypernetwork"
    num_gpus = 4
    instance_type = "a100.4gpu"
    depth = 12
    seed = 0

    shared_args = [
        f"--depth {depth}",
        "--window-pattern L",
        "--num-iterations 10000",
        # Reviewer-mandated (inherited): no in-training evaluation.
        "--eval-every -1",
        "--core-metric-every -1",
        "--sample-every -1",
        "--gist-placement sentence_nltk",
        "--num-gist-tokens 8",
        "--gist-hypernet gated",
        "--gist-engram-dim 128",
    ]

    arms = [
        ("hnet_gated_eng_b17", "d12 sentence-attn K=8 gated hypernet + engram bigram table 2^17x128, 10k steps", ["--gist-engram-bits 17"]),
        ("hnet_gated_eng_b18", "d12 sentence-attn K=8 gated hypernet + engram bigram table 2^18x128, 10k steps", ["--gist-engram-bits 18"]),
        ("hnet_gated_eng_b19", "d12 sentence-attn K=8 gated hypernet + engram bigram table 2^19x128, 10k steps", ["--gist-engram-bits 19"]),
        ("hnet_gated_eng_b20", "d12 sentence-attn K=8 gated hypernet + engram bigram table 2^20x128, 10k steps", ["--gist-engram-bits 20"]),
    ]

    configs = []
    for tag, description, extra_args in arms:
        args_parts = shared_args + extra_args + [f"--seed {seed}"]
        args_str = " ".join(args_parts).strip()
        cmd_hash = hashlib.sha1(args_str.encode("utf-8")).hexdigest()[:8]
        model_tag = f"d{depth}_sa_nltk_k8_{tag}"
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
    # (SHARED_ARTIFACTS_DIR = /workspace-SR004.nfs2/d.tarasov/nanochat-artifacts-sentence-attention), so the
    # symlink is auto-created for new worktrees and resolves inside worker containers
    # (which mount /workspace-SR004.nfs2, not /mnt/virtual_*). The local checkpoint-exists
    # check reads the shared store directly (absolute) so it is correct even before the
    # worktree symlink exists.
    base_dir_job = f"{workdir}/artifacts"
    base_dir_local = "/workspace-SR004.nfs2/d.tarasov/nanochat-artifacts-sentence-attention"

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
        *sentence_attention_experiments(),
        *gist_hypernetwork_experiments(),
        *gist_hypernetwork_engram_experiments(),
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
