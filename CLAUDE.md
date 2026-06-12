# nanochat — project guide

## Environment

- Python: `/home/jovyan/.mlspace/envs/nanochat/bin/python` (already on PATH — run `python`, `torchrun`, `pytest` directly).
- Source of truth for experiment design is **code**, not markdown. All hyperparameters, job configs, and model selection live in the training/job scripts.

## Experiments as code

Every experiment is a Python function in `scripts/jobs/run_training.py` that returns a
list of config dicts (one per `base_train.py` invocation). To add an experiment:

1. Write a `*_experiments()` function returning configs (each has `args`, `model_tag`,
   `description`, `instance_type`, `num_gpus`, `experiment_slug`).
2. Add it to the `experiment_configs` list inside `if __name__ == "__main__":`.
3. Add a plan in `experiments/<name>.md` (Hypothesis / Setup / Results / Conclusions / Changelog).

Default node: `a100.4gpu` (4 GPUs).

## Running jobs

Jobs are submitted to MLSpace via `scripts/jobs/`. The `mls` SDK is only imported at
launch time, so the experiment functions can be imported/tested without it.

### Training

```bash
# Preview the commands that would be submitted (no SDK / no submission)
python scripts/jobs/run_training.py --dry

# Submit all training jobs (skips experiments whose checkpoint already exists)
python scripts/jobs/run_training.py

# Re-run even if the checkpoint directory already exists
python scripts/jobs/run_training.py --force
```

### Evaluation

Discovers trained checkpoints under `$NANOCHAT_BASE_DIR/base_checkpoints/` (default
`~/.cache/nanochat`), picks the latest step per model, and submits `base_eval.py` jobs.

```bash
# Preview eval jobs
python scripts/jobs/run_evaluation.py --dry

# Submit eval jobs (CORE metric + BPB by default; skips if results already exist)
python scripts/jobs/run_evaluation.py

# Only run a subset of eval modes (core, bpb, sample)
python scripts/jobs/run_evaluation.py --eval core

# Only evaluate models whose tag contains a substring
python scripts/jobs/run_evaluation.py --model-filter d12

# Force re-evaluation
python scripts/jobs/run_evaluation.py --force
```

Common flags for both launchers: `--profile`, `--dry`, `--force`,
`--author_name`, `--telegram_nick`.

### How launch works

Both launchers build a command that runs `scripts/jobs/prepare_torchrun.sh`, which sets
up DDP env vars from MPI and calls `torchrun --nproc_per_node=$NUM_GPUS ...`. Training
runs `-m scripts.base_train <args> --model-tag <tag>`; evaluation runs
`-m scripts.base_eval --model-tag <tag> --step <step> --eval <modes>`.

### Local (non-job) runs

```bash
# Single-node training directly
torchrun --nproc_per_node=4 -m scripts.base_train --depth 12 --embed-proj-dim 512

# Single-node evaluation directly
torchrun --nproc_per_node=4 -m scripts.base_eval --model-tag d12 --eval core,bpb
```

## Git conventions

- Stage files explicitly (`git add <files>`), never `git add -A`.
- Conventional commit prefixes: `fix:`, `feat:`, `plan:`, `results:`, `test:`, `refactor:`.
- Commit after code changes with a descriptive message.
