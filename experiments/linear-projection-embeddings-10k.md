# Linear Projection Embeddings — 10k-step projection-dimension sweep

## Hypothesis

There exists a low-dim **linear embedding projection** setting at which the projected
embedding **matches or beats the full-embedding baseline** (no projection) at the **10k-step**
horizon, on both **CORE** and **BPB**.

The projection adds a low-rank learnable term `embed_proj(low_dim_embed(idx))` summed with
`wte` (see `nanochat/gpt.py`) — a low-rank factorization of the embedding correction whose rank
acts as a regularizer / capacity reallocation. We therefore expect a **sweet spot in rank**:
too small under-parameterizes the correction, too large recovers the baseline. This sweep is
designed to locate it. Iteration 1 of an ongoing beats-baseline recipe search; see
[[linear_projection_embeddings]] for the original short-horizon study.

## Setup

Training function: `linear_projection_embeddings_10k_experiments` in
`scripts/jobs/run_training.py` (source of truth for all hyperparameters, step counts, arms, and
job configs). The projection is gated by `--embed-proj-dim` in `scripts/base_train.py`
(`0` = baseline / no projection).

- **Node:** `num_gpus=4`, `instance_type=a100.4gpu`.
- **Horizon:** 10k steps, **single seed** (no multi-seed fan-out — one run per arm).
- **Arms:** baseline (no projection) plus the projection-dim arms, for direct comparison.
- **Evaluation:** `scripts/jobs/run_evaluation.py` → `scripts/base_eval.py` (CORE + BPB).
- **Artifacts:** checkpoints under `$NANOCHAT_BASE_DIR/base_checkpoints/<model_tag>/`.

## Results

_To be filled by a later stage._

## Conclusions

_To be filled by a later stage._

## Changelog

- 2026-06-13: Created the 10k-step single-seed group in `scripts/jobs/run_training.py`; first
  re-tested baseline vs proj_512 at d12 over the longer horizon.
- 2026-06-15: Extended the group into a **projection-dimension ablation** — sweep
  `--embed-proj-dim` over {128, 256, 512, 1024} plus the no-projection baseline at 10k steps,
  single seed, to find a dim that matches/beats baseline on CORE and BPB.
- 2026-06-15: Rewrote the plan to the standard format and corrected the Setup to reference the
  actual training function `linear_projection_embeddings_10k_experiments`. Results/Conclusions
  remain placeholders for the sweep.
