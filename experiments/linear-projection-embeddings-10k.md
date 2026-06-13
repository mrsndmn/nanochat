# Linear Projection Embeddings — 10k-step single-seed

## Hypothesis

Re-test whether the linear projection embedding configs improve over the no-projection
baseline when trained for a **longer horizon (10k steps)** with a **single seed**. The
prior d12 result established the proj_512 val_bpb advantage at a short training budget; this
phase asks whether that advantage persists, grows, or washes out once both arms are trained
much longer. See [[linear_projection_embeddings]] for the original short-horizon study.

## Setup

Training function: `linear_projection_embeddings_10k` in `scripts/jobs/run_training.py`
(source of truth for all hyperparameters, step counts, model selection, and job configs).
Evaluation via `scripts/jobs/run_evaluation.py` → `scripts/base_eval.py`. Default job:
`num_gpus=4`, `instance_type=a100.4gpu`; checkpoints/artifacts under
`$NANOCHAT_BASE_DIR/base_checkpoints/<model_tag>/`.

Rationale for this group:
- **(a) Longer horizon.** Training now uses **10k steps** for this group, instead of the
  short budget used by the original d12 phase.
- **(b) Single seed only.** No multi-seed fan-out — the prior multi-seed phase already showed
  the proj_512 advantage clears 2σ of training-seed variance with non-overlapping seed
  distributions, so one seed suffices to read the longer-horizon trend.
- **(c) d20 / d6 cancelled.** The d20 depth-scaling and d6 configs have been removed/cancelled
  and are **not** part of this group; this phase stays at d12 and varies only the training
  horizon.

val_bpb is the primary metric; CORE is reference-only (it did not reliably discriminate these
variants at d12).

## Results

_Pending — to be filled once the 10k-step runs complete._

## Conclusions

_Pending._

## Changelog

- 2026-06-13: Created new 10k-step single-seed experiment group
  (`linear_projection_embeddings_10k` in `scripts/jobs/run_training.py`); removed the d20
  depth-scaling and d6 configs. Plan re-tests baseline vs proj_512 at d12 over a 10k-step
  horizon with a single seed.
