# Linear Projection Embeddings — 10k-step dimension ablation

## Hypothesis

There exists a **low-dimensional linear projection dimension** at which the
parameter-efficient projected embeddings **match or exceed the dense baseline** on
CORE/BPB when trained for **10k steps** at d12. Earlier phases compared only a single
projection width (proj_512) against baseline; this phase reframes the question as a
**dimension sweep**: by ablating the projection dimension over a range of low values we
expect to find a dimension (and any accompanying recipe) where the projected model is at
least as good as the dense baseline while using fewer embedding parameters. See
[[linear_projection_embeddings]] for the original short-horizon study.

## Setup

Training function: `linear_projection_embeddings_10k_experiments` in
`scripts/jobs/run_training.py` (source of truth for all hyperparameters, step counts,
the ablation grid, model selection, and job configs). The projection width is controlled
by the `--embed-proj-dim` parameter on `scripts/base_train.py` (`embed_proj_dim=0` =
dense baseline, no projection). Evaluation via `scripts/jobs/run_evaluation.py` →
`scripts/base_eval.py`. Default job: `num_gpus=4`, `instance_type=a100.4gpu`;
checkpoints/artifacts under `$NANOCHAT_BASE_DIR/base_checkpoints/<model_tag>/`.

**Ablation grid.** Sweep `--embed-proj-dim` over a range of **low projection dimensions**
(small → moderate widths) at d12 / 10k steps, single seed, alongside the **dense baseline**
(`embed_proj_dim=0`) as the reference arm. One config is emitted per dimension with a
distinct `model_tag`. The exact set of dimensions lives in the training function — see code
for the concrete values. Single seed only (no multi-seed fan-out, per project convention).
Primary metrics are **CORE and BPB**; the objective is to identify the smallest projection
dimension whose CORE/BPB is **≥ baseline**.

## Results

_To be filled after the dimension-ablation runs complete._

## Conclusions

_Placeholder — to be filled once results are in: report whether any low projection
dimension beats the dense baseline on CORE/BPB, and which dimension/recipe wins._

## Changelog

- 2026-06-13: Created 10k-step single-seed group
  (`linear_projection_embeddings_10k_experiments`); removed d20/d6 configs. Initial phase
  compared baseline vs proj_512 only and was confounded by overfitting at the 10k horizon.
- 2026-06-14: **Restarted as a dimension ablation.** Reframed the hypothesis around finding
  the low projection dimension (plus any accompanying recipe) at which projected embeddings
  **beat the dense baseline on CORE/BPB** at 10k steps. Plan now describes an `--embed-proj-dim`
  sweep over a range of low dimensions against the dense baseline reference; Results/Conclusions
  reset to placeholders pending the new runs.
