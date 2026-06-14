# Linear Projection Embeddings — 10k-step projection-dimension sweep

## Hypothesis

There exists a projection dimension among **{128, 256, 512, 1024}** (or a recipe involving
one of them) at which the low-dim **linear embedding projection matches or beats the baseline**
(full embeddings, no projection) at **10k steps**, on both **CORE** and **BPB**.

Intuition: the projection adds a low-rank learnable term
`embed_proj(low_dim_embed(idx))` summed with `wte` (see `nanochat/gpt.py`), i.e. a low-rank
**factorization of the embedding correction**. This acts as a **regularizer / parameter
reallocation** — small rank constrains the embedding's effective degrees of freedom and shifts
capacity elsewhere, large rank approaches the full-embedding case. We therefore expect a
**sweet spot in rank**: too small under-parameterizes the correction, too large gives back the
baseline, and an intermediate dim should generalize best. The sweep is designed to locate it.

This is **iteration 1** of an ongoing autonomous search for a projection recipe that beats
baseline at the 10k horizon. See [[linear_projection_embeddings]] for the original short-horizon
study.

## Setup

Training function: `linear_projection_embedding_experiments` in `scripts/jobs/run_training.py`
(source of truth for all hyperparameters, step counts, model selection, and job configs). The
low-dim embedding projection is gated by the `--embed-proj-dim` flag in `scripts/base_train.py`
(`0` = baseline / no projection).

- **Horizon:** 10k steps; **single seed** (no multi-seed fan-out).
- **Arms:** the baseline (no projection) is included alongside the projection-dim arms for
  direct comparison.
- **Node:** `num_gpus=4`, `instance_type=a100.4gpu`.
- Evaluation via `scripts/jobs/run_evaluation.py` → `scripts/base_eval.py` (CORE + BPB).
- Checkpoints/artifacts under `$NANOCHAT_BASE_DIR/base_checkpoints/<model_tag>/`.

## Results

_To be filled by a later stage._

## Conclusions

_To be filled by a later stage._

## Changelog

- 2026-06-13: Created the 10k-step single-seed group
  (`linear_projection_embeddings_10k` in `scripts/jobs/run_training.py`); first re-tested
  baseline vs proj_512 at d12 over the longer horizon. Both arms severely overfit (≈58 epochs
  over the shard), the 2520-step proj_512 advantage washed out at the in-training optimum, and
  the horizon itself was found misconfigured for d12.
- 2026-06-15: Extended this group into a **projection-dimension ablation** — sweep
  `--embed-proj-dim` over {128, 256, 512, 1024} plus the no-projection baseline at 10k steps,
  single seed, to find a dim (or recipe) that matches/beats baseline on CORE and BPB. Reset
  Results/Conclusions to placeholders for the new sweep. Iteration 1 of an ongoing
  beats-baseline recipe search.
