# Linear Projection Embeddings — 10k-step projection-dimension sweep

## Hypothesis

There exists a low-dim **linear embedding projection** setting at which the projected
embedding **matches or beats the full-embedding baseline** (no projection) at the **10k-step**
horizon, on both **CORE** and **BPB**.

The projection adds a low-rank learnable term `embed_proj(low_dim_embed(idx))` summed with
`wte` (see `nanochat/gpt.py`) — a low-rank factorization of the embedding correction whose rank
acts as a regularizer / capacity reallocation. We therefore expect a **sweet spot in rank**:
too small under-parameterizes the correction, too large recovers the baseline. Iteration of an
ongoing beats-baseline recipe search; see [[linear_projection_embeddings]] for the original
short-horizon study.

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

10k-step, single-seed sweep over `--embed-proj-dim`. CORE is higher-is-better; `val_bpb` is
lower-is-better. `CORE_std` is the eval-side bootstrap std (CORE seeds 1337–1341), not
training-seed variance. All arms are d12 (n_layer 12, n_embd 768), step 10000.

| arm (embed_proj_dim) | CORE   | CORE_std | val_bpb |
|----------------------|--------|----------|---------|
| baseline (0)         | 0.1957 | 0.0019   | —       |
| proj128              | 0.1818 | 0.0020   | —       |
| proj256              | 0.1848 | 0.0015   | —       |
| proj512              | 0.1797 | 0.0030   | —       |
| proj1024             | 0.1864 | 0.0017   | 0.8040  |

**Relaunch validity.** Disabling intermediate evaluation and relaunching with `--force`
produced valid end-of-run checkpoints for all five arms, and **CORE completed cleanly** for
every arm (each with a bootstrap std). **BPB is incomplete**: `val_bpb` was recorded only for
`proj1024` (0.8040); the baseline and the other three projection arms have no BPB, so no
projection-vs-baseline BPB comparison is possible yet.

The shared `base_checkpoints/` dir also holds two foreign checkpoints from a sibling `sa_*`
branch — `d12_sa_baseline` (CORE 0.1920) and `d12_sa_nltk_k1` (skipped: vocab 32769 ≠ 32768).
These are **not** part of this experiment and are excluded from the comparison; the eval-robustness
fix simply lets the stage tolerate them instead of crashing.

## Conclusions

**The low-dim projection hurts CORE at 10k — the hypothesis is not supported.** No projection
dim matched or beat the no-projection baseline on CORE. The baseline (0.1957) tops every
projection arm; the best projection, `proj1024` (0.1864), still trails by ~0.009 — several times
the ~0.002 eval-side std, so the gap is not eval noise. The trend is non-monotonic (proj512 is
the weakest at 0.1797), but the largest dim is the closest to baseline, consistent with "large
dim recovers baseline": within 128–1024 the additive low-rank embedding correction is a net
regularization cost, not a gain. Recovering baseline would require dim ≫ 1024, which defeats the
low-dim goal.

**BPB is inconclusive** — only `proj1024` (0.8040) has a value and there is no baseline BPB to
compare against.

**Recommended next step:** re-run BPB eval with `--force` for the four arms missing `val_bpb`
(baseline, proj128/256/512) to complete the CORE+BPB picture. Pending that, treat the additive
low-rank embedding projection as not beneficial at d12/10k — either drop this direction or
revisit the projection design (e.g. `embed_proj` init / learning rate) before spending more
compute on further dim sweeps.

## Changelog

- 2026-06-13: Created the 10k-step single-seed group in `scripts/jobs/run_training.py`; first
  re-tested baseline vs the prior best projection arm at d12 over the longer horizon.
- 2026-06-15: Extended the group into a **projection-dimension ablation** — sweep
  `--embed-proj-dim` plus the no-projection baseline at 10k steps, single seed, to find a dim
  that matches/beats baseline on CORE and BPB.
- 2026-06-15: Rewrote the plan to the standard format and aligned the Setup to reference the
  actual training function `linear_projection_embeddings_10k_experiments`. Results/Conclusions
  remain placeholders for the sweep.
- 2026-06-15: Relaunched training after disabling intermediate evaluation
  (`--eval-every`/`--core-metric-every`/`--sample-every` = -1) and re-forcing stale checkpoints.
  This yielded valid checkpoints and clean final CORE for all five arms (baseline +
  proj128/256/512/1024). Baseline CORE 0.1957 beats every projection arm (best proj1024 0.1864);
  proj512 is weakest (0.1797). `val_bpb` recorded only for proj1024 (0.8040) — BPB still missing
  for the other four arms.
