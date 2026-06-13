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

**The single-epoch re-run actually trained this time.** Fresh job configs with distinct `_1ep` model
tags (`d12_baseline_10k_1ep`, `d12_proj512_10k_1ep`) were launched so the ≤1-epoch run could not be
skipped by the stale `d12_*_10k` checkpoints. Both arms now train over the expanded ClimbMix data at
≤1 epoch (150 train shards, no wrap), with the **global batch size unchanged** (524,288 tok/step,
10k steps, single seed). These are genuinely new weights, not re-read stale ones.

d12, single seed (s0), final step (10000):

| arm                | val_bpb (step 10000) | CORE   | CORE_std |
|--------------------|----------------------|--------|----------|
| baseline (`_1ep`)  | **0.8038**           | **0.1804** | 0.0014 |
| proj_512 (`_1ep`)  | 0.8050               | 0.1796 | 0.0023   |
| Δ (proj−base)      | +0.0012 (+0.15%)     | −0.0008 (−0.4%) | — |

**Removing data repetition dramatically helped both arms.** Versus the prior stale multi-epoch
(58-epoch) checkpoints, val_bpb fell from ~2.4 → **0.80** and CORE rose from ~0.04 → **0.18**. The
single-epoch runs are by a wide margin the best models in the whole eval table on *both* metrics. This
confirms the previous 10k numbers were a pure memorization/overfit artifact of cycling a tiny shard set
58×, not a property of the architecture.

**For the linear-projection mechanism specifically, the two arms are now tied.** proj_512 is
marginally *worse* than baseline on both metrics (val_bpb +0.0012; CORE −0.0008), and the CORE gap is
≈0.3σ of the per-arm std — i.e. statistically indistinguishable. The decisive short-horizon
projection advantage does **not** persist once data no longer repeats.

Comparison across regimes (proj_512 vs baseline):

| regime                              | baseline val_bpb | proj_512 val_bpb | Δ val_bpb        | verdict for projection |
|-------------------------------------|------------------|------------------|------------------|------------------------|
| 2520-step multi-seed                | 1.7889 ± 0.0025  | 1.7349 ± 0.0058  | −0.0540 (9.4σ)   | proj_512 clearly wins  |
| 10k multi-epoch (stale, overfit)    | 2.4524           | 2.3719           | −0.0805 (artifact) | slower memorization, not quality |
| **10k single-epoch (`_1ep`, new)**  | **0.8038**       | **0.8050**       | **+0.0012**      | **tied / no advantage** |

### Anomalies

- **The earlier "re-run" never trained** (documented previously): the prior pass reused the
  `d12_*_10k` tags whose 58-epoch checkpoints already existed, so the job was skipped and eval re-read
  stale weights. Fixed here via distinct `_1ep` tags. The stale `d12_baseline_10k` / `d12_proj512_10k`
  rows (val_bpb 2.45 / 2.37, CORE 0.043 / 0.033, `epoch=58`) remain in the table as the overfit
  reference only.
- **Eval still reads the final step, not the best checkpoint.** At ≤1 epoch this matters far less —
  there is no overfit U-curve, val_bpb at step 10000 (~0.80) is near the run minimum — but best-`val_bpb`
  selection is still missing.
- Across the wider table the next-best signal stays at 2520 steps: best val_bpb is the proj_512
  seeds/sweep (≈1.729) vs baseline seeds (≈1.787); best CORE there is also proj_512
  (s3 0.0684, s0 0.0677). `d12_proj512_s4` CORE 0.0464 is a low-seed outlier vs its ~0.066–0.068
  siblings. `d12` (step 250, CORE −0.0136) and `d6` (step 1000) are early/small reference checkpoints,
  not part of this group.

## Conclusions

**Single-epoch training (no data repetition) is a large, unambiguous win — but not for the projection
mechanism.** Removing the 58× shard wrap cut val_bpb roughly in third (2.4 → 0.80) and quadrupled CORE
(0.04 → 0.18) for *both* arms, making these the strongest checkpoints in the table. The earlier 10k
numbers were overfitting, exactly as suspected.

**The linear-projection advantage does not survive at the single-epoch 10k horizon.** With the
repetition artifact removed, baseline and proj_512 are statistically tied (val_bpb 0.8038 vs 0.8050,
Δ +0.0012; CORE 0.1804 vs 0.1796, ≈0.3σ) — proj_512 is, if anything, a hair worse. The decisive
2520-step result (−0.0540 bpb, 9.4σ; see [[linear_projection_embeddings]]) therefore appears to be a
**short-horizon / data-limited effect that washes out** once the model trains on more, non-repeated
data. The projection mechanism neither helps nor hurts at this scale and horizon.

**Recommended next steps (in priority order):**
1. **Treat the projection question as answered "no advantage" at d12 single-epoch.** The mechanism is
   not worth carrying as a default at this scale unless a different regime revives it.
2. **Probe whether the projection helps at larger scale / longer horizon**, where capacity and data are
   the binding constraint rather than the short-budget regime that first showed the gain (e.g. deeper
   model or more steps with still ≤1 epoch).
3. **Add best-`val_bpb` checkpoint selection** for cleanliness; less urgent now that the U-curve is gone.
4. **Keep val_bpb primary; CORE is now informative** at single-epoch (0.18 vs ~0.04 overfit) and agrees
   with val_bpb that the arms are tied — so it can be reported alongside, not gated on.

## Changelog

- 2026-06-13: Created new 10k-step single-seed experiment group
  (`linear_projection_embeddings_10k` in `scripts/jobs/run_training.py`); removed the d20
  depth-scaling and d6 configs. Plan re-tests baseline vs proj_512 at d12 over a 10k-step
  horizon with a single seed.
- 2026-06-13: Filled Results/Conclusions from the 10k single-seed eval. Both arms **severely
  overfit** (58 epochs over the shard; train bpb ~0.30; val-bpb U-curve, min_val_bpb ≈ 1.046 then
  back up to ~2.4 at step 10000, which is what eval reports). At the in-training optimum the arms
  are **tied** (baseline 1.0468 vs proj_512 1.0457, Δ −0.0011), so the decisive 2520-step proj_512
  advantage (−0.0540) **washes out** at the longer horizon; proj_512 leads only at the overfit
  final step (2.3719 vs 2.4524) via slower memorization, and is worse on CORE (0.0329 vs 0.0432).
  Conclusion: 10k does not show a persisting projection gain; the horizon itself is misconfigured.
  Next: fix data budget (≤1 epoch), evaluate/save best checkpoint, re-run at corrected horizon.
- 2026-06-13: Reconfigured the group to ≤1 epoch on the expanded data (cap to 150 train shards via
  `--num-train-shards`, **global batch size unchanged** at 524,288 tok/step, still 10k steps,
  single seed). Re-ran the eval pipeline — but the **single-epoch training never actually executed**:
  the arms reuse the same `d12_*_10k` tags, the existing checkpoints blocked the job (no `--force`),
  and eval re-read the stale 58-epoch weights. Outcome: CORE/BPB **identical** to the prior
  multi-epoch run (baseline 2.4524 / 0.0432, proj_512 2.3719 / 0.0329; both `epoch=58`, no
  `num_train_shards` in meta), so this pass yields **no** single-epoch signal. The linear-projection
  conclusion is unchanged and still rests on the 2520-step multi-seed result. Next: re-launch the
  single-epoch run with `--force` (or fresh tags), then compare at the best checkpoint.
- 2026-06-13: Created **new job configs with distinct model tags** (`d12_baseline_10k_1ep`,
  `d12_proj512_10k_1ep`) to dodge the checkpoint-skip collision with the stale `d12_*_10k` weights,
  and **actually launched the training jobs** (the prior pipeline run launched none). The expanded-
  dataset single-epoch configuration (150 train shards → ≤1 epoch, no data repetition) ran with the
  **identical global batch size** (524,288 tok/step, 10k steps, single seed). Result: removing data
  repetition is a large win for both arms — val_bpb 2.4 → **0.80**, CORE 0.04 → **0.18**, the best
  models in the table — confirming the prior 10k numbers were an overfit/memorization artifact. But
  for the projection mechanism the arms are now **tied** (baseline 0.8038 / 0.1804 vs proj_512
  0.8050 / 0.1796; Δ val_bpb +0.0012, CORE −0.0008 ≈ 0.3σ). The decisive 2520-step proj_512 advantage
  (−0.0540 bpb, 9.4σ) **washes out** once data no longer repeats — linear projection shows no
  advantage at the single-epoch 10k horizon.
