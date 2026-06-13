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

d12, single seed (s0). 10k-step runs (`d12_*_10k`). Eval reports the **final** step (10000).
`min_val_bpb` and `train_bpb` are read from the run's `loop_state` / in-training eval.

| arm           | final val_bpb (step 10000) | best val_bpb (min, in-training) | train bpb | epochs over shard | CORE (final) |
|---------------|----------------------------|---------------------------------|-----------|-------------------|--------------|
| baseline      | 2.4524                     | 1.0468                          | 0.2999    | 58                | 0.0432       |
| proj_512      | **2.3719**                 | 1.0457                          | 0.3447    | 58                | 0.0329       |
| Δ (proj−base) | **−0.0805** (−3.3%)        | −0.0011 (−0.1%)                 | +0.0448   | —                 | −0.0103      |

**Both arms severely overfit at the 10k horizon.** At d12 the data shard is small enough that
10k steps loop it **58 times**; train bpb collapses to ~0.30 while val bpb climbs back up. The
classic overfit U-curve is visible in the run state: val bpb bottoms out at **min_val_bpb ≈ 1.05**
(hit mid-run) and then degrades to **~2.4 by step 10000**, which is what the eval pipeline reports.

**Comparison vs the prior shorter-horizon runs (10k vs 2520 steps).** The prior multi-seed phase
at step 2520 gave baseline **1.7889 ± 0.0025** vs proj_512 **1.7349 ± 0.0058** (Δ −0.0540, ≈ 3.0%,
decisive 9.4σ). The 10k runs land at very different places depending on which checkpoint you read:
- *At the in-training optimum* (min_val_bpb ≈ 1.046 for both arms) the longer horizon is **much
  better than 2520** — the short runs were simply undertrained — but the proj_512 advantage
  **collapses to −0.0011 bpb, i.e. the two arms are tied.**
- *At the reported final step* proj_512 is nominally ahead (2.3719 vs 2.4524, Δ −0.0805, −3.3%),
  but this is in the overfit regime and is driven by proj_512 **memorizing slightly less** (its
  train bpb 0.3447 > baseline 0.2999), not by a genuine generalization gain.

**CORE.** proj_512 (0.0329) is *below* baseline (0.0432) at the final step, and both are below the
2520-step seeded means (baseline 0.0541, proj_512 0.0630). Consistent with the established finding
that CORE does not reliably discriminate these variants at d12 — and here both runs are degraded.

### Anomalies

- **Overfitting / data exhaustion is the dominant effect**, not the projection. 58 epochs over the
  shard, train bpb ~0.30, and a val-bpb U-curve mean the final-step val_bpb (2.37–2.45) measures
  memorization, not quality. The genuinely best checkpoints (~1.046 bpb) are never evaluated.
- **Eval reads the final step, not the best.** No best-checkpoint selection / early stopping is in
  place, so the reported numbers are the post-overfit ones.
- Single seed by design — no per-arm variance, so the −0.0805 final-step delta has no error bar.

## Conclusions

**The 10k longer-horizon test is confounded by overfitting and does not show the projection's
2520-step advantage persisting.** At the in-training val-bpb optimum the baseline and proj_512 arms
are effectively **tied** (1.0468 vs 1.0457, Δ −0.0011), so the decisive −0.0540 bpb edge seen at
2520 steps **washes out** once both arms are trained to their (much lower) best point. The only
place proj_512 still leads is the final, overfit step (Δ −0.0805), and that reflects slightly
slower memorization rather than better generalization. **At 10k steps the linear-projection
embedding does not clearly improve over baseline** on the metric that matters (best val_bpb), and
it is *worse* on CORE.

A second, equally important takeaway: **the 10k horizon itself is the wrong setup for d12** — it
loops the shard 58× and the model overfits long before step 10000. The useful signal (val bpb
~1.046, well below the 1.79 of the 2520 runs) sits at the in-training minimum, which the pipeline
neither saves nor evaluates.

**Recommended next steps (in priority order):**
1. **Fix the horizon / data budget.** Either add more data shards or cut the step count so training
   stays at ≤1 epoch (or close to a Chinchilla-style data:param ratio). The current 58-epoch loop
   makes any longer-horizon comparison a memorization test, not a quality test.
2. **Evaluate the best checkpoint, not the final step.** Add best-`val_bpb` checkpoint saving (or
   early stopping) so model selection and eval use the ~1.046-bpb minimum rather than the overfit
   step-10000 weights.
3. **Re-run baseline vs proj_512 at the corrected horizon** and compare at the best checkpoint.
   Single seed is acceptable to read the trend; only escalate to multi-seed if the arms are close.
4. **Keep val_bpb primary; do not gate on CORE at d12** — it is uninformative here (proj_512 even
   regresses) as already established by the multi-seed phase.

The decisive, trustworthy result remains the 2520-step multi-seed finding (proj_512 −0.0540 bpb,
9.4σ); see [[linear_projection_embeddings]]. This 10k phase does **not** extend it to a longer
horizon — it instead surfaces an overfitting / checkpoint-selection problem to fix first.

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
