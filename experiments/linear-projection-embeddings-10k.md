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

**The intended single-epoch re-run has not actually been trained yet; this eval pass re-read the
prior multi-epoch checkpoints.** The ≤1-epoch reconfiguration (cap to 150 train shards via
`--num-train-shards`, global batch size unchanged) is committed in code, but the two arms keep the
same tags (`d12_baseline_10k`, `d12_proj512_10k`) and the launcher skips a tag whose checkpoint
already exists. The evaluated `model_010000.pt` files predate the reconfiguration commit, their
`meta` still reports `dataloader_state_dict.epoch = 58`, and `user_config` carries **no**
`num_train_shards`. So the numbers below are **byte-identical to the previous multi-epoch run** — the
single-epoch regime was never exercised.

d12, single seed (s0), final step (10000):

| arm           | final val_bpb (step 10000) | best val_bpb (min, in-training) | train bpb | epochs over shard | CORE (final) |
|---------------|----------------------------|---------------------------------|-----------|-------------------|--------------|
| baseline      | 2.4524                     | 1.0468                          | 0.2999    | 58                | 0.0432       |
| proj_512      | **2.3719**                 | 1.0457                          | 0.3447    | 58                | 0.0329       |
| Δ (proj−base) | **−0.0805** (−3.3%)        | −0.0011 (−0.1%)                 | +0.0448   | —                 | −0.0103      |

**Did single-epoch training change CORE/BPB vs the multi-epoch run? Unknown — nothing changed
because nothing re-trained.** Both arms still show `epoch = 58`, train bpb ~0.30, and the classic
overfit U-curve (val bpb bottoms at min_val_bpb ≈ 1.05 mid-run, then degrades to ~2.4 at step 10000,
which is what eval reports). These are the *same* memorization-dominated checkpoints described in the
prior update, not a corrected ≤0.78-epoch run.

**Reference — prior shorter-horizon runs (2520 steps).** The multi-seed phase gave baseline
**1.7889 ± 0.0025** vs proj_512 **1.7349 ± 0.0058** (Δ −0.0540, ≈ 3.0%, decisive 9.4σ). The 58-epoch
10k checkpoints sit far worse at their final step (2.37–2.45) and only tie at their in-training
optimum (1.0468 vs 1.0457, Δ −0.0011). The nominal final-step proj_512 lead (−0.0805) reflects
slightly slower memorization (its train bpb 0.3447 > baseline 0.2999), not better generalization.

**CORE.** proj_512 (0.0329) is *below* baseline (0.0432) at the final step, and both are below the
2520-step seeded means (baseline 0.0541, proj_512 0.0630) — again the overfit-checkpoint artifact,
not a single-epoch signal.

### Anomalies

- **The single-epoch re-run did not run.** Same-tag checkpoints already existed, so the reconfigured
  job was skipped (no `--force`); eval re-evaluated the stale 58-epoch weights. The "re-run results"
  are therefore identical to the multi-epoch run and carry **no** information about the ≤1-epoch
  regime. This is the headline issue to fix before any comparison is meaningful.
- **Overfitting / data exhaustion still dominates** these (stale) checkpoints: 58 epochs over the
  shard, train bpb ~0.30, val-bpb U-curve. The genuinely best checkpoints (~1.046 bpb) are never
  evaluated.
- **Eval reads the final step, not the best** — no best-`val_bpb` checkpoint selection / early
  stopping, so reported numbers are the post-overfit ones.
- Across the wider table the trustworthy signal stays at 2520 steps: best val_bpb is the proj_512
  seeds/sweep (≈1.729) vs baseline seeds (≈1.787); best CORE is also proj_512
  (s3 0.0684, s0 0.0677). `d12_proj512_s4` CORE 0.0464 is a low-seed outlier vs its ~0.066–0.068
  siblings. `d12` (step 250) and `d6` (step 1000) are early/small reference checkpoints, not part of
  this group.

## Conclusions

**The corrected single-epoch comparison is still pending — it did not actually train.** The ≤1-epoch
data budget is in code and the global batch size was left unchanged (524,288 tok/step, 10k steps), but
because the arms reuse the existing `d12_*_10k` tags the reconfigured job was skipped and eval re-read
the stale 58-epoch checkpoints. **CORE and BPB are therefore unchanged simply because the underlying
weights are unchanged**; this pass gives no evidence either way about the single-epoch regime.

**The linear-projection conclusion is unaffected by this run and continues to rest on the 2520-step
multi-seed finding** (proj_512 −0.0540 bpb, 9.4σ; see [[linear_projection_embeddings]]). It is neither
confirmed nor refuted at a single-epoch 10k horizon yet. What the stale checkpoints still show — arms
tied at their in-training optimum (1.0468 vs 1.0457) and proj_512 leading only at the overfit final
step via slower memorization — remains a memorization artifact, not a quality signal, exactly as
before.

**Recommended next steps (in priority order):**
1. **Actually launch the single-epoch re-run.** Re-submit with `--force` (or under fresh tags) so the
   150-shard / ≤0.78-epoch config produces new checkpoints instead of being skipped. Until then there
   is nothing single-epoch to analyze.
2. **Evaluate the best checkpoint, not the final step.** Add best-`val_bpb` checkpoint saving / early
   stopping so model selection uses the val-bpb minimum rather than overfit step-10000 weights — this
   matters even once data no longer wraps.
3. **Then compare baseline vs proj_512 at the corrected horizon.** Single seed is acceptable to read
   the trend; escalate to multi-seed only if the arms are close.
4. **Keep val_bpb primary; do not gate on CORE at d12** — it is uninformative here (proj_512 even
   regresses on these checkpoints), as established by the multi-seed phase.

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
