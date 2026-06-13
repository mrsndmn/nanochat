# Linear Projection Embeddings

## Hypothesis

Adding a low-rank learnable correction to the pretrained embedding table via
`inputs_embeds = Linear(low_dim_embed(token_ids)) + wte(token_ids)` can improve
embedding quality with fewer trainable parameters than making the full embedding
table trainable. The projection starts at zero (no initial perturbation) and
learns a correction in a low-dimensional bottleneck space.

### Multi-seed validation (current phase)

The prior single-run result — `proj_512` (`embed_proj_dim=512`) lowering val_bpb by
−0.0588 vs the no-projection baseline (`embed_proj_dim=0`) — is a **single point estimate**
and needs multi-seed confirmation before it can drive the d12 default. In the same single-run
data the **CORE ordering was uncorrelated with both val_bpb and `embed_proj_dim`**, i.e.
consistent with noise rather than a real quality ranking, so CORE cannot yet be trusted to
discriminate these variants. This phase asks: **does the proj_512 val_bpb advantage survive
training-seed variance, and can CORE resolve the deltas at all?**

## Setup

Training function: `scripts/base_train.py` with `--embed-proj-dim` flag.
Configs in `scripts/jobs/run_training.py` (`linear_projection_embedding_experiments`);
evaluation via `scripts/jobs/run_evaluation.py` → `scripts/base_eval.py`. Code is the
single source of truth for all hyperparameters, model selection, and job configs.

### Investigation (reviewer-mandated): why is CORE insensitive across projection variants?

**Open question.** Across all six d12 input-embedding projection variants the CORE metric
comes out (nearly) identical (0.0603) even though val_bpb clearly separates them
(e.g. −0.0588 val_bpb at `embed_proj_dim=512`). Before launching the
[[low_rank_unembedding]] follow-up, the reviewer requires us to understand *why* CORE does
not move, so that we are not building on a metric that cannot discriminate these runs.

**Goal (a) — root-cause the CORE insensitivity.** Determine which of the following explains
the flat CORE, and rule the others in or out with evidence:
- *Genuine metric saturation / low resolution* — CORE is simply too coarse to resolve quality
  differences for a small (d12) model at this training budget, so equal scores reflect real
  (lack of) separation rather than a defect.
- *Too few eval examples* — the number of CORE examples per task (e.g. `--max-per-task`
  subsampling) is small enough that score quantization hides true differences; more examples
  would separate the variants.
- *A bug / pipeline artifact*, e.g. evaluation not actually loading the projection weights,
  reading a shared or wrong checkpoint (note: `base_eval` resolves checkpoints purely by
  `model_tag`/`step`, so if variants collide on a single tag they would all evaluate the same
  weights), or a hardcoded model path — any of which would make every variant evaluate an
  identical model and produce identical CORE by construction.

**Goal (b) — propose a path to reliable, discriminative results.** Recommend concrete ways to
obtain CORE (or an alternative downstream metric) that actually separates these low-dim
projection experiments — e.g. confirming per-variant checkpoints are distinct and loaded,
increasing the eval example count, evaluating at a larger depth / longer training budget where
CORE becomes discriminating, or adding a finer-grained complementary metric alongside val_bpb.

## Results

d12 models, step 2520. Config lives in `scripts/jobs/run_training.py`
(`linear_projection_embedding_experiments`).

Re-evaluated after the reporting fix (commit `47543b6`). CORE is now read per-`(model_tag,
step)` from the canonical `evaluation/eval_<step>.json`, so it is **no longer byte-identical**
across variants. **Single eval seed (1337)** per variant, so no cross-seed std yet
(`core_metric_std = 0.0` is degenerate, not a real uncertainty).

| variant       | embed_proj_dim | val_bpb    | Δbpb vs base | CORE (1 seed) |
|---------------|----------------|------------|--------------|---------------|
| baseline      | 0              | 1.7877     | —            | 0.0616        |
| proj_128      | 128            | 1.7599     | −0.0278      | 0.0536        |
| proj_256      | 256            | 1.7401     | −0.0476      | 0.0618        |
| **proj_512**  | **512**        | **1.7289** | **−0.0588**  | 0.0588        |
| proj_1024     | 1024           | 1.7384     | −0.0493      | **0.0657**    |
| proj_2048     | 2048           | 1.7315     | −0.0562      | 0.0586        |

- **BPB (primary, discriminative):** every projection variant beats the no-projection
  baseline. The gain grows with `embed_proj_dim` up to 512 (best, −0.0588 bpb / −3.3%
  relative) and then plateaus / slightly regresses — 1024 and 2048 are both worse than 512.
- **CORE (now distinct, but not yet reliable):** spread 0.0536–0.0657 across the six variants.
  Crucially the CORE ordering is **uncorrelated with val_bpb**: the best-bpb variant (proj_512)
  sits *below* baseline on CORE, proj_128 (a clear bpb win) is *worst* on CORE, and proj_1024
  tops CORE. With only one eval seed and no run-to-run estimate, this ~0.012 spread is
  consistent with sampling/run noise rather than a real quality ranking.
- **Parameter cost:** added params = `embed_proj_dim × (vocab 32768 + n_embd 768)` =
  `embed_proj_dim × 33536`. So 512 adds ~17M params; 2048 adds ~69M for a *worse* BPB than
  512. 512 is the clear efficiency sweet spot.

### Anomalies

- **CORE/val_bpb discordance.** CORE no longer separates variants in a way that tracks val_bpb
  (or even monotonically with `embed_proj_dim`); treat the per-variant CORE deltas as noise
  until multi-seed uncertainty is measured.
- `d12` (step 250) and `d6` (step 1000) are partially-trained reference runs with negative
  CORE; not part of this ablation. `d12_unembed_512` belongs to [[low_rank_unembedding]]
  (val_bpb 1.8686, worse than baseline) and is listed only for cross-reference.

## CORE-reliability investigation

This section answers the open question framed in *Setup → Investigation*: **why is CORE
byte-identical (0.0603) across all six d12 projection variants while val_bpb cleanly
separates them?** The investigation was done by reading the eval/reporting code (the source
of truth), not the prior markdown.

### Hypothesis

The flat CORE is **not** model behavior — it is a **reporting artifact**. CORE for every
variant is being read from a single results file that is keyed by training *step* only (not by
`model_tag`), so all six d12 runs at step 2520 read the *same* overwritten CSV and therefore
display one variant's number. val_bpb separates them only because it is read per-checkpoint
from a distinct file. Secondarily, even with correct wiring, CORE is a discrete
decision-based metric (argmin over option losses) and is expected to be far less sensitive
than the continuous val_bpb to the small quality deltas these low-rank corrections produce.

### Setup

Investigation target (training function): `scripts/base_train.py`
(`--embed-proj-dim`), configs in `scripts/jobs/run_training.py`
(`linear_projection_embedding_experiments`); eval via `scripts/jobs/run_evaluation.py` →
`scripts/base_eval.py`; results aggregated by `scripts/results.py`. Default job:
`num_gpus=4`, `instance_type=a100.4gpu`; checkpoints/artifacts under
`$NANOCHAT_BASE_DIR/base_checkpoints/<model_tag>/`. Code remains the single source of truth
for all configs/hyperparameters.

### Candidate root causes (with evidence)

- **(e) Results-file collision keyed by step, not model_tag — PRIMARY / decisive.**
  `base_eval.py` writes the CORE CSV to `base_model_{step:06d}.csv` — the slug is *step only*
  (`scripts/base_eval.py:213`, written at `:288`). `scripts/results.py` prefers a
  per-checkpoint `evaluation/eval_<step>.json` (`results.py:29,104`) but **`base_eval.py`
  never writes that JSON** (it only writes the CSV and logs to the report), so `results.py`
  always falls back to `_read_core_from_csv(artifacts_root, step)` →
  `base_eval/base_model_{step:06d}.csv` (`results.py:45,106`). All six variants finish at
  step 2520, so every row reads the *same* `base_model_002520.csv`, overwritten by whichever
  variant evaluated last → identical CORE by construction. val_bpb, in contrast, is read
  per-checkpoint from `meta_<step>.json` (`results.py:111`), which is why it differs. This
  exactly matches the symptom (CORE identical to 4 dp; val_bpb distinct).

- **(a) Same/duplicate checkpoint evaluated — RULED OUT.** `model_tag` embeds a sha1 of the
  training args (`run_training.py:54-55`), so each variant trains to a distinct directory;
  `run_evaluation.py` discovers tags by directory and resolves the latest step per dir
  independently (`run_evaluation.py:27-51,162`). The checkpoints on disk are genuinely
  distinct — the collision is downstream, at report aggregation, not at the weights.

- **(b) CORE is coarse / decision-based — CONTRIBUTORY, expected.** CORE accuracy is the mean
  of a 0/1 correctness tensor (`core_eval.py:251-262`); for MC/schema tasks correctness is an
  `argmin` over per-option mean loss (`core_eval.py:232-237`), and centering+task-averaging
  (`base_eval.py:162-167`) keeps it quantized at ~1/N per task. A −3% val_bpb change need not
  flip any discrete argmin decision, so even with correct wiring CORE is expected to move much
  less than val_bpb at d12 scale. This makes CORE a poor *primary* signal here regardless of
  the artifact.

- **(c) Eval token/example budget — minor.** Full eval uses all examples
  (`--max-per-task=-1`, `run_evaluation.py:88`; `base_eval.py:184`), so subsampling
  quantization is not the cause here; in-training CORE does subsample (500/task,
  `base_train.py:77`). Larger budgets reduce quantization but won't separate variants whose
  argmin decisions don't flip.

- **(d) Determinism/seeding — RULED OUT as a defect.** Few-shot selection and shuffling use
  fixed seeds (`core_eval.py:179` `Random(1234+idx)`; `base_eval.py:154` `Random(1337)`).
  This is correct and desirable — it holds the eval set fixed so only weights vary; it does
  *not* fabricate identical numbers.

- **Truncation fix (`1800511`) flattening differences — RULED OUT for d12.** `GPT.max_seq_len`
  returns the rotary cache size `sequence_len * 10` (`gpt.py:218,224-229`) = 20480 for d12,
  far above the longest CORE prompt (~5.4k tokens). The truncation path in
  `core_eval.py:198-213` never engages at d12, so it cannot be flattening differences (it
  matters only for tiny-context models like d6).

### Proposal (to be implemented in code)

1. **Make val_bpb / val loss the primary signal.** It is continuous and already separates the
   variants; treat CORE as secondary confirmation, not the decision metric.
2. **Fix the reporting artifact (highest priority).** Key the CORE output by `model_tag`
   (and step), and have `base_eval.py` write the canonical per-checkpoint
   `evaluation/eval_<step>.json` that `results.py`/`run_evaluation.py` already expect — so
   each variant's CORE is read from its own file and the step-only CSV fallback is no longer
   load-bearing.
3. **Add a distinct-checkpoint sanity check.** Before/at eval, assert each `model_tag`
   resolves to a unique checkpoint path + step and log the resolved `(model_tag, path, step)`
   so any future collision is visible in the logs.
4. **Run multiple seeds and report mean ± std.** Repeat each variant across a few seeds and
   report per-variant val_bpb (and CORE) with uncertainty, so deltas are distinguishable from
   noise.
5. **Increase the CORE example budget** for the final eval to cut quantization, and report
   per-variant **Δ vs baseline with uncertainty** rather than a single point estimate.

All concrete values (seed count, example budget, depth) live in
`scripts/jobs/run_training.py` / `run_evaluation.py`, not here.

### Expected results

- After the reporting fix, CORE values diverge across variants (no longer all 0.0603); the
  per-`model_tag` JSON/CSV confirms distinct numbers.
- val_bpb remains the cleanest separator; with multiple seeds, proj_512's advantage over
  baseline is expected to exceed the seed-to-seed std (i.e. a real, not noise, effect).
- CORE deltas, if present, are small and within the resolution limits of (b) — consistent
  with CORE being a weak discriminator at d12, motivating val_bpb as the primary metric.

### Results

**Q1 — artifact vs. real flatness: confirmed pure reporting artifact, now resolved.** After
the fix (`base_eval.py` writes the canonical per-checkpoint `evaluation/eval_<step>.json` keyed
by `(model_tag, step)`; `results.py` reads it per-tag), CORE comes out **distinct** for every
variant (0.0536 / 0.0618 / 0.0588 / 0.0657 / 0.0586 vs. baseline 0.0616) instead of the old
byte-identical 0.0603. The underlying checkpoints always differed — val_bpb separated them all
along — so there is **no residual model-behavior flatness**; the 0.0603 was entirely the
step-only CSV overwrite predicted by hypothesis (e).

**Q2 — reliability: not yet established.** The re-eval used a **single eval seed (1337)**, so
`core_metric_std = 0.0` is a single-sample placeholder, not a measured uncertainty, and the
`CORE_std` column is empty. Two independent caveats prevent calling the per-variant CORE deltas
real:
- *No variance estimate.* The CORE spread (0.0536–0.0657, ~20% relative) has no error bars.
- *Wrong-direction ordering.* CORE does not track val_bpb or `embed_proj_dim`, which is exactly
  what coarse, decision-based (argmin) noise looks like at d12 (investigation cause (b)).

Note the two noise sources are distinct: `--seeds` only varies `fewshot_seed = 1234 + seed` and
the subsample shuffle (`base_eval.py:150,160`), i.e. CORE's few-shot sampling — it does **not**
change val_bpb (deterministic per checkpoint) and does **not** capture training-run
stochasticity. Establishing reliability therefore needs *both* (i) multiple **eval** seeds to
bound few-shot sampling noise and (ii) multiple **training** seeds to bound run-to-run variance.

## Multi-seed validation phase

### Design

Only **two** variants are retrained — **baseline** (`embed_proj_dim=0`) and **proj_512**
(`embed_proj_dim=512`). The intermediate dims (128/256/1024/2048) are dropped; the single-run
sweep already located 512 as the val_bpb sweet spot, so this phase spends its budget on
*confirming* that one comparison rather than re-sweeping.

- **Training seeds:** each variant is trained with **≥3 (5 preferred) independent training
  seeds**, to bound run-to-run (training) variance — the noise source the single-run estimate
  could not see.
- **Eval seeds:** every checkpoint is evaluated with **≥5 eval seeds**, to bound CORE's
  few-shot-sampling noise (eval seeds do not affect val_bpb, which is deterministic per
  checkpoint).
- **Reporting:** report **val_bpb and CORE as mean ± std per variant**.
- **Metric roles:** **val_bpb is the PRIMARY metric**; CORE is **confirmatory only**.

Exact seeds, seed counts, and configs live in `scripts/jobs/run_training.py`
(`linear_projection_embedding_experiments`) and `scripts/jobs/run_evaluation.py` — code is the
single source of truth; the above states intent, not values.

### Decision rule

- **Adopt `embed_proj_dim=512` as the d12 default *only if*** its val_bpb advantage over the
  baseline **exceeds 2σ of the run-to-run (training-seed) variance**. Otherwise the
  single-run gain is not distinguishable from a lucky seed and is not adopted.
- **Drop CORE as a d12 selection metric** if, even at ≥5 eval seeds, it **cannot resolve the
  ~0.005 deltas** between variants — in that case rely on val_bpb at d12 and, if needed,
  confirm at greater depth / longer budget where CORE separates.

### Results

_(pending — multi-seed runs not yet completed)_

## Conclusions

The zero-initialized low-rank embedding correction **helps**, consistently lowering val_bpb
over the baseline, with **`embed_proj_dim=512` best** (1.7289, −0.0588 bpb vs 1.7877) at a
modest ~17M added parameters. Returns diminish past 512 — larger bottlenecks add parameters
without improving (or slightly hurting) BPB.

The flat CORE (0.0603) was a **pure reporting artifact** (step-keyed CSV overwrite), now fixed:
CORE is per-variant distinct. But the corrected CORE is **not yet reliable** — single eval
seed, no variance, and an ordering that contradicts val_bpb and `embed_proj_dim`. So **val_bpb
remains the only trustworthy discriminator**, and on val_bpb proj_512 is the clear winner; the
low-dim projection helps. Whether it helps *CORE* is currently unanswerable.

**Recommended next steps (in priority order):**
1. **Re-eval with ≥5 eval seeds** (cheap — no retraining) to put a std/SE on CORE and decide
   whether the ~0.012 variant spread survives few-shot sampling noise.
2. **Retrain the key comparison (baseline vs. proj_512) with ≥3 training seeds** (5 preferred)
   and report val_bpb and CORE as mean ± std — this is the only way to show the proj_512 gain
   exceeds run-to-run variance rather than being a lucky seed.
3. **Keep val_bpb as the primary metric** and adopt `embed_proj_dim=512` as the default; if
   CORE still fails to resolve ~0.005 deltas even with 5 seeds, confirm the bpb gain at greater
   depth / longer budget where CORE separates, rather than chasing CORE at d12.

## Changelog

- 2026-06-12: Initial implementation of linear projection embeddings in gpt.py and base_train.py
- 2026-06-13: Filled Results/Conclusions from step-2520 eval; proj_512 best on BPB, CORE saturated.
- 2026-06-13: Per reviewer feedback, pivoted to a CORE-reliability investigation — added an
  Investigation sub-section under Setup framing the open question (flat CORE despite separating
  val_bpb) with goals (a) root-cause the insensitivity (metric saturation vs. too few eval
  examples vs. an eval/checkpoint-loading bug) and (b) propose how to get discriminative results.
  The [[low_rank_unembedding]] training launch is deferred until this is resolved.
- 2026-06-13: Completed the CORE-reliability investigation from code (base_eval.py,
  run_evaluation.py, results.py, core_eval.py, gpt.py). Root cause: a reporting artifact —
  CORE is read from a step-only CSV (`base_model_<step>.csv`, base_eval.py:213/288) because
  the per-checkpoint `evaluation/eval_<step>.json` is never written, so all variants at step
  2520 read one overwritten file (results.py:45,106) → identical 0.0603; val_bpb differs
  because it is per-checkpoint (results.py:111). Ruled out duplicate-checkpoint (distinct tags
  via args hash), seeding (fixed, correct), and the max_seq_len truncation fix (cache 20480 ≫
  prompts at d12). CORE's argmin decision metric is a contributing coarseness factor. Added a
  CORE-reliability investigation section with hypothesis, evidence, and a proposal (val_bpb as
  primary; key CORE output by model_tag + write canonical JSON; distinct-checkpoint sanity
  check; multi-seed mean±std; larger CORE budget; report Δ with uncertainty).
- 2026-06-13: Re-evaluated all six d12 variants after the reporting fix (`47543b6`). Confirmed
  the flat 0.0603 was a **pure reporting artifact** — CORE is now per-variant distinct
  (0.0536–0.0657 vs baseline 0.0616), no residual model flatness. But reliability is **not yet
  established**: only one eval seed (1337, `core_metric_std=0.0`), and CORE ordering is
  uncorrelated with val_bpb / `embed_proj_dim` → deltas consistent with noise. Filled the
  investigation Results; val_bpb stays primary (proj_512 best). Next: ≥5 eval seeds for CORE
  variance, ≥3 training seeds for baseline-vs-proj_512 run-to-run, report mean±std.
- 2026-06-13: Started the **multi-seed validation phase**. Narrowed to two variants (baseline
  vs. proj_512), each retrained with ≥3 (5 preferred) training seeds and evaluated with ≥5 eval
  seeds, reporting val_bpb and CORE as mean ± std. Recorded the decision rule: adopt
  `embed_proj_dim=512` as the d12 default only if its val_bpb advantage exceeds 2σ of
  training-seed variance; drop CORE as a d12 selection metric if it cannot resolve ~0.005 deltas
  at 5 eval seeds. val_bpb is primary, CORE confirmatory. Single-run findings left intact above.
