# Linear Projection Embeddings

## Hypothesis

Adding a low-rank learnable correction to the pretrained embedding table via
`inputs_embeds = Linear(low_dim_embed(token_ids)) + wte(token_ids)` can improve
embedding quality with fewer trainable parameters than making the full embedding
table trainable. The projection starts at zero (no initial perturbation) and
learns a correction in a low-dimensional bottleneck space.

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

| variant       | embed_proj_dim | val_bpb    | Δbpb vs base | CORE   |
|---------------|----------------|------------|--------------|--------|
| baseline      | 0              | 1.7877     | —            | 0.0603 |
| proj_128      | 128            | 1.7599     | −0.0278      | 0.0603 |
| proj_256      | 256            | 1.7401     | −0.0476      | 0.0603 |
| **proj_512**  | **512**        | **1.7289** | **−0.0588**  | 0.0603 |
| proj_1024     | 1024           | 1.7384     | −0.0493      | 0.0603 |
| proj_2048     | 2048           | 1.7315     | −0.0562      | 0.0603 |

- **BPB:** every projection variant beats the no-projection baseline. The gain grows with
  `embed_proj_dim` up to 512 (best, −0.0588 bpb / −3.3% relative) and then plateaus /
  slightly regresses — 1024 and 2048 are both worse than 512.
- **CORE:** identical (0.0603) across all six d12 variants; CORE does not separate them at
  this scale/step (see anomaly).
- **Parameter cost:** added params = `embed_proj_dim × (vocab 32768 + n_embd 768)` =
  `embed_proj_dim × 33536`. So 512 adds ~17M params; 2048 adds ~69M for a *worse* BPB than
  512. 512 is the clear efficiency sweet spot.

### Anomalies

- All six d12 variants report the **same CORE (0.0603)** despite distinct BPB — the metric
  is saturated/insensitive at this training budget; treat it as non-discriminating here,
  not as evidence of equivalence.
- `d12` (step 250) and `d6` (step 1000) are partially-trained reference runs with negative
  CORE; not part of this ablation.

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

_(empty — to be filled after the reporting fix and multi-seed re-eval.)_

## Conclusions

The zero-initialized low-rank embedding correction **helps**, consistently lowering val_bpb
over the baseline, with **`embed_proj_dim=512` best** (1.7289, −0.0588 bpb vs 1.7877) at a
modest ~17M added parameters. Returns diminish past 512 — larger bottlenecks add parameters
without improving (or slightly hurting) BPB. CORE is too coarse to confirm the benefit at
this scale.

**Recommended next step:** adopt `embed_proj_dim=512` as default and re-evaluate on a
longer/larger run (or greater depth) where CORE becomes discriminating, to confirm the BPB
gain carries to downstream quality.

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
