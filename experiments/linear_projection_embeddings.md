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

### Depth-scaling at d20 (current phase)

The d12 result is established and decisive: `embed_proj_dim=512` lowers val_bpb to
1.7349 vs the 1.7889 baseline (9.4σ of training-seed variance, ~3% relative). The open
question is whether this gain is **depth-dependent**: **does the input-projection val_bpb
advantage scale with depth — persisting, or even growing, at d20 — or does it wash out as the
model gains capacity** and the low-rank embedding correction becomes redundant? This phase
re-runs the decisive baseline-vs-proj_512 comparison at d20 to find out.

## Setup

Training function: `scripts/base_train.py` with `--embed-proj-dim` flag.
Configs in `scripts/jobs/run_training.py` (`linear_projection_embedding_experiments`);
evaluation via `scripts/jobs/run_evaluation.py` → `scripts/base_eval.py`. Code is the
single source of truth for all hyperparameters, model selection, and job configs.

### Depth-scaling (d20) arms

Two arms, mirroring the decisive d12 comparison at greater depth:
- **d20 + `embed_proj_dim=512`** (projection on)
- **d20 + `embed_proj_dim=0`** (baseline, no projection)

Each arm is trained with **3 independent training seeds** → **6 runs total**. The
**primary metric is val_bpb, reported as mean ± std across the 3 seeds per arm**.
**CORE is NOT a gate** for this phase: the d12 multi-seed phase showed run-to-run CORE
variance of ±0.008, which exceeds the val_bpb-equivalent deltas of interest here, so CORE
cannot reliably discriminate the arms and is reported only for reference. Exact seeds,
depth, and all hyperparameters live in `scripts/jobs/run_training.py`
(`linear_projection_embedding_experiments`) — code is the single source of truth.

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

Five training seeds (s0–s4) per variant at step 2520, each evaluated with ≥5 eval seeds.
`CORE` below is the mean over eval seeds for that training seed; the `±` after the per-variant
means is the **training-seed** std (the run-to-run noise the decision rule is measured against).

| variant  | embed_proj_dim | val_bpb (mean ± std) | val_bpb range   | CORE (mean ± std) | CORE range      |
|----------|----------------|----------------------|-----------------|-------------------|-----------------|
| baseline | 0              | 1.7889 ± 0.0025      | 1.7861–1.7925   | 0.0541 ± 0.0076   | 0.0448–0.0658   |
| proj_512 | 512            | **1.7349 ± 0.0058**  | 1.7295–1.7432   | 0.0630 ± 0.0093   | 0.0464–0.0684   |

**val_bpb (PRIMARY) — proj_512 advantage holds far beyond 2σ → ADOPT.**
- Mean advantage = **0.0540 bpb** (≈ 3.0% relative), i.e. **9.4σ** of proj_512's own training-seed
  std and **21.6σ** of baseline's (Welch t ≈ 19.2). This clears the 2σ adoption threshold by an
  order of magnitude.
- The two seed distributions are **completely non-overlapping**: the *worst* proj_512 seed
  (1.7432) still beats the *best* baseline seed (1.7861) by 0.0429 bpb. The single-run −0.0588
  gain was therefore not a lucky seed — it reproduces (slightly smaller, 0.0540) across every
  seed pairing.

**CORE (CONFIRMATORY) — cannot resolve the deltas → DROP as a d12 selection metric.**
- Per-variant CORE difference is only 0.0090 (proj 0.0630 vs base 0.0541), **Welch t ≈ 1.67,
  i.e. < 2σ** — not significant.
- The dominant noise is **training-seed** variance: std ≈ 0.0076 (baseline) / 0.0093 (proj_512),
  both **larger than the ~0.005 delta** the metric would need to resolve and larger than the
  0.0090 between-variant gap itself. The tight per-row eval-seed std (`CORE_std` 0.0004–0.0030)
  is misleading precision — it bounds few-shot sampling noise only, while run-to-run variance is
  ~3–4× larger and makes the variant CORE ranges overlap heavily (proj 0.0464–0.0684 vs base
  0.0448–0.0658).
- Concretely, proj_512 seed s4 collapses to CORE 0.0464 — squarely inside the baseline range —
  even though the *same* checkpoint still wins on val_bpb (1.7432, below every baseline). CORE
  and val_bpb disagree at the seed level, the noise-signature predicted earlier. Even at ≥5 eval
  seeds CORE does not separate the variants, so it is dropped as a d12 selection metric.

#### Anomalies
- `proj_512_s4` is the weakest proj_512 seed on *both* metrics (val_bpb 1.7432, CORE 0.0464); it
  still beats all baselines on val_bpb but drives most of proj_512's CORE std. No checkpoint is
  missing; the legacy single-run tags (`*_2b0bc792`, `*_9077cd29`, etc.) remain only for
  cross-reference and agree with the seeded means.

## d20 depth-scaling phase

### Results

_Pending — d20 baseline vs proj_512, 3 training seeds per arm (6 runs). To be filled with
val_bpb mean ± std per arm once the runs complete._

### Conclusions

_Pending._

## Conclusions (d12)

**Adopt `embed_proj_dim=512` as the d12 default.** Across 5 training seeds the zero-initialized
low-rank embedding correction lowers val_bpb by **0.0540 ± (proj std 0.0058)** — from
1.7889 ± 0.0025 (baseline) to **1.7349 ± 0.0058** (proj_512). That advantage is **9.4σ** of the
run-to-run (training-seed) variance — far past the 2σ adoption rule — with the two seed
distributions **completely non-overlapping** (worst proj_512 seed beats best baseline seed by
0.0429 bpb). The earlier single-run −0.0588 gain was real, not a lucky seed.

**Drop CORE as a d12 selection metric.** Even with ≥5 eval seeds, CORE cannot resolve these
variants: the proj_512−baseline difference is only 0.0090 (Welch t ≈ 1.67, < 2σ), and the
**training-seed** std (0.0076–0.0093) exceeds both the ~0.005 delta of interest and the
between-variant gap itself. Tight per-row eval-seed `CORE_std` (0.0004–0.0030) is misleading
precision — it bounds few-shot sampling only, not run-to-run variance. CORE even disagrees with
val_bpb at the seed level (proj_512 s4: best-tier val_bpb but baseline-tier CORE). This confirms
the earlier noise-signature observation. **val_bpb is the trustworthy primary metric at d12;**
CORE should not gate d12 model selection.

**Recommended next steps (in priority order):**
1. **Promote `embed_proj_dim=512` to the d12 default** in the training config and use it as the
   baseline for subsequent d12 work.
2. **Test the projection at greater depth** (e.g. d20/d26) to see whether the val_bpb gain
   persists and scales — and whether CORE *does* separate variants once the model is large
   enough that argmin decisions flip; CORE only becomes a candidate selection metric there.
3. **Stop chasing CORE at d12.** Do not spend further eval-seed budget trying to resolve d12
   CORE deltas; rely on val_bpb. Resume the deferred [[low_rank_unembedding]] follow-up on the
   same val_bpb-primary footing.

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
- 2026-06-13: **Multi-seed results in — decision reached.** 5 training seeds × ≥5 eval seeds.
  val_bpb: baseline 1.7889 ± 0.0025 vs proj_512 1.7349 ± 0.0058; advantage 0.0540 bpb = **9.4σ**
  of training-seed variance (Welch t ≈ 19.2), seed distributions completely non-overlapping →
  **adopt `embed_proj_dim=512` as the d12 default.** CORE: baseline 0.0541 ± 0.0076 vs proj_512
  0.0630 ± 0.0093, difference 0.0090 (Welch t ≈ 1.67, < 2σ); training-seed std exceeds the
  ~0.005 delta and CORE disagrees with val_bpb at the seed level (proj_512 s4) → **drop CORE as
  a d12 selection metric**, confirming the noise-signature observation. Filled the multi-seed
  Results and rewrote Conclusions. Next: promote 512 to default, test at greater depth.
- 2026-06-13: Started the **d20 depth-scaling phase**. Extended the Hypothesis to ask whether
  the input-projection val_bpb gain scales with depth (persists/grows at d20) or washes out as
  capacity increases. Two arms — d20 `embed_proj_dim=512` vs d20 `embed_proj_dim=0`, 3 training
  seeds each (6 runs); primary metric val_bpb as mean ± std per arm. CORE is **not** a gate this
  phase (d12 run-to-run CORE variance ±0.008 exceeds the deltas of interest). Results/Conclusions
  left as placeholders. d12 content kept intact (decisive result above).
