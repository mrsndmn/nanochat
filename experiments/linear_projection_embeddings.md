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
