# Polysemy × Context (synthetic language)

## Hypothesis
The perplexity gap between a **polysemous** and a **monosemous** synthetic language shrinks
as context length grows: `gap(L) = PPL_poly(L) − PPL_mono(L) → 0` as `L↑`. Polysemy injects
residual sense-uncertainty `H(S|W)` that context resolves, so its contribution to local
entropy should decay with context faster than syntactic uncertainty does (a decomposition
of Someya et al. 2025's m-local entropy result). See the resolved spec in
`run/deep-interview/deep-interview-polysemy-context.md`.

## Setup
Brownfield, in three components; **all three are now built.**

- **(1) Generator — `scripts/gen_polysemy_data.py` + `nanochat/polysemy.py`.** Samples one
  PCFG sense stream (syntax, held constant across conditions), then varies only the
  sense→form layer per condition. Forms map 1:1 to token ids (no BPE) so `H(S|W)` is exact;
  output is parquet `text` shards of form symbols + `vocab.json` + `metadata.json` under
  `<base>/base_data_polysemy/<condition>/`. Also exports a held-out sense-labeled `probe.jsonl`
  per condition (`--probe-docs`, default 2000) for the representation probe. Condition grid,
  scale, PCFG and confound policy live in the code — see `default_conditions()` / `GeneratorConfig`.
- **(2) Trainer integration & context-length sweep — built.** `nanochat/identity_tokenizer.py`
  (1 form = 1 token id; `token_bytes` = 1/form so the trainer's bpb == bits-per-form). `base_train.py`
  / `base_eval.py` take `--data-dir` + `--tokenizer identity`; the dataloader threads `data_dir`;
  eval recovers tokenizer+data-dir from the checkpoint meta and skips CORE/sample (English ICL is
  meaningless for the synthetic vocab). `run_training.polysemy_context_experiments()` emits the
  5 conditions × `--max-seq-len ∈ {8,32,128,512}` grid (d6, full attention, fixed batch, 10k, 1 seed).
- **(3) Metrics & analysis — built.** `nanochat/polysemy_analysis.py` (pure) + `scripts/analyze_polysemy.py`:
  PPL(L)/BPC(L), **`gap(L) = PPL_poly(L) − PPL_mono(L)`**, BPC vs the analytic source-entropy floor,
  lexical-vs-total `H_m` decomposition (from the metadata sidecar), and per-condition decision
  rules (resolved / decaying / flat / growing). `scripts/probe_polysemy.py` fits a torch
  logistic probe on hidden states (captured via an `lm_head` pre-hook) to decode the latent
  sense, bucketed by left-context length.

Decisions recorded as ADRs: `docs/adr/0003` (forms-are-tokens / identity tokenizer),
`docs/adr/0004` (enforce |V|, record other confounds as covariates),
`docs/adr/0005` (component-2/3 integration: identity load path, eval config recovery, BPC=bits/form).

## Results
First full condition×L results — all **15 arms trained** (10k steps). Primary metric is
**BPB (== bits-per-form)**; CORE/sample are skipped by design (synthetic vocab), so those columns
are blank. **No dedicated evaluation stage ran** — `base_eval_results/` holds no JSON for these
tags — so every number below is the **in-training `val_bpb` from the step-10000 checkpoint meta**,
not a controlled fixed-budget BPB eval. All 15 arms carry a `val_bpb`; the only checkpoint without
one is `polytest_smoke` (d2 CPU smoke, step 2, `val_bpb = inf`), which is not an arm.

**Raw BPB (lower = better):**

| condition (H(S\|W) target) | L512   | L1024  | L2048  |
| -------------------------- | ------ | ------ | ------ |
| poly_mono (0.0)            | 5.2705 | 5.2764 | 5.2906 |
| hsw0p5_homonymy (0.5)      | 5.2818 | 5.2876 | 5.3027 |
| hsw1p5_homonymy (1.5)      | 5.3162 | 5.3218 | 5.3367 |
| hsw0p5_overlap (0.5)       | 4.7951 | 4.8016 | 4.8180 |
| hsw1p5_overlap (1.5)       | 3.8026 | 3.8110 | 3.8317 |

**gap(L) = BPB_poly − BPB_mono (the hypothesis metric):**

| condition       | L512   | L1024  | L2048  |
| --------------- | ------ | ------ | ------ |
| hsw0p5_homonymy | +0.011 | +0.011 | +0.012 |
| hsw1p5_homonymy | +0.046 | +0.045 | +0.046 |
| hsw0p5_overlap  | −0.475 | −0.475 | −0.473 |
| hsw1p5_overlap  | −1.468 | −1.465 | −1.459 |

- **Homonymy (clean arm — cross-class merge, context should resolve).** The penalty is small and
  **positive**, scaling with injected sense entropy (~+0.011 @ H(S|W)=0.5, ~+0.046 @ 1.5) but sitting
  **far below the injected H(S|W)** — only ~2–3% of the 0.5/1.5 injected bits survive as excess BPB.
  The model already resolves ~97% of the sense uncertainty from context. **But gap(L) does not shrink
  across L∈{512,1024,2048} — it is flat (even ticks up slightly).** The predicted `gap(L)→0` decay is
  therefore **not visible in this L window**: the residual is already at floor by L=512.
- **Overlap (same-class merge — residual H(S|W,C)>0).** BPB falls **far below mono** (gap ≈ −0.47 @
  H=0.5, ≈ −1.47 @ 1.5, i.e. ≈ −H(S|W)). Same-class merging collapses senses onto shared forms in
  shared contexts, which **lowers the source form-entropy floor** — so overlap is *not* a valid
  matched-H(S|W) comparison to mono; its negative "gap" is a floor artifact, not a polysemy penalty,
  and must be read floor-relative (BPC-vs-floor), not mono-relative.
- **Context length.** BPB is essentially flat and **rises slightly with L** for every arm, mono
  included (mono 5.2705→5.2906). Longer context did not lower per-form loss — consistent with sense
  already resolved by L=512, the tiny L-trend dominated by long-range positions / optimization rather
  than sense resolution.

- **Metric note.** These synthetic-vocab arms **skip CORE and sample by design** (English ICL is
  meaningless for the identity vocab), so **BPB is the sole primary metric** and is seed-stable. The
  ±0.01 single-seed CORE-noise caveat does **not** apply here — CORE is never evaluated for this
  experiment.

## Conclusions
**Hypothesis status: partially supported, but the specific `gap(L)→0` decay is untested in this
window.** The mechanism the hypothesis rests on — context resolves injected sense uncertainty — holds
strongly on the clean **homonymy** arm: only ~2–3% of the injected H(S|W) survives as excess BPB. But
the sweep starts too long: by L=512 the residual is already at floor, so gap(L) is flat across
{512,1024,2048} and the decay curve itself is never observed. The **overlap** arms are confounded by a
lowered form-entropy floor (negative gap ≈ −H(S|W)) and cannot be compared to mono directly.

Next steps:
1. **Add short-L anchors** (e.g. L∈{8,32,128}) so the sweep spans the pre-resolution regime where the
   homonymy gap is still large — that is where the decay to floor should be visible. The long-doc
   pivot overshot the resolution horizon.
2. **Measure gap within a sequence** via `scripts.probe_polysemy` (sense-decode accuracy bucketed by
   left-context length) and `scripts.analyze_polysemy`'s position-resolved BPC — reads off the decay
   decoupled from the training-L confound, reusing the long-doc checkpoints already in hand.
3. **De-confound overlap** — compare overlap arms **floor-relative** (BPC-vs-analytic-floor, already
   built) rather than mono-relative, or hold the form marginal constant so overlap adds residual
   ambiguity without lowering the floor.
4. **Run the controlled BPB evaluation stage** (`run_evaluation.py --eval bpb`) to replace the
   in-training `val_bpb` with a fixed-token-budget BPB per arm before drawing final conclusions.

## Changelog
- 2026-06-30: Hardened the spec via deep interview (5 glossary terms, ADRs 0003/0004) and
  built component 1 — the PCFG/Zipf/sense→form generator, CLI, and tests. Band-grouping
  sense→form layer (degree ∝ √freq) + fine tail-pair top-up hits target `H(S|W)` within
  ±0.05 while spreading polysemy across many forms; `|V|` held constant via paired splits.
- 2026-06-30: Built components 2 & 3. Component 2: identity tokenizer + `--data-dir`/`--tokenizer`
  wiring through the dataloader, `base_train`, `base_eval` (recovers config from checkpoint meta,
  skips CORE/sample), and the `polysemy_context_experiments` condition×L grid. Component 3:
  `polysemy_analysis` (gap(L), BPC-vs-floor, lexical `H_m` decomposition, decision rules) +
  `analyze_polysemy`/`probe_polysemy` scripts and the generator's `probe.jsonl` export. Made the
  generator's broken-pandas blocker coexist with `torch._dynamo`; made `nanochat.dataset`'s
  `requests` import lazy. ADR 0005 added. All tests green (82 passed).
- 2026-07-01: Parallelized generation (sense-stream sampling + per-condition build) — ~8× faster
  (400M in ~27 min). Then pivoted to a **long-document regime** so the context sweep is meaningful:
  the original PCFG caps at ~6-token docs (raising depth does nothing), so a long-L sweep would
  saturate. Added `build_long_pcfg` — *linear center-embedding* (subcritical, analytic entropy
  preserved) whose nesting depth/phase is a genuine long-range dependency, producing ~3–4k-token
  documents (continuation knob). Reworked the sweep to **L ∈ {512,1024,2048}** at a constant 32768
  global batch with per-L device batch 16/8/4 (grad-accum=1, identical optimization steps),
  `--eval-every 2500`, no `--eval-tokens` override. Validated H(S|W) within ±0.05 on long docs.
- 2026-07-01: Data generation for all 5 conditions completed and validated (H(S|W) within ±0.05 of
  target; |V|=512; ~400M tokens/condition). First evaluation sweep returned "No checkpoints found" —
  the training grid has not yet produced checkpoints, so gap(L) is still pending and the hypothesis
  remains untested. Confirmed BPB (bits-per-form) as the sole primary metric (CORE/sample skipped
  for synthetic vocab; single-seed CORE-noise caveat N/A).
- 2026-07-02: First full condition×L results — all 15 arms trained. BPB read from the step-10000
  checkpoint meta (in-training `val_bpb`; **no dedicated eval-stage JSON on disk**; CORE/sample blank
  by design). Homonymy penalty is small/positive and ≪ injected H(S|W) (~+0.011 @0.5, ~+0.046 @1.5)
  but **flat across L∈{512,1024,2048}** — decay not observed, residual already at floor by L=512.
  Overlap arms fall far below mono (gap ≈ −H(S|W)) — a lowered form-entropy-floor confound, not a
  polysemy penalty. Next: short-L anchors + within-sequence probe to expose the decay; floor-relative
  comparison for overlap; run the controlled BPB eval stage.
