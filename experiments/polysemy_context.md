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
- **Component 1** validated (`tests/test_polysemy_generator.py`) and smoke-run at K=512: every
  condition hits its target `H(S|W)` within ±0.05 bits, `|V|` constant at 512, polysemy spread
  across many forms (meaning-frequency law), corpora reload as a 1:1 form↔token stream.
- **Components 2 & 3** validated end to end (CPU smoke + unit tests, 82 passing): `base_train`
  trains a d2 identity-tokenizer model and saves a checkpoint+meta; `base_eval --tokenizer auto`
  recovers the config from meta, skips CORE, and reports bpb == bits/form (= loss/ln2);
  `analyze_polysemy` produces the PPL/BPC/gap tables + verdicts from checkpoint metas + the
  generator metadata; `probe_polysemy` extracts hidden states and fits the sense-decoding probe.
  The training/eval grid (20 jobs) is ready to launch; the headline `gap(L)` numbers come from
  running it.

## Conclusions
The full pipeline (generate → train across L → analyze gap(L) + probe) is implemented and tested.
Next: run `scripts.gen_polysemy_data`, launch `run_training.polysemy_context_experiments` (20
arms), then `scripts.analyze_polysemy` + `scripts.probe_polysemy` to read off whether
`gap(L) → 0` (expected for homonymy) vs plateaus above 0 (expected for overlapping polysemy).

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
