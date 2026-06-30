# Polysemy × Context (synthetic language)

## Hypothesis
The perplexity gap between a **polysemous** and a **monosemous** synthetic language shrinks
as context length grows: `gap(L) = PPL_poly(L) − PPL_mono(L) → 0` as `L↑`. Polysemy injects
residual sense-uncertainty `H(S|W)` that context resolves, so its contribution to local
entropy should decay with context faster than syntactic uncertainty does (a decomposition
of Someya et al. 2025's m-local entropy result). See the resolved spec in
`run/deep-interview/deep-interview-polysemy-context.md`.

## Setup
Brownfield, in three components; **only component 1 (the dataset generator) is built so far.**

- **(1) Generator — `scripts/gen_polysemy_data.py` + `nanochat/polysemy.py`.** Samples one
  PCFG sense stream (syntax, held constant across conditions), then varies only the
  sense→form layer per condition. Forms map 1:1 to token ids (no BPE) so `H(S|W)` is exact;
  output is parquet `text` shards of form symbols + `vocab.json` + `metadata.json` under
  `<base>/base_data_polysemy/<condition>/`. Condition grid, scale, PCFG and confound policy
  live in the code (source of truth) — see `default_conditions()` and `GeneratorConfig`.
- **(2) Trainer integration & context-length sweep — TODO.** Identity-tokenizer wiring +
  `base_train.py` over `--max-seq-len ∈ {8,32,128,512}`; `run_training.py` configs.
- **(3) Metrics & analysis — TODO.** m-local entropy decomposition, BPC vs analytic minimum,
  `gap(L)`, probes, decision rules.

Decisions recorded as ADRs: `docs/adr/0003` (forms-are-tokens / identity tokenizer),
`docs/adr/0004` (enforce |V|, record other confounds as covariates).

## Results
Component 1 validated (`tests/test_polysemy_generator.py`, 11 passing) and smoke-run at
K=512 over the default grid: every condition hits its target `H(S|W)` within ±0.05 bits
(0.00 / 0.46 / 0.46 / 1.50 / 1.48), `|V|` held constant at 512 across conditions, polysemy
spread across many forms per the meaning-frequency law (≈11–12 polysemous forms at
`H(S|W)=0.5`, ≈114–172 at `H(S|W)=1.5`), and corpora reload as a 1:1 form↔token stream.

## Conclusions
The synthetic-language generator is ready; the corpora carry exact analytic `H(S|W)` and
the recorded covariates the analysis needs. Next: build component 2 (identity-tokenizer
load path + per-condition context-length training configs), then component 3 (the `gap(L)`
readout and entropy decomposition).

## Changelog
- 2026-06-30: Hardened the spec via deep interview (5 glossary terms, ADRs 0003/0004) and
  built component 1 — the PCFG/Zipf/sense→form generator, CLI, and tests. Band-grouping
  sense→form layer (degree ∝ √freq) + fine tail-pair top-up hits target `H(S|W)` within
  ±0.05 while spreading polysemy across many forms; `|V|` held constant via paired splits.
