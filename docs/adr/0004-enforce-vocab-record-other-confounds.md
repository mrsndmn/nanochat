# 4. Enforce |V| exactly, record other confounds as covariates

Date: 2026-06-30
Status: Accepted

## Context
The spec's confound table uses the word "match" for several quantities (unigram entropy,
gzip-compressibility, `|V|`) to keep conditions causally comparable. Hard-enforcing all of
them *simultaneously* at generation time is brittle: unigram-entropy matching, gzip
matching and `|V|` matching can fight the primary `H(S|W)` target, and a feasible joint
solution may not exist for every condition. The spec itself hedges — for `|V|` it offers
"pair each merge with a split **OR** model `|V|` as covariate."

## Decision
The v1 generator **enforces only what is exact and cheap**: `|V|` is held equal across
conditions by pairing each merge (2 senses→1 form) with a split (+1 form), and the
analytic PCFG entropy is computed and exported. **All other confounds** — unigram entropy,
gzip size, lexical/total `H_m`, per-form `H(S|W)`, and the synonymy level introduced by
splits — are **measured and written to a metadata sidecar** as covariates for the
(deferred) analysis to control statistically, rather than matched during generation.

## Consequences
- The generator stays simple and robust: it never has to solve an infeasible joint
  matching problem, and the primary `H(S|W)` target is not compromised.
- Causal cleanliness shifts partly from design-time control to analysis-time covariate
  adjustment; the analysis (component 3) **must** use the recorded covariates or causal
  claims weaken.
- Paired splits introduce synonymy (many forms→one sense), a side effect that is recorded
  rather than hidden, so it can be regressed out later.
- The acceptance bar is concrete: each condition hits target `H(S|W)` within ±0.05 bits,
  `|V|` equal across conditions, full metadata present, corpus reloads 1:1.
- Rejected: hard-enforce all confounds (brittle, possibly infeasible) and enforce-nothing
  (weakest causal footing; `|V|` imbalance is cheap to avoid so we avoid it).
