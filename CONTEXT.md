# Project Glossary (CONTEXT.md)

Canonical domain terms for the nanochat experiments repo. Glossary only — no
implementation detail. Update inline as terms are resolved.

## Generic base (a.k.a. base repo / generic harness)
The `main` branch: code useful for *any* experiment, with no experiment-specific
mechanism implementations. Parent commit of every long-lived experiment branch.
_Avoid:_ "the repo", "trunk" (ambiguous about what it contains).

## Mechanism
A single experiment-specific feature woven into the model/data/training code. There
are three: **sentence attention**, **low-dim projection embedding**, **low-rank
unembedding**. Each is owned by exactly one long-lived experiment branch and is
absent from the generic base.
_Avoid:_ "feature" (overloaded), "experiment" (an experiment is a config/run, not the code).

## Long-lived experiment branch
A branch cut from the generic base that carries exactly one mechanism, is maintained
indefinitely, and is **never merged into `main`** (and `main` is never merged into it).
Named `main-<mechanism>` (e.g. `main-sentence-attention`).
_Avoid:_ "feature branch" (implies it will be merged), "fork".

## Sentence attention
Mechanism that replaces causal attention with a block-causal + global-gist mask
(per-document), inserting K gist / end-of-sentence tokens at NLTK sentence boundaries.
Owned by `main-sentence-attention`.
_Avoid:_ "gist attention" alone (gists are one part), "segment attention".

## Low-dim projection embedding
Mechanism adding a low-rank learnable correction to the token embedding
(`embed_proj_dim`: `low_dim_embed` → `embed_proj` summed with `wte`). Owned by
`main-low-dim-projection`.
_Avoid:_ "linear projection" alone (ambiguous), "projection embeddings".

## Low-rank unembedding
Mechanism adding a LoRA-style low-rank correction to the output logits
(`unembed_proj_dim`: `unembed_proj_down` → `unembed_proj_up` added to `lm_head`).
Distinct from low-dim projection embedding (output side, not input side). Owned by
`main-low-rank-unembedding`.
_Avoid:_ conflating with "low-dim projection embedding" — different side of the model.
