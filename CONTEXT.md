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

## Sense
A latent class `s ∈ {1..K}`; the terminal symbol of the PCFG. The unit the language's
syntax is defined over and that carries meaning. Never seen directly by the LM.
_Avoid:_ "word", "token" (those are forms), "concept" (vague).

## Form
A surface token the LM actually sees. In this experiment the form→token map is the
**identity** — exactly one form equals exactly one token id (no BPE merging) — so the
controlled per-form sense-ambiguity is preserved end to end.
_Avoid:_ "word" (overloaded), "subword"/"BPE token" (a form is never split).

## Polysemy
A non-injective sense→form map: multiple senses share one form, so a form carries
residual sense-uncertainty `H(S|W=w) > 0`. The monosemous control is the bijective
case `H(S|W)=0`. Split into **homonymy** (merged senses have disjoint context
distributions → context resolves them) vs **overlapping polysemy** (merged senses
share context mass → residual `H(S|W,C) > 0`).
_Avoid:_ "ambiguity" alone (covers syntactic uncertainty too); "synonymy" (the inverse:
many forms → one sense).

## Identity tokenizer
A trivial whitespace tokenizer mapping each form symbol to exactly one token id (fixed
vocab, no merges/splits), so the corpus stays a 1:1 form↔token stream. The counterpart
of nanochat's BPE tokenizer, which this experiment deliberately bypasses.
_Avoid:_ "tokenizer" unqualified (ambiguous with the BPE tokenizer).

## Context-overlap
A generator knob in [0,1] controlling how much the *context distributions* of the
senses merged onto one form coincide. `0` = pure **homonymy** (merged senses occur in
disjoint contexts → context fully disambiguates them); `partial` = the senses share
context mass → residual `H(S|W,C) > 0` that context cannot remove. Distinct from the
amount of polysemy `H(S|W)`: overlap is about *resolvability*, not quantity.
_Avoid:_ "ambiguity overlap", conflating it with the `H(S|W)` level.

## gap(L)
The headline readout: `gap(L) = PPL_poly(L) − PPL_mono(L)`, the extra next-form
perplexity a polysemous condition pays over the monosemous baseline at context length
`L` (model sequence length). The hypothesis is `gap(L) → 0` as `L` grows for homonymy
(context resolves the sense) and plateaus above 0 for overlapping polysemy. Also tracked
in bits/form (`gap_bpc`). Computed by `nanochat/polysemy_analysis.py`.
_Avoid:_ "perplexity gap" unqualified (always relative to the monosemous baseline at the
same `L`); confusing it with `H(S|W)` (the injected ambiguity, not the measured penalty).

## BPC (bits per form)
The per-token cross-entropy in bits (`loss / ln2`). Under the identity tokenizer one form
= one token = one "byte", so BPC equals the trainer's bpb and is directly comparable to
the analytic source-entropy floor (PCFG bits/sense). _Avoid:_ "bits per byte" (true only
because each form is defined to be 1 byte here) and "bits per character" in the
natural-text sense.
