# 3. Forms are tokens via an identity tokenizer, bypassing BPE

Date: 2026-06-30
Status: Accepted

## Context
The polysemy×context experiment defines a **form** as "the surface token the LM sees"
and controls per-form sense-ambiguity `H(S|W)` exactly, so that analytic ground-truth
entropy is known. nanochat's data path, however, consumes raw **text** and tokenizes it
on the fly with a BPE tokenizer (`nanochat/dataloader.py:119`,
`tokenizer.encode(doc_batch, prepend=bos_token, ...)`, vocab 32768). Running synthetic
forms through BPE would split/merge them into subword pieces, making the form→token map
non-injective and destroying the controlled `H(S|W)` the whole experiment rests on.

## Decision
The generator emits each form as a **whitespace-separated symbol** in the parquet `text`
column plus a **fixed `vocab.json` form→id table**, and the corpus is consumed (in the
deferred component 2) by a trivial **identity tokenizer**: one form ↔ exactly one token
id, no merges or splits. The BPE tokenizer is deliberately bypassed for this experiment.

## Consequences
- The 1:1 form↔token invariant holds end to end, so per-form and corpus-level `H(S|W)`
  stay analytically exact — the experiment's measurement instrument is sound.
- The corpus stays human-inspectable (readable symbol streams) and reuses the existing
  text→parquet pipeline shape (zstd, row-groups of 1024), minimizing trainer changes.
- A new identity-tokenizer code path and a polysemy-specific data dir
  (`base_data_polysemy/...`, not `base_data_climbmix/`) are required in component 2.
- The model's vocab size becomes `|V|` (hundreds), far smaller than 32768 — model init
  and embedding/unembedding sizing in the trainer must read it from the corpus vocab.
- Rejected alternatives: pre-tokenized integer shards (needs a deeper dataloader rewrite
  and loses human-inspectability) and "emit words, train BPE" (re-introduces the exact
  merge/split problem this decision avoids).
