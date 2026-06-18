# 1. Experiment mechanisms live on long-lived, never-merged branches; main is a generic base

Date: 2026-06-18
Status: Accepted

## Context
`main` currently carries three experiment-specific mechanisms — sentence attention,
low-dim projection embedding, and low-rank unembedding — woven into shared files
(`gpt.py`, `dataloader.py`, `tokenizer.py`, `base_train.py`, `base_eval.py`,
`checkpoint_manager.py`, `run_training.py`). The mechanisms are no-ops when disabled,
so they could be defended as reusable infrastructure. We want the base repo to contain
only code useful for *any* experiment, with each mechanism maintained in isolation.

## Decision
Strip `main` of all three mechanisms (full mechanism removal — implementation, CLI
flags, plans, and dedicated tests), making `main` a generic nanochat harness. Each
mechanism gets its own long-lived branch cut from the current main commit `edbc628`,
carrying **only** that one mechanism (the other two cross-removed). These branches are
never merged into `main`, and `main` is never merged into them. Branches are created
before `main` is stripped so no code is lost. As a consequence, the generic base must
tolerate loading existing checkpoints whose `meta.json` carries config keys the
stripped `GPTConfig` no longer declares (drop unknown keys with a warning).

## Consequences
- Clean separation: each mechanism has a single, focused maintenance home.
- Divergence cost: generic fixes on `main` must be manually cherry-picked into the
  branches; there is no automatic sync. The branches will drift over time.
- Existing checkpoints that *used* a removed mechanism are only loadable on that
  mechanism's branch; the generic base can load only checkpoints where the feature was
  disabled (thanks to unknown-key tolerance).
- Hard to reverse: re-merging would require reconciling intentional divergence.
