# 2. Low-rank unembedding gets its own branch despite being an orphan

Date: 2026-06-18
Status: Accepted

## Context
Low-rank unembedding (`unembed_proj_dim`) is a third mechanism in `gpt.py`, with a
`--unembed-proj-dim` flag and an `experiments/low_rank_unembedding.md` plan, but **no
active training config** in `run_training.py` and no trained checkpoint. Full removal
from `main` forces a decision on where it goes. It is neither sentence attention nor
low-dim *embedding* (it acts on the output/logit side, not the input embedding side),
so folding it into `main-low-dim-projection` would conflate two distinct mechanisms.

## Decision
Give it a dedicated long-lived branch `main-low-rank-unembedding` (+ worktree),
preserving the code rather than deleting it, even though it is currently an orphan with
no active experiment.

## Consequences
- The mechanism's code and plan are preserved and clearly separated for future revival.
- A third branch/worktree to maintain for code that is not currently exercised.
- No dedicated unit test exists upstream, so verification relies on import-smoke plus a
  minimal forward-shape check rather than a full test (an alternative to deleting it,
  which we rejected to avoid losing the implementation).
