# Gist Hypernetwork — content-conditioned gist embeddings (d12, strict regime)

## Hypothesis

In the **strict block-causal regime** — where gist tokens are the *only* cross-sentence
channel — replacing the content-blind fixed gist embedding rows with **content-conditioned
embeddings produced by a Gist Hypernetwork** (per-slot learned queries cross-attending over the
completed sentence's token embeddings) improves BPB versus the fixed-gist control.

Prior content-flavored variants (static inits `eos_clone`/`real_mean`, discrete Bloom-bucket
`hash` gists, `gist_head` projection specialization) were all neutral-to-worse — but none was a
trained *continuous* sentence→embedding map, so the content-conditioning hypothesis itself is
untested. There is a live counter-hypothesis that gist input content is simply ignored by the
trunk; the two-arm design below makes that outcome cleanly diagnosable (ADR 0002).

**Win condition** (decision rule): val BPB ≤ **0.8161** at the end of training, i.e. ≥0.003
better than the fixed-gist control `d12_sa_nltk_k8` (0.8191) — ~3× the largest static-init
effect ever measured and far above seed noise (~1e-4…5e-4). A −0.001…−0.003 delta is an
iterate signal, not a claim. CORE is reference-only (±0.01 single-seed noise).
**Null diagnosis**: read the final per-slot alpha gates from the gated checkpoint — alpha≈0
across slots means the content channel went unused (honest null); forced-arm-worse +
gated-at-control means the channel is useless/harmful and falsifies the direction, including
the deferred test-time-training follow-up.

## Setup

- **Training function (source of truth):** `gist_hypernetwork_experiments` in
  `scripts/jobs/run_training.py` — 2 arms (`gated`, `forced`) against the existing
  `d12_sa_nltk_k8` control checkpoint (reused, not retrained). All hyperparameters, arms and
  protocol flags live there; the mechanism lives in `nanochat/gpt.py` (`GistHypernet`,
  `_gist_cross_attn_mask`, `GPT._apply_gist_hypernet`).
- Protocol matches the sentence-attention group exactly (d12, 10k steps, single seed,
  `a100.4gpu`, no in-training evaluation; end-of-training CORE+BPB via
  `scripts/jobs/run_evaluation.py`). The hypernet is excluded from the scaling-params horizon
  computation so both arms train under the identical schedule as the control.
- Requirements interview record: `run/deep-interview/deep-interview-gist-hypernetwork.md`
  (glossary: `CONTEXT.md`; decisions: `docs/adr/0001`, `docs/adr/0002`).

## Results

_Pending — to be filled after the post-training evaluation stage completes._

## Conclusions

_Pending._

## Changelog

- 2026-08-27: Deep-interview spec resolved (strict K=8 host regime, −0.003 BPB win gate,
  gated+forced two-arm design). Implemented `GistHypernet` (masked cross-attention, per-slot
  zero-init alpha gates, nonzero c_proj init per the dead-path lesson) + experiment configs +
  tests. TTT gist refinement explicitly deferred pending this round's outcome.
