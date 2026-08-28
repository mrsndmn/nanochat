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

### Iteration 2 — Engram sparse bigram memory

Iteration 1 established that *context-derived* content is redundant (the trunk computes it
anyway) while the channel itself is used at full strength (open α gates). Iteration 2 tests
the axis the null left open: **stored parametric memory the trunk cannot compute from
context**. A zero-init hashed bigram table (engram-lite recipe, dev/LOG 2026-01-27; token-
level variant was this repo's best low-dim-projection result) feeds retrieved n-gram
associations into the hypernet's key/value inputs; the K slot queries select over them, and
the own-sentence mask keeps retrieval sentence-local. Two arms sweep table capacity
(2^18 vs 2^20 rows × 128). Zero-init ⇒ bit-exact equal to the gated arm at init;
dead-path-safe because the projection is nonzero and α is known to open. Win gate unchanged
(val BPB ≤ 0.8161); post-hoc readouts: α gates + table row-norm stats (used capacity).
Param-fairness caveat: the table adds ~33M/~134M embedding-like params — inherent to the
Engram thesis (sparse memory is cheap capacity), reported, not matched.

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

End-of-training (step 10000) evaluation, val BPB (primary) and CORE (reference-only):

| Arm | val BPB | Δ vs control | CORE (mean over 5 eval seeds) |
|-----|---------|--------------|-------------------------------|
| control `d12_sa_nltk_k8` (fixed gists) | 0.81906 | — | 0.1435 |
| `hnet_gated` | 0.81826 | −0.00080 | 0.1464 ± 0.0018 |
| `hnet_forced` | 0.81863 | −0.00043 | — |

- **Win gate (≤ 0.8161, i.e. −0.003) NOT met.** The gated delta (−0.0008) is below even the
  −0.001 iterate-signal threshold; both deltas sit in the range static-init tweaks reached.
- **Mechanistic readout — the gate OPENED**: final per-slot alphas from the gated checkpoint
  are far from zero: `[0.995, 1.169, 1.796, 1.044, 0.809, −1.218, −0.674, −1.466]`
  (mean |α| = 1.15). The model actively consumed the content channel.
- **Forced ≈ control** (−0.0004): pure content-conditioned embeddings with no fixed-row
  fallback lose nothing — the content channel is fully *sufficient*, not harmful.

## Conclusions

**Null, with a sharp mechanism story: content-conditioned gist input embeddings are USED but
REDUNDANT.** The alpha readout rules out the "gate stayed shut" null mode (ADR 0002): the
model wired the hypernetwork in at full strength, and the forced arm shows sentence-content
embeddings alone suffice — yet neither moves BPB beyond noise. The natural reading is that in
the strict regime the trunk's own attention already writes the sentence's content into gist
positions (gist queries see their whole sentence block across 12 layers), so injecting the
same information at the input adds nothing the model couldn't compute. The strict-regime gap
to full causal (~0.017 BPB) therefore does not live in gist *input content*; it lives
elsewhere (gist channel bandwidth K, or block-causality itself).

**Implication for the deferred TTT follow-up:** TTT gist refinement would inject
gradient-computed content into this same input channel. With amortized content already shown
redundant, input-level TTT is a low-odds bet at this regime and should not proceed as
designed; a TTT variant would need to target a different bottleneck (e.g. refining deeper
gist KV states, not input embeddings).

Next steps considered: (a) close the strict-regime gap via channel capacity (wider gist
bandwidth / windows) rather than content; (b) port nothing to the alt regime — with gists
barely load-bearing there, a redundant-content mechanism has even less room.

## Changelog

- 2026-08-27: Deep-interview spec resolved (strict K=8 host regime, −0.003 BPB win gate,
  gated+forced two-arm design). Implemented `GistHypernet` (masked cross-attention, per-slot
  zero-init alpha gates, nonzero c_proj init per the dead-path lesson) + experiment configs +
  tests. TTT gist refinement explicitly deferred pending this round's outcome.
- 2026-08-27: Submitted both training jobs from branch `experiment/gist-hypernetwork-1`
  (commit `cb02a09`):
  - `d12_sa_nltk_k8_hnet_gated` → `lm-mpi-job-30b3a922-9b55-471a-ab09-9df09482b27f`
  - `d12_sa_nltk_k8_hnet_forced` → `lm-mpi-job-5a858e88-261b-43c9-94cb-a897797eb153`
- 2026-08-28: Both training jobs completed (~3h45m each, error_code 0); step-10000
  checkpoints verified on disk. Eval jobs submitted and completed
  (`lm-mpi-job-283ab0b3-5ac6-4d77-b1ff-79b595398078` gated,
  `lm-mpi-job-b5f7bb10-22b9-428c-a353-a446e09db555` forced). Results + conclusions
  recorded: NULL vs the −0.003 win gate; alpha gates open (mean |α|=1.15) ⇒ content
  channel used-but-redundant; input-level TTT follow-up deprioritized.
- 2026-08-28: Iteration 2 implemented — Engram-style sparse bigram memory in the hypernet
  KV stream (`gist_hypernetwork_engram_experiments`, `--gist-engram-bits/-dim`,
  `_bigram_engram_lookup` kept eager via compiler.disable per the Inductor int32 lesson).
  Two capacity arms (2^18, 2^20 × 128), bit-exact vs the gated arm at init.
- 2026-08-28: Extended the engram sweep to 4 capacity arms (2^17..2^20 × 128) and submitted
  all training jobs from commit `032ed79`:
  - `d12_sa_nltk_k8_hnet_gated_eng_b17` → `lm-mpi-job-1abca184-9ef1-4231-8fa0-3b05bb55bcec`
  - `d12_sa_nltk_k8_hnet_gated_eng_b18` → `lm-mpi-job-86c694a4-07f0-497d-b955-5157affa5884`
  - `d12_sa_nltk_k8_hnet_gated_eng_b19` → `lm-mpi-job-d07cc51a-3cb3-4125-a1cc-b0711e8512ef`
  - `d12_sa_nltk_k8_hnet_gated_eng_b20` → `lm-mpi-job-00301972-7ac5-400d-9415-6d304f116f1b`
  Evaluation (bpb+core) chained automatically after training completion.
