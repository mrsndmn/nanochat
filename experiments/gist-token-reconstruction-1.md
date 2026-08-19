# Gist-Token Reconstruction — auxiliary sentence-reconstruction loss (d12)

Direct follow-up to `experiments/sentence-attention.md`; same architecture, same protocol.

## Hypothesis

In the sentence-attention architecture, **gist tokens are the only channel through which
information crosses sentence blocks** — a token attends to its own sentence block plus the gist
tokens emitted at earlier sentence boundaries. Yet **nothing in the next-token objective explicitly
forces a gist to summarize its own sentence**: the gists are free to become whatever happens to be
locally convenient for predicting the next few tokens, and any cross-sentence content they carry is
incidental.

Adding an **auxiliary reconstruction loss** — from the K gist tokens emitted at a sentence boundary,
predict/reconstruct the token ids of the corresponding (just-ended) sentence — should force the
gists into a **denser, more faithful summary** of their sentence, improving cross-sentence
information flow.

Prediction: a sentence-attention arm trained with the auxiliary reconstruction loss achieves
**better BPB (primary metric, seed-stable)** and **better CORE (secondary, reference-only)** than an
otherwise identical sentence-attention arm with the auxiliary loss disabled (`lambda = 0`).

## Setup

- **Training function (source of truth):** `gist_reconstruction_experiments` in
  `scripts/jobs/run_training.py`. It emits every arm of this experiment — each a dict with
  `model_tag`, `args`, `description`, `instance_type`, `num_gpus`, and `experiment_slug`. **All
  hyperparameters, arm lists, reconstruction weights and arg strings live in that function; this
  plan does not duplicate them.**
- **Mechanism (source of truth):**
  - `nanochat/gpt.py` — auxiliary reconstruction head on top of the gist positions + the
    reconstruction loss term.
  - `nanochat/dataloader.py` — the per-sentence segmentation info (sentence spans / gist→sentence
    association) that the loss needs as targets.
  - `scripts/base_train.py` — the new CLI flag(s) that enable the auxiliary loss and set its weight,
    alongside the existing `--gist-placement` / `--num-gist-tokens` flags.
- **Node:** `num_gpus = 4`, `instance_type = a100.4gpu` (project default).
- **Artifacts / checkpoints:** under `$NANOCHAT_BASE_DIR/base_checkpoints/<model_tag>/` (i.e.
  `artifacts/base_checkpoints/<model_tag>/` in the workspace).
- **Depth:** 12, with `--window-pattern L` (as in sentence-attention, so the comparison isolates the
  sentence mechanism rather than confounding it with the default sliding-window pattern).
- **Training horizon:** exactly **10k optimization steps** (`--num-iterations 10000`).
- **Seeds:** **single seed only** (seed 0) — one training run per config, no multi-seed fan-out
  (per project convention).

### Evaluation protocol (inherited from sentence-attention)

- **NO intermediate evaluation during training**: `--eval-every -1`, `--core-metric-every -1`,
  `--sample-every -1`. The 10k steps run uninterrupted.
- The model is scored **exactly once, at the end**, by the separate post-training evaluation stage
  (`scripts/jobs/run_evaluation.py` → `scripts/base_eval.py`) on **CORE + BPB** of the final
  checkpoint.
- No in-training / running-minimum validation values are used to decide the outcome.

### Ablation design

- One **`lambda = 0` sentence-attention control** at a fixed K — this is the *internal* reference and
  the only arm the decision rule compares against.
- **2–3 additional arms with increasing reconstruction weight**, same K, everything else identical.
- The existing **d12 full-causal baseline** (`model_tag d12_sa_baseline`) from
  `experiments/sentence-attention.md` is the *external* reference; it is not retrained here.
- Total **new** training runs kept small (**≤ 3–4 new arms**) — each is a full 10k-step 4×A100 run.
  Exact K and the weight ladder live in `gist_reconstruction_experiments`.

### Decision rule

- **Primary: BPB.** The auxiliary loss "wins" only if BPB improves over the `lambda = 0` control by
  **more than run-to-run noise**. No BPB improvement ⇒ the hypothesis is not supported, regardless of
  what CORE does.
- **Secondary: CORE, reference-only.** Recorded convention: CORE carries **~±0.01 single-seed noise**
  at d12 / 10k steps. **A CORE delta ≤ 0.01 is NOT evidence** — it cannot be distinguished from a
  training-seed artifact under the single-seed protocol.

### Known threats / confounds

- **(a) Metric purity.** The auxiliary reconstruction loss must **not leak into the reported
  validation loss or BPB**. Reported metrics must stay pure next-token loss over **real (non-gist)
  tokens** only; the auxiliary term is a training-time addition to the optimized objective and must
  be excluded from everything that gets reported/compared.
- **(b) Parameter-count asymmetry.** The reconstruction head adds parameters relative to the
  `lambda = 0` control. The plan must state explicitly that the head is **discarded at evaluation
  time** (it is not used by `base_eval.py`) and that it must **not touch the `lm_head` path** —
  no weight sharing or gradient path that changes next-token prediction capacity. If the head does
  share weights with `lm_head`, that is a different experiment and must be recorded as such.
- **(c) Task difficulty is sentence-length dependent.** Non-autoregressive reconstruction of a whole
  sentence from K gists may be trivially easy for short sentences and near-impossible for long ones,
  so the effective supervision signal is unevenly distributed across the corpus; a null result may
  reflect a degenerate auxiliary task rather than a false hypothesis.
- **(d) No supervision at inference.** Gists receive no reconstruction signal at inference time, so
  **any gain must show up in plain next-token metrics** (BPB / CORE). A low auxiliary reconstruction
  loss on its own is not a result.

## Results

All four arms trained and were scored once at 10k steps. **BPB is primary; CORE is reference-only.**
`val_bpb` is read from the end-of-training eval (`evaluation/bpb_010000.json`); the aggregate table
shipped with the eval stage left the BPB column blank and `min_val_bpb = inf` — the `inf` is expected
(in-training eval is disabled by protocol, so no running minimum exists), the blank column is an
aggregation-side miss, and the numbers below were read directly from the eval artifacts.

| arm                     | lambda | val_bpb | ΔBPB vs control | CORE   | ΔCORE vs control | final recon loss (nats) |
|-------------------------|--------|---------|-----------------|--------|------------------|-------------------------|
| `d12_gr_k4_w0p0` (ctrl) | 0      | 0.8230  | —               | 0.1382 | —                | — (no head)             |
| `d12_gr_k4_w0p1`        | 0.1    | 0.8265  | **+0.0035**     | 0.1487 | +0.0105          | 0.368                   |
| `d12_gr_k4_w0p5`        | 0.5    | 0.8396  | **+0.0166**     | 0.1192 | −0.0190          | 0.184                   |
| `d12_gr_k4_w1p0`        | 1.0    | 0.8532  | **+0.0302**     | 0.1182 | −0.0200          | 0.109                   |
| `d12_sa_baseline` (ext) | —      | 0.8022  | −0.0208         | 0.1830 | +0.0448          | —                       |

(Higher BPB = worse. ΔBPB is signed so that **positive = worse than the lambda=0 control**.)

**Noise calibration, measured inside this project (not assumed).**
- **BPB is very seed-stable:** the three available seed pairs (`d12_sa_baseline`, `d12_sa_alt_short64`,
  `d20_sa_baseline`) differ by 0.0001 / 0.0005 / 0.0004 BPB. Take **±0.0005** as the BPB noise floor.
- **CORE is not:** the lambda=0 control reproduces the earlier, config-identical
  `d12_sa_nltk_k4` run to **0.0002 BPB** (0.8230 vs 0.8228 — the new flag is confirmed inert at 0),
  yet the same two runs differ by **0.0149 CORE** (0.1382 vs 0.1233). This is a direct, in-experiment
  confirmation of the ~±0.01 CORE convention.

**(1) No improvement, and the effect is monotone downward — there is no interior optimum.**
Every nonzero lambda is worse than the lambda=0 control at the same K, and the damage grows
monotonically with lambda (+0.0035 → +0.0166 → +0.0302 BPB). Even the smallest weight tested,
lambda=0.1, is **~7× the BPB noise floor** — a real regression, not run-to-run scatter. The
degradation is present in the pure next-token training CE as well (EMA at 10k: 2.3122 → 2.3223 →
2.3585 → 2.3973 nats), so it is a property of the optimized model, not an evaluation artifact.
Against the external full-causal reference the result is worse still: the control already sits
+0.0208 BPB behind `d12_sa_baseline`, and the auxiliary loss **widens** the very gap it was
introduced to close.

**(2) The auxiliary task was decisively learned — the ablation did test something.** The final
reconstruction loss falls to **0.368 / 0.184 / 0.109 nats** at lambda = 0.1 / 0.5 / 1.0, versus a
**unigram reference of ≈7.60 nats** (measured on a training shard) and **uniform chance of
ln(32772) = 10.40 nats**. That is 20–70× below unigram, and it decreases monotonically with lambda
exactly as a genuinely optimized term should. This is **not** a flat/degenerate auxiliary loss: the
gists demonstrably *can* be forced to carry their sentence's token ids. Caveat on how much this
proves — supervision is a strided subsample of a length-capped sentence, so the head only has to
recover on the order of 8 positions per sentence, not the whole sentence.

**(3) Yes — the cost is a capacity tax, and it is the whole story.** Because the head scores against
its own output projection and never touches `lm_head`, the only channel by which the auxiliary term
can hurt next-token prediction is the **gist hidden states themselves**. Combining (1) and (2): as
lambda rises the model *succeeds* at packing sentence token-identity into the gists (recon loss
0.368 → 0.109) and *simultaneously* gets worse at next-token prediction (BPB +0.0035 → +0.0302), in
lockstep. Gist capacity is scarce and this is a rival use of it, not a complementary one. Wall-clock
cost was negligible (3.11h control vs 3.16h with the head).

**CORE, read with its noise band.** The λ=0.1 CORE delta (+0.0105) is **not evidence of anything** —
it is at the noise threshold, and the control's own config-identical replicate spans 0.1233–0.1382,
which brackets nothing useful. The high-lambda CORE drops (−0.019 / −0.020) exceed 0.01 and share
the sign of the BPB regression, so they are weakly corroborating, but their magnitude is unreliable
for the same reason. **BPB carries the verdict on its own; no CORE number changes it.**

## Conclusions

**The hypothesis is not supported, and the sign is the opposite of the prediction.** Forcing gists to
reconstruct their own sentence does not improve next-token quality at d12/K=4/10k — it degrades it
monotonically in lambda, with no peak and no safe small-lambda regime among the weights tested. The
mechanism is now pinned rather than merely suspected: the auxiliary objective was *learned* (recon
loss 20–70× below unigram) and next-token quality got *worse* in proportion, so faithful
sentence-reconstruction and useful-for-prediction are **competing** demands on a fixed K-token gist
budget. The negative result is informative: it says the gists in the baseline sentence-attention
model are not idle capacity waiting for a better training signal, and that "make the gist a faithful
summary" is the wrong prior for what a gist should hold.

Per the decision rule, no BPB improvement ⇒ hypothesis rejected, regardless of CORE.

**Still confounded / not established by this run:**
- **Single seed** per arm. BPB's ±0.0005 noise floor makes the ranking safe, but nothing here
  separates seed from condition beyond that floor.
- **One K (K=4) only.** A larger gist budget might absorb the reconstruction task without taxing
  prediction; the tax and the optimal lambda could both be K-dependent. This is the single biggest gap.
- **One head design.** Non-autoregressive, shallow, position-conditioned, own output projection. A
  parallel head must encode token *order* into the gists explicitly, which may be exactly the wasteful
  part; a different decoder could change the trade.
- **One supervision density.** Strided, length-capped targets — the task is a subsample, and its
  difficulty is unevenly distributed across sentence lengths (threat (c), unaddressed).
- **Added head parameters** exist during training (discarded at eval), so the arms differ in
  trainable parameter count as well as in objective, even though they are equal at evaluation time.
- **Interaction with lambda scheduling** untested: a term that is harmful at convergence could still
  help as a warm-up-only shaping signal that is annealed to 0.

**Proposed next step (not launched).** The cleanest single follow-up is a **K sweep at a fixed small
lambda** — take lambda=0.1 (the least damaging nonzero weight) and run K ∈ {1, 8, 16} against the
matching lambda=0 sentence-attention control at each K, reusing the existing K-sweep arms as
controls. That directly tests whether the capacity tax is a fixed cost or shrinks as the gist budget
grows, and it is the only way to tell "reconstruction is the wrong objective" from "K=4 was simply
too small to afford it". A second, independent variant worth one arm: an **ordered/autoregressive
reconstruction head** (decode the sentence left-to-right from the gists instead of predicting each
position in parallel), which removes the need for the gists to redundantly encode absolute position
and is the most likely way the head design is at fault. Lower priority, given the monotone result:
a **lambda anneal-to-zero schedule**, to check whether reconstruction is useful early and only
harmful late.

## Changelog

- 2026-08-19: Created this experiment as a follow-up to `experiments/sentence-attention.md`.
  Adds an **auxiliary gist-reconstruction loss**: at each sentence boundary the K gist tokens are
  trained to reconstruct the token ids of the just-ended sentence, via a new reconstruction head in
  `nanochat/gpt.py`, per-sentence segmentation targets from `nanochat/dataloader.py`, and new
  enable/weight CLI flag(s) in `scripts/base_train.py`. Arms are defined by
  `gist_reconstruction_experiments` in `scripts/jobs/run_training.py`: a `lambda = 0`
  sentence-attention control at fixed K plus 2–3 arms of increasing reconstruction weight, at d12 /
  `--window-pattern L` / 10k steps / single seed (seed 0) / `a100.4gpu` (4 GPUs), with no
  in-training evaluation and a single end-of-training CORE + BPB scoring pass.
- 2026-08-19: Implemented. Design decisions the plan left open (code is the source of truth —
  `GistReconstructionHead` / `gist_reconstruction_targets` in `nanochat/gpt.py`): the head scores
  against its **own** output projection, never `lm_head` (threat (b)); tokens in the **last
  sentence of a document are skipped** (no boundary follows them, and attaching them to the next
  document's boundary would leak across documents); gist tokens, BOS and ignore positions are
  excluded from the targets; supervision is taken on a fixed strided subsample of positions
  (`--gist-recon-stride`) and bounded by `--gist-recon-max-sentence-len`. `--gist-recon-weight 0`
  allocates no head at all. The head is discarded when a checkpoint is loaded for eval/inference
  (`checkpoint_manager.build_model`), so BPB/CORE stay pure next-token metrics (threat (a)).
  No sentence-attention arm was relaunched.
- 2026-08-19: Review fix — a sentence closed only by an **incomplete gist run** is now excluded
  too. The best-fit dataloader crops the last document of a packed row at an arbitrary offset,
  so a row can end mid-gist-run; that partial run still registered as a boundary, and the head's
  fixed K-state gather then reached past the run's start into the **sentence's own token
  states** — i.e. it could predict a target token from that token's own hidden state.
  Reconstruction boundaries now require the complete K-token run
  (`_next_boundary_idx(..., run_len=K)`). The sentence-attention mask's boundary notion is
  unchanged, so the `lambda = 0` path stays bit-identical.
- 2026-08-20: Results in. **Hypothesis rejected.** All four arms trained and were scored once at
  10k steps. BPB degrades monotonically with lambda against the same-K `lambda = 0` control
  (+0.0035 / +0.0166 / +0.0302 at lambda = 0.1 / 0.5 / 1.0) — no peak, no safe small-lambda regime,
  and even the smallest weight is ~7× the measured BPB noise floor (±0.0005, from three seed pairs).
  The auxiliary task was genuinely learned (final reconstruction loss 0.368 / 0.184 / 0.109 nats vs
  ≈7.60 unigram and 10.40 chance), so the ablation tested a real signal: the model can pack sentences
  into the gists, and doing so costs next-token quality — a capacity tax, since the head never
  touches `lm_head`. The `lambda = 0` control reproduced the earlier `d12_sa_nltk_k4` run to 0.0002
  BPB (flag confirmed inert at 0) while differing 0.0149 in CORE, re-confirming that CORE deltas
  ≤ 0.01 are not evidence here. Next step (not launched): sweep K at fixed lambda = 0.1 to test
  whether the tax shrinks with a larger gist budget; secondarily, an ordered/autoregressive
  reconstruction head.
