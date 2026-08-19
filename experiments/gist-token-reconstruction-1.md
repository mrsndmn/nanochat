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

_Pending — to be filled in after the post-training evaluation stage completes._

## Conclusions

_Pending._

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
