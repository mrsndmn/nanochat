# Sentence Attention — block-causal + global-gist (d12)

## Hypothesis

**Sentence attention** — block-causal attention restricted to sentence/segment blocks, plus a
**global-gist (summary) token** that lets information flow across blocks — can **match or improve
quality (CORE) and/or bits-per-byte (BPB)** versus standard full causal attention at **depth 12**,
potentially with **better efficiency** (long-range context is carried only through a small number
of gist tokens rather than the full key/value history).

Concretely: a token attends only to (a) the tokens of its own current sentence block and (b) the
gist tokens emitted at earlier sentence boundaries within the same document. If gists are an
adequate summary channel, the sentence-attention model should track the full-causal baseline at
d12.

## Setup

- **Training function (source of truth):** `sentence_attention_experiments` in
  `scripts/jobs/run_training.py`. It emits the d12 sentence-attention configs — each a dict with
  `model_tag`, `args`, `description`, `instance_type`, `num_gpus`, and `experiment_slug`. **All
  hyperparameters, arms, and model selection live in that function; this plan does not duplicate
  them.** The mechanism itself lives in `nanochat/gpt.py` (forward-built block-causal +
  global-gist per-document mask on the SDPA path), `nanochat/dataloader.py` (gist insertion at
  sentence boundaries), and `nanochat/tokenizer.py` (gist token-id reservation + NLTK-Punkt
  sentence splitter); see commit `ee7de75`.
- **Node:** `num_gpus = 4`, `instance_type = a100.4gpu` (the project default).
- **Artifacts / checkpoints:** under `$NANOCHAT_BASE_DIR/base_checkpoints/<model_tag>/` (i.e.
  `artifacts/base_checkpoints/<model_tag>/` in the workspace).
- **Training horizon:** exactly **10k optimization steps** (`--num-iterations 10000`).
- **Seeds:** **single seed only** (seed 0) — one training run per config, no multi-seed fan-out
  (per project convention).

### Evaluation protocol (reviewer-mandated)

- **NO intermediate evaluations are performed during training.** The model is evaluated **only
  once, at the END of training**, after all 10k optimization steps have completed.
- Evaluation is **deferred entirely to the separate post-training evaluation stage**
  (`scripts/jobs/run_evaluation.py` → `scripts/base_eval.py`), which scores the final checkpoint
  on **CORE** and **BPB**.
- The comparison therefore uses end-of-training metrics only; no in-training / running-minimum
  validation values are used to decide the outcome.

## Results

_Pending — to be filled in after the post-training evaluation stage completes._

## Conclusions

_Pending._

## Changelog

- 2026-06-15: Created the sentence-attention experiment group
  (`sentence_attention_experiments` in `scripts/jobs/run_training.py`) and implemented the
  mechanism: tokenizer gist utils + NLTK Punkt splitter (`nanochat/tokenizer.py`), dataloader gist
  insertion (`nanochat/dataloader.py`), forward-built block-causal + global-gist per-document mask
  on the SDPA path (`nanochat/gpt.py`), vocab growth + gist-aware eval (`scripts/base_train.py`,
  `scripts/base_eval.py`, `nanochat/checkpoint_manager.py`); see commit `ee7de75`.
- 2026-06-15: Finalized the plan and started the **first end-to-end run** at d12 / 10k steps
  (`--num-iterations 10000`) / single seed (seed 0) on `a100.4gpu`. Per reviewer feedback:
  **no intermediate evaluations during training** — the model is evaluated **only once, at the end
  of training**, with evaluation deferred entirely to the separate post-training evaluation stage.
