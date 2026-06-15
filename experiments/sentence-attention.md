# Sentence Attention — block-causal + global-gist (d12, 10k, gist-token sweep)

## Hypothesis

Replacing the standard causal mask with a **block-causal + global-gist** mask preserves (or
improves) validation quality while compressing long-range context into a small number of
**gist ("end-of-sentence") tokens**. Concretely: a token attends only to (a) the tokens of its
own current sentence block (back to the most recent boundary) and (b) all gist tokens from
earlier sentences in the same document — long-range information flows only through the gists.

If gists are an adequate summary channel, a sentence-attention model should track the
full-causal baseline at d12; the **sweep over the number of gist tokens per boundary
K ∈ {1, 4, 8, 16}** locates how much gist capacity is needed and where the quality/budget
trade-off sits. (This is the data-side + mask mechanism only; KV-cache compression — the
eventual payoff of being able to drop completed sentences' non-gist KV — is a follow-up.)

## Setup

Training function: `sentence_attention_experiments` in `scripts/jobs/run_training.py` (source
of truth for all hyperparameters, step counts, model selection, and job configs). Evaluation
via `scripts/jobs/run_evaluation.py` → `scripts/base_eval.py`. Default job: `num_gpus=4`,
`instance_type=a100.4gpu`; checkpoints/artifacts under
`$NANOCHAT_BASE_DIR/base_checkpoints/<model_tag>/`.

**Arms (5, d12 / 10k steps / single seed 0 / `--window-pattern L`):**

| model_tag | gist placement | K (gists/boundary) |
|-----------|----------------|--------------------|
| `d12_sa_baseline` | — (full causal) | 0 |
| `d12_sa_nltk_k1`  | sentence_nltk   | 1 |
| `d12_sa_nltk_k4`  | sentence_nltk   | 4 |
| `d12_sa_nltk_k8`  | sentence_nltk   | 8 |
| `d12_sa_nltk_k16` | sentence_nltk   | 16 |

**Mechanism (where it lives, in code):**
- **Data side** (`nanochat/dataloader.py` `refill_buffer` + `nanochat/tokenizer.py`): each
  document is split into sentences with NLTK Punkt; K gist ids are inserted after every
  sentence boundary except the last; every document still starts with exactly one BOS. Gist
  ids are reserved just past the real vocab (`tokenizer.gist_token_ids`) and inserted by id —
  the BPE tokenizer is never retrained. The model vocab/embedding is grown by K in
  `base_train.py`.
- **Model side** (`nanochat/gpt.py` `GPT._build_sentence_mask`): from the input ids alone the
  forward builds a boolean `[B,1,T,T]` mask `allowed = (block_causal | special_visible) &
  same_doc` (self always allowed), where `block_causal` uses a vectorized cummax closest-EOS
  index, `special_visible` exposes every gist/BOS key, and `same_doc = (idx==bos).cumsum`
  confines attention to the query's own document within a packed row (no cross-doc leakage).
  Masked layers route through `F.scaled_dot_product_attention(attn_mask=...)`. On the
  experiment node (A100 = sm80) FA3 never loads, so SDPA is the real path either way; the
  baseline arm (no gist ids) is a byte-for-byte no-op on the existing flash path.

**Design choices (locked):** pure sentence attention (no `full_attention_layers`); supervise
everything (gist tokens are scored as normal tokens in the loss); NLTK-Punkt placement only
(uniform/regex not run).

**Metrics.**
- **Primary: val nats/token over real tokens at the in-training minimum (`min_val_loss`).**
  Gist tokens have 0 bytes, so they are excluded from both bpb and the nats/token denominator —
  nats/token therefore measures real-token language-model quality and is the fair cross-arm
  comparison. Read from each run's `loop_state.min_val_loss` (surfaced by `scripts/results.py`
  as `min_val_nats`).
- **Secondary:** `min_val_bpb` (best in-training bpb).
- **Reference-only:** `val_bpb`/`val_loss` at the final step; CORE.

**Decision rule.** A sentence arm "preserves quality" if its `min_val_nats` is within the
known d12/10k tie-noise band of the baseline (the prior 10k study at this depth showed arm
ties of ~0.001 bpb — treat sub-noise deltas on the secondary `min_val_bpb` as ties; for the
primary nats/token metric, treat differences not clearly above run-to-run noise as ties).
Select the K with the lowest `min_val_nats`. **No error bars** (single seed by mandate), so
only deltas clearly above noise are called; otherwise report "tied".

**Known threats to validity (read results with these in mind):**
1. **10k/d12 overfitting (data exhaustion).** At d12 the shard loops ~58× over 10k steps and
   both arms overfit; the final-step `val_bpb` measures memorization, not quality. **Compare
   at the in-training minimum (`min_val_*`), not the final step.** This affects all arms
   equally, so the *relative* comparison stays valid. (See [[linear-projection-embeddings-10k]],
   whose own recommendation was to fix the horizon / read the best checkpoint.)
2. **K-dependent position-budget confound.** Larger K inflates sequence length per real token,
   shifting RoPE phase and the real:context ratio per arm — so the K-ordering is not perfectly
   apples-to-apples. A position-matched control is a follow-up.
3. **bpb K-inflation.** Under supervise-everything the model spends training loss predicting
   deterministic gists; bpb excludes gists but the inflation motivates using nats/token as the
   headline.
4. **Cropped-doc partial gist runs.** Best-fit packing can truncate a document mid sentence/gist
   run at row ends; the mask math is robust (cummax + forced self-diagonal), but a block can be
   truncated at a crop boundary.
5. **CORE mask mismatch.** CORE prompts carry no gist tokens, so a sentence model runs under an
   effectively full-causal mask at CORE time (train/eval mismatch). CORE is reference-only here;
   launch sentence evals with `--eval bpb` to skip the (wasted) CORE passes.

## Results

_To be filled after the runs complete._ Report per arm: `min_val_nats` (primary), `min_val_bpb`,
final `val_bpb`/`val_nats`, epochs over shard, and CORE (reference). Compare each sentence arm to
`d12_sa_baseline` at the in-training minimum.

## Conclusions

_To be filled after analysis._

## Changelog

- 2026-06-15: Created the sentence-attention experiment group
  (`sentence_attention_experiments` in `scripts/jobs/run_training.py`): 1 full-causal baseline
  + 4 NLTK-sentence-anchored gist arms K ∈ {1,4,8,16} at d12 / 10k / seed 0 / `--window-pattern L`.
  Implemented the mechanism: tokenizer gist utils + NLTK Punkt splitter (`nanochat/tokenizer.py`),
  dataloader gist insertion (`nanochat/dataloader.py`), forward-built block-causal + global-gist
  per-document mask on the SDPA path (`nanochat/gpt.py`), vocab growth + `token_bytes` extension +
  nats/min-val logging (`scripts/base_train.py`), gist-aware eval + relaxed vocab assert
  (`scripts/base_eval.py`, `nanochat/checkpoint_manager.py`), and `min_val_nats`/`min_val_bpb`
  columns in `scripts/results.py`. Plan + consensus review under `.omc/`.
