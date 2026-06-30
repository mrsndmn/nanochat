# 5. Polysemy × Context trainer + analysis integration (components 2 & 3)

Date: 2026-06-30
Status: Accepted

## Context
Component 1 (the synthetic generator, ADR 0003/0004) emits, per condition, a parquet corpus
of whitespace form symbols + `vocab.json` + `metadata.json`. Components 2 (train across
context length `L`) and 3 (read off `gap(L)` and the entropy decomposition) had to be wired
into nanochat's BPE/ClimbMix-shaped train/eval path without disturbing the generic base.

## Decision
**Identity tokenizer + data-dir, selected by flags.** `nanochat/identity_tokenizer.py` maps
each form to one id (forms at `0..|V|-1` from `vocab.json`, special tokens appended). The
dataloader gained a `data_dir` kwarg; `base_train.py` / `base_eval.py` gained `--data-dir`
and `--tokenizer {bpe,identity}`. No change to the default (BPE/ClimbMix) path.

**BPC = bits per form.** The identity `token_bytes` is `1` per real form and `0` per special
token, so the trainer's existing bytes-normalized metric (`evaluate_bpb`) equals
bits-per-form, and BOS is masked out. Perplexity comes from the raw cross-entropy
(`loss`, nats/token). This is the "BPC" component 3 compares to the analytic floor.

**Eval recovers its config from the checkpoint.** `base_train` already saves `user_config`
into the checkpoint meta. `base_eval --tokenizer auto` (default) reads `tokenizer` + `data_dir`
back from the meta, builds the identity tokenizer, injects it into `build_model`
(`checkpoint_manager` gained a `tokenizer=` param), and **drops CORE + sample** (English ICL
tasks / prompts are meaningless for a synthetic-symbol vocab). So `run_evaluation.py` works
unchanged — each polysemy checkpoint evaluates itself correctly.

**Analysis reads numbers, not models.** `nanochat/polysemy_analysis.py` is pure (numpy only):
`gap(L)`, BPC-vs-floor, lexical-vs-total `H_m` decomposition (from the metadata sidecar), and
decision rules. `scripts/analyze_polysemy.py` discovers `poly_<slug>_L<seqlen>` checkpoints,
takes per-cell val loss from `evaluation/bpb_*.json` if present else the training meta, and
emits CSV + markdown tables (rows = conditions, columns = L).

**Representation probe via an `lm_head` pre-hook.** `scripts/probe_polysemy.py` captures the
normed final hidden state (input to `lm_head`) and fits a torch logistic probe (no sklearn) to
decode the latent sense, bucketed by left-context length. The generator exports a held-out,
sense-labeled `probe.jsonl` (disjoint seed; shared across conditions so only the surface forms
differ). Pure helpers live in `nanochat/probe_utils.py` (torch only) so they unit-test without
the model stack.

## Consequences
- The experiment runs entirely through the existing job launchers; the only experiment-specific
  flags are `--data-dir` + `--tokenizer identity`, and the checkpoint remembers them for eval.
- Two incidental robustness fixes were required so the analysis/probe coexist with the
  generator in one process and in the `nanochat-polysemy` env: (a) `nanochat.dataset`'s
  `requests` import is now lazy (download-only), so the data path loads where `requests` is
  broken; (b) `nanochat.polysemy`'s broken-pandas blocker now raises only on a *real* `import
  pandas` and returns "not found" for a bare `importlib.util.find_spec` probe, so it no longer
  breaks `torch._dynamo`/`torch.compile`.
- Cross-condition comparability rests on holding model, batch, horizon and the sense stream
  fixed and sweeping only `L` with **full attention** (`--window-pattern L`); a sliding window
  would cap usable context and confound the very axis under study.
- Rejected: pre-tokenized integer shards (deeper dataloader rewrite, ADR 0003); re-specifying
  tokenizer/data-dir at eval time (brittle — the checkpoint is the source of truth); sklearn for
  the probe (not in the env).
