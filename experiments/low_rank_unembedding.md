# Low-Rank Unembedding Correction

## Hypothesis

The validated input-side low-rank embedding correction
(`inputs_embeds = wte(idx) + Linear(low_dim_embed(idx))`, best at `embed_proj_dim=512`,
−0.0588 val_bpb in [[linear_projection_embeddings]]) has a natural **output-side analog**:
a zero-initialized LoRA-style low-rank term added to the `lm_head` logits

```
logits = lm_head(x) + unembed_proj_up(unembed_proj_down(x))
```

where `unembed_proj_down: n_embd → r`, `unembed_proj_up: r → vocab`, with the up
projection zero-initialized so the correction starts at zero. At matched rank `r=512`
the added parameter count is the same formula as the input projection
(`r × (vocab + n_embd)`), enabling a clean parameter-matched comparison of where the
low-rank correction is most useful — input embedding, output unembedding, or both.

Two questions:
1. Does an output-side low-rank correction help on its own, like the input-side one?
2. Do input + output corrections **compose** (additive gains) or overlap (redundant)?

## Setup

Training: `scripts/base_train.py` with the new `--unembed-proj-dim` flag (model code in
`nanochat/gpt.py`: `unembed_proj_down` / `unembed_proj_up`, AdamW group at `unembedding_lr`).
Configs in `scripts/jobs/run_training.py` (`low_rank_unembedding_experiments`), d12,
`--window-pattern SSSL`, matching linear-projection-embeddings for direct comparability.

Variants (all d12):
- `baseline` — no projection (reuses prior checkpoint)
- `proj_512` — input `embed_proj_dim=512` (reuses prior checkpoint, prior best)
- `unembed_512` — output `unembed_proj_dim=512`
- `both_512` — input + output, both `=512`

## Results

_Pending training + eval (CORE + BPB)._

| variant      | embed_proj | unembed_proj | val_bpb | Δbpb vs base | CORE |
|--------------|------------|--------------|---------|--------------|------|
| baseline     | 0          | 0            | TBD     | —            | TBD  |
| proj_512     | 512        | 0            | TBD     | TBD          | TBD  |
| unembed_512  | 0          | 512          | TBD     | TBD          | TBD  |
| both_512     | 512        | 512          | TBD     | TBD          | TBD  |

## Conclusions

_Pending._

## Changelog

- 2026-06-13: Initial implementation. Added `unembed_proj_dim` to GPTConfig/gpt.py
  (zero-init LoRA-style correction on lm_head logits, AdamW group at unembedding_lr),
  `--unembed-proj-dim` flag in base_train.py, and `low_rank_unembedding_experiments` in
  run_training.py. Validated at init (correction exactly zero, identical loss to baseline;
  param-count and optimizer-group asserts pass; matched param cost vs input projection).
