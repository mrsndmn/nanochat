# Linear Projection Embeddings

## Hypothesis

Adding a low-rank learnable correction to the pretrained embedding table via
`inputs_embeds = Linear(low_dim_embed(token_ids)) + wte(token_ids)` can improve
embedding quality with fewer trainable parameters than making the full embedding
table trainable. The projection starts at zero (no initial perturbation) and
learns a correction in a low-dimensional bottleneck space.

## Setup

Training function: `scripts/base_train.py` with `--embed-proj-dim` flag.

## Results

(pending)

## Conclusions

(pending)

## Changelog

- 2026-06-12: Initial implementation of linear projection embeddings in gpt.py and base_train.py
