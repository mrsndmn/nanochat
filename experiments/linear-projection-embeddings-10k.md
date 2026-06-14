# Linear Projection Embeddings — 10k-step

## Hypothesis

A zero-initialized low-rank correction to the embedding table
(`inputs_embeds = Linear(low_dim_embed(token_ids)) + wte(token_ids)`) can match or beat
the dense baseline on CORE/BPB with fewer trainable parameters. At d12 / 2520 steps,
`proj_512` (`embed_proj_dim=512`) lowered val_bpb decisively. This phase tests whether that
advantage survives at a longer **10k-step** horizon trained on non-repeated data (<=1 epoch).

## Setup

Training function: `linear_projection_embedding_experiments` in
`scripts/jobs/run_training.py` (single source of truth for all hyperparameters, step counts,
model selection, and job configs). Projection width is set by `--embed-proj-dim` on
`scripts/base_train.py` (`embed_proj_dim=0` = dense baseline). Evaluation via
`scripts/jobs/run_evaluation.py` → `scripts/base_eval.py`. Default job: `num_gpus=4`,
`instance_type=a100.4gpu`; checkpoints under `$NANOCHAT_BASE_DIR/base_checkpoints/<model_tag>/`.

Two arms at d12 / 10k steps, single seed: dense **baseline** (`embed_proj_dim=0`) vs
**proj_512** (`embed_proj_dim=512`). Data is capped to <=1 epoch (150 train shards, no wrap)
with the global batch size unchanged. Primary metric **val_bpb**; CORE secondary.

## Results

_Pending._

## Conclusions

_Pending._

## Changelog
