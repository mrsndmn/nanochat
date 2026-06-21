# Linear Projection Embeddings — 10k-step

## Hypothesis

At d12 / 10k steps / <=1 epoch, the embedding-side **additive per-token** projection is
capped at a **tie** with the dense baseline (baseline val_bpb 0.8058; proj512 0.8066): a
zero-init low-rank correction summed into a full-rank trainable `wte` is **absorbable** —
anything it can express as a function of the current token-id alone, `wte` can already learn.
Separable **context** terms then regressed (prevtok512 0.8099, adapter512 0.8075) because they
are **redundant** with the existing smear gate + attention, which already carry separable
previous-token information.

The remaining untested embedding-side idea is a **genuinely joint** `(token_t, token_{t-1})`
interaction that is simultaneously:

- **non-absorbable** — its value depends on the *pair*, not on the current token-id alone, so
  it cannot be folded back into `wte`; and
- **non-redundant** — it enters as a *product / identity* (a multiplicative or
  pair-keyed term), not as a separable sum that attention/smear already approximate.

We test whether such a joint input-side term can push val_bpb **below 0.8058**. This phase
continues strictly on the **embedding / input side** (no pivot to base-model LR sweeps).

Two new arms are compared against the reused dense baseline (`d12_baseline_10k_bb2`, 0.8058):

- **Arm A — gated multiplicative joint-bigram path.** The previous token's low-dim vector
  modulates the current token's low-dim embedding **element-wise** (a Hadamard product, the
  non-separable interaction), up-projected and added to `wte` behind a learned **scalar gate**.
  Gate is **zero-init** (path starts as a no-op = baseline) but the projections are
  **small-nonzero** init so the path has gradient and trains from step 0 (a zero gate with both
  projections also zero would be a frozen no-op).
- **Arm B — hashed (prev,cur) bigram-identity embedding.** A hashed `(prev, cur)` pair index
  (≈262k buckets, ≈64-d, bias-free up-projection) supplies a **pair-keyed identity** term added
  to `wte` behind a learned gate (small-nonzero init). This is non-absorbable by construction —
  the lookup key is the pair, not the current token.

Both arms share the same design constraints as the existing input path: a **causal
right-shift with a position-0 sentinel** for the previous token (no future leakage), the term
is **added to `wte` before the input norm**, and the path must support the **KV-cache**
(previous-token state carried across decode steps).

## Setup

Training function: `linear_projection_embedding_experiments` in
`scripts/jobs/run_training.py` (single source of truth for all hyperparameters, step counts,
model selection, and job configs). The input-side mechanism lives in `nanochat/gpt.py`
(`low_dim_embed` / `embed_proj` summed into the hidden state before the input norm, alongside
the smear gate and KV-cache `prev_embedding`); training flags such as `--embed-proj-dim`,
`--num-train-shards`, `--window-pattern`, `--num-iterations`, and `--seed` are defined in
`scripts/base_train.py`. Evaluation via `scripts/jobs/run_evaluation.py` → `scripts/base_eval.py`.

Default job: `num_gpus=4`, `instance_type=a100.4gpu`; checkpoints under
`$NANOCHAT_BASE_DIR/base_checkpoints/<model_tag>/` (shared store
`nanochat-artifacts-low-dim-projection`).

Fixed invariants for this line — depth 12, 10k iterations, single seed, `SSSL` window pattern,
150 train shards (<=1 epoch, no wrap), the unchanged 524,288 tok/step global batch, and the
`a100.4gpu` / 4-GPU job — are defined in code (`scripts/jobs/run_training.py`,
`scripts/base_train.py`) and are not duplicated here.

**Success criterion:** an arm drops **val_bpb below 0.8058** (target −0.001 to −0.003).
val_bpb is primary; CORE is secondary (±0.02 noise band).

## Results

_Pending._

## Conclusions

_Pending._

## Changelog

- **2026-06-21** — New embedding-side phase. Prior phases established that the additive
  per-token projection ties the dense baseline (baseline 0.8058 vs proj512 0.8066, absorbable
  into `wte`) and that separable context terms regress (prevtok512 0.8099, adapter512 0.8075,
  redundant with smear + attention). Next phase adds two joint `(token_t, token_{t-1})` input
  arms against the reused baseline `d12_baseline_10k_bb2` (0.8058): (A) a gated multiplicative
  joint-bigram path and (B) a hashed (prev,cur) bigram-identity embedding — both non-absorbable
  and non-redundant. Continuing on the embedding side per the user override; no pivot to
  base-model LR sweeps. Results/Conclusions pending.
</content>
</invoke>
