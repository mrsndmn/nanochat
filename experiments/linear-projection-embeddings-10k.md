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

Single seed per arm, 10k steps. val_bpb is primary; the two reused dense baselines both
landed at exactly **0.8058**, so run-to-run val_bpb noise on this line is small and a
−0.002 move is meaningful.

| arm                                   | val_bpb    | Δ vs 0.8058 | CORE   |
|---------------------------------------|------------|-------------|--------|
| d12_baseline_10k_bb2 (reference)      | 0.8058     | —           | 0.1880 |
| **d12_bigramhash512_10k_bb2 (Arm B)** | **0.8037** | **−0.0021** | 0.1925 |
| d12_multbigram512_10k_bb2 (Arm A)     | 0.8072     | +0.0014     | 0.1815 |

- **Arm B (hashed (prev,cur) bigram-identity)** is the **best val_bpb of the entire
  linear-projection line** (0.8037), beating every additive-projection arm, every context
  arm, and the dense baseline by −0.0021 — inside the −0.001…−0.003 target.
- **Arm A (gated multiplicative joint-bigram)** regressed (+0.0014); the element-wise
  Hadamard interaction alone does not help.
- CORE is secondary and noisy here (±0.02): Arm B's 0.1925 is among the top values in the
  table and moves with val_bpb in the right direction; Arm A's 0.1815 dips slightly. No
  anomalies or missing evaluations — all models report val_bpb, val_nats, and CORE.

## Conclusions

**Verdict: SUCCESS.** Arm B (the hashed pair-keyed bigram-identity term) drops val_bpb to
**0.8037**, −0.0021 below the dense baseline (0.8058) and the lowest of any arm in this
line. This is the first embedding-side mechanism to break the baseline tie, and it
**validates the diagnosis**: the prior additive-projection ceiling was a
**redundancy/separability limit**, not a hard input-side ceiling. A term that is
**non-absorbable** (keyed on the *pair*, not the current token-id, so `wte` cannot refold
it) and **non-redundant** (a pair-keyed identity lookup, not a separable sum that
smear/attention already approximate) does add usable headroom at d12/10k.

Arm A (multiplicative Hadamard joint-bigram) **regressed** (+0.0014), so the win is
specific to the *identity/lookup* form of the joint interaction, not the multiplicative
form — the effective ingredient is the explicit pair-keyed embedding, not element-wise
modulation.

Per the user override, this line stays on the **embedding / input side**; the bigram-hash
result is the lead to push further. Next steps (all embedding-side):

- **Scale the winning path** — sweep hash bucket counts / hash widths and the joint
  embedding dim for `bigramhash`, and give the joint path a longer training horizon to see
  whether the −0.002 gain widens.
- **Higher n-gram order** — extend the pair-keyed identity to a hashed
  (prev2, prev1, cur) **trigram** term using the same non-absorbable construction.
- **Learned gate schedules** — anneal/warm the joint-path gate instead of a static
  zero-init scalar, and try per-dimension rather than scalar gating.
- **Joint + additive combination** — combine the bigram-hash identity with the additive
  per-token projection (proj512), since they capture different (pair vs token) structure.

## Changelog

- **2026-06-21** — New embedding-side phase. Prior phases established that the additive
  per-token projection ties the dense baseline (baseline 0.8058 vs proj512 0.8066, absorbable
  into `wte`) and that separable context terms regress (prevtok512 0.8099, adapter512 0.8075,
  redundant with smear + attention). Next phase adds two joint `(token_t, token_{t-1})` input
  arms against the reused baseline `d12_baseline_10k_bb2` (0.8058): (A) a gated multiplicative
  joint-bigram path and (B) a hashed (prev,cur) bigram-identity embedding — both non-absorbable
  and non-redundant. Continuing on the embedding side per the user override; no pivot to
  base-model LR sweeps. Results/Conclusions pending.
- **2026-06-21** — Results in. **Arm B (`d12_bigramhash512_10k_bb2`) wins: val_bpb 0.8037,
  −0.0021 vs baseline 0.8058 — the best of the line and the first arm to break the tie.**
  Arm A (multiplicative, `d12_multbigram512_10k_bb2`) regressed to 0.8072. Confirms the
  embedding-side ceiling was a redundancy/separability limit; the pair-keyed identity (lookup)
  form is the effective joint mechanism, not the multiplicative form. Next (embedding-side):
  scale the bigram-hash path (buckets/width/dim, longer horizon), trigram identity, learned
  gate schedules, joint+additive combination.
</content>
</invoke>
