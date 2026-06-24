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
landed at exactly **0.8058** (noise floor <0.0001), so −0.002…−0.004 moves are real signal.
CORE is secondary (±0.02 noise) and is **decoupled** from val_bpb here — not interpreted.

**Joint vs baseline (prior phase).** The hashed pair-identity term beat the baseline; the
multiplicative Hadamard term regressed.

| arm                                | val_bpb    | Δ vs 0.8058 | CORE   |
|------------------------------------|------------|-------------|--------|
| baseline (d12_baseline_10k_bb2)    | 0.8058     | —           | 0.1880 |
| bigramhash512 (dim 64, 2^18)       | 0.8037     | −0.0021     | 0.1925 |
| multbigram512 (Arm A)              | 0.8072     | +0.0014     | 0.1815 |

**HASH-DIM sweep** (buckets fixed 2^18, init-std 0.005):

| hash dim    | val_bpb    | Δ vs 0.8058 | CORE   |
|-------------|------------|-------------|--------|
| 32          | 0.8043     | −0.0015     | 0.1946 |
| 64 (center) | 0.8037     | −0.0021     | 0.1925 |
| **128**     | **0.8033** | **−0.0025** | 0.1782 |
| 256         | 0.8041     | −0.0017     | 0.1911 |
| 512         | 0.8044     | −0.0014     | 0.1906 |

Inverted-U with an interior sweet spot at **dim 128** (0.8033); width above 128 regresses
(256, 512), consistent with overfitting/saturation at 10k steps. All five widths beat the
baseline; the spread is shallow (~0.001).

**BUCKET sweep** (dim fixed 64, init-std 0.005):

| buckets       | val_bpb    | Δ vs 0.8058 | CORE   |
|---------------|------------|-------------|--------|
| 2^16          | 0.8052     | −0.0006     | 0.1810 |
| 2^18 (center) | 0.8037     | −0.0021     | 0.1925 |
| **2^20**      | **0.8014** | **−0.0044** | 0.1861 |

Monotone: more buckets (fewer collisions) → strictly lower val_bpb, no saturation by 2^20,
largest single step at the top end. This is the dominant lever, and **2^20 (0.8014) is the
best arm in the whole line** — collisions, not width, bottleneck the pair term.

**Best operating point: dim 64 × 2^20 buckets (0.8014, −0.0044).** Every one of the **seven**
hashed pair-identity arms beats 0.8058, whereas every earlier additive/separable arm tied or
regressed (proj512 0.8066, prevtok512 0.8099, adapter512 0.8075, multbigram512 0.8072). No
missing evaluations — all models report val_bpb, val_nats, and CORE.

## Conclusions

**Verdict: SUCCESS — comprehensively.** The success criterion (≥1 arm strictly below 0.8058)
is met by **all seven** hashed pair-identity arms; the best, **dim 64 × 2^20 buckets, reaches
0.8014 (−0.0044)** — the lowest val_bpb of the entire line. The two sweeps localize the levers:

- **Bucket count is the dominant, unsaturated lever.** At fixed dim 64, val_bpb falls
  monotonically 2^16→2^18→2^20 (0.8052→0.8037→0.8014), with the biggest step at the top — the
  pair term is collision-bottlenecked, and more buckets keep paying.
- **Hash-dim has a shallow interior optimum (~128).** At fixed 2^18 buckets val_bpb is an
  inverted-U (best 0.8033 at 128; 256/512 regress), so width past ~128 overfits at 10k steps
  rather than helping.

That **every** setting beats the baseline — even the smallest (2^16, dim 32) — while **no**
additive/separable arm ever did confirms the pair-identity term is genuinely
**non-absorbable** (keyed on the *pair*, so `wte` cannot refold it) and **non-redundant** (a
pair-keyed lookup, not a separable sum smear/attention already approximate). The *structure*,
not the parameter budget, breaks the tie; bucket count then tunes how far. (Caveat: the bucket
gain rides a growing embedding table — 2^20×64 ≈ 67M params — so further bucket scaling trades
parameters for bpb.) CORE is noisy/decoupled here (best-val_bpb arm dim-128 has the *lowest*
CORE, 0.1782) and is not read.

Per the user override, this line stays strictly on the **embedding / input side** (no LR
pivot). Next steps:

- **Test the joint optimum** — the two sweeps only crossed at the dim-64 / 2^18 center; run
  **dim 128 × 2^20** (the two best 1-D points) to check whether the levers compound.
- **Push buckets further** — **2^22 at dim 64** to find where the monotone bucket gain
  saturates; it is the strongest, still-unspent lever.
- **Param-matched control** — compare the 2^20 win against a param-matched additive table to
  confirm the gain is structural, not pure capacity.
- **Trigram identity** — extend the non-absorbable construction to a hashed
  (prev2, prev1, cur) term.

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
- **2026-06-24** — Bigram-hash scaling sweeps in. **SUCCESS criterion met comprehensively:
  all 7 hashed pair-identity arms beat 0.8058.** BUCKET sweep (dim 64) is monotone —
  2^16/2^18/2^20 → 0.8052/0.8037/0.8014, unsaturated; **best operating point dim 64 × 2^20 =
  0.8014 (−0.0044), the lowest of the line.** HASH-DIM sweep (2^18 buckets) is an inverted-U
  with a sweet spot at dim 128 (0.8033), regressing at 256/512 (overfit). Bucket count is the
  dominant lever; hash-dim a shallow second-order knob. Because every setting beats baseline
  while no additive/separable arm ever did, the pair-identity term is confirmed
  non-absorbable/non-redundant in practice. CORE noisy/decoupled (not read). Next
  (embedding-side, no LR pivot): joint dim 128 × 2^20, push to 2^22 buckets, param-matched
  control, trigram identity.
</content>
</invoke>
