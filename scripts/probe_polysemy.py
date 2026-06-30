"""
Representation probe for the Polysemy × Context experiment (component 3).

Asks the direct question behind the hypothesis: **does the trained model use left-context
to resolve a form's latent sense, and does that resolution improve as context grows?**

For a trained checkpoint we run the held-out, sense-labeled probe set (``probe.jsonl``,
written by the generator) through the model, capture the final hidden state at each token
via a forward pre-hook on ``lm_head``, and fit a linear probe (torch logistic regression,
no sklearn dependency) to decode the latent sense from that hidden state. We then report
the probe's accuracy bucketed by **how many tokens of left-context the form had**. Rising
accuracy with context is evidence the model resolves polysemy from context — and the
contrast across conditions (mono ~flat-high; homonymy rising-to-high; overlapping
polysemy rising-then-plateauing below 1) mirrors the gap(L) story from the BPC analysis.

The probe is fit on a train split of tokens and evaluated on a disjoint test split, so a
high bucketed accuracy reflects linearly-decodable sense information, not memorization.

Usage:
    torchrun --nproc_per_node=1 -m scripts.probe_polysemy --model-tag poly_hsw1p5_homonymy_L128
    python -m scripts.probe_polysemy --model-tag poly_mono_L128 --device-type cpu --max-docs 500
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model, find_last_step
from nanochat.identity_tokenizer import get_identity_tokenizer
from nanochat.polysemy_analysis import probe_resolution_summary
from nanochat.probe_utils import CTX_EDGES, ctx_bucket, fit_linear_probe, bucket_accuracy


@torch.no_grad()
def extract_features(model, tokenizer, probe_records, device, seq_len, max_docs=None):
    """Run probe docs through the model and collect (hidden, sense, context_len) per token.

    Captures the input to ``lm_head`` (the normed final hidden state) via a pre-hook. With
    a BOS prepended, input position p (p>=1) holds the form whose ground-truth sense is
    senses[p-1]; its left-context length (in form tokens) is p-1. Docs are truncated to the
    model's seq_len (rotary cache bound)."""
    feats, labels, ctx = [], [], []
    captured = {}

    def hook(module, inputs):
        captured["h"] = inputs[0].detach()
    handle = model.lm_head.register_forward_pre_hook(hook)
    bos = tokenizer.get_bos_token_id()
    try:
        records = probe_records if max_docs is None else probe_records[:max_docs]
        for rec in records:
            forms, senses = rec["forms"], rec["senses"]
            if not forms:
                continue
            # ids = [bos] + form ids, truncated so T <= seq_len
            ids = [bos] + [tokenizer.stoi[f] for f in forms]
            ids = ids[:seq_len]
            if len(ids) < 2:
                continue
            x = torch.tensor([ids], dtype=torch.long, device=device)
            model(x)  # triggers the lm_head pre-hook
            h = captured["h"][0].float().cpu().numpy()  # (T, dim)
            T = h.shape[0]
            # position p (1..T-1) -> form at index p-1, context p-1 form tokens
            for p in range(1, T):
                feats.append(h[p])
                labels.append(int(senses[p - 1]))
                ctx.append(p - 1)
    finally:
        handle.remove()
    if not feats:
        return np.zeros((0, 1)), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=np.int64), np.asarray(ctx, dtype=np.int64)


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Representation probe for polysemy×context.")
    p.add_argument("--model-tag", required=True, help="checkpoint tag, e.g. poly_hsw1p5_homonymy_L128")
    p.add_argument("--step", type=int, default=None, help="checkpoint step (default = last)")
    p.add_argument("--data-dir", default=None, help="probe data dir (default = recover from checkpoint meta)")
    p.add_argument("--device-type", default="", help="cuda|cpu|mps (empty = autodetect)")
    p.add_argument("--max-docs", type=int, default=2000, help="cap probe docs processed")
    p.add_argument("--test-frac", type=float, default=0.3, help="held-out fraction of tokens for probe eval")
    p.add_argument("--steps", type=int, default=300, help="probe optimizer steps")
    p.add_argument("--out-dir", default=None, help="output dir (default: <base_dir>/polysemy_analysis/probe)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = build_args()
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    base_dir = get_base_dir()
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.model_tag)
    step = args.step if args.step is not None else find_last_step(checkpoint_dir)

    # Recover the training data dir (holds vocab.json + probe.jsonl) from the checkpoint meta.
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    train_cfg = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            train_cfg = json.load(f).get("user_config", {}) or {}
    data_dir = args.data_dir or train_cfg.get("data_dir")
    assert data_dir, "could not determine data dir (pass --data-dir or train with --tokenizer identity)"
    probe_path = os.path.join(data_dir, "probe.jsonl")
    assert os.path.exists(probe_path), f"probe set not found: {probe_path} (regenerate with --probe-docs > 0)"

    tokenizer = get_identity_tokenizer(data_dir)
    model, _, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=step, tokenizer=tokenizer)
    seq_len = meta["model_config"]["sequence_len"]
    num_classes = meta["model_config"]["vocab_size"]  # >= max sense id; senses are a subset of form-id range
    print0(f"Probing {args.model_tag} step={step} | seq_len={seq_len} | data_dir={data_dir}")

    with open(probe_path, "r", encoding="utf-8") as f:
        probe_records = [json.loads(line) for line in f]

    X, y, ctx = extract_features(model, tokenizer, probe_records, device, seq_len, max_docs=args.max_docs)
    print0(f"Collected {len(y):,} probe tokens (dim={X.shape[1] if len(y) else 0}), "
           f"contexts {int(ctx.min()) if len(ctx) else 0}..{int(ctx.max()) if len(ctx) else 0}")
    if len(y) < 50:
        print0("Too few probe tokens to fit a probe.")
        compute_cleanup()
        return 1

    # Train/test split over tokens (deterministic).
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(y))
    n_test = max(1, int(args.test_frac * len(y)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    n_classes = int(max(num_classes, y.max() + 1))
    preds = fit_linear_probe(X[train_idx], y[train_idx], X[test_idx], n_classes,
                             device=device, steps=args.steps, seed=args.seed)
    acc_by_bucket, overall = bucket_accuracy(preds, y[test_idx], ctx[test_idx])
    summary = probe_resolution_summary({b: a for b, (a, _) in acc_by_bucket.items()})

    print0("\nSense-decoding accuracy by left-context length:")
    print0(f"{'context>=':>10} | {'accuracy':>8} | {'n':>7}")
    for b, (a, n) in acc_by_bucket.items():
        print0(f"{b:>10} | {a:>8.4f} | {n:>7}")
    print0(f"\noverall test accuracy: {overall:.4f}")
    print0(f"resolves_with_context: {summary.get('resolves_with_context')} "
           f"(acc {summary.get('acc_at_min_ctx')}→{summary.get('acc_at_max_ctx')}, "
           f"slope {summary.get('logctx_slope')})")

    if ddp_rank == 0:
        out_dir = args.out_dir or os.path.join(base_dir, "polysemy_analysis", "probe")
        os.makedirs(out_dir, exist_ok=True)
        out = {
            "model_tag": args.model_tag, "step": step, "data_dir": data_dir,
            "seq_len": seq_len, "num_probe_tokens": int(len(y)),
            "overall_test_accuracy": overall,
            "accuracy_by_context_bucket": {str(b): {"accuracy": a, "n": n} for b, (a, n) in acc_by_bucket.items()},
            "summary": summary,
        }
        path = os.path.join(out_dir, f"{args.model_tag}_probe.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print0(f"Probe result written to: {path}")

    compute_cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
