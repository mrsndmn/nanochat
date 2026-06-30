"""
Generate the Polysemy × Context synthetic corpora (component 1).

For one run we sample ONE sense stream (the syntax) and reuse it across every condition,
so the syntactic / global entropy is held constant by construction; only the sense->form
(lexical) layer varies. Each condition is written to its own directory under
``<out-dir>/<condition_slug>/`` as:

    shard_XXXXX.parquet   # 'text' column of whitespace-separated form symbols (zstd)
    vocab.json            # form symbol -> token id (identity tokenizer table)
    metadata.json         # H(S|W) target/measured, H_m, unigram entropy, gzip, |V|, ...

Usage:
    # Preview the plan (sizes, conditions) without writing anything
    python -m scripts.gen_polysemy_data --dry

    # Generate the default v1 grid to the shared artifacts store
    python -m scripts.gen_polysemy_data

    # Small local smoke run
    python -m scripts.gen_polysemy_data --out-dir /tmp/poly --num-tokens 200000 --num-senses 64
"""

import argparse
import os
import sys
import time


def _default_workers() -> int:
    """A sensible default worker count: usable CPUs, capped (diminishing returns + overhead)."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(32, n))

from nanochat.polysemy import (
    GeneratorConfig, POS_CLASSES, PROBE_SEED_OFFSET, build_default_pcfg, build_sense_inventory,
    generate_sense_corpus, _sense_probabilities, build_condition, default_conditions,
    render_documents_with_senses, write_parquet_shards, write_vocab, write_metadata,
    write_probe_jsonl,
)


def _class_sizes(num_senses: int) -> dict:
    """Split K senses across POS classes: content classes (N,V) large, function (DET,P) small."""
    fractions = {"N": 0.45, "V": 0.30, "DET": 0.10, "P": 0.15}
    sizes = {c: max(1, int(round(num_senses * f))) for c, f in fractions.items()}
    # fix rounding drift so the totals match num_senses exactly
    drift = num_senses - sum(sizes.values())
    sizes["N"] += drift
    return {c: sizes[c] for c in POS_CLASSES}


# Heavy shared inputs for parallel condition builds. Forked Pool workers inherit these
# copy-on-write, so the multi-GB sense stream is never pickled to children. Populated in the
# parent immediately before the (fork-context) Pool is created.
_COND_STATE = {}


def _build_one(cond, cond_dir, cfg, pcfg, inventory, sense_docs, sense_prob,
               probe_sense_docs, shard_chars, seed):
    """Build + write one condition; return a small summary (never returns the big objects)."""
    t1 = time.time()
    documents, smap, metadata = build_condition(cfg, pcfg, inventory, sense_docs, sense_prob, cond)
    paths = write_parquet_shards(documents, cond_dir, shard_chars=shard_chars)
    write_vocab(smap, cond_dir)
    write_metadata(metadata, cond_dir)
    if probe_sense_docs is not None:
        write_probe_jsonl(render_documents_with_senses(probe_sense_docs, smap, seed=seed), cond_dir)
    hsw = metadata["h_s_given_w"]
    return {"slug": cond.slug, "vocab_size": smap.vocab_size, "target": hsw["target_bits"],
            "measured": hsw["measured_bits"], "within_tol": hsw["within_tolerance"],
            "shards": len(paths), "secs": time.time() - t1}


def _build_condition_worker(task):
    """Pool worker: read the fork-inherited shared state and build one condition."""
    cond, cond_dir, shard_chars, seed = task
    s = _COND_STATE
    return _build_one(cond, cond_dir, s["cfg"], s["pcfg"], s["inventory"], s["sense_docs"],
                      s["sense_prob"], s["probe_sense_docs"], shard_chars, seed)


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate polysemy×context synthetic corpora.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="output root (default: <NANOCHAT_BASE_DIR>/base_data_polysemy)")
    p.add_argument("--num-senses", type=int, default=512, help="total senses K")
    p.add_argument("--num-tokens", type=int, default=10_000_000, help="approx sense tokens to generate")
    p.add_argument("--seed", type=int, default=0, help="seed for the whole run (deterministic)")
    p.add_argument("--zipf-exponent", type=float, default=1.0, help="within-class Zipf exponent")
    p.add_argument("--max-depth", type=int, default=5, help="PCFG recursion depth cap")
    p.add_argument("--min-len", type=int, default=8, help="min derivation length (senses)")
    p.add_argument("--max-len", type=int, default=40, help="max derivation length (senses)")
    p.add_argument("--tolerance", type=float, default=0.05, help="H(S|W) target tolerance (bits)")
    p.add_argument("--shard-chars", type=int, default=50_000_000, help="approx chars per parquet shard")
    p.add_argument("--probe-docs", type=int, default=2000, help="held-out sense-labeled docs per condition for the representation probe (0 = skip)")
    p.add_argument("--num-workers", type=int, default=_default_workers(),
                   help="parallel processes for sense-stream sampling (the dominant cost). "
                        "Deterministic given (seed, num_workers); 1 = original single-process behavior.")
    p.add_argument("--condition-workers", type=int, default=len(default_conditions()),
                   help="parallel processes for the per-condition build (render + stats + write). "
                        "Each holds ~3x the per-condition corpus in RAM, so lower it for very large "
                        "--num-tokens; 1 = serial.")
    p.add_argument("--dry", action="store_true", help="print the plan and exit without writing")
    p.add_argument("--force", action="store_true", help="overwrite a condition dir if it already exists")
    return p.parse_args()


def main() -> int:
    args = build_args()

    out_root = args.out_dir
    if out_root is None:
        from nanochat.common import get_base_dir
        out_root = os.path.join(get_base_dir(), "base_data_polysemy")

    class_sizes = _class_sizes(args.num_senses)
    cfg = GeneratorConfig(
        class_sizes=class_sizes, num_tokens=args.num_tokens, seed=args.seed,
        zipf_exponent=args.zipf_exponent, max_depth=args.max_depth,
        min_len=args.min_len, max_len=args.max_len, tolerance=args.tolerance,
        num_workers=args.num_workers,
    )
    conditions = default_conditions()

    print(f"out_root      = {out_root}")
    print(f"num_senses K  = {args.num_senses}  (class sizes: {class_sizes})")
    print(f"num_tokens    ~ {args.num_tokens:,}")
    print(f"seed          = {args.seed}")
    print(f"num_workers   = {args.num_workers} (sampling) | {args.condition_workers} (conditions)")
    print(f"conditions    = {[c.slug for c in conditions]}")
    if args.dry:
        print("[DRY] no data written.")
        return 0

    pcfg = build_default_pcfg()
    inventory = build_sense_inventory(class_sizes, zipf_exponent=args.zipf_exponent)

    print("Generating the shared sense stream (syntax held constant across conditions)...")
    t0 = time.time()
    sense_docs = generate_sense_corpus(
        pcfg, inventory, num_tokens=args.num_tokens, max_depth=args.max_depth,
        min_len=args.min_len, max_len=args.max_len, seed=args.seed, num_workers=args.num_workers,
    )
    sense_prob = _sense_probabilities(sense_docs, inventory.num_senses)
    n_sense_tokens = sum(len(d) for d in sense_docs)
    print(f"  {len(sense_docs):,} docs | {n_sense_tokens:,} sense tokens | {time.time()-t0:.1f}s")

    # Held-out sense stream for the representation probe (disjoint seed, shared across
    # conditions so its syntax is held constant just like the training stream).
    probe_sense_docs = None
    if args.probe_docs > 0:
        probe_sense_docs = generate_sense_corpus(
            pcfg, inventory, num_tokens=args.probe_docs * args.max_len,
            max_depth=args.max_depth, min_len=args.min_len, max_len=args.max_len,
            seed=args.seed + PROBE_SEED_OFFSET, num_workers=args.num_workers,
        )[: args.probe_docs]
        print(f"Probe held-out stream: {len(probe_sense_docs):,} docs (sense-labeled)")

    # Which conditions still need building (idempotency / --force)?
    pending = []
    for cond in conditions:
        cond_dir = os.path.join(out_root, cond.slug)
        if os.path.isdir(cond_dir) and os.listdir(cond_dir) and not args.force:
            print(f"Skipping {cond.slug}: {cond_dir} already exists (use --force).")
            continue
        pending.append((cond, cond_dir))

    cond_workers = max(1, min(args.condition_workers, len(pending)))
    t_build = time.time()
    summaries = []
    if pending and cond_workers > 1:
        # Build conditions in parallel. fork context so workers inherit the (multi-GB) shared
        # sense stream copy-on-write instead of pickling it; only the small task tuples are sent.
        from multiprocessing import get_context
        _COND_STATE.update(cfg=cfg, pcfg=pcfg, inventory=inventory, sense_docs=sense_docs,
                           sense_prob=sense_prob, probe_sense_docs=probe_sense_docs)
        tasks = [(cond, cond_dir, args.shard_chars, args.seed) for cond, cond_dir in pending]
        print(f"Building {len(pending)} conditions with {cond_workers} parallel workers...")
        with get_context("fork").Pool(cond_workers) as pool:
            summaries = pool.map(_build_condition_worker, tasks)
        _COND_STATE.clear()
    else:
        for cond, cond_dir in pending:
            summaries.append(_build_one(cond, cond_dir, cfg, pcfg, inventory, sense_docs,
                                        sense_prob, probe_sense_docs, args.shard_chars, args.seed))
    if pending:
        print(f"Built {len(pending)} conditions in {time.time()-t_build:.1f}s")

    vocab_sizes = set()
    for s in summaries:
        vocab_sizes.add(s["vocab_size"])
        ok = "ok" if s["within_tol"] else "OUT-OF-TOL"
        print(f"  [{s['slug']}] |V|={s['vocab_size']} H(S|W) target={s['target']:.2f} "
              f"measured={s['measured']:.3f} ({ok}) | {s['shards']} shards | {s['secs']:.1f}s")

    if len(vocab_sizes) > 1:
        print(f"WARNING: |V| differs across generated conditions: {sorted(vocab_sizes)}")
    elif vocab_sizes:
        print(f"|V| held constant across conditions: {vocab_sizes.pop()}")
    print(f"Done. Corpora under: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
