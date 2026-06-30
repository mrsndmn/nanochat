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

from nanochat.polysemy import (
    GeneratorConfig, POS_CLASSES, build_default_pcfg, build_sense_inventory,
    generate_sense_corpus, _sense_probabilities, build_condition, default_conditions,
    write_parquet_shards, write_vocab, write_metadata,
)


def _class_sizes(num_senses: int) -> dict:
    """Split K senses across POS classes: content classes (N,V) large, function (DET,P) small."""
    fractions = {"N": 0.45, "V": 0.30, "DET": 0.10, "P": 0.15}
    sizes = {c: max(1, int(round(num_senses * f))) for c, f in fractions.items()}
    # fix rounding drift so the totals match num_senses exactly
    drift = num_senses - sum(sizes.values())
    sizes["N"] += drift
    return {c: sizes[c] for c in POS_CLASSES}


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
    )
    conditions = default_conditions()

    print(f"out_root      = {out_root}")
    print(f"num_senses K  = {args.num_senses}  (class sizes: {class_sizes})")
    print(f"num_tokens    ~ {args.num_tokens:,}")
    print(f"seed          = {args.seed}")
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
        min_len=args.min_len, max_len=args.max_len, seed=args.seed,
    )
    sense_prob = _sense_probabilities(sense_docs, inventory.num_senses)
    n_sense_tokens = sum(len(d) for d in sense_docs)
    print(f"  {len(sense_docs):,} docs | {n_sense_tokens:,} sense tokens | {time.time()-t0:.1f}s")

    vocab_sizes = set()
    for cond in conditions:
        cond_dir = os.path.join(out_root, cond.slug)
        if os.path.isdir(cond_dir) and os.listdir(cond_dir) and not args.force:
            print(f"Skipping {cond.slug}: {cond_dir} already exists (use --force).")
            continue
        t1 = time.time()
        documents, smap, metadata = build_condition(cfg, pcfg, inventory, sense_docs, sense_prob, cond)
        paths = write_parquet_shards(documents, cond_dir, shard_chars=args.shard_chars)
        write_vocab(smap, cond_dir)
        write_metadata(metadata, cond_dir)
        vocab_sizes.add(smap.vocab_size)
        hsw = metadata["h_s_given_w"]
        ok = "ok" if hsw["within_tolerance"] else "OUT-OF-TOL"
        print(f"  [{cond.slug}] |V|={smap.vocab_size} H(S|W) target={hsw['target_bits']:.2f} "
              f"measured={hsw['measured_bits']:.3f} ({ok}) | {len(paths)} shards | {time.time()-t1:.1f}s")

    if len(vocab_sizes) > 1:
        print(f"WARNING: |V| differs across generated conditions: {sorted(vocab_sizes)}")
    elif vocab_sizes:
        print(f"|V| held constant across conditions: {vocab_sizes.pop()}")
    print(f"Done. Corpora under: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
