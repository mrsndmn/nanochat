"""Measure how many attention tokens sentence attention saves vs full causal.

Builds the **exact** training mask (``GPT._build_sentence_mask``) over real val
rows with K gist tokens inserted at NLTK sentence boundaries — the same dataloader
path the BPB eval uses — and compares the per-query attended-key count against the
full-causal reference (all tokens of the query's own document at-or-before q).
Prints the headline numbers hardcoded into the fig4 token-savings panel.

Run from the repo root:

    python experiments/figures/gist-hypernetwork/measure_token_savings.py
"""
import argparse
import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, REPO_ROOT)

from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.gpt import GPT
from nanochat.tokenizer import get_tokenizer


def count_pairs(idx, gist_ids, bos_id):
    """Per-row (causal_pairs, same_doc_pairs, sentence_attention_pairs).

    causal  = full-row keys at-or-before q — what the plain causal baseline
              attends to (its sequences carry no gists and no doc mask).
    same_doc= the causal reference confined to q's own document (the packaging
              the sentence-attention mask also enforces).
    sa      = the boolean mask the model actually applies (GPT._build_sentence_mask).
    """
    dummy = SimpleNamespace(
        config=SimpleNamespace(
            end_of_sentence_token_ids=tuple(gist_ids),
            bos_token_id=bos_id,
        )
    )
    mask = GPT._build_sentence_mask(dummy, idx)  # (B, 1, T, T) bool
    B, T = idx.shape
    seg = (idx == bos_id).cumsum(dim=1)  # (B, T) document id within the row
    q = torch.arange(T).view(1, T, 1)
    k = torch.arange(T).view(1, 1, T)
    causal = k <= q  # (1, T, T)
    same_doc = causal & (seg.unsqueeze(2) == seg.unsqueeze(1))  # (B, T, T)
    causal_pairs = causal.expand(B, -1, -1).sum(dim=(-1, -2)).long()
    return (
        causal_pairs,
        same_doc.sum(dim=(-1, -2)).long(),
        mask.sum(dim=(-1, -2)).long(),
    )


def measure(K, n_batches, B=8, T=2048):
    gist_ids = [32768 + i for i in range(K)]
    tokenizer = get_tokenizer()
    bos = tokenizer.get_bos_token_id()
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer,
        B,
        T,
        "val",
        device="cpu",
        gist_token_ids=gist_ids,
        gist_placement="sentence_nltk",
    )
    causal_tot = same_doc_tot = sa_tot = tokens = gist_tokens = 0
    for _ in range(n_batches):
        idx, _ = next(loader)
        causal, same_doc, sa = count_pairs(idx, gist_ids, bos)
        causal_tot += causal.sum().item()
        same_doc_tot += same_doc.sum().item()
        sa_tot += sa.sum().item()
        tokens += idx.numel()
        gist_tokens += (idx >= 32768).sum().item()
    causal_pq = causal_tot / tokens
    same_doc_pq = same_doc_tot / tokens
    sa_pq = sa_tot / tokens
    print(f"K={K}: {n_batches} batches x {B}x{T} = {tokens:,} tokens "
          f"({tokens // 2048} rows)")
    print(f"  attended keys per query:  full causal {causal_pq:,.1f}  |  "
          f"same-doc causal {same_doc_pq:,.1f}  ->  "
          f"sentence attention {sa_pq:,.1f}")
    print(f"  attention pairs saved:    vs baseline "
          f"{(1.0 - sa_tot / causal_tot) * 100.0:.1f}%  |  "
          f"vs same-doc {(1.0 - sa_tot / same_doc_tot) * 100.0:.1f}%")
    print(f"  gist tokens in sequence:  {gist_tokens / tokens * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--batches", type=int, default=25)
    args = parser.parse_args()
    for K in args.k:
        measure(K, args.batches)
