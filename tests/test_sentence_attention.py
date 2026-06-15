"""
Tests for the sentence-attention experiment (block-causal + global-gist).

Covers the three new, data/GPU-free pieces introduced for this experiment:

  1. The experiment config function
     `scripts.jobs.run_training.sentence_attention_experiments` — arms, single seed, 10k
     horizon, the reviewer-mandated "no in-training evaluation" flags, and node settings.
  2. The tokenizer gist utilities in `nanochat.tokenizer`
     (`gist_token_ids`, `split_sentences_nltk`).
  3. The forward-built sentence mask in `nanochat.gpt`
     (`GPT._build_sentence_mask`, `_closest_boundary_idx`) checked against an independent
     brute-force reference, including multi-document (no cross-doc leakage) packing.

Run: python -m pytest tests/test_sentence_attention.py -v
"""

import re

import numpy as np
import pytest
import torch

from scripts.jobs.run_training import sentence_attention_experiments
from nanochat.tokenizer import gist_token_ids, split_sentences_nltk
from nanochat.gpt import GPT, GPTConfig, _closest_boundary_idx


REQUIRED_KEYS = {
    "args", "model_tag", "description", "cmd_hash", "instance_type",
    "experiment_slug", "num_gpus",
}

EXPECTED_TAGS = [
    "d12_sa_baseline",
    "d12_sa_nltk_k1",
    "d12_sa_nltk_k4",
    "d12_sa_nltk_k8",
    "d12_sa_nltk_k16",
]


# ---------------------------------------------------------------------------
# 1) Experiment config function
# ---------------------------------------------------------------------------
class TestSentenceAttentionConfigs:

    def test_arms_and_tags(self):
        configs = sentence_attention_experiments()
        tags = [c["model_tag"] for c in configs]
        # One full-causal baseline + 4 NLTK gist arms (K in {1,4,8,16}).
        assert tags == EXPECTED_TAGS
        # One run per config (no duplicate tags / no fan-out).
        assert len(set(tags)) == len(tags)

    def test_required_keys_and_node_settings(self):
        configs = sentence_attention_experiments()
        for c in configs:
            assert REQUIRED_KEYS.issubset(c.keys())
            assert c["experiment_slug"] == "sentence-attention"
            assert c["instance_type"] == "a100.4gpu"
            assert c["num_gpus"] == 4

    def test_single_seed_only(self):
        configs = sentence_attention_experiments()
        seeds = set()
        for c in configs:
            m = re.search(r"--seed (\d+)", c["args"])
            assert m, f"no --seed in args: {c['args']}"
            seeds.add(m.group(1))
        # Every run uses the same single seed (no multi-seed fan-out).
        assert seeds == {"0"}

    def test_ten_thousand_steps_and_d12(self):
        configs = sentence_attention_experiments()
        for c in configs:
            assert "--num-iterations 10000" in c["args"]
            assert "--depth 12" in c["args"]
            # Sentence mechanism isolated from sliding-window confound.
            assert "--window-pattern L" in c["args"]

    def test_no_intermediate_evaluation(self):
        """Reviewer-mandated: training must NOT run any periodic eval/sampling. All three
        cadence flags must be disabled (-1) on every arm so eval happens only post-training."""
        configs = sentence_attention_experiments()
        for c in configs:
            assert "--eval-every -1" in c["args"], c["model_tag"]
            assert "--core-metric-every -1" in c["args"], c["model_tag"]
            assert "--sample-every -1" in c["args"], c["model_tag"]

    def test_baseline_has_no_gist_flags(self):
        configs = sentence_attention_experiments()
        baseline = next(c for c in configs if c["model_tag"] == "d12_sa_baseline")
        # Full-causal baseline: no gist placement, no gist tokens.
        assert "--gist-placement" not in baseline["args"]
        assert "--num-gist-tokens" not in baseline["args"]

    def test_gist_arms_have_correct_k(self):
        configs = sentence_attention_experiments()
        by_tag = {c["model_tag"]: c for c in configs}
        for k in (1, 4, 8, 16):
            args = by_tag[f"d12_sa_nltk_k{k}"]["args"]
            assert "--gist-placement sentence_nltk" in args
            assert f"--num-gist-tokens {k}" in args


# ---------------------------------------------------------------------------
# 2) Tokenizer gist utilities
# ---------------------------------------------------------------------------
class TestGistTokenizerUtils:

    def test_gist_token_ids_layout(self):
        # K gist ids placed immediately past the real vocab, contiguous and ordered.
        assert gist_token_ids(32768, 1) == [32768]
        assert gist_token_ids(100, 4) == [100, 101, 102, 103]
        assert gist_token_ids(50, 0) == []

    def test_split_sentences_is_lossless(self):
        texts = [
            "Hello world. This is a test! Third one here?",
            "No terminal punctuation",
            "",
            "One sentence only.",
            "Dr. Smith went to Washington. He left at 5 p.m. sharp.",
        ]
        for t in texts:
            pieces = split_sentences_nltk(t)
            # Lossless reconstruction is what makes "encode pieces then concat" == "encode whole".
            assert "".join(pieces) == t
            # Always returns a non-empty list of pieces (even for empty / unsplittable input).
            assert len(pieces) >= 1

    def test_split_multiple_sentences(self):
        pieces = split_sentences_nltk("First. Second. Third.")
        assert len(pieces) == 3


# ---------------------------------------------------------------------------
# 3) Forward-built sentence mask
# ---------------------------------------------------------------------------
class _MaskHolder:
    """Minimal stand-in so we can exercise GPT._build_sentence_mask without instantiating
    the full model (the method only reads self.config and operates on the input ids)."""
    def __init__(self, gist_ids, bos_id):
        self.config = GPTConfig(end_of_sentence_token_ids=tuple(gist_ids), bos_token_id=bos_id)


def _reference_mask(idx, gist_ids, bos_id):
    """Independent brute-force reference for the sentence-attention mask (True = attend).

    allowed[q, k] is True iff (causal) and (same document) and (k in q's own sentence block
    OR k is a gist/BOS special token); the diagonal is always True. The block for query q
    starts at the most recent sentence boundary strictly before q (a boundary = the last
    token of a contiguous gist run)."""
    gist_set = set(gist_ids)
    B, T = idx.shape
    out = np.zeros((B, T, T), dtype=bool)
    for b in range(B):
        tok = [idx[b, t].item() for t in range(T)]
        is_gist = [x in gist_set for x in tok]
        boundary = [is_gist[t] and not (t + 1 < T and is_gist[t + 1]) for t in range(T)]
        seg, c = [], 0
        for t in range(T):
            if bos_id >= 0 and tok[t] == bos_id:
                c += 1
            seg.append(c)
        for q in range(T):
            eos = 0
            for p in range(q):  # most recent boundary strictly before q
                if boundary[p]:
                    eos = p
            for k in range(q + 1):  # causal
                special = is_gist[k] or (bos_id >= 0 and tok[k] == bos_id)
                ok = (k >= eos) or special
                if bos_id >= 0:
                    ok = ok and (seg[q] == seg[k])
                if k == q:
                    ok = True
                out[b, q, k] = ok
    return out


class TestSentenceMask:

    def test_closest_boundary_idx(self):
        # gist run boundary = LAST token of each contiguous run of gist positions.
        # positions:        0  1  2  3(g) 4(g) 5  6  7(g) 8
        gist_mask = torch.tensor([[False, False, False, True, True, False, False, True, False]])
        bidx = _closest_boundary_idx(gist_mask)[0].tolist()
        # boundaries are at index 4 (end of run 3-4) and 7. For q<=4 -> 0, q in {5,6,7} -> 4,
        # q == 8 -> 7 (boundary at 7 is strictly before 8).
        assert bidx == [0, 0, 0, 0, 0, 4, 4, 4, 7]

    def test_mask_matches_reference_single_doc(self):
        gist_ids = (50, 51)
        bos = 1
        idx = torch.tensor([[1, 7, 8, 50, 51, 9, 10, 11, 50, 51, 12, 13]])
        mask = GPT._build_sentence_mask(_MaskHolder(gist_ids, bos), idx)
        assert mask.shape == (1, 1, idx.shape[1], idx.shape[1])
        ref = _reference_mask(idx, gist_ids, bos)
        assert np.array_equal(mask[:, 0].numpy(), ref)

    def test_mask_matches_reference_multi_doc(self):
        # Two documents packed in one row; a second row with a different layout.
        gist_ids = (50, 51)
        bos = 1
        idx = torch.tensor([
            [1, 7, 8, 50, 51, 9, 1, 11, 50, 51, 12, 13],
            [1, 2, 50, 51, 3, 4, 5, 1, 6, 7, 50, 8],
        ])
        mask = GPT._build_sentence_mask(_MaskHolder(gist_ids, bos), idx)
        ref = _reference_mask(idx, gist_ids, bos)
        assert np.array_equal(mask[:, 0].numpy(), ref)

    def test_no_cross_document_leakage(self):
        gist_ids = (50,)
        bos = 1
        # doc1 = positions 0..5, doc2 = positions 6..11 (second BOS at index 6).
        idx = torch.tensor([[1, 2, 3, 50, 4, 5, 1, 7, 8, 50, 9, 10]])
        mask = GPT._build_sentence_mask(_MaskHolder(gist_ids, bos), idx)[0, 0]
        # No query in doc2 may attend to any key in doc1 (k < 6).
        assert not mask[6:, :6].any()
        # Causality preserved: no attention to strictly-future keys.
        T = idx.shape[1]
        future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        assert not (mask & future).any()

    def test_earlier_gist_globally_visible_within_doc(self):
        gist_ids = (50,)
        bos = 1
        idx = torch.tensor([[1, 2, 3, 50, 4, 5, 6, 7]])
        mask = GPT._build_sentence_mask(_MaskHolder(gist_ids, bos), idx)[0, 0]
        # The gist at position 3 must be visible to every later query in the same document.
        assert mask[4:, 3].all()

    def test_baseline_no_gist_means_no_mask_built(self):
        # With no gist ids configured, forward leaves attn_mask=None (standard causal path);
        # _build_sentence_mask is simply never invoked. Sanity: the config flag is empty.
        cfg = GPTConfig()
        assert cfg.end_of_sentence_token_ids == ()
