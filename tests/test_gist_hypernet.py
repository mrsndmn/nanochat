"""
Tests for the gist-hypernetwork experiment (content-conditioned gist embeddings).

Covers the data/GPU-free pieces:

  1. The experiment config function `scripts.jobs.run_training.gist_hypernetwork_experiments`
     — two arms (gated/forced), protocol identical to the fixed-gist control arm.
  2. The cross-attention mask `nanochat.gpt._gist_cross_attn_mask` against an independent
     brute-force reference (own-sentence keys only, no gist/BOS keys, per-document, causal,
     non-empty rows).
  3. The mechanism on a tiny CPU model: gated arm is bit-exact equal to the fixed-gist
     control at init (alpha=0 and no NaN leakage), forced arm actually replaces gist
     embeddings, causality/content isolation of `GPT._apply_gist_hypernet`, optimizer
     param partition, scaling-params accounting, meta-device construction, and the
     checkpoint-manager config patch.

Run: python -m pytest tests/test_gist_hypernet.py -v
"""

import re

import numpy as np
import pytest
import torch

from scripts.jobs.run_training import gist_hypernetwork_experiments, sentence_attention_experiments
from nanochat.gpt import GPT, GPTConfig, _gist_cross_attn_mask
from nanochat.checkpoint_manager import _patch_missing_config_keys


# ---------------------------------------------------------------------------
# 1) Experiment config function
# ---------------------------------------------------------------------------
class TestGistHypernetConfigs:

    def test_arms_and_tags(self):
        configs = gist_hypernetwork_experiments()
        tags = [c["model_tag"] for c in configs]
        assert tags == ["d12_sa_nltk_k8_hnet_gated", "d12_sa_nltk_k8_hnet_forced"]
        assert len(set(tags)) == len(tags)

    def test_arm_flags(self):
        by_tag = {c["model_tag"]: c for c in gist_hypernetwork_experiments()}
        assert "--gist-hypernet gated" in by_tag["d12_sa_nltk_k8_hnet_gated"]["args"]
        assert "--gist-hypernet forced" in by_tag["d12_sa_nltk_k8_hnet_forced"]["args"]

    def test_protocol_matches_fixed_gist_control(self):
        """The ONLY difference vs the d12_sa_nltk_k8 control must be the --gist-hypernet flag,
        so the comparison isolates content-conditioning from every other knob."""
        control = next(c for c in sentence_attention_experiments() if c["model_tag"] == "d12_sa_nltk_k8")
        control_flags = set(control["args"].split())
        for c in gist_hypernetwork_experiments():
            arm_flags = set(c["args"].split())
            extra = arm_flags - control_flags
            missing = control_flags - arm_flags
            assert missing == set(), f"{c['model_tag']} dropped control flags: {missing}"
            assert extra == {"--gist-hypernet", "gated"} or extra == {"--gist-hypernet", "forced"}, extra

    def test_single_seed_10k_no_intermediate_eval(self):
        for c in gist_hypernetwork_experiments():
            assert re.search(r"--seed 0(\s|$)", c["args"])
            assert "--num-iterations 10000" in c["args"]
            assert "--eval-every -1" in c["args"]
            assert "--core-metric-every -1" in c["args"]
            assert "--sample-every -1" in c["args"]
            assert c["instance_type"] == "a100.4gpu"
            assert c["num_gpus"] == 4
            assert c["experiment_slug"] == "gist-hypernetwork"


# ---------------------------------------------------------------------------
# 2) Cross-attention mask vs brute-force reference
# ---------------------------------------------------------------------------
def _reference_hypernet_mask(idx, gist_start, bos_id):
    """Brute-force reference: q attends k iff k is a non-gist non-BOS token of q's own
    sentence (same most-recent-boundary strictly before), k <= q, same document; the
    diagonal is always True."""
    B, T = idx.shape
    out = np.zeros((B, T, T), dtype=bool)
    for b in range(B):
        tok = [idx[b, t].item() for t in range(T)]
        is_gist = [x >= gist_start for x in tok]
        boundary = [is_gist[t] and not (t + 1 < T and is_gist[t + 1]) for t in range(T)]
        seg, c = [], 0
        for t in range(T):
            if bos_id >= 0 and tok[t] == bos_id:
                c += 1
            seg.append(c)
        eos = []
        for qi in range(T):
            e = 0
            for p in range(qi):
                if boundary[p]:
                    e = p
            eos.append(e)
        for qi in range(T):
            for ki in range(T):
                ok = (ki <= qi) and (eos[ki] == eos[qi]) and not is_gist[ki]
                if bos_id >= 0:
                    ok = ok and tok[ki] != bos_id and seg[qi] == seg[ki]
                if ki == qi:
                    ok = True
                out[b, qi, ki] = ok
    return out


class TestGistCrossAttnMask:

    GIST_START = 50  # ids >= 50 are gists in these tests
    BOS = 1

    def test_matches_reference_single_doc(self):
        idx = torch.tensor([[1, 7, 8, 50, 51, 9, 10, 11, 50, 51, 12, 13]])
        mask = _gist_cross_attn_mask(idx, self.GIST_START, self.BOS)
        assert mask.shape == (1, 1, idx.shape[1], idx.shape[1])
        ref = _reference_hypernet_mask(idx, self.GIST_START, self.BOS)
        assert np.array_equal(mask[:, 0].numpy(), ref)

    def test_matches_reference_multi_doc(self):
        idx = torch.tensor([
            [1, 7, 8, 50, 51, 9, 1, 11, 50, 51, 12, 13],
            [1, 2, 50, 51, 3, 4, 5, 1, 6, 7, 50, 51],
        ])
        mask = _gist_cross_attn_mask(idx, self.GIST_START, self.BOS)
        ref = _reference_hypernet_mask(idx, self.GIST_START, self.BOS)
        assert np.array_equal(mask[:, 0].numpy(), ref)

    def test_gist_sees_only_own_sentence(self):
        #        0    1  2  3(g) 4(g) 5  6  7(g) 8(g) 9
        idx = torch.tensor([[1, 7, 8, 50, 51, 9, 10, 50, 51, 12]])
        m = _gist_cross_attn_mask(idx, self.GIST_START, self.BOS)[0, 0]
        # First run (positions 3,4): sentence keys {1, 2} plus self only.
        assert m[3].nonzero().flatten().tolist() == [1, 2, 3]
        assert m[4].nonzero().flatten().tolist() == [1, 2, 4]
        # Second run (positions 7,8): sentence keys {5, 6} plus self only — NOT {1, 2}.
        assert m[7].nonzero().flatten().tolist() == [5, 6, 7]
        assert m[8].nonzero().flatten().tolist() == [5, 6, 8]

    def test_no_empty_rows(self):
        # Degenerate inputs (BOS-only doc, gist right after BOS) must still leave every
        # query row non-empty — SDPA turns fully-masked rows into NaNs.
        idx = torch.tensor([[1, 50, 51, 2, 1, 1, 3, 50, 51, 4, 5, 6]])
        m = _gist_cross_attn_mask(idx, self.GIST_START, self.BOS)[0, 0]
        assert m.any(dim=-1).all()

    def test_no_cross_document_leakage_and_causal(self):
        idx = torch.tensor([[1, 2, 3, 50, 4, 5, 1, 7, 8, 50, 9, 10]])
        m = _gist_cross_attn_mask(idx, self.GIST_START, self.BOS)[0, 0]
        assert not m[6:, :6].any()
        T = idx.shape[1]
        future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        assert not (m & future).any()


# ---------------------------------------------------------------------------
# 3) Mechanism on a tiny CPU model
# ---------------------------------------------------------------------------
VOCAB = 64          # pads to 64 exactly
GIST_IDS = (62, 63) # K=2, contiguous at the top of the vocab
BOS = 1


def tiny_config(gist_hypernet="none"):
    return GPTConfig(
        sequence_len=32, vocab_size=VOCAB, n_layer=2, n_head=2, n_kv_head=2, n_embd=64,
        window_pattern="L", end_of_sentence_token_ids=GIST_IDS, bos_token_id=BOS,
        gist_hypernet=gist_hypernet,
    )


def build_tiny(gist_hypernet="none", seed=0):
    model = GPT(tiny_config(gist_hypernet))
    # Re-seed AFTER construction: module constructors consume RNG (extra hypernet Linears
    # would desync the shared-weight init). Real training constructs on meta device, where
    # construction consumes no RNG, and the hypernet init draws come last in init_weights.
    torch.manual_seed(seed)
    model.init_weights()
    return model.eval()


# Two docs packed in one row; sentences separated by the (62, 63) gist runs.
IDX = torch.tensor([[BOS, 5, 6, 7, 62, 63, 8, 9, 62, 63, 10, 11, BOS, 12, 13, 62, 63, 14, 15, 2]])


class TestGistHypernetMechanism:

    def test_gated_bit_exact_at_init(self):
        """alpha=0 must make the gated arm bit-for-bit identical to the fixed-gist control
        (same seed => identical shared weights; also catches NaN leakage through 0*h)."""
        control = build_tiny("none")
        gated = build_tiny("gated")
        with torch.no_grad():
            assert torch.equal(control.forward(IDX), gated.forward(IDX))

    def test_forced_no_nans_and_differs(self):
        forced = build_tiny("forced")
        control = build_tiny("none")
        with torch.no_grad():
            logits = forced.forward(IDX)
            assert not torch.isnan(logits).any()
            assert not torch.equal(logits, control.forward(IDX))

    def _apply(self, model, idx):
        with torch.no_grad():
            x = model.transformer.wte(idx).float()
            T = idx.shape[1]
            cos_sin = (model.cos[:, :T].float(), model.sin[:, :T].float())
            return model._apply_gist_hypernet(x, idx, cos_sin), x

    def test_non_gist_positions_unchanged(self):
        forced = build_tiny("forced")
        out, x = self._apply(forced, IDX)
        non_gist = (IDX < min(GIST_IDS)).squeeze(0)
        assert torch.equal(out[0, non_gist], x[0, non_gist])
        gist = ~non_gist
        assert not torch.equal(out[0, gist], x[0, gist])

    def test_gist_content_isolated_to_own_sentence(self):
        """Forced arm: a gist run's output depends on its own sentence's tokens only —
        not on later tokens, other sentences, or the other document."""
        forced = build_tiny("forced")
        out, _ = self._apply(forced, IDX)
        run1, run2, run3 = [4, 5], [8, 9], [15, 16]  # gist positions per run

        # Perturb sentence 1 of doc 1 (position 2): run1 changes, run2/run3 don't.
        idx2 = IDX.clone(); idx2[0, 2] = 21
        out2, _ = self._apply(forced, idx2)
        assert not torch.equal(out2[0, run1], out[0, run1])
        assert torch.equal(out2[0, run2], out[0, run2])
        assert torch.equal(out2[0, run3], out[0, run3])

        # Perturb sentence 2 of doc 1 (position 6): run1 unchanged (causality), run2 changes.
        idx3 = IDX.clone(); idx3[0, 6] = 22
        out3, _ = self._apply(forced, idx3)
        assert torch.equal(out3[0, run1], out[0, run1])
        assert not torch.equal(out3[0, run2], out[0, run2])

        # Perturb doc 2 (position 14): doc 1 runs unchanged, doc 2 run changes.
        idx4 = IDX.clone(); idx4[0, 14] = 23
        out4, _ = self._apply(forced, idx4)
        assert torch.equal(out4[0, run1], out[0, run1])
        assert torch.equal(out4[0, run2], out[0, run2])
        assert not torch.equal(out4[0, run3], out[0, run3])

    def test_gated_path_not_dead_at_init(self):
        """The known dead-path failure: gate AND projections zero-init => zero gradient
        everywhere. Here alpha=0 but projections are nonzero, so alpha itself must receive
        a nonzero gradient from a real backward pass."""
        gated = build_tiny("gated").train()
        targets = torch.roll(IDX, -1, dims=1)
        loss = gated.forward(IDX, targets=targets)
        loss.backward()
        alphas = gated.gist_hypernet.alphas
        assert alphas.grad is not None
        assert alphas.grad.abs().sum().item() > 0

    def test_optimizer_partition_and_groups(self):
        gated = build_tiny("gated")
        optimizer = gated.setup_optimizer()  # internal assert covers the full partition
        all_group_params = [p for g in optimizer.param_groups for p in g["params"]]
        assert len(all_group_params) == len(list(gated.parameters()))
        # alphas sit in an AdamW group; the 3 hypernet matrices are in Muon groups.
        hn = gated.gist_hypernet
        kinds = {id(p): g["kind"] for g in optimizer.param_groups for p in g["params"]}
        assert kinds[id(hn.alphas)] == "adamw"
        assert kinds[id(hn.queries)] == "adamw"
        for w in (hn.c_k.weight, hn.c_v.weight, hn.c_proj.weight):
            assert kinds[id(w)] == "muon"

    def test_scaling_params_accounting(self):
        gated = build_tiny("gated")
        counts = gated.num_scaling_params()  # internal assert: groups sum to total
        assert counts["gist_hypernet"] > 0
        # The hypernet must NOT leak into the horizon-defining groups: same values as control.
        control = build_tiny("none")
        control_counts = control.num_scaling_params()
        assert counts["transformer_matrices"] == control_counts["transformer_matrices"]
        assert counts["lm_head"] == control_counts["lm_head"]

    def test_meta_device_construction(self):
        for mode in ("gated", "forced"):
            with torch.device("meta"):
                GPT(tiny_config(mode))

    def test_requires_gist_tokens(self):
        cfg = GPTConfig(sequence_len=32, vocab_size=VOCAB, n_layer=2, n_head=2, n_kv_head=2,
                        n_embd=64, gist_hypernet="gated")  # no gist ids configured
        with pytest.raises(AssertionError):
            GPT(cfg)

    def test_checkpoint_config_patch_defaults_to_none(self):
        kwargs = {}
        _patch_missing_config_keys(kwargs)
        assert kwargs["gist_hypernet"] == "none"
