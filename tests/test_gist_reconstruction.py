"""
Tests for the auxiliary gist-reconstruction loss (experiments/gist-token-reconstruction-1.md).

Covers the pieces introduced for this experiment, all data/GPU-free:

  1. The experiment config function
     `scripts.jobs.run_training.gist_reconstruction_experiments` — the lambda ladder, the fixed
     K, single seed, 10k horizon, "no in-training evaluation" flags and node settings.
  2. The boundary/target machinery in `nanochat.gpt` (`_next_boundary_idx`,
     `gist_reconstruction_targets`) against an independent brute-force reference, including the
     exclusion rules (gist tokens, BOS, last sentence of a document, span bound).
  3. The auxiliary head itself (`GistReconstructionHead`): finite loss that decreases when
     overfitting a toy batch.
  4. The two invariants the experiment depends on:
     - gist_recon_weight = 0 (no head) is NUMERICALLY IDENTICAL to the pre-existing path;
     - a checkpoint trained with the head loads into the eval path (head discarded).

Run: PYTHONPATH=. python -m pytest tests/test_gist_reconstruction.py -v
"""

import re

import pytest
import torch
import torch.nn.functional as F

from scripts.jobs.run_training import gist_reconstruction_experiments
from nanochat.gpt import (
    GPT,
    GPTConfig,
    GistReconstructionHead,
    _next_boundary_idx,
    gist_reconstruction_targets,
)


REQUIRED_KEYS = {
    "args", "model_tag", "description", "cmd_hash", "instance_type",
    "experiment_slug", "num_gpus",
}

EXPECTED_TAGS = [
    "d12_gr_k4_w0p0",
    "d12_gr_k4_w0p1",
    "d12_gr_k4_w0p5",
    "d12_gr_k4_w1p0",
]


# ---------------------------------------------------------------------------
# 1) Experiment config function
# ---------------------------------------------------------------------------
class TestGistReconstructionConfigs:

    def test_arms_and_tags(self):
        configs = gist_reconstruction_experiments()
        tags = [c["model_tag"] for c in configs]
        assert tags == EXPECTED_TAGS
        assert len(set(tags)) == len(tags)  # one run per config, no fan-out
        # Small arm count: each is a full 10k-step 4xA100 run.
        assert len(configs) <= 4

    def test_required_keys_and_node_settings(self):
        for c in gist_reconstruction_experiments():
            assert REQUIRED_KEYS.issubset(c.keys())
            assert c["experiment_slug"] == "gist-token-reconstruction-1"
            assert c["instance_type"] == "a100.4gpu"
            assert c["num_gpus"] == 4

    def test_single_seed_only(self):
        seeds = set()
        for c in gist_reconstruction_experiments():
            m = re.search(r"--seed (\d+)", c["args"])
            assert m, f"no --seed in args: {c['args']}"
            seeds.add(m.group(1))
        assert seeds == {"0"}

    def test_ten_thousand_steps_and_d12(self):
        for c in gist_reconstruction_experiments():
            assert "--num-iterations 10000" in c["args"]
            assert "--depth 12" in c["args"]
            assert "--window-pattern L" in c["args"]

    def test_no_intermediate_evaluation(self):
        for c in gist_reconstruction_experiments():
            assert "--eval-every -1" in c["args"], c["model_tag"]
            assert "--core-metric-every -1" in c["args"], c["model_tag"]
            assert "--sample-every -1" in c["args"], c["model_tag"]

    def test_only_the_weight_varies(self):
        """Every arm must be identical except for --gist-recon-weight (that is the ablation)."""
        stripped = set()
        weights = []
        for c in gist_reconstruction_experiments():
            m = re.search(r"--gist-recon-weight (\S+)", c["args"])
            assert m, c["args"]
            weights.append(float(m.group(1)))
            stripped.add(re.sub(r"--gist-recon-weight \S+ ?", "", c["args"]))
            # K is fixed across arms, sentence attention always on.
            assert "--gist-placement sentence_nltk" in c["args"]
            assert "--num-gist-tokens 4" in c["args"]
        assert len(stripped) == 1, f"arms differ by more than the weight: {stripped}"
        # Exactly one lambda=0 control, and an increasing ladder of nonzero weights.
        assert weights[0] == 0.0
        assert sum(w == 0.0 for w in weights) == 1
        assert weights == sorted(weights)
        assert all(w > 0 for w in weights[1:])

    def test_does_not_relaunch_the_external_baseline(self):
        tags = {c["model_tag"] for c in gist_reconstruction_experiments()}
        assert "d12_sa_baseline" not in tags

    def test_distinct_cmd_hashes(self):
        hashes = [c["cmd_hash"] for c in gist_reconstruction_experiments()]
        assert len(set(hashes)) == len(hashes)


# ---------------------------------------------------------------------------
# 2) Boundary + target machinery
# ---------------------------------------------------------------------------
def _reference_targets(idx, targets, gist_ids, bos_id, max_len, stride):
    """Independent brute-force reference for gist_reconstruction_targets.

    For every position t: find the first gist boundary at-or-after t (a boundary = the last
    token of a contiguous gist run). t is supervised iff it is a real token (not a gist, not
    BOS), not an ignore position, that boundary exists and lies in the same document, and t is
    within max_len tokens of the start of its sentence.
    """
    gist_set = set(gist_ids)
    B, T = idx.shape
    sel = list(range(0, T, stride))
    boundary_out, rel_out, target_out = [], [], []
    for b in range(B):
        tok = [int(idx[b, t]) for t in range(T)]
        is_gist = [x in gist_set for x in tok]
        is_bos = [bos_id >= 0 and x == bos_id for x in tok]
        boundary = [is_gist[t] and not (t + 1 < T and is_gist[t + 1]) for t in range(T)]
        seg, c = [], 0
        for t in range(T):
            if is_bos[t]:
                c += 1
            seg.append(c)
        row_b, row_rel, row_tgt = [], [], []
        for t in sel:
            # first boundary at or after t
            nb = None
            for p in range(t, T):
                if boundary[p]:
                    nb = p
                    break
            # start of t's sentence: after the previous boundary, but not before the doc start
            prev_b = 0
            for p in range(t):
                if boundary[p]:
                    prev_b = p
            doc_start = 0
            for p in range(t + 1):
                if is_bos[p]:
                    doc_start = p
            sent_start = max(prev_b + 1, doc_start + 1)
            rel = t - sent_start
            ok = (
                not is_gist[t] and not is_bos[t]
                and nb is not None and seg[nb] == seg[t]
                and 0 <= rel < max_len
                and int(targets[b, t]) >= 0
            )
            row_b.append(min(nb if nb is not None else T - 1, T - 1))
            row_rel.append(min(max(rel, 0), max_len - 1))
            row_tgt.append(tok[t] if ok else -1)
        boundary_out.append(row_b)
        rel_out.append(row_rel)
        target_out.append(row_tgt)
    return sel, boundary_out, rel_out, target_out


class TestBoundaryMachinery:

    def test_next_boundary_idx(self):
        # positions:  0  1  2  3(g) 4(g) 5  6  7(g) 8
        gist_mask = torch.tensor([[False, False, False, True, True, False, False, True, False]])
        nb = _next_boundary_idx(gist_mask)[0].tolist()
        # boundaries at 4 (end of run 3-4) and 7. Position 8 has none -> T == 9.
        assert nb == [4, 4, 4, 4, 4, 7, 7, 7, 9]

    def test_next_boundary_idx_no_gists(self):
        gist_mask = torch.zeros(2, 5, dtype=torch.bool)
        assert (_next_boundary_idx(gist_mask) == 5).all()

    @pytest.mark.parametrize("stride", [1, 2, 3])
    def test_targets_match_reference(self, stride):
        gist_ids = (50, 51)
        bos = 1
        idx = torch.tensor([
            # one doc, three sentences (the third has no closing gist run)
            [1, 7, 8, 50, 51, 9, 10, 11, 50, 51, 12, 13],
            # two docs packed in one row (second BOS at position 6)
            [1, 2, 50, 51, 3, 4, 1, 6, 7, 50, 51, 8],
        ])
        targets = torch.cat([idx[:, 1:], torch.full((2, 1), -1)], dim=1)
        max_len = 8
        sel, boundary, rel, target = gist_reconstruction_targets(
            idx, targets, gist_ids, bos, max_len, stride)
        r_sel, r_b, r_rel, r_tgt = _reference_targets(idx, targets, gist_ids, bos, max_len, stride)
        assert sel.tolist() == r_sel
        assert target.tolist() == r_tgt
        assert rel.tolist() == r_rel
        # boundary is only meaningful where the position is supervised
        for b in range(idx.shape[0]):
            for i in range(len(r_sel)):
                if r_tgt[b][i] >= 0:
                    assert boundary[b][i] == r_b[b][i]

    def test_gist_and_bos_positions_are_excluded(self):
        gist_ids = (50, 51)
        bos = 1
        idx = torch.tensor([[1, 7, 8, 50, 51, 9, 10, 50, 51, 11, 12, 13]])
        targets = torch.cat([idx[:, 1:], torch.full((1, 1), -1)], dim=1)
        _, _, _, target = gist_reconstruction_targets(idx, targets, gist_ids, bos, 32, 1)
        t = target[0]
        # gist tokens (positions 3,4,7,8) are the encoder, never a target
        for p in (3, 4, 7, 8):
            assert t[p].item() == -1, f"gist position {p} was supervised"
        # BOS (position 0) is a document delimiter, not sentence content
        assert t[0].item() == -1
        # and no supervised target is ever a gist id or the BOS id
        supervised = t[t >= 0].tolist()
        assert supervised, "expected at least some supervised positions"
        assert not (set(supervised) & set(gist_ids))
        assert bos not in supervised

    def test_last_sentence_of_document_is_skipped(self):
        """Tokens after the final gist run of a document have no boundary to reconstruct them
        from and must be skipped -- never attached to the next document's boundary."""
        gist_ids = (50,)
        bos = 1
        # doc1 = 0..5 (last sentence = positions 4,5), doc2 = 6..11 (last sentence = 10,11)
        idx = torch.tensor([[1, 2, 3, 50, 4, 5, 1, 7, 8, 50, 9, 10]])
        targets = torch.cat([idx[:, 1:], torch.full((1, 1), -1)], dim=1)
        _, _, _, target = gist_reconstruction_targets(idx, targets, gist_ids, bos, 32, 1)
        t = target[0]
        for p in (4, 5, 10, 11):
            assert t[p].item() == -1, f"tail position {p} should have no reconstruction target"
        # the first sentence of each document IS supervised
        assert t[1].item() == 2 and t[2].item() == 3
        assert t[7].item() == 7 and t[8].item() == 8

    def test_ignore_positions_are_excluded(self):
        gist_ids = (50,)
        bos = 1
        idx = torch.tensor([[1, 2, 3, 4, 50, 5, 6, 50, 7, 8]])
        targets = torch.cat([idx[:, 1:], torch.full((1, 1), -1)], dim=1)
        targets[0, 2] = -1  # mark one position as ignore/padding
        _, _, _, target = gist_reconstruction_targets(idx, targets, gist_ids, bos, 32, 1)
        assert target[0, 2].item() == -1
        assert target[0, 1].item() == 2  # neighbours unaffected

    def test_max_sentence_len_bounds_the_span(self):
        gist_ids = (50,)
        bos = 1
        idx = torch.tensor([[1] + list(range(2, 12)) + [50, 99]])
        targets = torch.cat([idx[:, 1:], torch.full((1, 1), -1)], dim=1)
        _, _, rel, target = gist_reconstruction_targets(idx, targets, gist_ids, bos, 4, 1)
        # sentence starts at position 1, so only positions 1..4 (rel 0..3) are supervised
        assert [p for p in range(idx.shape[1]) if target[0, p] >= 0] == [1, 2, 3, 4]
        assert rel[0, 1:5].tolist() == [0, 1, 2, 3]

    def test_boundary_points_at_the_gist_run_end(self):
        gist_ids = (50, 51, 52, 53)  # K = 4
        bos = 1
        idx = torch.tensor([[1, 7, 8, 9, 50, 51, 52, 53, 10, 11, 50, 51, 52, 53, 12]])
        targets = torch.cat([idx[:, 1:], torch.full((1, 1), -1)], dim=1)
        _, boundary, _, target = gist_reconstruction_targets(idx, targets, gist_ids, bos, 32, 1)
        # tokens 1..3 belong to the sentence closed by the run ending at 7
        assert boundary[0, 1:4].tolist() == [7, 7, 7]
        # tokens 8,9 belong to the sentence closed by the run ending at 13
        assert boundary[0, 8:10].tolist() == [13, 13]
        assert (target[0, 1:4] >= 0).all() and (target[0, 8:10] >= 0).all()


# ---------------------------------------------------------------------------
# 3) The auxiliary head
# ---------------------------------------------------------------------------
class TestReconstructionHead:

    def _head(self, n_embd=32, num_gist=3, max_len=16, vocab=64):
        cfg = GPTConfig(
            n_embd=n_embd,
            end_of_sentence_token_ids=tuple(range(num_gist)),
            gist_recon_enabled=True,
            gist_recon_max_sentence_len=max_len,
        )
        head = GistReconstructionHead(cfg, vocab)
        head.init_weights()
        return head

    def test_shapes_and_position_table_padding(self):
        head = self._head(max_len=40)
        # the table is padded up to a multiple of 64 so its dim 0 stays reduce_scatter-friendly
        assert head.pos_table_size == 64
        g = torch.randn(2, 5, 3 * 32)
        rel = torch.zeros(2, 5, dtype=torch.long)
        assert head(g, rel).shape == (2, 5, 64)

    def test_loss_is_finite_and_decreases_on_overfit_batch(self):
        torch.manual_seed(0)
        vocab, num_gist, n_embd, n_pos = 64, 3, 32, 12
        head = self._head(n_embd=n_embd, num_gist=num_gist, vocab=vocab)
        # A toy batch: a few sentences (each with its own gist states) x within-sentence slots.
        gists = torch.randn(4, n_pos, num_gist * n_embd)
        rel = torch.arange(n_pos).unsqueeze(0).expand(4, -1).contiguous()
        target = torch.randint(0, vocab, (4, n_pos))
        opt = torch.optim.AdamW(head.parameters(), lr=3e-3)

        def loss_fn():
            logits = head(gists, rel).float()
            return F.cross_entropy(logits.view(-1, vocab), target.view(-1))

        first = loss_fn().item()
        assert torch.isfinite(torch.tensor(first))
        for _ in range(120):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            opt.step()
        last = loss_fn().item()
        assert torch.isfinite(torch.tensor(last))
        assert last < first - 0.5, f"reconstruction loss did not decrease: {first} -> {last}"

    def test_prediction_depends_on_position_and_content(self):
        """The head must model (content x position) jointly: two different within-sentence
        positions of the same sentence must be able to get different distributions."""
        torch.manual_seed(0)
        head = self._head()
        g = torch.randn(1, 2, 3 * 32).repeat(1, 1, 1)
        g[:, 1] = g[:, 0]  # same gist content
        rel = torch.tensor([[0, 5]])
        out = head(g, rel)
        assert not torch.allclose(out[0, 0], out[0, 1])


# ---------------------------------------------------------------------------
# 4) End-to-end invariants on a tiny GPT
# ---------------------------------------------------------------------------
GIST_IDS = (60, 61)
BOS_ID = 1


def _tiny_config(gist_recon_enabled, **overrides):
    kwargs = dict(
        sequence_len=32, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32,
        window_pattern="L",
        end_of_sentence_token_ids=GIST_IDS,
        bos_token_id=BOS_ID,
        gist_recon_enabled=gist_recon_enabled,
        gist_recon_max_sentence_len=8,
        gist_recon_stride=1,
    )
    kwargs.update(overrides)
    return GPTConfig(**kwargs)


def _tiny_model(gist_recon_enabled, seed=0, **overrides):
    model = GPT(_tiny_config(gist_recon_enabled, **overrides), pad_vocab_size_to=8)
    torch.manual_seed(seed)  # seed AFTER construction: init_weights re-inits everything
    model.init_weights()
    return model


def _tiny_batch():
    # two rows, each holding sentences separated by the 2-token gist run
    rows = [
        [BOS_ID, 5, 6, 7, 60, 61, 8, 9, 60, 61, 10, 11, 12, 13, 14, 15],
        [BOS_ID, 5, 6, 60, 61, 7, BOS_ID, 8, 9, 10, 60, 61, 11, 12, 13, 14],
    ]
    idx = torch.tensor(rows)
    targets = torch.cat([idx[:, 1:], torch.full((2, 1), -1)], dim=1)
    return idx, targets


class TestForwardIntegration:

    def test_weight_zero_path_is_numerically_identical(self):
        """gist_recon_weight = 0 allocates no head; the trunk init and the next-token loss must
        be bit-for-bit what they were before this experiment existed."""
        plain = _tiny_model(False)
        withhead = _tiny_model(True)
        assert plain.gist_recon is None
        assert withhead.gist_recon is not None
        # The head is built and initialized LAST, so every shared parameter is identical.
        plain_params = dict(plain.named_parameters())
        for name, p in withhead.named_parameters():
            if name.startswith("gist_recon."):
                continue
            assert name in plain_params
            assert torch.equal(p, plain_params[name]), f"{name} differs"
        idx, targets = _tiny_batch()
        with torch.no_grad():
            loss_plain = plain(idx, targets)
            loss_head = withhead(idx, targets)  # return_aux defaults to False
        assert torch.equal(loss_plain, loss_head)
        # ... and the aux path leaves the next-token loss untouched too
        with torch.no_grad():
            loss_aux, aux = withhead(idx, targets, return_aux=True)
        assert torch.equal(loss_plain, loss_aux)
        assert torch.isfinite(aux) and aux > 0

    def test_return_aux_without_head_returns_zero(self):
        plain = _tiny_model(False)
        idx, targets = _tiny_batch()
        with torch.no_grad():
            loss, aux = plain(idx, targets, return_aux=True)
            ref = plain(idx, targets)
        assert torch.equal(loss, ref)
        assert aux.item() == 0.0

    def test_aux_loss_is_finite_and_differentiable_into_the_trunk(self):
        model = _tiny_model(True)
        idx, targets = _tiny_batch()
        loss, aux = model(idx, targets, return_aux=True)
        assert torch.isfinite(aux)
        aux.backward()
        # every auxiliary head parameter receives a gradient (no unused params for the
        # distributed optimizer, which does not tolerate a None grad)
        for name, p in model.gist_recon.named_parameters():
            assert p.grad is not None, f"gist_recon.{name} has no grad"
        # and the auxiliary gradient reaches the trunk (through the gist hidden states).
        # NOTE: attn.c_q/c_k/c_v legitimately get a zero gradient at init from ANY loss, because
        # attn.c_proj is zero-initialized; probe the params that do carry gradient at init.
        assert model.transformer.wte.weight.grad.abs().sum() > 0
        assert model.transformer.h[-1].attn.c_proj.weight.grad.abs().sum() > 0
        # the aux loss must NOT push gradient through lm_head (no shared weights / no path)
        assert model.lm_head.weight.grad is None

    def test_no_aux_in_the_eval_paths(self):
        """loss_reduction='none' (the bpb path) and the logits path must be unaffected by the
        presence of the head -- these are the pure next-token metrics."""
        plain = _tiny_model(False)
        withhead = _tiny_model(True)
        idx, targets = _tiny_batch()
        with torch.no_grad():
            assert torch.equal(plain(idx, targets, loss_reduction='none'),
                               withhead(idx, targets, loss_reduction='none'))
            assert torch.equal(plain(idx), withhead(idx))

    def test_kv_cache_path_skips_the_aux_loss(self):
        """Requirement: the auxiliary loss is training-only and must be skipped whenever a
        kv-cache is in play (generate) or there are no targets."""
        model = _tiny_model(True)
        idx, _ = _tiny_batch()
        with torch.no_grad():
            out = model(idx)  # targets=None -> logits, never a tuple
        assert isinstance(out, torch.Tensor) and out.shape[-1] == model.config.vocab_size

    def test_head_params_are_isolated_in_the_optimizer(self):
        model = _tiny_model(True)
        opt = model.setup_optimizer()
        head_params = {id(p) for p in model.gist_recon.parameters()}
        grouped = [p for g in opt.param_groups for p in g["params"]]
        assert len(grouped) == len(list(model.parameters()))
        # the head never shares a group with the trunk / lm_head
        for g in opt.param_groups:
            ids = {id(p) for p in g["params"]}
            assert ids <= head_params or not (ids & head_params)
        # and lm_head weights are not touched by the head (no weight sharing)
        assert id(model.lm_head.weight) not in head_params
        assert model.gist_recon.out.weight is not model.lm_head.weight

    def test_scaling_params_exclude_the_head(self):
        """base_train derives the batch size / horizon from transformer_matrices + lm_head; the
        training-only head must not shift them (else the arms would not be comparable)."""
        plain = _tiny_model(False)
        withhead = _tiny_model(True)
        a, b = plain.num_scaling_params(), withhead.num_scaling_params()
        assert a["transformer_matrices"] == b["transformer_matrices"]
        assert a["lm_head"] == b["lm_head"]
        assert a["gist_recon"] == 0 and b["gist_recon"] > 0
        assert b["total"] == a["total"] + b["gist_recon"]
        # FLOPs/token describes the deployed (eval) model, so it is unchanged too
        assert plain.estimate_flops() == withhead.estimate_flops()


class TestCheckpointCompatibility:

    def test_eval_load_discards_the_head(self, tmp_path, monkeypatch):
        """A checkpoint trained WITH the head must load into the eval path (which builds the
        model with strict=True and never allocates the head)."""
        import json
        from dataclasses import asdict
        import nanochat.checkpoint_manager as cm

        model = _tiny_model(True)
        state = {"_orig_mod." + k: v for k, v in model.state_dict().items()}  # as torch.compile saves it
        assert any(k.startswith("_orig_mod.gist_recon.") for k in state)
        torch.save(state, tmp_path / "model_000010.pt")
        meta = {"step": 10, "model_config": asdict(model.config)}
        (tmp_path / "meta_000010.json").write_text(json.dumps(meta))

        # stub the tokenizer: build_model only uses it for a vocab-size sanity check
        class _Tok:
            def get_vocab_size(self):
                return model.config.vocab_size - len(GIST_IDS)
        monkeypatch.setattr(cm, "get_tokenizer", lambda: _Tok())

        loaded, _, meta_data = cm.build_model(str(tmp_path), 10, torch.device("cpu"), "eval")
        assert loaded.gist_recon is None, "eval must not allocate the training-only head"
        assert meta_data["model_config"]["gist_recon_enabled"] is False
        # the surviving weights are the ones that matter, and the model still runs
        idx, targets = _tiny_batch()
        with torch.no_grad():
            assert torch.isfinite(loaded(idx, targets))

    def test_old_checkpoints_get_defaults(self):
        from nanochat.checkpoint_manager import _patch_missing_config_keys
        cfg = {"vocab_size": 64, "n_layer": 2}
        _patch_missing_config_keys(cfg)
        assert cfg["gist_recon_enabled"] is False
        assert cfg["gist_recon_max_sentence_len"] > 0
        assert cfg["gist_recon_stride"] > 0
        assert GPTConfig(gist_recon_enabled=False).gist_recon_enabled is False
