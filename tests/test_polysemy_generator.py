"""
Tests for the Polysemy × Context synthetic dataset generator (nanochat/polysemy.py).

These cover the spec's acceptance criteria for component 1:
- monosemous condition has H(S|W) == 0 exactly;
- polysemous conditions hit the target H(S|W) within tolerance;
- |V| is equal across all conditions in a run;
- the corpus reloads as a 1:1 form<->token stream (decode(encode(text)) == text);
- generation is deterministic given a seed;
- parquet shards + vocab.json + metadata.json are written and well-formed.

Run: PYTHONPATH=. pytest -q tests/test_polysemy_generator.py
"""

import json
import os

import pytest

from nanochat.polysemy import (
    GeneratorConfig, IdentityVocab, build_condition, build_default_pcfg,
    build_sense_form_map, build_sense_inventory, corpus_hsw, default_conditions,
    generate_sense_corpus, render_documents, render_documents_with_senses,
    write_metadata, write_parquet_shards, write_probe_jsonl, write_vocab,
    _sense_probabilities,
)

POS_FRACS = {"N": 0.45, "V": 0.30, "DET": 0.10, "P": 0.15}


def _class_sizes(k):
    sizes = {c: max(1, round(k * f)) for c, f in POS_FRACS.items()}
    sizes["N"] += k - sum(sizes.values())
    return sizes


@pytest.fixture(scope="module")
def small_run():
    """A tiny but non-trivial run reused by several tests."""
    k = 64
    class_sizes = _class_sizes(k)
    cfg = GeneratorConfig(class_sizes=class_sizes, num_tokens=40_000, seed=0,
                          min_len=6, max_len=30, tolerance=0.05, hm_max_tokens=20_000)
    pcfg = build_default_pcfg()
    inventory = build_sense_inventory(class_sizes)
    sense_docs = generate_sense_corpus(pcfg, inventory, num_tokens=cfg.num_tokens,
                                       max_depth=cfg.max_depth, min_len=cfg.min_len,
                                       max_len=cfg.max_len, seed=cfg.seed)
    sense_prob = _sense_probabilities(sense_docs, inventory.num_senses)
    return cfg, pcfg, inventory, sense_docs, sense_prob


def test_sense_inventory_sizes():
    class_sizes = _class_sizes(128)
    inv = build_sense_inventory(class_sizes)
    assert inv.num_senses == 128
    assert sum(len(v) for v in inv.senses_by_class.values()) == 128


def test_sense_stream_lengths_within_bounds(small_run):
    cfg, pcfg, inventory, sense_docs, _ = small_run
    assert len(sense_docs) > 10
    assert all(cfg.min_len <= len(d) <= cfg.max_len for d in sense_docs)


def test_monosemous_has_zero_hsw(small_run):
    cfg, pcfg, inventory, sense_docs, sense_prob = small_run
    smap = build_sense_form_map(inventory, sense_prob, target_hsw=0.0, overlap="none")
    measured, per_form = corpus_hsw(smap.form_to_senses, sense_prob)
    assert measured == 0.0
    assert smap.vocab_size == inventory.num_senses  # bijective
    assert all(h == 0.0 for h in per_form.values())


@pytest.mark.parametrize("target,overlap", [(0.5, "none"), (0.5, "partial"), (1.0, "partial")])
def test_polysemous_hits_target_within_tolerance(small_run, target, overlap):
    cfg, pcfg, inventory, sense_docs, sense_prob = small_run
    smap = build_sense_form_map(inventory, sense_prob, target_hsw=target,
                                overlap=overlap, tolerance=cfg.tolerance)
    measured, _ = corpus_hsw(smap.form_to_senses, sense_prob)
    assert abs(measured - target) <= cfg.tolerance, f"measured {measured} vs target {target}"


def test_vocab_size_constant_across_conditions(small_run):
    cfg, pcfg, inventory, sense_docs, sense_prob = small_run
    sizes = set()
    for cond in default_conditions():
        overlap = "none" if cond.overlap == "mono" else cond.overlap
        smap = build_sense_form_map(inventory, sense_prob, target_hsw=cond.target_hsw,
                                    overlap=overlap, tolerance=cfg.tolerance)
        sizes.add(smap.vocab_size)
    assert sizes == {inventory.num_senses}, f"|V| not constant: {sizes}"


def test_homonymy_merges_cross_class_overlap_merges_within_class(small_run):
    cfg, pcfg, inventory, sense_docs, sense_prob = small_run
    # homonymy: every polysemous form mixes >1 POS class
    smap_h = build_sense_form_map(inventory, sense_prob, target_hsw=0.5, overlap="none")
    for w, senses in smap_h.form_to_senses.items():
        if len(senses) > 1:
            classes = {inventory.class_of[s] for s in senses}
            assert len(classes) > 1, f"homonymy form {w} merged within one class {classes}"
    # overlapping: every polysemous form stays within one POS class
    smap_o = build_sense_form_map(inventory, sense_prob, target_hsw=0.5, overlap="partial")
    found_poly = False
    for w, senses in smap_o.form_to_senses.items():
        if len(senses) > 1:
            found_poly = True
            classes = {inventory.class_of[s] for s in senses}
            assert len(classes) == 1, f"overlapping form {w} merged across classes {classes}"
    assert found_poly


def test_form_token_roundtrip_identity(small_run):
    cfg, pcfg, inventory, sense_docs, sense_prob = small_run
    smap = build_sense_form_map(inventory, sense_prob, target_hsw=0.5, overlap="partial")
    documents = render_documents(sense_docs, smap, seed=cfg.seed)
    vocab = IdentityVocab(smap.vocab)
    for doc in documents[:200]:
        assert vocab.decode(vocab.encode(doc)) == doc
        # every emitted symbol is in the vocab (1:1 form<->token)
        for tok in doc.split():
            assert tok in vocab.stoi


def test_parallel_generation_is_deterministic_and_honors_budget():
    pcfg = build_default_pcfg()
    inv = build_sense_inventory(_class_sizes(64))
    kw = dict(num_tokens=30_000, seed=0, min_len=6, max_len=30)
    a = generate_sense_corpus(pcfg, inv, num_workers=4, **kw)
    b = generate_sense_corpus(pcfg, inv, num_workers=4, **kw)
    assert a == b, "parallel generation must be deterministic for fixed (seed, num_workers)"
    assert sum(len(d) for d in a) >= kw["num_tokens"]  # budget honored
    # a different worker count yields a different (but still valid) stream
    c = generate_sense_corpus(pcfg, inv, num_workers=2, **kw)
    assert a != c
    # all docs respect the length filter regardless of worker count
    assert all(6 <= len(d) <= 30 for d in a)


def test_generation_is_deterministic(small_run):
    cfg, pcfg, inventory, sense_docs, sense_prob = small_run
    a = render_documents(sense_docs, build_sense_form_map(inventory, sense_prob, target_hsw=0.5,
                         overlap="partial", seed=cfg.seed), seed=cfg.seed)
    b = render_documents(sense_docs, build_sense_form_map(inventory, sense_prob, target_hsw=0.5,
                         overlap="partial", seed=cfg.seed), seed=cfg.seed)
    assert a == b


def test_probe_set_export_aligns_forms_and_senses(tmp_path, small_run):
    cfg, pcfg, inventory, sense_docs, sense_prob = small_run
    smap = build_sense_form_map(inventory, sense_prob, target_hsw=1.5, overlap="none")
    records = render_documents_with_senses(sense_docs[:20], smap, seed=cfg.seed)
    assert len(records) == 20
    for rec in records:
        assert len(rec["forms"]) == len(rec["senses"]) > 0
        # every recorded sense renders to its recorded form under the map (consistency)
        for form, sense in zip(rec["forms"], rec["senses"]):
            assert form in smap.sense_to_forms[sense]
            assert sense in smap.form_to_senses[form]
    # jsonl writer round-trips
    path = write_probe_jsonl(records, str(tmp_path))
    loaded = [json.loads(l) for l in open(path)]
    assert loaded == records


def test_probe_set_senses_independent_of_condition(small_run):
    """The held-out sense stream is shared; only forms change across conditions (so a probe
    decodes the SAME latent senses, with condition-specific surface forms)."""
    cfg, pcfg, inventory, sense_docs, sense_prob = small_run
    mono = build_sense_form_map(inventory, sense_prob, target_hsw=0.0, overlap="none")
    homo = build_sense_form_map(inventory, sense_prob, target_hsw=1.5, overlap="none")
    rec_mono = render_documents_with_senses(sense_docs[:10], mono, seed=cfg.seed)
    rec_homo = render_documents_with_senses(sense_docs[:10], homo, seed=cfg.seed)
    assert [r["senses"] for r in rec_mono] == [r["senses"] for r in rec_homo]


def test_build_condition_and_write(tmp_path, small_run):
    cfg, pcfg, inventory, sense_docs, sense_prob = small_run
    cond = default_conditions()[2]  # hsw0p5_overlap
    documents, smap, metadata = build_condition(cfg, pcfg, inventory, sense_docs, sense_prob, cond)

    out_dir = str(tmp_path / cond.slug)
    paths = write_parquet_shards(documents, out_dir, shard_chars=200_000)
    vpath = write_vocab(smap, out_dir)
    mpath = write_metadata(metadata, out_dir)

    assert len(paths) >= 2  # at least one train + one val shard
    assert all(p.endswith(".parquet") and os.path.exists(p) for p in paths)
    assert os.path.exists(vpath) and os.path.exists(mpath)

    # metadata sidecar carries the required fields
    with open(mpath) as f:
        md = json.load(f)
    assert md["vocab_size"] == smap.vocab_size
    assert md["h_s_given_w"]["within_tolerance"] is True
    assert set(md["h_m_bits"]["forms_total"].keys()) == {str(m) for m in cfg.hm_ms}
    assert md["gzip"]["bytes"] > 0

    # parquet round-trips back to the same documents under the identity vocab
    import pyarrow.parquet as pq
    vocab = IdentityVocab(smap.vocab)
    read_docs = []
    for p in paths:
        read_docs.extend(pq.read_table(p).column("text").to_pylist())
    assert read_docs == documents
    for d in read_docs[:50]:
        assert vocab.decode(vocab.encode(d)) == d
