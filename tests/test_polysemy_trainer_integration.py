"""
Integration tests for component 2 (trainer/data wiring) of the polysemy experiment.

- The dataloader, pointed at a generated condition dir via data_dir + the identity
  tokenizer, yields BOS-aligned batches of in-vocab token ids (no BPE, no climbmix).
- list_parquet_files with an explicit data_dir never falls back to the legacy path.
- run_training.polysemy_context_experiments() emits the expected condition×L grid with
  unique tags and the fixed knobs the experiment requires (identity tok, full attention).

Run: PYTHONPATH=. pytest -q tests/test_polysemy_trainer_integration.py
"""

import importlib.util
import re

import pytest

# Importing nanochat.polysemy first installs the broken-pandas blocker some envs need
# before pyarrow touches pandas (see nanochat/polysemy.py header).
import nanochat.polysemy as P
from nanochat.identity_tokenizer import get_identity_tokenizer


@pytest.fixture(scope="module")
def condition_dir(tmp_path_factory):
    """Generate a tiny single-condition corpus on disk (parquet shards + vocab.json)."""
    out = tmp_path_factory.mktemp("cond")
    class_sizes = {"N": 8, "V": 5, "DET": 2, "P": 3}
    pcfg = P.build_default_pcfg()
    inv = P.build_sense_inventory(class_sizes)
    sense_docs = P.generate_sense_corpus(pcfg, inv, num_tokens=8000, min_len=5, max_len=20, seed=0)
    sp = P._sense_probabilities(sense_docs, inv.num_senses)
    smap = P.build_sense_form_map(inv, sp, target_hsw=0.5, overlap="partial")
    docs = P.render_documents(sense_docs, smap, seed=0)
    P.write_parquet_shards(docs, str(out), shard_chars=40_000)
    P.write_vocab(smap, str(out))
    return str(out)


def test_list_parquet_files_explicit_dir(condition_dir):
    from nanochat.dataset import list_parquet_files
    paths = list_parquet_files(data_dir=condition_dir)
    assert len(paths) >= 2  # at least one train + one val shard
    assert all(p.endswith(".parquet") for p in paths)


def test_list_parquet_files_missing_explicit_dir_raises(tmp_path):
    from nanochat.dataset import list_parquet_files
    with pytest.raises(AssertionError):
        list_parquet_files(data_dir=str(tmp_path / "does_not_exist"))


def test_dataloader_reads_condition_dir(condition_dir):
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    tok = get_identity_tokenizer(condition_dir)
    B, T = 4, 8
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, B=B, T=T, split="train", device="cpu", data_dir=condition_dir, buffer_size=64)
    x, y = next(loader)
    assert tuple(x.shape) == (B, T) and tuple(y.shape) == (B, T)
    # every row starts with BOS; all ids are valid (in [0, vocab_size))
    assert (x[:, 0] == tok.get_bos_token_id()).all()
    assert int(x.min()) >= 0 and int(x.max()) < tok.get_vocab_size()
    # targets are inputs shifted by one (bestfit packs token t+1 as the target of token t)
    assert (y[:, :-1] == x[:, 1:]).all()
    # val split also loads
    vloader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, B=B, T=T, split="val", device="cpu", data_dir=condition_dir, buffer_size=64)
    xv, _ = next(vloader)
    assert tuple(xv.shape) == (B, T)


@pytest.fixture(scope="module")
def run_training():
    spec = importlib.util.spec_from_file_location("run_training", "scripts/jobs/run_training.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_polysemy_configs_grid(run_training):
    cfgs = run_training.polysemy_context_experiments()
    # 5 default conditions × 4 context lengths
    assert len(cfgs) == len(P.default_conditions()) * len(run_training.POLYSEMY_SEQ_LENS)
    tags = [c["model_tag"] for c in cfgs]
    assert len(set(tags)) == len(tags), "model tags must be unique"
    slugs = {c.slug for c in P.default_conditions()}
    for c in cfgs:
        a = c["args"]
        assert "--tokenizer identity" in a
        assert "--window-pattern L" in a            # full attention: required for the L sweep
        assert "--core-metric-every -1" in a        # CORE meaningless for synthetic vocab
        assert "--total-batch-size 32768" in a      # global batch held constant across L
        assert "--eval-every 2500" in a
        assert "--eval-tokens" not in a             # left at base default (not overridden)
        L = int(re.search(r"--max-seq-len (\d+)", a).group(1))
        assert L in run_training.POLYSEMY_SEQ_LENS
        # per-L device batch gives grad-accum == 1 at the fixed 32768-token global batch on 4 GPUs
        dbs = int(re.search(r"--device-batch-size (\d+)", a).group(1))
        assert dbs * L * 4 == 32768, f"grad-accum != 1 for L={L} (dbs={dbs})"
        slug = re.match(r"^poly_(.+)_L\d+$", c["model_tag"]).group(1)
        assert slug in slugs
        assert f"--data-dir {run_training.POLYSEMY_DATA_ROOT}/{slug}" in a
        assert c["experiment_slug"] == "polysemy-context"


def test_polysemy_configs_custom_data_root(run_training):
    cfgs = run_training.polysemy_context_experiments(data_root="/some/root")
    assert all("--data-dir /some/root/" in c["args"] for c in cfgs)
