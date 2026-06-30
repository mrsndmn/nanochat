"""
Tests for the train-shard cap (--num-train-shards) added to the bestfit dataloader.

`_document_batches` must select only the first N train shards when num_train_shards is set
(pinning the epoch / data budget), always keep the val split on the last shard, and fall back
to using every train shard when the cap is disabled (-1). A cap larger than what is available
must raise a clear error rather than silently using fewer shards.

These tests stub out list_parquet_files and parquet reading so no real dataset is required.

Run: python -m pytest tests/test_dataloader_shard_budget.py -v
"""

import pytest

import nanochat.dataloader as dl


@pytest.fixture
def fake_shards(monkeypatch):
    """Patch the file listing + DDP info so _document_batches runs without real data."""
    paths = [f"/fake/shard_{i:05d}.parquet" for i in range(10)]  # 9 train + 1 val
    monkeypatch.setattr(dl, "list_parquet_files", lambda data_dir=None, warn_on_legacy=False: list(paths))
    # single-process: rank 0 of world size 1
    monkeypatch.setattr(dl, "get_dist_info", lambda: (False, 0, 0, 1))
    return paths


class _FakeRowGroup:
    def column(self, name):
        return self

    def to_pylist(self):
        return ["doc"]  # one tiny document per row group


def _opened_paths(monkeypatch, split, num_train_shards):
    """Drive _document_batches over exactly one epoch and record which shards it opens."""
    opened = []

    class FakeParquetFile:
        num_row_groups = 1  # one row group -> exactly one yield per shard

        def __init__(self, filepath):
            opened.append(filepath)

        def read_row_group(self, idx):
            return _FakeRowGroup()

    monkeypatch.setattr(dl.pq, "ParquetFile", FakeParquetFile)

    gen = dl._document_batches(split, None, 128, num_train_shards=num_train_shards)
    # Stop as soon as the epoch wraps (epoch == 2): one full pass observed.
    for _ in range(1000):
        _, (_, _, epoch) = next(gen)
        if epoch >= 2:
            break
    return opened


def test_cap_selects_first_n_train_shards(fake_shards, monkeypatch):
    opened = _opened_paths(monkeypatch, "train", num_train_shards=3)
    first_pass = opened[:3]
    assert first_pass == fake_shards[:3]
    # shard index 3..8 (still train) must not be touched under a cap of 3
    assert fake_shards[5] not in opened


def test_disabled_cap_uses_all_train_shards(fake_shards, monkeypatch):
    opened = _opened_paths(monkeypatch, "train", num_train_shards=-1)
    # all 9 train shards (everything except the last/val shard) are used
    assert set(opened[:9]) == set(fake_shards[:-1])
    assert fake_shards[-1] not in opened  # val shard never used by train split


def test_val_split_ignores_cap_and_uses_last_shard(fake_shards, monkeypatch):
    opened = _opened_paths(monkeypatch, "val", num_train_shards=3)
    assert set(opened) == {fake_shards[-1]}


def test_cap_exceeding_available_raises(fake_shards, monkeypatch):
    # 9 train shards available; asking for 50 must fail loudly.
    with pytest.raises(AssertionError):
        _opened_paths(monkeypatch, "train", num_train_shards=50)
