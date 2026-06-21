"""
Tests for nanochat.common._ensure_worktree_artifacts_symlink — the guard that points a
conventional ``artifacts`` base dir at the shared artifacts store (SHARED_ARTIFACTS_DIR).

Training/eval jobs and direct get_base_dir callers may resolve their base dir to
``<checkout>/artifacts``. Every checkout and worktree should have that ``artifacts`` entry
be a symlink to the single absolute shared store on the workspace volume, so all runs
share one tokenizer + base data + checkpoints and the link resolves inside job containers
(which only mount /workspace-SR004.nfs2, not /mnt/virtual_*). New worktrees are often
created without the symlink; without this guard get_base_dir() would os.makedirs an empty
real dir and training fails fast at get_tokenizer().

The guard (re)creates the symlink for an ``artifacts`` base dir that is missing, an empty
real dir, or a stale/broken symlink, while never clobbering a populated real dir.

Run: python -m pytest tests/test_worktree_artifacts_symlink.py -v
"""

import os

import pytest

import nanochat.common as common
from nanochat.common import _ensure_worktree_artifacts_symlink


@pytest.fixture
def shared(tmp_path, monkeypatch):
    """A stand-in shared artifacts store with the module constant pointed at it."""
    store = tmp_path / "shared" / "nanochat-artifacts"
    store.mkdir(parents=True)
    (store / "tokenizer.pkl").write_text("x")
    monkeypatch.setattr(common, "SHARED_ARTIFACTS_DIR", str(store))
    return store


def test_missing_symlink_is_created(tmp_path, shared):
    base = tmp_path / "wt" / "artifacts"
    base.parent.mkdir()
    assert not os.path.lexists(str(base))

    _ensure_worktree_artifacts_symlink(str(base))

    assert os.path.islink(str(base))
    assert os.path.realpath(str(base)) == os.path.realpath(str(shared))
    # The shared tokenizer is now reachable through the worktree base dir.
    assert (base / "tokenizer.pkl").read_text() == "x"


def test_empty_real_dir_is_replaced(tmp_path, shared):
    base = tmp_path / "wt" / "artifacts"
    base.mkdir(parents=True)  # the buggy empty real directory
    assert base.is_dir() and not os.path.islink(str(base))

    _ensure_worktree_artifacts_symlink(str(base))

    assert os.path.islink(str(base))
    assert os.path.realpath(str(base)) == os.path.realpath(str(shared))


def test_non_empty_real_dir_is_left_alone(tmp_path, shared):
    base = tmp_path / "wt" / "artifacts"
    base.mkdir(parents=True)
    (base / "local.pkl").write_text("keep")

    _ensure_worktree_artifacts_symlink(str(base))

    assert not os.path.islink(str(base))
    assert (base / "local.pkl").read_text() == "keep"


def test_stale_symlink_is_repointed(tmp_path, shared):
    """A symlink to the old (e.g. /mnt/...) location is repointed at the shared store."""
    base = tmp_path / "wt" / "artifacts"
    base.parent.mkdir()
    old = tmp_path / "old-artifacts"
    old.mkdir()
    os.symlink(str(old), str(base))

    _ensure_worktree_artifacts_symlink(str(base))

    assert os.path.islink(str(base))
    assert os.path.realpath(str(base)) == os.path.realpath(str(shared))


def test_broken_symlink_is_repointed(tmp_path, shared):
    base = tmp_path / "wt" / "artifacts"
    base.parent.mkdir()
    os.symlink(str(tmp_path / "does-not-exist"), str(base))
    assert os.path.islink(str(base)) and not os.path.exists(str(base))

    _ensure_worktree_artifacts_symlink(str(base))

    assert os.path.realpath(str(base)) == os.path.realpath(str(shared))


def test_already_correct_symlink_is_idempotent(tmp_path, shared):
    base = tmp_path / "wt" / "artifacts"
    base.parent.mkdir()
    os.symlink(str(shared), str(base))

    _ensure_worktree_artifacts_symlink(str(base))

    assert os.path.islink(str(base))
    assert os.path.realpath(str(base)) == os.path.realpath(str(shared))


def test_non_artifacts_basename_is_ignored(tmp_path, shared):
    # The default base dir (SHARED_ARTIFACTS_DIR, basename != "artifacts") is never linked.
    base = tmp_path / ".cache" / "nanochat"
    base.mkdir(parents=True)
    _ensure_worktree_artifacts_symlink(str(base))
    assert not os.path.islink(str(base))


def test_shared_store_itself_is_not_linked(tmp_path, monkeypatch):
    """If the shared store's own basename is 'artifacts', do not self-link it."""
    store = tmp_path / "artifacts"
    store.mkdir()
    monkeypatch.setattr(common, "SHARED_ARTIFACTS_DIR", str(store))

    _ensure_worktree_artifacts_symlink(str(store))

    assert not os.path.islink(str(store))
