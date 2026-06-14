"""
Tests for nanochat.common._ensure_worktree_artifacts_symlink — the guard that
recreates the per-worktree ``artifacts`` symlink when an Arkhip git worktree was
created without it.

Training/eval jobs export NANOCHAT_BASE_DIR=<workdir>/artifacts. In the main
checkout that ``artifacts`` is a symlink to the shared artifacts dir (tokenizer +
base data). New worktrees are sometimes created without it; without this guard
get_base_dir() would os.makedirs an empty real dir and training fails fast at
get_tokenizer(). The guard must repoint such worktrees at the main checkout's
artifacts, while leaving non-worktree / non-artifacts / populated dirs alone.

Run: python -m pytest tests/test_worktree_artifacts_symlink.py -v
"""

import os
import subprocess

import pytest

from nanochat.common import _ensure_worktree_artifacts_symlink


def _git(cwd, *args):
    subprocess.run(["git", "-C", cwd, *args], check=True,
                   capture_output=True, text=True)


@pytest.fixture
def main_and_worktree(tmp_path):
    """A real git main checkout (with an artifacts dir) plus a linked worktree."""
    main = tmp_path / "main"
    main.mkdir()
    _git(str(main), "init", "-q")
    _git(str(main), "config", "user.email", "t@t.t")
    _git(str(main), "config", "user.name", "t")
    (main / "f.txt").write_text("hi")
    _git(str(main), "add", "f.txt")
    _git(str(main), "commit", "-q", "-m", "init")

    # The shared artifacts dir lives in the main checkout.
    main_artifacts = main / "artifacts"
    main_artifacts.mkdir()
    (main_artifacts / "tokenizer.pkl").write_text("x")

    wt = tmp_path / "wt-4"
    _git(str(main), "worktree", "add", "-q", str(wt))
    return main, wt, main_artifacts


def test_missing_symlink_is_created(main_and_worktree):
    main, wt, main_artifacts = main_and_worktree
    base = wt / "artifacts"
    assert not base.exists()

    _ensure_worktree_artifacts_symlink(str(base))

    assert os.path.islink(str(base))
    assert os.path.realpath(str(base)) == os.path.realpath(str(main_artifacts))
    # The shared tokenizer is now reachable through the worktree base dir.
    assert (base / "tokenizer.pkl").read_text() == "x"


def test_empty_real_dir_is_replaced(main_and_worktree):
    main, wt, main_artifacts = main_and_worktree
    base = wt / "artifacts"
    base.mkdir()  # the buggy empty real directory
    assert base.is_dir() and not os.path.islink(str(base))

    _ensure_worktree_artifacts_symlink(str(base))

    assert os.path.islink(str(base))
    assert os.path.realpath(str(base)) == os.path.realpath(str(main_artifacts))


def test_non_empty_real_dir_is_left_alone(main_and_worktree):
    main, wt, _ = main_and_worktree
    base = wt / "artifacts"
    base.mkdir()
    (base / "local.pkl").write_text("keep")

    _ensure_worktree_artifacts_symlink(str(base))

    assert not os.path.islink(str(base))
    assert (base / "local.pkl").read_text() == "keep"


def test_existing_symlink_is_respected(main_and_worktree):
    main, wt, _ = main_and_worktree
    base = wt / "artifacts"
    other = wt / "somewhere_else"
    other.mkdir()
    os.symlink(str(other), str(base))

    _ensure_worktree_artifacts_symlink(str(base))

    assert os.path.realpath(str(base)) == os.path.realpath(str(other))


def test_main_checkout_is_not_touched(main_and_worktree):
    """In the main checkout (not a linked worktree), do nothing even if artifacts
    is missing — only the worktree convention is auto-repaired."""
    main, _, _ = main_and_worktree
    # The main checkout's own artifacts is a real dir with data -> must be left alone.
    base = main / "artifacts"
    _ensure_worktree_artifacts_symlink(str(base))
    assert not os.path.islink(str(base))
    assert (base / "tokenizer.pkl").read_text() == "x"


def test_non_artifacts_basename_is_ignored(tmp_path):
    base = tmp_path / ".cache" / "nanochat"
    base.mkdir(parents=True)
    _ensure_worktree_artifacts_symlink(str(base))
    assert not os.path.islink(str(base))


def test_non_git_dir_is_ignored(tmp_path):
    base = tmp_path / "artifacts"
    _ensure_worktree_artifacts_symlink(str(base))
    # No git repo above it -> nothing created, no crash.
    assert not os.path.exists(str(base))
