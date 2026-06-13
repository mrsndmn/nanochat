"""
Tests for nanochat.job_progress.build_job_progress — the factory that decides
where (and whether) a training run writes its Arkhip /status progress file.

Arkhip's /status reads <RL_RUN_JOBS_METRICS_DIR>/<job_name>.json, where job_name
is the MLSpace job name exported as ARKHIP_JOB_NAME by prepare_torchrun.sh. With
no job-name env var (local/dummy runs) no progress file must be created.

Run: python -m pytest tests/test_job_progress.py -v
"""

import json

import pytest

from nanochat.job_progress import (
    DEFAULT_METRICS_DIR,
    JOB_NAME_ENV,
    METRICS_DIR_ENV,
    JobProgress,
    build_job_progress,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(JOB_NAME_ENV, raising=False)
    monkeypatch.delenv(METRICS_DIR_ENV, raising=False)


def test_no_job_name_returns_none(monkeypatch):
    # Absent ARKHIP_JOB_NAME (local run) -> no progress file at all.
    assert build_job_progress() is None


def test_blank_job_name_returns_none(monkeypatch):
    monkeypatch.setenv(JOB_NAME_ENV, "   ")
    assert build_job_progress() is None


def test_uses_metrics_dir_override_and_job_name(monkeypatch, tmp_path):
    monkeypatch.setenv(JOB_NAME_ENV, "lm-mpi-job-abc123")
    monkeypatch.setenv(METRICS_DIR_ENV, str(tmp_path))
    progress = build_job_progress()
    assert isinstance(progress, JobProgress)
    assert progress.output_file == tmp_path / "lm-mpi-job-abc123.json"


def test_defaults_to_arkhip_metrics_dir(monkeypatch):
    monkeypatch.setenv(JOB_NAME_ENV, "lm-mpi-job-abc123")
    progress = build_job_progress()
    assert progress is not None
    assert str(progress.output_file) == f"{DEFAULT_METRICS_DIR}/lm-mpi-job-abc123.json"


def test_eta_none_at_step_zero_then_populated(monkeypatch, tmp_path):
    monkeypatch.setenv(JOB_NAME_ENV, "lm-mpi-job-abc123")
    monkeypatch.setenv(METRICS_DIR_ENV, str(tmp_path))
    progress = build_job_progress()
    progress.save_interval_seconds = 0  # persist on every on_log
    progress.on_train_begin(max_steps=100)

    progress.on_log(step=0, metrics={"train/loss": 5.0})
    payload = json.loads(progress.output_file.read_text())
    assert payload["global_step"] == 0
    assert payload["estimated_remaining_hms"] is None

    progress.on_log(step=10, metrics={"train/loss": 4.0})
    payload = json.loads(progress.output_file.read_text())
    assert payload["global_step"] == 10
    assert payload["progress_ratio"] == pytest.approx(0.1)
    assert isinstance(payload["estimated_remaining_hms"], str)
