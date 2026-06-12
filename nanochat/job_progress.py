"""
Job progress tracker that writes a JSON status file periodically.

Adapted from llm_embeds_optim's JobProgressCallback for nanochat's custom training loop.
Usage:

    progress = JobProgress(output_file=Path("run/progress.json"))
    progress.on_train_begin(max_steps=1000)
    # in training loop:
    progress.on_log(step=42, metrics={"train/loss": 1.23})
    # at end:
    progress.on_train_end()
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _format_duration(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass
class JobProgress:
    output_file: Path
    save_interval_seconds: int = 600
    start_time: float = field(default_factory=time.time)
    max_steps: int = field(default=0, init=False)
    _last_persist_time: float = field(default=0.0, init=False)
    _latest_payload: dict[str, Any] | None = field(default=None, init=False)

    def _estimate_remaining_seconds(self, step: int) -> float | None:
        if step <= 0 or self.max_steps <= 0:
            return None
        elapsed = max(0.0, time.time() - self.start_time)
        seconds_per_step = elapsed / step
        remaining_steps = max(0, self.max_steps - step)
        return seconds_per_step * remaining_steps

    def _persist_payload(self, payload: dict[str, Any]) -> None:
        try:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = self.output_file.with_suffix(self.output_file.suffix + ".tmp")
            tmp_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
            os.replace(tmp_file, self.output_file)
        except (OSError, FileNotFoundError):
            pass

    def _build_payload(self, step: int, metrics: dict[str, Any]) -> dict[str, Any]:
        eta_seconds = self._estimate_remaining_seconds(step)
        return {
            "timestamp": int(time.time()),
            "global_step": step,
            "max_steps": self.max_steps,
            "progress_ratio": (step / self.max_steps) if self.max_steps > 0 else None,
            "elapsed_seconds": max(0.0, time.time() - self.start_time),
            "estimated_remaining_seconds": eta_seconds,
            "estimated_remaining_hms": _format_duration(eta_seconds),
            "metrics": metrics,
        }

    def _print_payload(self, payload: dict[str, Any]) -> None:
        eta_text = payload.get("estimated_remaining_hms") or "unknown"
        progress = payload.get("progress_ratio")
        progress_text = f"{progress * 100:.2f}%" if isinstance(progress, float) else "unknown"
        print(
            f"[job-progress] step={payload['global_step']}/{payload['max_steps']} "
            f"progress={progress_text} eta={eta_text} metrics={json.dumps(payload['metrics'], ensure_ascii=True)}",
            flush=True,
        )

    def on_train_begin(self, max_steps: int) -> None:
        self.start_time = time.time()
        self.max_steps = max_steps
        self._last_persist_time = 0.0
        self._latest_payload = None

    def on_log(self, step: int, metrics: dict[str, Any]) -> None:
        if not metrics:
            return

        eta_seconds = self._estimate_remaining_seconds(step)
        metrics = dict(metrics)
        metrics["estimated_remaining_seconds"] = eta_seconds
        metrics["estimated_remaining_hms"] = _format_duration(eta_seconds)

        payload = self._build_payload(step, metrics)
        self._latest_payload = payload
        self._print_payload(payload)

        now = time.time()
        if now - self._last_persist_time >= self.save_interval_seconds:
            self._persist_payload(payload)
            self._last_persist_time = now

    def on_train_end(self) -> None:
        if self._latest_payload is not None:
            self._persist_payload(self._latest_payload)
