"""
TensorBoard logger with the same interface as wandb / DummyWandb.

Drop-in replacement so training scripts can switch with minimal changes:

    from nanochat.tb_logger import TBLogger
    logger = TBLogger(log_dir="runs/my_run", config={...})
    logger.log({"train/loss": 0.5, "step": 42})
    logger.finish()
"""

from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    def __init__(self, log_dir: str | Path, config: dict[str, Any] | None = None):
        self.writer = SummaryWriter(log_dir=str(log_dir))
        # Store config as text for reference
        if config:
            self.writer.add_text("config", _dict_to_markdown(config))

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        # Extract step from data if not provided explicitly
        if step is None:
            step = data.get("step", None)
        for key, value in data.items():
            if key == "step":
                continue
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, global_step=step)
            # Skip non-scalar values (dicts, strings, etc.)

    def finish(self) -> None:
        self.writer.flush()
        self.writer.close()


def _dict_to_markdown(d: dict) -> str:
    lines = []
    for k, v in d.items():
        lines.append(f"- **{k}**: `{v}`")
    return "\n".join(lines)
