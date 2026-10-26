"""Optional experiment tracking: logs to Weights & Biases when ``ISCO_WANDB=1`` and the
``wandb`` package is installed; otherwise a no-op (JSONL results are always written)."""
from __future__ import annotations

import os


class Tracker:
    def __init__(self, project: str = "isco", run_name: str | None = None, config: dict | None = None):
        self.run = None
        if os.environ.get("ISCO_WANDB") == "1":
            try:
                import wandb

                self.run = wandb.init(project=project, name=run_name, config=config or {}, reinit=True)
            except Exception:  # pragma: no cover - optional dependency
                self.run = None

    def log(self, metrics: dict, step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
