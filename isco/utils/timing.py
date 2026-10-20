"""Lightweight wall-clock timing."""
from __future__ import annotations

import time
from contextlib import contextmanager
from collections import defaultdict


class Timer:
    """Accumulates named wall-clock sections: ``with timer("fpfh"): ...``."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] += time.perf_counter() - t0
            self.counts[name] += 1

    def summary(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in sorted(self.totals.items())}
