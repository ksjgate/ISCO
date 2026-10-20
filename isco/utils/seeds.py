"""Deterministic seeding helpers built on numpy.random.SeedSequence."""
from __future__ import annotations

import hashlib
import random

import numpy as np


def stable_int(*parts: object) -> int:
    """A process-independent 32-bit integer derived from ``parts`` (unlike hash())."""
    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "little")


def rng_for(*parts: object) -> np.random.Generator:
    """A numpy Generator deterministically keyed by arbitrary labels."""
    return np.random.default_rng(stable_int(*parts))


def spawn(seed: int, n: int) -> list[np.random.Generator]:
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:  # torch is optional for most modules
        import torch

        torch.manual_seed(seed)
    except Exception:  # pragma: no cover
        pass
