"""Deterministic seeding helpers built on numpy.random.SeedSequence."""
from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager

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


def idle_core_order(sample_s: float = 0.2) -> list[int]:
    """Allowed cores ordered from least to most busy over a short /proc/stat sample."""
    import os
    import time

    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))

    def snap() -> dict[int, tuple[int, int]]:
        out = {}
        try:
            with open("/proc/stat") as f:
                for line in f:
                    if line.startswith("cpu") and line[3].isdigit():
                        p = line.split()
                        v = [int(x) for x in p[1:]]
                        out[int(p[0][3:])] = (v[3] + v[4], sum(v))      # idle + iowait, total
        except OSError:
            pass
        return out

    a = snap()
    time.sleep(sample_s)
    b = snap()

    def busy(c: int) -> float:
        if c not in a or c not in b:
            return 0.0
        dt = b[c][1] - a[c][1]
        return 1.0 - (b[c][0] - a[c][0]) / dt if dt > 0 else 0.0

    return sorted(allowed, key=lambda c: (round(busy(c), 1), c))


@contextmanager
def core_order_env():
    """Around a ``Pool(...)`` call: publish the currently idle-first core order to the
    spawned workers, so a pool started while another one runs pins to the idle cores
    (``pin_worker``) instead of piling onto cores 0, 1, ..."""
    import os

    saved = os.environ.get("ISCO_CORE_ORDER")
    os.environ["ISCO_CORE_ORDER"] = ",".join(str(c) for c in idle_core_order())
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("ISCO_CORE_ORDER", None)
        else:
            os.environ["ISCO_CORE_ORDER"] = saved


def cores_for(ident: int, cores_per_worker: int, order: list[int]) -> list[int]:
    """The block of ``order`` that worker number ``ident`` (1-based) pins to."""
    n = len(order)
    start = ((ident - 1) * cores_per_worker) % n
    return sorted({order[(start + i) % n] for i in range(max(1, cores_per_worker))})


def pin_worker(cores_per_worker: int = 1) -> list[int] | None:
    """Pin the current multiprocessing worker to its own block of CPU cores.

    TBB (Embree inside open3d) and OpenMP size their thread pools from the process
    affinity mask, so pinning is the only reliable way to stop every worker from running
    one thread per core (observed: 400+ runnable threads on 24 cores). Blocks are taken
    from the order the parent published with ``core_order_env`` (idle cores first), else
    from core 0 upwards."""
    import multiprocessing as mp
    import os

    ident = mp.current_process()._identity
    if not ident or not hasattr(os, "sched_setaffinity"):
        return None
    env = os.environ.get("ISCO_CORE_ORDER")
    order = [int(c) for c in env.split(",")] if env else list(range(os.cpu_count() or 1))
    cores = cores_for(ident[0], cores_per_worker, order)
    try:
        os.sched_setaffinity(0, cores)
    except OSError:
        return None
    return cores
