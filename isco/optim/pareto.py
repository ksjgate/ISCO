"""Pareto front extraction and hypervolume (minimisation)."""
from __future__ import annotations

import numpy as np


def nondominated_mask(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    n = len(F)
    mask = np.ones(n, bool)
    for i in range(n):
        if not mask[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dom.any():
            mask[i] = False
    return mask


def hypervolume(F: np.ndarray, ref: list[float] | np.ndarray | None = None) -> float:
    F = np.asarray(F, dtype=float)
    if len(F) == 0:
        return 0.0
    ref = np.asarray(ref if ref is not None else [1.1] * F.shape[1], dtype=float)
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    from pymoo.indicators.hv import HV

    return float(HV(ref_point=ref)(F))


def hv_curve(F_seq: np.ndarray, ref=None) -> np.ndarray:
    """Hypervolume of the first k points for k = 1..len(F_seq) (search efficiency, E8)."""
    return np.array([hypervolume(F_seq[: k + 1], ref) for k in range(len(F_seq))])
