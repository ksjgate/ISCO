"""Part descriptors for the experiment on part shape features and optimal inspection
configurations (pre-registration §12, 2026-10-26): geometry and slot-visibility features,
all computed before any detection evaluation of that experiment.

Visibility comes from the same ray casts as stage 1: 2,000 area-weighted samples of the
nominal surface x 96 slots (``nominal_vis``) and 200 samples per defect instance x 96 slots
(``vis``). A sample is *reachable* if at least one slot sees it.

* ``height_ratio``       smallest / largest extent (flatness)
* ``reachable``          share of the nominal surface seen from at least one slot
* ``best_1view``         best single slot: share of the reachable surface it sees
* ``rotation_gain``      best single slot plus a 90-degree turn of the part, minus ``best_1view``
* ``defect_hard_share``  share of the search and report defects (unreachable controls
                         excluded) whose best single slot sees less than half of the defect
* ``top_minus_low``      reachable share seen from the 80-degree ring minus the 20-degree ring
* ``views_to_90pct``     slots a greedy cover needs to see 90 % of the reachable surface
"""
from __future__ import annotations

import numpy as np

from ..geometry.camera_model import SlotGrid

FEATURES = ("height_ratio", "reachable", "best_1view", "rotation_gain", "defect_hard_share", "top_minus_low",
            "views_to_90pct")
SELECTION_FEATURES = ("height_ratio", "best_1view", "rotation_gain")
HARD_DEFECT_SHARE = 0.5


def greedy_cover(V: np.ndarray, n: int) -> np.ndarray:
    """Samples covered by n greedily chosen views (V: views x samples, bool)."""
    cov = np.zeros(V.shape[1], bool)
    for _ in range(n):
        cov |= V[int((V & ~cov).sum(1).argmax())]
    return cov


def nominal_descriptors(nominal_vis: np.ndarray, extents_m, grid: SlotGrid | None = None) -> dict:
    grid = grid or SlotGrid()
    ext = np.sort(np.asarray(extents_m, float))[::-1]
    reach = nominal_vis.any(0)
    R = nominal_vis[:, reach]
    n = max(R.shape[1], 1)
    one = R.sum(1).max() / n
    turn = max((R[k] | R[grid.rotate(k, 90.0)]).sum() for k in range(grid.n_slots)) / n
    ring = lambda i_el: R[[grid.slot_id(a, i_el, d) for a in range(grid.n_az) for d in range(grid.n_dist)]].any(0).mean()  # noqa: E731
    by_el = [float(ring(i)) for i in range(grid.n_el)]
    v90 = next((m for m in range(1, grid.n_slots + 1) if greedy_cover(R, m).mean() >= 0.9), grid.n_slots)
    return {"height_ratio": float(ext[-1] / ext[0]), "reachable": float(reach.mean()), "best_1view": float(one),
            "rotation_gain": float(turn - one), "top_minus_low": by_el[-1] - by_el[0], "views_to_90pct": int(v90),
            "by_elevation": by_el}


def defect_hard_share(defect_vis: np.ndarray, threshold: float = HARD_DEFECT_SHARE) -> float:
    """defect_vis: (defects, slots, samples) bool; share of defects whose best slot sees < threshold."""
    best = defect_vis.mean(2).max(1)
    return float((best < threshold).mean())


def part_descriptors(part_id: str, set_id: str = "mvp", sensor_id: str = "sl_default") -> dict:
    """All features from a built variant set's visibility table."""
    from ..datagen.parts import load_part
    from ..datagen.variants import load_manifest
    from ..visibility.raycast import vis_dir

    z = np.load(vis_dir(part_id, set_id, sensor_id) / "vis.npz")
    d = nominal_descriptors(z["nominal_vis"], load_part(part_id).extents_m)
    roles = {r["inst_id"]: r["role"] for r in load_manifest(part_id, set_id)["instances"]}
    keep = np.array([roles.get(str(i)) in ("search", "report") for i in z["inst_ids"]])
    d["defect_hard_share"] = defect_hard_share(z["vis"][keep])
    d["n_defects_used"] = int(keep.sum())
    return d


def screening_descriptors(part_id: str) -> dict:
    """Nominal-surface features of a prepared part (no variant set needed)."""
    from ..datagen.parts import load_part
    from ..visibility.raycast import nominal_visibility

    nv, _, _ = nominal_visibility(part_id)
    return nominal_descriptors(nv, load_part(part_id).extents_m)


def standardise(rows: dict[str, dict], keys) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    X = np.array([[rows[p][k] for k in keys] for p in rows], float)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    return {p: (X[i] - mu) / sd for i, p in enumerate(rows)}, mu, sd


def farthest_point_selection(existing: list[str], candidates: list[str], z: dict[str, np.ndarray], k: int) -> list[str]:
    """Greedy max-min selection: each step adds the candidate farthest (Euclidean, standardised
    features) from everything chosen so far, starting from ``existing``; ties -> earlier in
    ``candidates``."""
    chosen, picked = list(existing), []
    pool = list(candidates)
    for _ in range(min(k, len(pool))):
        dmin = [min(np.linalg.norm(z[c] - z[s]) for s in chosen) for c in pool]
        c = pool[int(np.argmax(dmin))]
        chosen.append(c)
        picked.append(c)
        pool.remove(c)
    return picked
