"""E3 (exploratory): the random "+1 camera" pairs of hypotheses.e3, rebuilt with the same
RNG and read from the evaluation cache (no new evaluations), so the violation rate can be
reported at several FNR thresholds and on the report split for both detectors.

    python scripts/analyze_e3_additions.py  ->  runs/exp/e3/additions.json (+ table on stdout)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from isco.eval import hypotheses as H
from isco.schemas.config import InspectionConfig

ROOT = Path(__file__).resolve().parents[1] / "runs" / "exp" / "e3"
PARTS = ["tless_13", "tless_18", "tless_29", "tless_30", "mulsen_nut", "mulsen_flat_pad"]
DETECTORS = ("fpfh_bank", "cad_distance")


def pairs(pid: str, det: str, n_additions: int = 20, seed: int = 0) -> list[dict]:
    ev = H._ev(pid, "mvp", 1, detector=det)
    rng = np.random.default_rng(seed + 7)                       # as in hypotheses.e3
    out = []
    for _ in range(n_additions):
        n0 = int(rng.integers(1, 5))
        base = InspectionConfig(slots=ev.space.random_slots(n0, rng), rotation="none", speed_level=1)
        cand = [s for s in ev.space.valid_slots if all(ev.space.compatible(s, k) for k in base.slots)]
        if not cand:
            continue
        plus = InspectionConfig(slots=base.slots + [int(rng.choice(cand))], rotation="none", speed_level=1)
        a, b = (ev.cache.get(pid, ev._key(c, seed)) for c in (base, plus))
        if a is None or b is None:
            raise RuntimeError(f"{pid} {det}: pair not in the evaluation cache")
        out.append({"n0": n0, "slots": [base.slots, plus.slots],
                    **{k: [a[sp][m], b[sp][m]] for k, sp, m in (("search", "search", "fnr"), ("report", "report", "fnr"),
                                                                ("auroc", "search", "auroc"), ("tau", "search", "tau"))}})
    return out


def main() -> None:
    data = {pid: {det: pairs(pid, det) for det in DETECTORS} for pid in PARTS}
    (ROOT / "additions.json").write_text(json.dumps(data))
    print("| detector | split | > 0.02 (rule) | > 0.05 | > 0.10 | > 0.15 | improves (< −0.02) | mean ΔFNR |")
    print("|---|---|---|---|---|---|---|---|")
    for det in DETECTORS:
        for split in ("search", "report"):
            d = np.array([p[split][1] - p[split][0] for pid in PARTS for p in data[pid][det]])
            print(f"| {det} | {split} | {np.mean(d > 0.02):.3f} | {np.mean(d > 0.05):.3f} | {np.mean(d > 0.10):.3f} | "
                  f"{np.mean(d > 0.15):.3f} | {np.mean(d < -0.02):.3f} | {d.mean():+.3f} |")


if __name__ == "__main__":
    main()
