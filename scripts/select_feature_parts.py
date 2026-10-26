"""Selection of the six added T-LESS parts for the experiment on part shape features and
optimal inspection configurations (pre-registration §12, 2026-10-26). Uses geometry only
(nominal-surface slot visibility), never detection results.

Rule: candidates are the T-LESS objects not yet used whose coarse and fine meshes are both
watertight (boolean defects need it). Features: flatness (height_ratio), best single-slot
visible share (best_1view) and the gain of one 90-degree turn (rotation_gain), standardised
over the ten existing parts plus the candidates. Starting from the ten existing parts, the
candidate farthest (max-min Euclidean distance) from all chosen parts is added, six times;
ties -> lower T-LESS id.

    python scripts/select_feature_parts.py  ->  runs/exp/feature_map/selection.json
"""
from __future__ import annotations

import json
from pathlib import Path

from isco.datagen.parts import load_part
from isco.eval.part_features import SELECTION_FEATURES, farthest_point_selection, screening_descriptors, standardise

EXISTING = ["tless_13", "tless_18", "tless_29", "tless_30", "mulsen_nut", "mulsen_flat_pad",
            "mulsen_capsule", "real3d_candybar", "real3d_starfish", "dowel"]
CANDIDATES = [f"tless_{i:02d}" for i in range(1, 31) if f"tless_{i:02d}" not in EXISTING]
OUT = Path(__file__).resolve().parents[1] / "runs" / "exp" / "feature_map" / "selection.json"


def watertight(pid: str) -> bool:
    st = load_part(pid).stats
    return bool(st["coarse"]["watertight"] and st["fine"]["watertight"])


def main(k: int = 6) -> None:
    excluded = {}
    eligible = []
    for c in CANDIDATES:
        try:
            ok = watertight(c)
        except FileNotFoundError:
            excluded[c] = "not prepared"
            continue
        if ok:
            eligible.append(c)
        else:
            excluded[c] = "mesh not watertight"
    desc = {p: screening_descriptors(p) for p in EXISTING + eligible}
    z, mu, sd = standardise({p: desc[p] for p in EXISTING + eligible}, SELECTION_FEATURES)
    picked = farthest_point_selection(EXISTING, eligible, z, k)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rule": __doc__, "features": list(SELECTION_FEATURES), "existing": EXISTING,
                               "eligible": eligible, "excluded": excluded, "selected": picked,
                               "standardisation": {"mean": mu.tolist(), "sd": sd.tolist()},
                               "descriptors": desc}, indent=1))
    print(f"{'part':16s} {'h/L':>5s} {'1view':>6s} {'+rot':>6s}  status")
    for p in EXISTING + eligible:
        tag = "existing" if p in EXISTING else ("SELECTED" if p in picked else "")
        print(f"{p:16s} {desc[p]['height_ratio']:5.2f} {desc[p]['best_1view']:6.2f} {desc[p]['rotation_gain']:+6.2f}  {tag}")
    print("excluded:", excluded)
    print("selected (in order):", picked)


if __name__ == "__main__":
    main()
