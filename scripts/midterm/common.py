"""Shared inputs of the mid-term report material (``scripts/midterm/*``): the parts of the
confirmatory experiments, their recommended configurations (the per-part optimum of the
cross-application experiment) and, per part, one held-out defect shown in the demo.

All results are read from ``runs/exp`` of the repository that holds the data (the
installed ``isco`` package's repository); intermediate outputs go to ``runs/midterm_reports``
next to these scripts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from isco.utils.io import data_root

ROOT = data_root().parent          # the checkout holding data/ and runs/ (ISCO_DATA_ROOT may point at another one)
EXP = ROOT / "runs" / "exp"
WORK = Path(__file__).resolve().parents[2] / "runs" / "midterm_reports"
PARTS = ["tless_13", "tless_18", "tless_29", "tless_30", "mulsen_nut", "mulsen_flat_pad"]
TYPES = ("dent", "bulge", "hole", "cut", "crack", "warp")
KO_TYPE = {"dent": "눌림", "bulge": "돌출", "hole": "구멍", "cut": "절단", "crack": "균열", "warp": "휨"}
KO_PART = {"tless_13": "T-LESS 13번", "tless_18": "T-LESS 18번", "tless_29": "T-LESS 29번",
           "tless_30": "T-LESS 30번", "mulsen_nut": "MulSen-AD 너트", "mulsen_flat_pad": "MulSen-AD 평판 패드",
           "mulsen_capsule": "MulSen-AD 캡슐"}
KO_ROT = {"none": "회전 없음", "r90x1": "90° 한 번", "r180x1": "180° 한 번", "r90x3": "90°씩 세 번"}
SPEEDS = (0.1, 0.3, 0.6)


def best_record(pid: str) -> dict:
    """The evaluation of the part's recommended configuration (cross-application experiment)."""
    best = json.loads((EXP / "e1" / "summary.json").read_text())["best_configs"][pid]
    rows = [json.loads(line) for line in open(EXP / "e1" / f"search_{pid}.jsonl")]
    return next(r for r in rows if r["config"] == best)


def showcase(parts=PARTS) -> dict:
    """Per part one detected held-out defect (score above the 1 %-false-alarm threshold,
    at least 30 points on the defect, mostly visible), defect types spread over the parts."""
    used, out = set(), {}
    for pid in parts:
        rec = best_record(pid)
        tau = rec["report"]["tau"]
        cands = [d for d in rec["report_defects"]
                 if d["score"] > tau and d["n_defect_points"] >= 30 and (d["V"] if d["V"] == d["V"] else 0) >= 0.5]
        cands.sort(key=lambda d: (d["type"] in used, -(d["score"] - tau)))
        if not cands:
            raise RuntimeError(f"{pid}: no detected defect to show")
        d = cands[0]
        used.add(d["type"])
        out[pid] = {"inst_id": d["inst_id"], "type": d["type"], "score": d["score"], "feat_score": d["feat_score"],
                    "cov_score": d["cov_score"], "tau": tau, "V": d["V"], "n_defect_points": d["n_defect_points"],
                    "config": rec["config"]}
    return out


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))


if __name__ == "__main__":
    s = showcase()
    save_json(WORK / "showcase.json", s)
    for k, v in s.items():
        print(k, v["inst_id"], v["type"], f"score {v['score']:.2f} > tau {v['tau']:.2f}", f"V {v['V']:.2f}",
              v["n_defect_points"])
