"""E7 / E7-N0 (exploratory): per-configuration FNR of the E2 configurations at every noise
level, read from the evaluation cache (no new evaluations), with the visibility surrogate
of E2: mean FNR, Kendall tau vs N2, the chance that N2's best configuration is in the
level's top 10 (ties broken at random) and Spearman rho(V_area, FNR) per level.

    python scripts/analyze_noise_levels.py  ->  runs/exp/e7/noise_levels.json (+ table on stdout)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from isco.eval import hypotheses as H
from isco.schemas.config import InspectionConfig
from isco.utils.io import read_jsonl

ROOT = Path(__file__).resolve().parents[1] / "runs" / "exp"
PARTS = ["tless_13", "tless_29"]
LEVELS = ["N0", "N1", "N2", "N3", "N2MS"]


def main(budget: int = 40, draws: int = 2000) -> None:
    rng = np.random.default_rng(0)
    out = {}
    print("| part | level | n | mean FNR | Kendall τ vs N2 | P(N2 best in top 10) | ρ(V_area, FNR) |")
    print("|---|---|---|---|---|---|---|")
    for pid in PARTS:
        rows = list(read_jsonl(ROOT / "e2" / f"evals_{pid}.jsonl"))[:budget]
        cfgs = [InspectionConfig(**r["config"]) for r in rows]
        V = np.array([r["surrogate"]["V_area"] for r in rows])
        fnr = {}
        for nz in LEVELS:
            ev = H._ev(pid, "mvp", 1, noise=nz)
            recs = [ev.cache.get(pid, ev._key(c, 0)) for c in cfgs]
            if all(r is not None for r in recs):
                fnr[nz] = np.array([r["report"]["fnr"] for r in recs])
        ref = fnr["N2"]
        best_ref = [np.lexsort((rng.random(len(ref)), ref))[0] for _ in range(draws)]
        out[pid] = {}
        for nz, x in fnr.items():
            top10 = [set(np.lexsort((rng.random(len(x)), x))[:10]) for _ in range(draws)]
            p_in = float(np.mean([b in t for b, t in zip(best_ref, top10)]))
            res = {"mean_fnr": float(x.mean()), "kendall_vs_N2": H._kendall(ref, x), "p_N2_best_in_top10": p_in,
                   "spearman_V_area": H._spearman(V, x), "fnr": x.tolist()}
            out[pid][nz] = res
            print(f"| {pid} | {nz} | {len(x)} | {res['mean_fnr']:.3f} | {res['kendall_vs_N2']:.2f} | {p_in:.2f} | "
                  f"{res['spearman_V_area']:.2f} |")
    (ROOT / "e7" / "noise_levels.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
