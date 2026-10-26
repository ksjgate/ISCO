"""The two search stages of one product for the demo video: stage 1 (visibility surrogate,
recomputed with the experiment's seed) and stage 2 (the 40 detector evaluations of the
cross-application experiment's search, read from its results).

    python scripts/midterm/search_data.py [--part tless_29]  -> runs/midterm_reports/search_<part>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXP, WORK, best_record, save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="tless_29")
    a = ap.parse_args()
    from isco.eval import hypotheses as H
    from isco.optim.search import SearchRunner

    ev = H._ev(a.part, "mvp", 1)
    r1 = SearchRunner(ev, None, None)
    cands = r1.stage1(1500, 500, seed=0)                     # as hypotheses._search (seed 0)
    warm = r1.shortlist(cands, 20, 10, seed=0)               # budget 40 -> k = 20
    rows = [json.loads(line) for line in open(EXP / "e1" / f"search_{a.part}.jsonl")]
    best = best_record(a.part)
    out = {"part": a.part,
           "stage1": [{"V": c["surrogate"]["V_area"], "T": c["objectives"]["cycle_time_s"], "n": c["n_cam"],
                       "id": c["config_id"]} for c in cands],
           "shortlist": [c.config_id for c in warm],
           "stage2": [{"fnr": r["report"]["fnr"], "fnr_search": r["search"]["fnr"], "T": r["objectives"]["cycle_time_s"],
                       "n": r["n_cam"], "id": r["config_id"], "V": r["surrogate"]["V_area"]} for r in rows],
           "best": {"id": best["config_id"], "fnr": best["report"]["fnr"], "T": best["objectives"]["cycle_time_s"],
                    "n": best["n_cam"], "config": best["config"]}}
    save_json(WORK / f"search_{a.part}.json", out)
    print(f"[search_data] {a.part}: {len(cands)} stage-1 candidates, {len(warm)} warm starts, {len(rows)} evaluations")


if __name__ == "__main__":
    main()
