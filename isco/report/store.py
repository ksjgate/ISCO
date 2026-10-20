"""Append-only JSONL results store (one row per evaluated configuration / stage)."""
from __future__ import annotations

from pathlib import Path

from ..utils.io import append_jsonl, read_jsonl


class ResultsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict) -> None:
        append_jsonl(self.path, row)

    def rows(self, **where) -> list[dict]:
        out = []
        for r in read_jsonl(self.path):
            if all(r.get(k) == v for k, v in where.items()):
                out.append(r)
        return out

    def frame(self, **where):
        import pandas as pd

        flat = []
        for r in self.rows(**where):
            o = r.get("objectives", {})
            s = r.get("surrogate", {}) or {}
            rep = r.get("report") or {}
            sea = r.get("search") or {}
            flat.append({
                "method": r.get("method"), "stage": r.get("stage"), "seed": r.get("search_seed", r.get("seed")),
                "trial": r.get("trial"), "config_id": r.get("config_id"), "n_cam": r.get("n_cam"),
                "slots": tuple(r.get("config", {}).get("slots", [])), "rotation": r.get("config", {}).get("rotation"),
                "speed_level": r.get("config", {}).get("speed_level"), "trigger": r.get("config", {}).get("trigger_mode"),
                "fnr_search": sea.get("fnr", o.get("fnr")), "fnr_report": rep.get("fnr"),
                "fnr_report_lo": (rep.get("fnr_ci") or [None, None])[0], "fnr_report_hi": (rep.get("fnr_ci") or [None, None])[1],
                "auroc_report": rep.get("auroc"), "cycle_time_s": o.get("cycle_time_s"),
                "V_area": s.get("V_area"), "V_density": s.get("V_density"), "V50": s.get("V50"),
                "V_multi": s.get("V_multi"), "V_min_type": s.get("V_min_type"), "coverage_normal": s.get("coverage_normal"),
            })
        return pd.DataFrame(flat)
