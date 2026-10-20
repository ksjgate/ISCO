"""Run report: Pareto front, recommended configuration under user constraints,
per-defect-type visibility/detection, surrogate validity, search summary.

Writes ``<run_dir>/report/{index.md, index.html, recommended.json, *.png}``.
Numbers shown for configurations come from the held-out *report* defect set; the
optimiser only ever saw the *search* set (plan D3).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..utils.io import read_jsonl, write_json
from . import figures

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Noto Sans KR,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;
color:#0b0b0b;background:#fcfcfb;line-height:1.5}
table{border-collapse:collapse;font-size:13px;margin:8px 0}th,td{border-bottom:1px solid #e4e3df;padding:4px 8px;text-align:right}
th{color:#52514e;font-weight:600}td:first-child,th:first-child{text-align:left}img{max-width:100%}
code{background:#f0efec;padding:1px 4px;border-radius:3px}h1,h2{font-weight:600}
@media (prefers-color-scheme: dark){body{background:#1a1a19;color:#f0efec}th{color:#c3c2b7}code{background:#383835}}
"""


def _fmt(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{d}f}"


def _cams_table(cfg: dict, grid) -> str:
    lines = ["| camera | slot | azimuth (°) | elevation (°) | distance (m) |", "|---|---|---|---|---|"]
    for i, k in enumerate(cfg["slots"]):
        s = grid.slot(k)
        lines.append(f"| {i + 1} | {k} | {s.azimuth_deg:.0f} | {s.elevation_deg:.0f} | {s.distance_m:.2f} |")
    return "\n".join(lines)


def build_report(run_dir: str | Path, max_cams: int | None = 3, max_cycle_s: float | None = 1.5,
                 title: str | None = None) -> Path:
    from ..datagen.parts import load_part, load_part_mesh
    from ..optim.cost import CostModel
    from ..optim.pareto import hypervolume, nondominated_mask
    from ..optim.search import recommend
    from ..render.slot_cache import SlotCache
    from ..schemas.process import ProcessParams

    run_dir = Path(run_dir)
    out = run_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else {}
    rows2 = read_jsonl(run_dir / "stage2.jsonl")
    rows1 = read_jsonl(run_dir / "stage1.jsonl")
    part_id = meta.get("part_id") or (rows2[0]["part_id"] if rows2 else None)
    set_id = meta.get("set_id") or (rows2[0]["set_id"] if rows2 else "mvp")
    process = ProcessParams.model_validate(meta["process"]) if "process" in meta else ProcessParams()
    cost = CostModel(process)
    cache = SlotCache.open(part_id, set_id)
    grid = cache.grid
    part = load_part(part_id)
    uniq = {r["config_id"]: r for r in rows2 if r.get("stage") == 2}
    recs = list(uniq.values())
    F = np.array([cost.normalised(r["objectives"]) for r in recs]) if recs else np.zeros((0, 3))
    front_mask = nondominated_mask(F) if len(F) else np.zeros(0, bool)
    front = [r for r, m in zip(recs, front_mask) if m]
    front_ids = {r["config_id"] for r in front}
    rec = recommend(recs, max_cams, max_cycle_s)
    figs = {}
    if recs:
        figs["pareto"] = figures.pareto_plot(recs, front_ids, out / "pareto.png", use_report=True,
                                             title=f"{part_id}: evaluated configurations")
        figs["vis_fnr"], rho = figures.vis_vs_fnr(recs, out / "vis_vs_fnr.png", metric="V_area")
    else:
        rho = float("nan")
    if rows1:
        figs["stage1"] = figures.stage1_plot(rows1, out / "stage1.png")
    if rec is not None:
        fine = load_part_mesh(part_id, "fine")
        figs["rig"] = figures.rig_plot(rec["config"], grid, np.asarray(part.focus_m), fine.vertices, fine.faces,
                                       out / "recommended_rig.png")
        figs["types"] = figures.per_type_bars(rec, out / "recommended_types.png")
        write_json(out / "recommended.json", {
            "constraints": {"max_cams": max_cams, "max_cycle_s": max_cycle_s},
            "config": rec["config"], "config_id": rec["config_id"],
            "cameras": [{"slot": k, "azimuth_deg": grid.slot(k).azimuth_deg, "elevation_deg": grid.slot(k).elevation_deg,
                         "distance_m": grid.slot(k).distance_m} for k in rec["config"]["slots"]],
            "fnr_search": rec["search"]["fnr"], "fnr_report": rec["report"]["fnr"], "fnr_report_ci": rec["report"]["fnr_ci"],
            "auroc_report": rec["report"]["auroc"], "cycle_time_s": rec["objectives"]["cycle_time_s"],
            "n_cam": rec["n_cam"], "fnr_by_type_report": rec["report"]["fnr_by_type"]})
    # method summary
    methods = sorted({r.get("method") for r in rows2 if r.get("method")})
    msum = []
    for m in methods:
        mr = [r for r in rows2 if r.get("method") == m]
        Fm = np.array([cost.normalised(r["objectives"]) for r in mr])
        best = min(mr, key=lambda r: r["objectives"]["fnr"])
        msum.append((m, len(mr), hypervolume(Fm), best["objectives"]["fnr"], best["report"]["fnr"]))
    # markdown
    L = [f"# {title or 'ISCO run report'}", "",
         f"- part: `{part_id}` ({part.source}, licence: {part.license})",
         f"- variant set: `{set_id}`; process: `{process.process_id}` ({process.mode}); "
         f"detector: `{recs[0]['detector_id'] if recs else '–'}`; noise: `{recs[0]['noise'] if recs else '–'}`",
         f"- stage 1 candidates: {len(rows1)}; stage 2 evaluated configurations: {len(recs)}",
         f"- constraints for the recommendation: cameras ≤ {max_cams}, cycle time ≤ {max_cycle_s} s", ""]
    L += ["## Recommended configuration", ""]
    if rec is None:
        L += ["No evaluated configuration satisfies the constraints.", ""]
    else:
        r = rec["report"]
        L += [_cams_table(rec["config"], grid), "",
              f"- rotation stage: **{rec['config']['rotation']}**, belt speed level **{rec['config']['speed_level']}** "
              f"({process.speeds_mps[rec['config']['speed_level']]} m/s), trigger **{rec['config']['trigger_mode']}**",
              f"- FNR @ 1% FPR (held-out): **{_fmt(r['fnr'])}** (95% CI {_fmt(r['fnr_ci'][0])}–{_fmt(r['fnr_ci'][1])}); "
              f"search set {_fmt(rec['search']['fnr'])}; AUROC {_fmt(r['auroc'])}; FNR @ 5% FPR {_fmt(r['fnr_5'])}",
              f"- cost: {rec['n_cam']} cameras, cycle time {_fmt(rec['objectives']['cycle_time_s'])} s/part",
              f"- FNR by defect type: " + ", ".join(f"{k} {_fmt(v)}" for k, v in r["fnr_by_type"].items()),
              f"- FNR by difficulty: " + ", ".join(f"{k} {_fmt(v)}" for k, v in r["fnr_by_difficulty"].items()), ""]
        if "rig" in figs:
            L += ["![rig](recommended_rig.png)", "", "![types](recommended_types.png)", ""]
    L += ["## Pareto front (search objectives; values on the held-out set)", "",
          "| cameras | slots | rotation | speed | trigger | cycle (s) | FNR search | FNR report | 95% CI | AUROC |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(front, key=lambda r: (r["n_cam"], r["objectives"]["cycle_time_s"])):
        c = r["config"]
        L.append(f"| {r['n_cam']} | {c['slots']} | {c['rotation']} | {c['speed_level']} | {c['trigger_mode']} | "
                 f"{_fmt(r['objectives']['cycle_time_s'])} | {_fmt(r['search']['fnr'])} | {_fmt(r['report']['fnr'])} | "
                 f"{_fmt(r['report']['fnr_ci'][0])}–{_fmt(r['report']['fnr_ci'][1])} | {_fmt(r['report']['auroc'])} |")
    L += ["", "![pareto](pareto.png)", ""]
    L += ["## Surrogate validity (stage 1 vs stage 2)", "",
          f"Spearman correlation between the visibility surrogate V_area and the held-out FNR over the "
          f"{len(recs)} evaluated configurations: **{_fmt(rho)}** (negative = more visible, fewer misses).", "",
          "![vis](vis_vs_fnr.png)", ""]
    if msum:
        L += ["## Search methods", "", "| method | evaluations | hypervolume | best FNR (search) | its FNR (report) |",
              "|---|---|---|---|---|"]
        for m, n, hv, fs, fr in msum:
            L.append(f"| {m} | {n} | {_fmt(hv, 3)} | {_fmt(fs)} | {_fmt(fr)} |")
        L.append("")
    if "stage1" in figs:
        L += ["![stage1](stage1.png)", ""]
    if rec is not None and rec.get("unreachable"):
        L += ["## Visibility ceiling", "",
              f"Control defects placed on surfaces no upper-hemisphere camera can see (belt contact) were detected at "
              f"rate {_fmt(rec['unreachable']['detected_frac'])} (n = {rec['unreachable']['n']}); "
              "they are excluded from the FNR above.", ""]
    md = "\n".join(L)
    (out / "index.md").write_text(md)
    import markdown

    html = markdown.markdown(md, extensions=["tables"])
    (out / "index.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
                                    f"content='width=device-width,initial-scale=1'><title>ISCO report</title>"
                                    f"<style>{CSS}</style></head><body>{html}</body></html>")
    return out / "index.md"
