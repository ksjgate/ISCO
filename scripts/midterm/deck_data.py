"""Numbers and stills for the mid-term slides (``deck.js`` reads them).

    python scripts/midterm/deck_data.py  -> runs/midterm_reports/deck/{deck_data.json, stills/*}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXP, KO_PART, KO_ROT, PARTS, SPEEDS, TYPES, WORK, best_record  # noqa: E402

OUT = WORK / "deck"
ISAAC = WORK / "isaac"
SHORT = {"tless_13": "T-LESS 13", "tless_18": "T-LESS 18", "tless_29": "T-LESS 29", "tless_30": "T-LESS 30",
         "mulsen_nut": "너트", "mulsen_flat_pad": "평판 패드"}


def jl(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p)]


def stills() -> None:
    import imageio
    from PIL import Image

    d = OUT / "stills"
    d.mkdir(parents=True, exist_ok=True)

    def grab(name: str, frac: float | None = None, frame: int | None = None) -> Image.Image:
        r = imageio.get_reader(str(ISAAC / f"{name}.mp4"))
        n = r.count_frames()
        k = frame if frame is not None else int(frac * n)
        f = np.asarray(r.get_data(min(n - 1, max(0, k))))
        r.close()
        return Image.fromarray(f)

    grab("line", 0.97).save(d / "line.jpg", quality=90)
    grab("line", 0.05).save(d / "line_wide.jpg", quality=90)
    grab("slots", 0.55).save(d / "slots.jpg", quality=90)
    grab("turntable", 0.42).save(d / "turntable.jpg", quality=90)
    for p in PARTS:
        meta = json.loads((ISAAC / f"part_{p}.json").read_text())
        grab(f"part_{p}", frame=meta["capture_frames"][0] - 2).save(d / f"part_{p}.jpg", quality=90)
        grab(f"part_{p}", frame=8).save(d / f"part_{p}_wide.jpg", quality=90)
    Image.open(d / "part_tless_29.jpg").convert("RGB").resize((1280, 720)).save(d / "video_cover.png")  # embedded-video poster
    from video import camera_panel  # noqa: E402

    for p in ("tless_29", "mulsen_nut"):
        camera_panel(p).save(d / f"capture_{p}.png")
    for p in PARTS:                     # product stills, centred crop
        im = Image.open(ISAAC / "thumbs" / f"{p}.png").convert("RGB")
        w, h = im.size
        im.crop((int(w * 0.2), int(h * 0.04), int(w * 0.8), int(h * 0.96))).save(d / f"thumb_{p}.jpg", quality=90)
    for t in TYPES:                     # defect close-ups (T-LESS 29)
        grab(f"defect_{t}", 0.5).crop((200, 60, 1080, 690)).save(d / f"defect_{t}.jpg", quality=90)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data: dict = {"parts": PARTS, "short": SHORT, "ko": {p: KO_PART[p] for p in PARTS}}
    # recommended configurations
    from isco.geometry.camera_model import SlotGrid

    g = SlotGrid()
    rows = []
    for p in PARTS:
        r = best_record(p)
        c = r["config"]
        rows.append({"part": p, "n": len(c["slots"]),
                     "cams": [[g.slot(k).azimuth_deg, g.slot(k).elevation_deg, g.slot(k).distance_m] for k in c["slots"]],
                     "rotation": KO_ROT[c["rotation"]] if c["rotation"] != "none" else "없음",
                     "speed": SPEEDS[c["speed_level"]], "cycle": r["objectives"]["cycle_time_s"],
                     "fnr": r["report"]["fnr"], "ci": r["report"]["fnr_ci"], "auroc": r["report"]["auroc"],
                     "by_type": [r["report"]["fnr_by_type"][t] for t in TYPES]})
    data["recommended"] = rows
    # cross-application
    e1 = json.loads((EXP / "e1" / "summary.json").read_text())
    data["cross"] = {"M": e1["fnr_matrix_report"], "median_delta": e1["median_offdiag_delta"],
                     "n_sig": e1["n_pairs_significant_and_ge_0.05"], "n_pairs": e1["n_pairs"]}
    # visibility
    e2 = json.loads((EXP / "e2" / "summary.json").read_text())
    data["visibility"] = {}
    for p in ("tless_13", "tless_18", "tless_29", "tless_30"):
        R = jl(EXP / "e2" / f"evals_{p}.jsonl")
        data["visibility"][p] = {"rho": e2[p]["spearman_V_area"], "recall": e2[p]["recall30_V_area"],
                                 "ok": e2[p]["supported"],
                                 "pts": [[r["surrogate"]["V_area"], r["report"]["fnr"], r["config"]["rotation"] != "none"]
                                         for r in R]}
    # camera addition
    e3 = json.loads((EXP / "e3" / "summary.json").read_text())
    data["camera_addition"] = {p: {"curves": e3[p]["fpfh_bank"]["best_fnr_by_n"],
                                   "cad": e3[p]["cad_distance"]["best_fnr_by_n"]["greedy_vis"],
                                   "gain": -e3[p]["bo_minus_greedy_vis_at_n3"],
                                   "viol_bank": e3[p]["fpfh_bank"]["violation_rate_random_additions"],
                                   "viol_cad": e3[p]["cad_distance"]["violation_rate_random_additions"]}
                               for p in PARTS}
    # static vs inline
    e4 = json.loads((EXP / "e4" / "summary.json").read_text())
    data["static_inline"] = {p: {"static_rot": KO_ROT[e4[p]["static_opt"]["rotation"]],
                                 "static_n": len(e4[p]["static_opt"]["slots"]),
                                 "inline_rot": KO_ROT[e4[p]["inline_opt"]["rotation"]],
                                 "inline_n": len(e4[p]["inline_opt"]["slots"]),
                                 "static_fnr": e4[p]["static_opt_deployed_inline"]["2"]["fnr_report"],
                                 "static_T": e4[p]["static_opt_deployed_inline"]["2"]["cycle_time_s"],
                                 "inline_fnr": e4[p]["inline_opt_fnr_report"], "inline_T": e4[p]["inline_opt_cycle_time_s"],
                                 "factors": e4[p]["factor_decomposition_fnr_at_0.6mps"]}
                             for p in ("tless_13", "tless_29", "tless_30")}
    # detectors
    e6 = json.loads((EXP / "e6" / "summary.json").read_text())
    data["detectors"] = {p: {"bank": e6[p]["fnr"]["fpfh_bank"], "cad": e6[p]["fnr"]["cad_distance"],
                             "tau": e6[p]["pairs"]["fpfh_bank~cad_distance"]["kendall_tau"], "robust": e6[p]["robust"]}
                         for p in e6}
    # noise
    nl = json.loads((EXP / "e7" / "noise_levels.json").read_text())
    data["noise"] = {p: {lv: {"mean": v["mean_fnr"], "tau": v["kendall_vs_N2"]} for lv, v in nl[p].items()} for p in nl}
    # sensor gap (dowel)
    e5b = json.loads((EXP / "e5b" / "summary.json").read_text())["dowel"]
    conds = ["real", "N0", "N1@native", "N2@native", "N3@native", "N2"]
    data["sensor"] = {"conds": conds, "btf": [e5b["detection"][c]["btf_fpfh"]["auroc_test"] for c in conds],
                      "isco": [e5b["detection"][c]["isco_fpfh"]["auroc_test"] for c in conds],
                      "sigma": [e5b["stats"][c]["sigma_part_mm"] for c in conds]}
    # search efficiency (interim)
    e8 = json.loads((WORK / "e8_interim.json").read_text())
    data["search"] = {"curves": e8["tless_13"]["curves_median"], "at30": e8["tless_13"]["hv_at_30_median"]}
    dm = json.loads((ISAAC / "defects.json").read_text())
    sizes = {}
    for t in TYPES:
        d = dm[t]["defect"]
        sz = []
        if d.get("diameter_m"):
            sz.append(f"지름 {d['diameter_m'] * 1e3:.1f} mm")
        if d.get("length_m"):
            sz.append(f"길이 {d['length_m'] * 1e3:.1f} mm")
        if d.get("width_m"):
            sz.append(f"폭 {d['width_m'] * 1e3:.2f} mm")
        if d.get("depth_m"):
            sz.append(("처짐 " if t == "warp" else "깊이 ") + f"{d['depth_m'] * 1e3:.2f} mm")
        sizes[t] = ", ".join(sz[:2])
    data["defect_sizes"] = sizes
    data["types"] = list(TYPES)
    def clean(o):                       # NaN (camera counts a method never reached) -> null
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return None if isinstance(o, float) and o != o else o

    (OUT / "deck_data.json").write_text(json.dumps(clean(data), indent=1))
    if "--no-stills" not in sys.argv:
        stills()
    print(f"[deck_data] {OUT / 'deck_data.json'}")


if __name__ == "__main__":
    main()
