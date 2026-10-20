"""Collect runs/exp/*/summary.json into one markdown overview (decisions per hypothesis).

    python scripts/summarize_experiments.py  ->  runs/exp/SUMMARY.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "runs" / "exp"


def load(name):
    p = ROOT / name / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def f(x, d=2):
    if x is None:
        return "–"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    try:
        if not np.isfinite(x):
            return "–"
        return f"{x:.{d}f}"
    except TypeError:
        return str(x)


def main() -> None:
    L = ["# ISCO experiment summary", ""]
    s = load("e0_4")
    if s:
        L += ["## E0.4 timing", "", "| part | render ms/view | stage-2 ref (s) | stage-2 6 cam x 4 stages (s) | stage-1 ms |", "|---|---|---|---|---|"]
        for r in s:
            L.append(f"| {r['part']} | {f(r['render_ms_per_view'], 1)} | {r['stage2_s_ref_3cam_first']} | {r['stage2_s_6cam_x4stages']} | {r['stage1_ms_per_config']} |")
        L.append("")
    s = load("e0_5")
    if s:
        L += ["## E0.5 severity calibration", "", f"mean pooled FNR at N2 = {f(s['mean_fnr_N2'])} (target 0.2–0.6) → "
              f"in target: {f(s['in_target'])}; scale_k recommended {s['scale_k_recommended']}", "",
              "| part | noise | FNR pooled | FNR report | AUROC |", "|---|---|---|---|---|"]
        for r in s["rows"]:
            L.append(f"| {r['part']} | {r['noise']} | {f(r['fnr_pooled'])} | {f(r['fnr_report'])} | {f(r['auroc_report'])} |")
        L.append("")
    for name, title in (("e0_2", "E0.2 cache vs direct render"), ("e0_3", "E0.3 GT masks")):
        p = ROOT / name / "summary.md"
        if p.exists():
            L += [f"## {title}", "", p.read_text().split("\n", 2)[-1], ""]
    s = load("e1")
    if s:
        L += ["## E1 / H1 cross-application", "", f"median off-diagonal ΔFNR = {f(s['median_offdiag_delta'])}; "
              f"significant pairs (Holm, ≥0.05) {s['n_pairs_significant_and_ge_0.05']}/{s['n_pairs']}; **H1 supported: {f(s['supported'])}**", "",
              "best configurations: " + "; ".join(f"{k}: {v['slots']} {v['rotation']} v{v['speed_level']}" for k, v in s["best_configs"].items()), ""]
    s = load("e2")
    if s:
        L += ["## E2 / H2 surrogate validity", "", "| part | ρ(V_area) | ρ(V_density) | ρ(V50) | recall@30 (V_area) | supported |", "|---|---|---|---|---|---|"]
        for k, v in s.items():
            L.append(f"| {k} | {f(v['spearman_V_area'])} | {f(v['spearman_V_density'])} | {f(v['spearman_V50'])} | {f(v['recall30_V_area'])} | {f(v['supported'])} |")
        n_sup = sum(v["supported"] for v in s.values())
        L += ["", f"**H2 valid in {n_sup}/{len(s)} parts (rule: ≥ 3/4)**", ""]
    s = load("e3")
    if s:
        L += ["## E3 / C1 non-additivity", "", "| part | BO − greedy-vis at n=3 (FNR gain) | violation rate (random +1 cam) | violation rate CAD | supported |", "|---|---|---|---|---|"]
        for k, v in s.items():
            L.append(f"| {k} | {f(v['bo_minus_greedy_vis_at_n3'])} | {f(v['fpfh_bank']['violation_rate_random_additions'])} | "
                     f"{f(v['cad_distance']['violation_rate_random_additions'])} | {f(v['supported'])} |")
        L.append("")
    s = load("e4")
    if s:
        L += ["## E4 / H4 static vs inline", "", "| part | static optimum | inline optimum | ΔFNR static-opt @0.6 m/s | config changed | supported |", "|---|---|---|---|---|---|"]
        for k, v in s.items():
            if not isinstance(v, dict):
                continue
            so, io = v["static_opt"], v["inline_opt"]
            L.append(f"| {k} | {so['slots']} {so['rotation']} | {io['slots']} {io['rotation']} v{io['speed_level']} | "
                     f"{f(v['delta_fnr_static_opt_at_0.6mps'])} | {f(v['config_changed(n or rotation)'])} | {f(v['supported_for_part'])} |")
        L += ["", f"**H4 supported: {f(s.get('supported'))}**", ""]
    s = load("e5")
    if s:
        L += ["## E5 / H3 real-scan ranking (MulSen-AD)", "", "| part | Kendall τ | top-5 overlap | supported |", "|---|---|---|---|"]
        for k, v in s.items():
            L.append(f"| {k} | {f(v['kendall_tau'])} | {v['top5_overlap']} | {f(v['supported_for_part'])} |")
        L.append("")
    for name in ("e6", "e6_dino"):
        s = load(name)
        if s:
            L += [f"## {name.upper()} detector robustness", ""]
            for k, v in s.items():
                L.append(f"- {k}: " + "; ".join(f"{pk}: τ={f(pv['kendall_tau'])}, best-in-top10={f(pv['a_best_in_b_top10'])}" for pk, pv in v["pairs"].items()))
            L.append("")
    s = load("e7")
    if s:
        L += ["## E7 noise ablation", ""]
        for k, v in s.items():
            L.append(f"- {k}: mean FNR {', '.join(f'{n} {f(x)}' for n, x in v['mean_fnr'].items())}; Kendall vs N2 "
                     + ", ".join(f"{n} {f(x)}" for n, x in v["kendall_vs_N2"].items()))
        L.append("")
    s = load("e7b")
    if s:
        L += ["## E7b yaw ablation", ""] + [f"- {k}: " + "; ".join(f"{sid}: {b['config']['slots']} {b['config']['rotation']} FNR {f(b['fnr_report'])}" for sid, b in v.items()) for k, v in s.items()] + [""]
    s = load("e8")
    if s:
        L += ["## E8 search efficiency", "", "| part | method | HV@30 (median) | HV final |", "|---|---|---|---|"]
        for k, v in s.items():
            for m in v["hv_final_median"]:
                L.append(f"| {k} | {m} | {f(v['hv_at_30_median'][m], 3)} | {f(v['hv_final_median'][m], 3)} |")
        L.append("")
    out = ROOT / "SUMMARY.md"
    out.write_text("\n".join(L) + "\n")
    print(out)


if __name__ == "__main__":
    main()
