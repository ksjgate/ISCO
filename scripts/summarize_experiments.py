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
    s = load("e0_6")
    if s:
        L += ["## E0.6 detector reproduction (MVTec 3D-AD)", "",
              "| category | BTF FPFH I-AUROC | published | Δ | within ±0.02 | AUPRO (published) | ISCO point FPFH I-AUROC |",
              "|---|---|---|---|---|---|---|"]
        for k, v in s.items():
            if "btf_fpfh" not in v:
                L.append(f"| {k} | {v.get('status')} | | | | | |")
                continue
            b, pub = v["btf_fpfh"], v["btf_fpfh"]["published"] or {}
            L.append(f"| {k} | {f(b['image_auroc'], 3)} | {f(pub.get('image_auroc'), 3)} | {f(b['delta_vs_published'], 3)} | "
                     f"{f(b['within_0.02'])} | {f(b['aupro_0.3'], 3)} ({f(pub.get('aupro'), 3)}) | {f(v['isco_fpfh']['image_auroc'], 3)} |")
        L.append("")
    s = load("e5b")
    if s:
        L += ["## E5b sensor gap (real MVTec scans vs ISCO re-renders)", ""]
        for k, v in s.items():
            if "stats" not in v:
                L += [f"- {k}: {v.get('status')}", ""]
                continue
            conds = list(v["stats"])
            L += [f"### {k}", "", "| condition | σ part (mm) | σ support (mm) | valid on object | outliers near object | "
                  "BTF FNR@1% (CI) | BTF FNR@5% | BTF AUROC | ISCO FNR@1% (CI) | ISCO FNR@5% | ISCO AUROC |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for c in conds:
                st, d = v["stats"][c], v["detection"][c]
                b, i = d["btf_fpfh"], d["isco_fpfh"]
                L.append(f"| {c} | {f(st['sigma_part_mm'], 3)} | {f(st['sigma_support_mm'], 3)} | {f(st['valid_object'])} | "
                         f"{f(st['outlier_frac_near_object'], 3)} | {f(b['fnr_1'])} ({f(b['fnr_1_ci'][0])}–{f(b['fnr_1_ci'][1])}) | "
                         f"{f(b['fnr_5'])} | {f(b['auroc_test'])} | {f(i['fnr_1'])} ({f(i['fnr_1_ci'][0])}–{f(i['fnr_1_ci'][1])}) | "
                         f"{f(i['fnr_5'])} | {f(i['auroc_test'])} |")
            types = list(v["detection"]["real"]["btf_fpfh"]["fnr_1_by_type"])
            for det in ("btf_fpfh", "isco_fpfh"):
                L += ["", f"{det}: FNR@1%FPR by type (decision: |Δ| ≤ 0.15 in ≥ 2/3 types)", "",
                      "| condition | " + " | ".join(types) + " | types within 0.15 | pass |", "|---" * (len(types) + 3) + "|"]
                for c in conds:
                    row = v["detection"][c][det]["fnr_1_by_type"]
                    dec = v["decision"][det].get(c, {})
                    L.append(f"| {c} | " + " | ".join(f(row[t]) for t in types)
                             + f" | {dec.get('types_within_0.15', '–')} | {f(dec.get('pass')) if dec else '–'} |")
                L.append(f"\nclosest level ({det}): {v['decision'][det]['closest_level']}")
            L += ["", f"closest level by σ on the part: {v['closest_level_by_sigma_part']}; "
                  f"N0 re-render vs real depth on the object: median {f(v['n0_vs_real_depth_median_mm'], 3)} mm", ""]
    for name, title in (("e1", "E1 / H1 cross-application"),
                        ("e1_full", "E1-full (exploratory: seeds 0,1,2, budget 60)")):
        s = load(name)
        if not s:
            continue
        L += [f"## {title}", "", f"median off-diagonal ΔFNR = {f(s['median_offdiag_delta'], 3)} (rule ≥ 0.10); "
              f"significant pairs (Holm, ≥0.05) {s['n_pairs_significant_and_ge_0.05']}/{s['n_pairs']}; **H1 supported: {f(s['supported'])}**", "",
              "best configurations: " + "; ".join(f"{k}: {v['slots']} {v['rotation']} v{v['speed_level']}" for k, v in s["best_configs"].items()), ""]
        D = np.array(s["delta_matrix"], float)
        L += [f"negative off-diagonal pairs (another part's optimum better): {int(np.nansum(D < 0))}/{s['n_pairs']}; "
              "mean FNR of each optimum over all parts: " + ", ".join(f"{k} {f(v)}" for k, v in s["universal_config_mean_fnr"].items()), ""]
    s = load("e2")
    if s:
        L += ["## E2 / H2 surrogate validity", "", "| part | ρ(V_area) | ρ(V_density) | ρ(V50) | recall@30 (V_area) | supported |", "|---|---|---|---|---|---|"]
        for k, v in s.items():
            L.append(f"| {k} | {f(v['spearman_V_area'])} | {f(v['spearman_V_density'])} | {f(v['spearman_V50'])} | {f(v['recall30_V_area'])} | {f(v['supported'])} |")
        n_sup = sum(v["supported"] for v in s.values())
        L += ["", f"**H2 valid in {n_sup}/{len(s)} parts (rule: ≥ 3/4)**", ""]
    s = load("e3")
    if s:
        L += ["## E3 / C1 non-additivity", "", "| part | FNR gain of BO over greedy-vis at n=3 | violation rate (random +1 cam) | violation rate CAD | supported (per part) |", "|---|---|---|---|---|"]
        for k, v in s.items():
            gain = None if v["bo_minus_greedy_vis_at_n3"] is None else -v["bo_minus_greedy_vis_at_n3"]   # stored as FNR(BO) - FNR(greedy)
            L.append(f"| {k} | {f(gain)} | {f(v['fpfh_bank']['violation_rate_random_additions'])} | "
                     f"{f(v['cad_distance']['violation_rate_random_additions'])} | {f(v['supported'])} |")
        # pre-registered C1 rule over parts: gap >= 0.05 in >= 4/6 parts and the violation rate
        # pooled over all random additions > 10 % for the bank and < 3 % for CAD-distance
        def pooled(det):
            n = sum(v[det]["n_additions"] or 0 for v in s.values())
            k = sum((v[det]["violation_rate_random_additions"] or 0) * (v[det]["n_additions"] or 0) for v in s.values())
            return k / n if n else None
        n_gap = sum(v["bo_minus_greedy_vis_at_n3"] is not None and -v["bo_minus_greedy_vis_at_n3"] >= 0.05 for v in s.values())
        vb, vc = pooled("fpfh_bank"), pooled("cad_distance")
        ok = n_gap >= (2 / 3) * len(s) and vb is not None and vb > 0.10 and vc is not None and vc < 0.03
        L += ["", f"parts where BO beats greedy-vis by ≥ 0.05 FNR at n = 3: {n_gap}/{len(s)} (rule ≥ 4/6); pooled violation rate "
              f"FPFH bank {f(vb, 3)} (rule > 0.10), CAD-distance {f(vc, 3)} (rule < 0.03) → **C1 supported: {f(ok)}**", ""]
    s = load("e4")
    if s:
        L += ["## E4 / H4 static vs inline", "", "| part | static optimum | inline optimum | ΔFNR static-opt @0.6 m/s | "
              "static-opt T @0.6 m/s | config changed | supported (any speed) | supported (T at 0.6 m/s) |", "|---|---|---|---|---|---|---|---|"]
        strict = []
        for k, v in s.items():
            if not isinstance(v, dict):
                continue
            so, io = v["static_opt"], v["inline_opt"]
            fast = v["static_opt_deployed_inline"]["2"]
            # the per-part flag accepts a T violation at any speed, which 0.1 m/s always gives;
            # the strict reading asks for a violation at the fastest speed
            ok = v["config_changed(n or rotation)"] and (v["delta_fnr_static_opt_at_0.6mps"] >= 0.10 or fast["violates_T"])
            strict.append(ok)
            L.append(f"| {k} | {so['slots']} {so['rotation']} | {io['slots']} {io['rotation']} v{io['speed_level']} | "
                     f"{f(v['delta_fnr_static_opt_at_0.6mps'])} | {f(fast['cycle_time_s'], 1)} s | {f(v['config_changed(n or rotation)'])} | "
                     f"{f(v['supported_for_part'])} | {f(ok)} |")
        L += ["", f"**H4 supported: {f(s.get('supported'))}** (strict reading: {sum(strict)}/{len(strict)} parts, rule ≥ 2/3)", ""]
    s = load("e5")
    if s:
        L += ["## E5 / H3 real-scan ranking (MulSen-AD)", "", "FNR_real over all accepted real defects (nothing is selected on "
              "the real set); report-only as sensitivity. Top-5 overlap: mean over random tie-breaks.", "",
              "| part | real defects / val normals | Kendall τ | top-5 overlap | τ (report only) | overlap (report only) | supported |",
              "|---|---|---|---|---|---|---|"]
        for k, v in s.items():
            L.append(f"| {k} | {v.get('n_real_defects', '–')} / {v.get('n_real_val', '–')} | {f(v['kendall_tau'])} | "
                     f"{f(v['top5_overlap'], 1)} | {f(v.get('kendall_tau_report'))} | {f(v.get('top5_overlap_report'), 1)} | "
                     f"{f(v['supported_for_part'])} |")
        n_sup = sum(v["supported_for_part"] for v in s.values())
        L += ["", f"**H3 supported in {n_sup}/{len(s)} parts (rule: ≥ 2/3) → {f(n_sup >= (2 / 3) * len(s))}**", ""]
    for name in ("e6", "e6_dino"):
        s = load(name)
        if s:
            L += [f"## {name.upper()} detector robustness", ""]
            for k, v in s.items():
                ref = v["pairs"].get(v.get("reference_pair") or "", {})
                robust = bool(ref) and ref["kendall_tau"] >= 0.5 and ref["a_best_in_b_top10"]   # both conditions of the rule
                L.append(f"- {k}: " + "; ".join(f"{pk}: τ={f(pv['kendall_tau'])}, best-in-top10={f(pv['a_best_in_b_top10'])}" for pk, pv in v["pairs"].items())
                         + f" → **detector-robust: {f(robust)}**")
            L.append("")
    for name, title in (("e7", "E7 noise ablation"), ("e7_n0", "E7-N0 (exploratory: noise-free level)")):
        s = load(name)
        if s:
            L += [f"## {title}", ""]
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
    s = load("specialization")
    if s:
        dU, dDA = s["delta_universal_minus_own"], s["delta_diag_advantage_full_minus_budget40"]
        L += ["## Part-specific optimum vs universal configuration (held-out report set)", "",
              f"universal configuration: {s['universal']['config']['slots']} {s['universal']['config']['rotation']} "
              f"v{s['universal']['config']['speed_level']} (mean search FNR {f(s['universal']['mean_search_fnr'])}; {', '.join(s['universal']['origin'])})", "",
              f"FNR(universal) − FNR(own, full search) = {f(dU['point'], 3)} [{f(dU['ci'][0], 3)}, {f(dU['ci'][1], 3)}] → "
              f"**{s['decision_specialisation']}** (part-specific if ≥ 0.05 and lower > 0; universal-equivalent if upper < 0.05)",
              f"diagonal advantage: full search {f(s['diag_advantage_full']['point'], 3)} (exact p {f(s['diag_perm_p_full'], 3)}), "
              f"budget 40 {f(s['diag_advantage_budget40']['point'], 3)} (exact p {f(s['diag_perm_p_budget40'], 3)}); "
              f"difference {f(dDA['point'], 3)} [{f(dDA['ci'][0], 3)}, {f(dDA['ci'][1], 3)}] → **{s['decision_search_budget']}**",
              f"expected parts whose own optimum ranks first among 7: {f(s['expected_parts_with_own_first'], 1)}", "",
              "| part | FNR own (full) | FNR own (budget 40) | FNR universal | own − universal [CI] | own − best other [CI] | E[rank of own] |",
              "|---|---|---|---|---|---|---|"]
        for k, v in s["per_part"].items():
            a, b = v["own_minus_universal"], v["own_minus_best_other"]
            L.append(f"| {k} | {f(v['fnr_own_full'])} | {f(v['fnr_own_budget40'])} | {f(v['fnr_universal'])} | "
                     f"{f(a['point'])} [{f(a['ci'][0])}, {f(a['ci'][1])}] | {f(b['point'])} [{f(b['ci'][0])}, {f(b['ci'][1])}] | "
                     f"{f(v['expected_rank_of_own_among_7'], 1)} |")
        if "without_flat_pad" in s:
            w = s["without_flat_pad"]["delta_universal_minus_own"]
            L.append(f"\nwithout the flat pad: universal − own {f(w['point'], 3)} [{f(w['ci'][0], 3)}, {f(w['ci'][1], 3)}]")
        L.append("")
    s = load("feature_map")
    if s:
        L += ["## Part shape features vs optimal configurations", "",
              "| hypothesis | feature | outcome | Spearman ρ | p | p (Holm) | supported |", "|---|---|---|---|---|---|---|"]
        for k, v in s["tests"].items():
            L.append(f"| {k} | {v['feature']} | {v['outcome']} | {f(v['rho'])} | {f(v['p'], 4)} | {f(v['p_holm'], 4)} | {f(v['supported'])} |")
        r = s["practical_rule"]
        L += ["", f"practical rule (leave one part out, {s['k_nn']} nearest parts): gap to own optimum {f(r['gap_to_own_optimum_mean'], 3)} "
              f"(rule ≤ 0.05), gain over the universal fixed configuration {f(r['gain_over_universal_mean'], 3)} "
              f"[{f(r['gain_over_universal_ci'][0], 3)}, {f(r['gain_over_universal_ci'][1], 3)}] → **supported: {f(r['supported'])}**", "",
              "| part | flatness | best 1 view | rotation gain | hard defects | target loss | rotation advantage | fixed-grid FNR sd | own optimum |",
              "|---|---|---|---|---|---|---|---|---|"]
        for p_ in s["parts"]:
            ft, o = s["features"][p_], s["optima"][p_]
            L.append(f"| {p_} | {f(ft['height_ratio'])} | {f(ft['best_1view'])} | {f(ft['rotation_gain'])} | {f(ft['defect_hard_share'])} | "
                     f"{f(s['target_loss'][p_])} | {f(s['rotation_advantage'][p_])} | {f(s['fixed_grid_fnr_sd'][p_])} | "
                     f"{o['slots']} {o['rotation']} v{o['speed_level']} |")
        L.append("")
    out = ROOT / "SUMMARY.md"
    out.write_text("\n".join(L) + "\n")
    print(out)


if __name__ == "__main__":
    main()
