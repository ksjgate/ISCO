"""Paper experiments E1-E8 (plan "실험 목록"; decision rules are pre-registered in the
private preregistration document). All evaluations go through ``ConfigEvaluator`` and
are cached, so experiments share work (e.g. E1's searches feed E2/E3/E8).

=====  ==================================================================================
E1     H1 cross-application: optimise per part (n<=3, T<=1.5 s), apply c*_A to part B ->
       DeltaFNR matrix (paired bootstrap CI, Holm) + "universal config" regret
E2     H2 surrogate validity: V-stratified configs -> Spearman(V, FNR), recall@30 of the
       top-10, failure taxonomy of high-V / high-FNR configs
E3     C1 non-additivity: greedy-visibility vs greedy-FNR vs random vs BO at equal budget;
       monotonicity-violation rate when adding a camera; threshold inflation; CAD-distance
       as an additive control
E4     H4 static vs inline: optimum under static capture vs inline process; static optimum
       deployed inline; factor decomposition (blur / trigger jitter / repositioning)
E5     H3 real-scan ranking: same configs on synthetic-defect vs real-defect (MulSen-AD)
       sets -> Kendall tau, top-5 overlap
E6     detector robustness: FPFH bank vs CAD distance (vs FPFH without xyz) rankings
E7     noise ablation N1/N2/N3/N2MS; E7b yaw-random variant set
E8     search efficiency: hypervolume vs evaluations for BO-warm / BO-cold / NSGA-II / random
=====  ==================================================================================

Follow-ups registered on 2026-10-26: ``specialization`` (part-specific optimum vs one
universal configuration, on a larger held-out report set) and ``feature_map`` (part shape
features vs optimal configurations on 16 parts).
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np

from .. import config_loader as C
from ..detect import scorer
from ..optim.evaluator import ConfigEvaluator
from ..optim.pareto import hv_curve
from ..optim.search import SearchRunner, recommend
from ..report import figures
from ..report.store import ResultsStore
from ..schemas.config import InspectionConfig
from ..utils.io import read_json, read_jsonl, repo_root, write_json
from ..utils.logging import get_logger
from .mvtec3d import e0_6, e5b  # noqa: F401  (CLI entry points)
from .sanity import e0_2, e0_3, e0_4, e0_5, exp_dir, reference_config, result_sanity  # noqa: F401  (CLI entry points)

log = get_logger(__name__)

CONSTRAINT = {"max_cams": 3, "max_cycle_s": 1.5}


def _ev(part, set_id, n_workers, detector="fpfh_bank", process="inline_default", noise="N2", **det_over):
    return ConfigEvaluator(part, set_id, C.detector(detector, **det_over), C.process(process), C.noise(noise),
                           n_workers=n_workers)


def _search(ev: ConfigEvaluator, d: Path, tag: str, method: str, budget: int, seed: int, fixed=None,
            stage1_trials: int = 1500, stage1_random: int = 500, k: int = 30, k_random: int = 10) -> list[dict]:
    """Stage 1 + shortlist + one stage-2 method; rows also appended to d/<tag>.jsonl."""
    st = ResultsStore(d / f"{tag}.jsonl")
    r1 = SearchRunner(ev, None, fixed)
    warm = []
    if method == "bo_warm":
        cands = r1.stage1(stage1_trials, stage1_random, seed=seed)
        warm = r1.shortlist(cands, min(k, budget // 2), k_random, seed=seed)
    return SearchRunner(ev, st, fixed).run(method, budget, seed, warm=warm)


def _best(rows: list[dict], max_cams: int | None = 3, max_cycle_s: float | None = 1.5) -> dict:
    """Recommended configuration under the constraints; falls back to the overall best
    search-set FNR if no evaluated configuration satisfies them (logged)."""
    b = recommend(rows, max_cams, max_cycle_s)
    if b is None:
        log.warning("no evaluated configuration satisfies the constraints; using the overall best")
        b = recommend(rows, None, None)
    return b


def _kendall(a, b) -> float:
    from scipy.stats import kendalltau

    return float(kendalltau(a, b).statistic)


def _top_overlap(a, b, k: int, rng: np.random.Generator, n_draws: int = 1000) -> float:
    """Mean size of the intersection of the k lowest entries of a and b, ties broken at
    random (index order would favour the configurations listed first)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    hits = [len(set(np.lexsort((rng.random(len(a)), a))[:k]) & set(np.lexsort((rng.random(len(b)), b))[:k]))
            for _ in range(n_draws)]
    return float(np.mean(hits))


def _defects(rec: dict) -> list[dict]:
    return (rec.get("search_defects") or []) + (rec.get("report_defects") or [])


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr

    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * pvals[j])
        adj[j] = min(1.0, run)
    return adj.tolist()


def _paired(rec_a: dict, rec_b: dict, B: int = 1000, seed: int = 0) -> tuple[float, float, float, float]:
    """FNR_a - FNR_b on the report set with a paired bootstrap; returns (diff, lo, hi, p)."""
    va, vb = np.asarray(rec_a["val_scores"]), np.asarray(rec_b["val_scores"])
    da = {d["inst_id"]: d["score"] for d in rec_a["report_defects"]}
    db = {d["inst_id"]: d["score"] for d in rec_b["report_defects"]}
    ids = sorted(set(da) & set(db))
    A, Bs = np.array([da[i] for i in ids]), np.array([db[i] for i in ids])
    rng = np.random.default_rng(seed)
    diffs = np.empty(B)
    for i in range(B):
        iv = rng.integers(len(va), size=len(va))
        idd = rng.integers(len(ids), size=len(ids))
        diffs[i] = np.mean(A[idd] <= scorer.threshold_at_fpr(va[iv])) - np.mean(Bs[idd] <= scorer.threshold_at_fpr(vb[iv]))
    point = np.mean(A <= scorer.threshold_at_fpr(va)) - np.mean(Bs <= scorer.threshold_at_fpr(vb))
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return float(point), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)), float(min(1.0, p))


# ---------------------------------------------------------------------------------- E1
def e1(parts: list[str], set_id: str = "mvp", seeds=(0,), budget: int = 40, n_workers: int = 20,
       out: str | None = None, **_) -> Path:
    d = exp_dir("e1", out)
    best = {}
    for pid in parts:
        ev = _ev(pid, set_id, n_workers)
        try:
            rows = []
            for s in seeds:
                rows += _search(ev, d, f"search_{pid}", "bo_warm", budget, s)
            b = _best(rows, **CONSTRAINT)
            best[pid] = b["config"]
            log.info(f"E1 {pid}: best {b['config']} FNR(search) {b['objectives']['fnr']:.2f}")
        finally:
            ev.close()
    # cross-application matrix
    n = len(parts)
    M = np.full((n, n), np.nan)
    D = np.full((n, n), np.nan)
    CI = {}
    pv = []
    recs = {}
    for j, pb in enumerate(parts):
        ev = _ev(pb, set_id, n_workers)
        try:
            for i, pa in enumerate(parts):
                recs[(pa, pb)] = ev.evaluate(InspectionConfig(**best[pa]), seeds[0])
        finally:
            ev.close()
    for i, pa in enumerate(parts):
        for j, pb in enumerate(parts):
            M[i, j] = recs[(pa, pb)]["report"]["fnr"]
            if i != j:
                diff, lo, hi, p = _paired(recs[(pa, pb)], recs[(pb, pb)])
                D[i, j] = diff
                CI[f"{pa}->{pb}"] = [diff, lo, hi, p]
                pv.append((f"{pa}->{pb}", p))
    adj = dict(zip([k for k, _ in pv], _holm([p for _, p in pv]))) if pv else {}
    off = D[~np.eye(n, dtype=bool)]
    sig = [k for k, v in CI.items() if adj.get(k, 1) < 0.05 and v[0] >= 0.05]
    univ = {pa: float(np.nanmean(M[i])) for i, pa in enumerate(parts)}
    summ = {"best_configs": best, "fnr_matrix_report": M.tolist(), "delta_matrix": D.tolist(), "pairs": CI,
            "holm_p": adj, "median_offdiag_delta": float(np.nanmedian(off)) if off.size else None,
            "n_pairs_significant_and_ge_0.05": len(sig), "n_pairs": len(CI),
            "universal_config_mean_fnr": univ,
            "decision_rule": "support H1 if median off-diagonal DeltaFNR >= 0.10 and >= 2/3 of pairs have CI excluding 0 (Holm)",
            "supported": bool(off.size and np.nanmedian(off) >= 0.10 and len(sig) >= (2 / 3) * len(CI))}
    write_json(d / "summary.json", summ)
    figures.heatmap(D, [f"opt for {p}" for p in parts], [f"on {p}" for p in parts], d / "fig5_cross_application.png",
                    title="E1: FNR(B | config optimised for A) - FNR(B | own optimum)", label="DeltaFNR")
    return d


# ---------------------------------------------------------------------------------- E2
def e2(parts: list[str], set_id: str = "mvp", seeds=(0,), budget: int = 60, n_workers: int = 20,
       out: str | None = None, n_pool: int = 3000, **_) -> Path:
    d = exp_dir("e2", out)
    summ = {}
    for pid in parts:
        ev = _ev(pid, set_id, n_workers)
        try:
            rng = np.random.default_rng(seeds[0])
            pool = {}
            while len(pool) < n_pool:
                c = ev.space.random_config(rng)
                pool[c.config_id] = c
            pool = list(pool.values())
            V = np.array([ev.vis.metrics(c)["V_area"] for c in pool])
            q = np.quantile(V, [1 / 3, 2 / 3])
            strata = [np.flatnonzero(V <= q[0]), np.flatnonzero((V > q[0]) & (V <= q[1])), np.flatnonzero(V > q[1])]
            per = budget // 3
            chosen = [pool[i] for s in strata for i in rng.choice(s, min(per, len(s)), replace=False)]
            rows = []
            st = ResultsStore(d / f"evals_{pid}.jsonl")
            for i, c in enumerate(chosen):
                r = ev.evaluate(c, seeds[0])
                st.append({**r, "method": "e2_stratified", "trial": i})
                rows.append(r)
        finally:
            ev.close()
        fnr = np.array([r["report"]["fnr"] for r in rows])
        res = {"n": len(rows)}
        for m in ("V_area", "V_density", "V50", "V_multi", "V_min_type", "coverage_normal"):
            res[f"spearman_{m}"] = _spearman([r["surrogate"][m] for r in rows], fnr)
        # recall@30: are the 10 best configs (FNR) inside the top-30 by V?
        best10 = set(np.argsort(fnr)[:10])
        for m in ("V_area", "V_density"):
            topV = set(np.argsort([-r["surrogate"][m] for r in rows])[:30])
            res[f"recall30_{m}"] = len(best10 & topV) / 10.0
        # failure taxonomy on high-V, high-FNR configurations
        v = np.array([r["surrogate"]["V_area"] for r in rows])
        bad = [r for r, vv, ff in zip(rows, v, fnr) if vv >= np.quantile(v, 0.66) and ff >= np.quantile(fnr, 0.66)]
        tax = {"grazing(mean V_multi<0.3)": 0, "few defect points(<30)": 0, "thin crack": 0, "threshold inflation": 0}
        tau_all = np.median([scorer.threshold_at_fpr(r["val_scores"]) for r in rows])
        for r in bad:
            missed = [x for x in r["report_defects"] if x["score"] <= scorer.threshold_at_fpr(r["val_scores"])]
            if r["surrogate"]["V_multi"] < 0.3:
                tax["grazing(mean V_multi<0.3)"] += 1
            if missed and np.median([x["n_defect_points"] for x in missed]) < 30:
                tax["few defect points(<30)"] += 1
            if missed and np.mean([x["type"] == "crack" for x in missed]) > 0.3:
                tax["thin crack"] += 1
            if scorer.threshold_at_fpr(r["val_scores"]) > 1.2 * tau_all:
                tax["threshold inflation"] += 1
        res["failure_taxonomy"] = {"n_high_V_high_FNR": len(bad), **tax}
        res["supported"] = bool(res["spearman_V_area"] <= -0.5 and res["recall30_V_area"] >= 0.7)
        summ[pid] = res
        figures.vis_vs_fnr(rows, d / f"fig4_vis_vs_fnr_{pid}.png", "V_area")
        log.info(f"E2 {pid}: rho(V_area, FNR) = {res['spearman_V_area']:.2f}")
    write_json(d / "summary.json", summ)
    return d


# ---------------------------------------------------------------------------------- E3
def e3(parts: list[str], set_id: str = "mvp", seeds=(0,), budget: int = 40, n_workers: int = 20,
       out: str | None = None, n_additions: int = 20, **_) -> Path:
    d = exp_dir("e3", out)
    summ = {}
    for pid in parts:
        res = {}
        for det in ("fpfh_bank", "cad_distance"):
            ev = _ev(pid, set_id, n_workers, detector=det)
            try:
                methods = ("greedy_vis", "greedy_fnr", "random", "bo_warm") if det == "fpfh_bank" else ("greedy_vis",)
                by = {}
                for m in methods:
                    rows = []
                    for s in seeds:
                        rows += _search(ev, d, f"{pid}_{det}_{m}", m, budget, s,
                                        fixed={"rotation": "none", "speed": 1, "trigger": "simultaneous"})
                    by[m] = rows
                best_by_n = {m: {n: min([r["objectives"]["fnr"] for r in rows if r["n_cam"] == n], default=np.nan)
                                 for n in range(1, 7)} for m, rows in by.items()}
                # monotonicity: add one compatible camera to random configs
                rng = np.random.default_rng(seeds[0] + 7)
                viol, inflation = [], []
                for _ in range(n_additions):
                    n0 = int(rng.integers(1, 5))
                    base = InspectionConfig(slots=ev.space.random_slots(n0, rng), rotation="none", speed_level=1)
                    cand = [s for s in ev.space.valid_slots if all(ev.space.compatible(s, k) for k in base.slots)]
                    if not cand:
                        continue
                    plus = InspectionConfig(slots=base.slots + [int(rng.choice(cand))], rotation="none", speed_level=1)
                    a, b = ev.evaluate(base, seeds[0]), ev.evaluate(plus, seeds[0])
                    viol.append(b["objectives"]["fnr"] > a["objectives"]["fnr"] + 0.02)
                    inflation.append(scorer.threshold_at_fpr(b["val_scores"]) - scorer.threshold_at_fpr(a["val_scores"]))
                gv = sorted(by["greedy_vis"], key=lambda r: r.get("greedy_n", r["n_cam"]))
                nested = [gv[i + 1]["objectives"]["fnr"] > gv[i]["objectives"]["fnr"] + 0.02 for i in range(len(gv) - 1)]
                res[det] = {"best_fnr_by_n": best_by_n, "violation_rate_random_additions": float(np.mean(viol)) if viol else None,
                            "violation_rate_greedy_nested": float(np.mean(nested)) if nested else None,
                            "tau_change_on_add_mean": float(np.mean(inflation)) if inflation else None,
                            "n_additions": len(viol)}
                if det == "fpfh_bank":
                    figures.lines_by_n({m: {n: v for n, v in best_by_n[m].items() if np.isfinite(v)} for m in methods},
                                       d / f"fig8_best_fnr_by_n_{pid}.png", "best FNR@1%FPR (search set)",
                                       title=f"E3 {pid}: equal budget {budget}")
            finally:
                ev.close()
        f = res["fpfh_bank"]
        gap3 = f["best_fnr_by_n"]["greedy_vis"].get(3, np.nan) - f["best_fnr_by_n"]["bo_warm"].get(3, np.nan)
        res["bo_minus_greedy_vis_at_n3"] = float(-gap3) if np.isfinite(gap3) else None
        res["supported"] = bool(np.isfinite(gap3) and gap3 >= 0.05 and (f["violation_rate_random_additions"] or 0) > 0.10)
        summ[pid] = res
    write_json(d / "summary.json", summ)
    return d


# ---------------------------------------------------------------------------------- E4
def e4(parts: list[str], set_id: str = "mvp", seeds=(0,), budget: int = 40, n_workers: int = 20,
       out: str | None = None, **_) -> Path:
    d = exp_dir("e4", out)
    summ = {}
    for pid in parts:
        opt = {}
        for proc in ("static_capture", "inline_default"):
            ev = _ev(pid, set_id, n_workers, process=proc)
            try:
                rows = []
                for s in seeds:
                    rows += _search(ev, d, f"{pid}_{proc}", "bo_warm", budget, s)
                cons = {"max_cams": 3, "max_cycle_s": None if proc == "static_capture" else 1.5}
                opt[proc] = _best(rows, **cons)
            finally:
                ev.close()
        s_cfg = InspectionConfig(**opt["static_capture"]["config"])
        i_cfg = InspectionConfig(**opt["inline_default"]["config"])
        ev = _ev(pid, set_id, n_workers, process="inline_default")
        try:
            deploy = {}
            for lvl in (0, 1, 2):
                c = InspectionConfig(**{**s_cfg.model_dump(), "speed_level": lvl})
                r = ev.evaluate(c, seeds[0])
                deploy[lvl] = {"fnr_report": r["report"]["fnr"], "cycle_time_s": r["objectives"]["cycle_time_s"],
                               "violates_T": r["objectives"]["cycle_time_s"] > 1.5}
            r_inline = ev.evaluate(i_cfg, seeds[0])
        finally:
            ev.close()
        # factor decomposition of the static optimum at the fastest speed
        fac = {}
        base = InspectionConfig(**{**s_cfg.model_dump(), "speed_level": 2})
        variants = {"full_inline": ("inline_default", "N2"), "no_blur": ("inline_default", None),
                    "no_trigger_jitter": ("_nojitter", "N2"), "static": ("static_capture", "N2")}
        for name, (proc, nz) in variants.items():
            if proc == "_nojitter":
                pp = C.process("inline_default").model_copy(update={"trigger_sigma_s": 0.0, "process_id": "inline_nojitter"})
            else:
                pp = C.process(proc)
            noise = C.noise("N2") if nz else C.noise("N2").model_copy(update={"t_acq_s": 0.0, "name": "N2_noblur"})
            ev2 = ConfigEvaluator(pid, set_id, C.detector("fpfh_bank"), pp, noise, n_workers=n_workers)
            try:
                fac[name] = ev2.evaluate(base, seeds[0])["report"]["fnr"]
            finally:
                ev2.close()
        diff = deploy[2]["fnr_report"] - r_inline["report"]["fnr"]
        changed = (s_cfg.n_cam != i_cfg.n_cam) or (s_cfg.rotation != i_cfg.rotation)
        summ[pid] = {"static_opt": s_cfg.model_dump(), "inline_opt": i_cfg.model_dump(),
                     "static_opt_deployed_inline": deploy, "inline_opt_fnr_report": r_inline["report"]["fnr"],
                     "inline_opt_cycle_time_s": r_inline["objectives"]["cycle_time_s"],
                     "delta_fnr_static_opt_at_0.6mps": diff, "config_changed(n or rotation)": changed,
                     "factor_decomposition_fnr_at_0.6mps": fac,
                     "supported_for_part": bool(changed and (diff >= 0.10 or any(v["violates_T"] for v in deploy.values())))}
    summ["supported"] = bool(sum(v["supported_for_part"] for v in summ.values() if isinstance(v, dict)) >= (2 / 3) * len(parts))
    write_json(d / "summary.json", summ)
    return d


# ---------------------------------------------------------------------------------- E5
def e5(parts: list[str], set_id: str = "mvp", seeds=(0,), budget: int = 40, n_workers: int = 20,
       out: str | None = None, real_set: str = "real", n_top: int = 20, n_random: int = 20, **_) -> Path:
    """parts: MulSen-AD parts (``mulsen_<cat>``) that have both a synthetic set and a real set."""
    d = exp_dir("e5", out)
    summ = {}
    for pid in parts:
        ev_s = _ev(pid, set_id, n_workers)
        try:
            rows = _search(ev_s, d, f"{pid}_synthetic_search", "bo_warm", budget, seeds[0])
            uniq = {r["config_id"]: r for r in rows}
            top = sorted(uniq.values(), key=lambda r: r["objectives"]["fnr"])[:n_top]
            rng = np.random.default_rng(seeds[0])
            cfgs = [InspectionConfig(**r["config"]) for r in top]
            while len(cfgs) < n_top + n_random:
                c = ev_s.space.random_config(rng)
                if c.config_id not in {x.config_id for x in cfgs}:
                    cfgs.append(c)
            syn = [ev_s.evaluate(c, seeds[0]) for c in cfgs]
        finally:
            ev_s.close()
        ev_r = _ev(pid, real_set, n_workers)
        try:
            real = [ev_r.evaluate(c, seeds[0]) for c in cfgs]
        finally:
            ev_r.close()
        fs = np.array([r["report"]["fnr"] for r in syn])
        # nothing is selected on the real set (the configurations come from the synthetic
        # search), so its search/report split carries no selection: FNR_real uses every
        # accepted real defect; the report-only value is kept as a sensitivity check
        fr = np.array([scorer.fnr_at_fpr(r["val_scores"], [x["score"] for x in _defects(r)]) for r in real])
        fr_rep = np.array([r["report"]["fnr"] for r in real])
        rng_t = np.random.default_rng(seeds[0])
        tau, overlap = _kendall(fs, fr), _top_overlap(fs, fr, 5, rng_t)
        summ[pid] = {"kendall_tau": tau, "top5_overlap": overlap, "n_configs": len(cfgs),
                     "n_real_defects": len(_defects(real[0])), "n_real_val": len(real[0]["val_scores"]),
                     "fnr_synthetic": fs.tolist(), "fnr_real": fr.tolist(), "fnr_real_report": fr_rep.tolist(),
                     "kendall_tau_report": _kendall(fs, fr_rep), "top5_overlap_report": _top_overlap(fs, fr_rep, 5, rng_t),
                     "supported_for_part": bool(tau >= 0.5 and overlap >= 3)}
        log.info(f"E5 {pid}: Kendall tau {tau:.2f}, top-5 overlap {overlap:.2f}")
    write_json(d / "summary.json", summ)
    return d


# ---------------------------------------------------------------------------------- E6
DETECTORS = {"fpfh_bank": ("fpfh_bank", {}), "cad_distance": ("cad_distance", {}),
             "fpfh_bank_noxyz": ("fpfh_bank", {"xyz_weight": 0.0}), "dinov2_bank": ("dinov2_bank", {})}


def e6(parts: list[str], set_id: str = "mvp", seeds=(0,), budget: int = 40, n_workers: int = 20,
       out: str | None = None, e2_dir: str | None = None,
       detectors=("fpfh_bank", "cad_distance", "fpfh_bank_noxyz"), **_) -> Path:
    """Detector robustness on the E2 configurations (FPFH bank as the reference vs CAD
    distance, FPFH without coordinates, and the DINOv2 view backbone)."""
    d = exp_dir("e6", out)
    src = Path(e2_dir) if e2_dir else exp_dir("e2")
    detectors = ["fpfh_bank"] + [x for x in detectors if x != "fpfh_bank"]
    summ = {}
    for pid in parts:
        cfgs = [InspectionConfig(**r["config"]) for r in read_jsonl(src / f"evals_{pid}.jsonl")][:budget]
        res = {}
        for name in detectors:
            det, over = DETECTORS[name]
            ev = _ev(pid, set_id, n_workers, detector=det, **over)
            try:
                res[name] = np.array([ev.evaluate(c, seeds[0])["report"]["fnr"] for c in cfgs])
            finally:
                ev.close()
        pairs = {}
        for a, b in itertools.combinations(res, 2):
            top10_b = set(np.argsort(res[b])[:10])
            pairs[f"{a}~{b}"] = {"kendall_tau": _kendall(res[a], res[b]),
                                 "a_best_in_b_top10": int(np.argmin(res[a])) in top10_b}
        ref = next((k for k in pairs if k.startswith("fpfh_bank~") and k != "fpfh_bank~fpfh_bank_noxyz"), None)
        summ[pid] = {"pairs": pairs, "fnr": {k: v.tolist() for k, v in res.items()},
                     "robust": bool(ref and pairs[ref]["kendall_tau"] >= 0.5), "reference_pair": ref}
    write_json(d / "summary.json", summ)
    return d


# ---------------------------------------------------------------------------------- E7
def e7(parts: list[str], set_id: str = "mvp", seeds=(0,), budget: int = 40, n_workers: int = 20,
       out: str | None = None, e2_dir: str | None = None, noises=("N1", "N2", "N3", "N2MS"), **_) -> Path:
    d = exp_dir("e7", out)
    src = Path(e2_dir) if e2_dir else exp_dir("e2")
    summ = {}
    for pid in parts:
        cfgs = [InspectionConfig(**r["config"]) for r in read_jsonl(src / f"evals_{pid}.jsonl")][:budget]
        res = {}
        for nz in noises:
            ev = _ev(pid, set_id, n_workers, noise=nz)
            try:
                res[nz] = np.array([ev.evaluate(c, seeds[0])["report"]["fnr"] for c in cfgs])
            finally:
                ev.close()
        summ[pid] = {"mean_fnr": {k: float(v.mean()) for k, v in res.items()},
                     "kendall_vs_N2": {k: _kendall(res["N2"], v) for k, v in res.items() if k != "N2"},
                     "best_config": {k: cfgs[int(np.argmin(v))].model_dump() for k, v in res.items()}}
    write_json(d / "summary.json", summ)
    return d


def e7b(parts: list[str], seeds=(0,), budget: int = 40, n_workers: int = 20, out: str | None = None, **_) -> Path:
    """Stable-pose / yaw ablation: guided (mvp) vs uniform-yaw (mvp_yawrand) variant sets."""
    d = exp_dir("e7b", out)
    summ = {}
    for pid in parts:
        best = {}
        for sid in ("mvp", "mvp_yawrand"):
            ev = _ev(pid, sid, n_workers)
            try:
                rows = _search(ev, d, f"{pid}_{sid}", "bo_warm", budget, seeds[0])
                b = _best(rows, **CONSTRAINT)
                best[sid] = {"config": b["config"], "fnr_report": b["report"]["fnr"]}
            finally:
                ev.close()
        summ[pid] = best
    write_json(d / "summary.json", summ)
    return d


# ---------------------------------------------------------------------------------- E8
def e8(parts: list[str], set_id: str = "mvp", seeds=(0, 1), budget: int = 40, n_workers: int = 20,
       out: str | None = None, methods=("bo_warm", "bo_cold", "nsga2", "random"), **_) -> Path:
    d = exp_dir("e8", out)
    summ = {}
    for pid in parts:
        ev = _ev(pid, set_id, n_workers)
        curves = {}
        try:
            for m in methods:
                cs = []
                for s in seeds:
                    rows = _search(ev, d, f"{pid}_{m}_s{s}", m, budget, s)
                    F = np.array([ev.cost.normalised(r["objectives"]) for r in rows])
                    c = hv_curve(F)
                    cs.append(np.r_[c, np.full(budget - len(c), c[-1] if len(c) else 0.0)][:budget])
                curves[m] = np.array(cs)
        finally:
            ev.close()
        figures.hv_curves(curves, d / f"fig9_hv_{pid}.png", title=f"E8 {pid}")
        final = {m: float(np.median(c[:, -1])) for m, c in curves.items()}
        at30 = {m: float(np.median(c[:, min(29, c.shape[1] - 1)])) for m, c in curves.items()}
        summ[pid] = {"hv_final_median": final, "hv_at_30_median": at30,
                     "bo_warm_vs_random_at_30": (at30.get("bo_warm", np.nan) / max(at30.get("random", np.nan), 1e-12) - 1)
                     if "random" in at30 else None,
                     "curves": {m: c.tolist() for m, c in curves.items()}}
    write_json(d / "summary.json", summ)
    return d


# ------------------------------------------------- follow-ups (pre-registration §12, 2026-10-26)
def _aligned_scores(rec: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Validation scores (manifest order) and defect scores sorted by instance id."""
    ds = sorted(_defects(rec), key=lambda x: x["inst_id"])
    return np.asarray(rec["val_scores"], float), np.array([x["score"] for x in ds], float), [x["inst_id"] for x in ds]


def _fnr_table(recs: dict, cfg_ids: list[str], parts: list[str], B: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """FNR at 1 % FPR of every (configuration, part) and its joint bootstrap: per part the
    validation normals and the defects are resampled once per replicate and shared by all
    configurations (paired). Returns point (C, P) and replicates (B, C, P)."""
    rng = np.random.default_rng(seed)
    point = np.empty((len(cfg_ids), len(parts)))
    boot = np.empty((B, len(cfg_ids), len(parts)))
    for j, p in enumerate(parts):
        sc = [_aligned_scores(recs[(c, p)]) for c in cfg_ids]
        nv, ids = len(sc[0][0]), sc[0][2]
        if any(len(v) != nv or i != ids for v, _, i in sc):
            raise ValueError(f"{p}: configurations were scored on different instances")
        for i, (v, s, _) in enumerate(sc):
            point[i, j] = np.mean(s <= scorer.threshold_at_fpr(v))
        for b in range(B):
            iv, idd = rng.integers(nv, size=nv), rng.integers(len(ids), size=len(ids))
            for i, (v, s, _) in enumerate(sc):
                boot[b, i, j] = np.mean(s[idd] <= scorer.threshold_at_fpr(v[iv]))
    return point, boot


def _diag_advantage(M: np.ndarray) -> np.ndarray:
    """M[..., i, j] = FNR on part j of the configuration optimised for part i: mean over parts
    of (mean FNR of the other parts' optima - FNR of the own optimum); batched over leading axes."""
    n = M.shape[-1]
    diag = np.diagonal(M, axis1=-2, axis2=-1)
    others = (M.sum(-2) - diag) / (n - 1)
    return (others - diag).mean(-1)


def _diag_perm_p(M: np.ndarray) -> float:
    """Exact permutation p of the diagonal advantage: rows (which optimum belongs to which
    part) relabelled in every possible way."""
    obs = _diag_advantage(M)
    perms = np.array(list(itertools.permutations(range(M.shape[0]))))
    vals = _diag_advantage(M[perms])
    return float(np.mean(vals >= obs - 1e-12))


def _ci_decision(point: float, boot: np.ndarray) -> dict:
    lo, hi = (float(x) for x in np.quantile(boot, [0.025, 0.975]))
    return {"point": float(point), "ci": [lo, hi]}


def specialization(parts: list[str], set_id: str = "mvp", ext_set: str = "mvp_r144", seeds=(0,), n_workers: int = 20,
                   out: str | None = None, e1_dir: str | None = None, e1_full_dir: str | None = None, top_k: int = 5,
                   B: int = 2000, margin: float = 0.05, **_) -> Path:
    """부품별 최적 설정 대 범용 설정 비교 실험 (part-specific optimum vs one universal configuration).

    Per-part optima: the cross-application experiment (budget 40, seed 0; ``e1``) and its rerun
    with the full search settings (budget 60 x seeds 0-2; ``e1_full``). Universal configuration:
    among the top-``top_k`` feasible configurations of every part's full search plus the six
    budget-40 optima, the one with the lowest mean search-split FNR over all parts (seed 0).
    All optima and the universal configuration are then scored on ``ext_set`` -- a held-out
    report set with 144 new defects and 200 validation normals per part, never used for any
    selection. Decision rules: pre-registration §12 (2026-10-26)."""
    d = exp_dir("specialization", out)
    exp_root = repo_root() / "runs" / "exp"
    s40 = read_json(Path(e1_dir or exp_root / "e1") / "summary.json")["best_configs"]
    full_dir = Path(e1_full_dir or exp_root / "e1_full")
    sfull = read_json(full_dir / "summary.json")["best_configs"]
    opt40 = {p: InspectionConfig(**s40[p]) for p in parts}
    optf = {p: InspectionConfig(**sfull[p]) for p in parts}
    ok = lambda r: r.get("stage") == 2 and r["n_cam"] <= CONSTRAINT["max_cams"] and \
        r["objectives"]["cycle_time_s"] <= CONSTRAINT["max_cycle_s"] + 1e-9  # noqa: E731
    cands: dict[str, InspectionConfig] = {}
    origin: dict[str, list[str]] = {}
    for p in parts:
        rows = sorted((r for r in read_jsonl(full_dir / f"search_{p}.jsonl") if ok(r)),
                      key=lambda r: (r["objectives"]["fnr"], r["n_cam"], r["objectives"]["cycle_time_s"], r["config_id"]))
        taken = []
        for r in rows:
            if r["config_id"] not in taken:
                taken.append(r["config_id"])
                cands.setdefault(r["config_id"], InspectionConfig(**r["config"]))
                origin.setdefault(r["config_id"], []).append(f"top{len(taken)}:{p}")
            if len(taken) == top_k:
                break
    for p, c in opt40.items():
        cands.setdefault(c.config_id, c)
        origin.setdefault(c.config_id, []).append(f"budget40_optimum:{p}")
    search_fnr: dict[str, dict[str, float]] = {cid: {} for cid in cands}
    cycle: dict[str, float] = {}
    for p in parts:
        ev = _ev(p, set_id, n_workers)
        try:
            for cid, c in cands.items():
                r = ev.evaluate(c, seeds[0])
                search_fnr[cid][p] = r["search"]["fnr"]
                cycle[cid] = r["objectives"]["cycle_time_s"]
        finally:
            ev.close()
    mean_search = {cid: float(np.mean(list(v.values()))) for cid, v in search_fnr.items()}
    uni = min(cands, key=lambda cid: (mean_search[cid], cands[cid].n_cam, cycle[cid], cid))
    log.info(f"universal configuration {cands[uni].model_dump()} (mean search FNR {mean_search[uni]:.3f})")

    finals = {c.config_id: c for c in [*opt40.values(), *optf.values(), cands[uni]]}
    recs = {}
    for p in parts:
        ev = _ev(p, ext_set, n_workers)
        try:
            for cid, c in finals.items():
                recs[(cid, p)] = ev.evaluate(c, seeds[0])
        finally:
            ev.close()
    ids = list(finals)
    F, Fb = _fnr_table(recs, ids, parts, B, seeds[0])
    row = {cid: i for i, cid in enumerate(ids)}
    J = np.arange(len(parts))
    own_f = np.array([row[optf[p].config_id] for p in parts])
    own_40 = np.array([row[opt40[p].config_id] for p in parts])
    u = row[uni]

    def stats_for(pp: list[int]) -> dict:
        own = F[own_f[pp], pp]
        own_b = Fb[:, own_f[pp], pp]
        dU = _ci_decision((F[u, pp] - own).mean(), (Fb[:, u, pp] - own_b).mean(1))
        Mf, M40 = F[np.ix_(own_f[pp], pp)], F[np.ix_(own_40[pp], pp)]
        Mfb = Fb[:, own_f[pp]][:, :, pp]
        M40b = Fb[:, own_40[pp]][:, :, pp]
        dDA = _ci_decision(_diag_advantage(Mf) - _diag_advantage(M40), _diag_advantage(Mfb) - _diag_advantage(M40b))
        return {"delta_universal_minus_own": dU, "diag_advantage_full": _ci_decision(_diag_advantage(Mf), _diag_advantage(Mfb)),
                "diag_advantage_budget40": _ci_decision(_diag_advantage(M40), _diag_advantage(M40b)),
                "delta_diag_advantage_full_minus_budget40": dDA,
                "diag_perm_p_full": _diag_perm_p(Mf), "diag_perm_p_budget40": _diag_perm_p(M40),
                "matrix_full": Mf.tolist(), "matrix_budget40": M40.tolist()}

    allp = list(J)
    res = stats_for(allp)
    dU, dDA = res["delta_universal_minus_own"], res["delta_diag_advantage_full_minus_budget40"]
    res["decision_specialisation"] = ("part-specific" if dU["point"] >= margin and dU["ci"][0] > 0 else
                                      "universal-equivalent" if dU["ci"][1] < margin else "inconclusive")
    res["decision_search_budget"] = ("stronger" if dDA["ci"][0] > 0 else "weaker" if dDA["ci"][1] < 0 else "inconclusive")
    rng = np.random.default_rng(seeds[0])
    per_part = {}
    for j, p in enumerate(parts):
        pool = list(own_f) + [u]                     # the six full-search optima and the universal one
        vals = F[pool, j]
        ranks = [1 + int(np.where(np.lexsort((rng.random(len(vals)), vals)) == pool.index(own_f[j]))[0][0])
                 for _ in range(1000)]
        others = [i for i in own_f if i != own_f[j]]
        best_other = others[int(np.argmin(F[others, j]))]
        per_part[p] = {"fnr_own_full": float(F[own_f[j], j]), "fnr_own_budget40": float(F[own_40[j], j]),
                       "fnr_universal": float(F[u, j]),
                       "own_minus_universal": _ci_decision(F[own_f[j], j] - F[u, j], Fb[:, own_f[j], j] - Fb[:, u, j]),
                       "own_minus_best_other": _ci_decision(F[own_f[j], j] - F[best_other, j],
                                                            Fb[:, own_f[j], j] - Fb[:, best_other, j]),
                       "expected_rank_of_own_among_7": float(np.mean(ranks)), "p_own_first": float(np.mean(np.array(ranks) == 1))}
    res["per_part"] = per_part
    res["expected_parts_with_own_first"] = float(sum(v["p_own_first"] for v in per_part.values()))
    if "mulsen_flat_pad" in parts:
        res["without_flat_pad"] = stats_for([j for j, p in enumerate(parts) if p != "mulsen_flat_pad"])
    res.update({"parts": parts, "configs": {cid: c.model_dump() for cid, c in finals.items()},
                "universal": {"config_id": uni, "config": cands[uni].model_dump(), "mean_search_fnr": mean_search[uni],
                              "origin": origin.get(uni, [])},
                "candidates": {cid: {"config": c.model_dump(), "origin": origin.get(cid, []), "search_fnr": search_fnr[cid],
                                     "mean_search_fnr": mean_search[cid]} for cid, c in cands.items()},
                "optima_full": {p: c.config_id for p, c in optf.items()}, "optima_budget40": {p: c.config_id for p, c in opt40.items()},
                "fnr_point": {cid: dict(zip(parts, F[row[cid]].tolist())) for cid in ids},
                "n_defects": {p: len(_aligned_scores(recs[(ids[0], p)])[2]) for p in parts},
                "n_val": {p: len(recs[(ids[0], p)]["val_scores"]) for p in parts}, "B": B, "margin": margin})
    write_json(d / "summary.json", res)
    figures.heatmap(np.array(res["matrix_full"]), [f"opt for {p}" for p in parts], [f"on {p}" for p in parts],
                    d / "fig_specialization_matrix.png", title="FNR on the held-out report set (full-search optima)", label="FNR")
    log.info(f"specialization: universal - own {dU['point']:+.3f} {dU['ci']} -> {res['decision_specialisation']}; "
             f"diag advantage full - budget40 {dDA['point']:+.3f} {dDA['ci']} -> {res['decision_search_budget']}")
    return d


FIXED_GRID = [(n, el, rot) for el in (20.0, 40.0, 60.0) for n in (1, 2, 3) for rot in ("none", "r90x1")] + \
             [(1, 80.0, "none"), (1, 80.0, "r90x1")]


def fixed_configs(space) -> dict[str, InspectionConfig]:
    """The fixed configuration grid of the feature experiment: n evenly spaced cameras
    (azimuth from 0) at one elevation, 0.25 m, 0.6 m/s, simultaneous trigger, with and
    without one 90-degree turn (cycle 0.5 s / 1.4 s). Two or three cameras at 80 degrees
    would be closer than the 30-degree minimum separation, so 80 degrees has one camera."""
    out = {}
    for n, el, rot in FIXED_GRID:
        slots = space.ring(n, el, space.grid.distances_m[0])
        if len(slots) != n or any(space.grid.slot(k).elevation_deg != el for k in slots):
            raise ValueError(f"fixed configuration n={n} el={el} is not realisable: {slots}")
        out[f"n{n}_el{int(el)}_{rot}"] = InspectionConfig(slots=slots, rotation=rot, speed_level=2, trigger_mode="simultaneous")
    return out


def _perm_spearman(x, y, n_perm: int, rng: np.random.Generator, sign: int = 1) -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = _spearman(x, y)
    if not np.isfinite(obs):                     # a constant variable: no evidence either way
        return float("nan"), 1.0
    perm = np.array([_spearman(x, rng.permutation(y)) for _ in range(n_perm)])
    p = (1 + np.sum(sign * perm >= sign * obs - 1e-12)) / (1 + n_perm)
    return float(obs), float(p)


def _mantel(dist: np.ndarray, loss: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, float]:
    """Spearman correlation of two symmetric matrices over the pairs i < j; part labels of
    ``dist`` permuted; one-sided (positive)."""
    iu = np.triu_indices(len(dist), 1)
    obs = _spearman(dist[iu], loss[iu])
    if not np.isfinite(obs):
        return float("nan"), 1.0
    perm = []
    for _ in range(n_perm):
        o = rng.permutation(len(dist))
        perm.append(_spearman(dist[np.ix_(o, o)][iu], loss[iu]))
    return float(obs), float((1 + np.sum(np.array(perm) >= obs - 1e-12)) / (1 + n_perm))


def _loo_rule(feats: dict[str, np.ndarray], fixed: dict[str, dict[str, float]], labels: list[str], k: int) -> dict:
    """Leave one part out: standardise the features on the other parts, take the k nearest
    parts, pick the fixed configuration with the lowest mean FNR on them (the rule) and the
    one with the lowest mean FNR on all other parts (the universal configuration)."""
    parts = list(feats)
    order = sorted(range(len(labels)), key=lambda i: (int(labels[i].split("_")[0][1:]), labels[i].endswith("r90x1"), i))
    out = {}
    for p in parts:
        rest = [q for q in parts if q != p]
        X = np.array([feats[q] for q in rest])
        mu, sd = X.mean(0), X.std(0) + 1e-12
        dist = np.linalg.norm((X - mu) / sd - (feats[p] - mu) / sd, axis=1)
        nn = [rest[i] for i in np.argsort(dist, kind="stable")[:k]]
        pick = lambda group: min(order, key=lambda i: (np.mean([fixed[q][labels[i]] for q in group]), order.index(i)))  # noqa: E731
        out[p] = {"neighbours": nn, "rule": labels[pick(nn)], "universal": labels[pick(rest)]}
    return out


def feature_map(parts: list[str], set_id: str = "mvp", seeds=(0,), budget: int = 40, n_workers: int = 20,
                out: str | None = None, e1_dir: str | None = None, n_perm: int = 10000, k_nn: int = 3, **_) -> Path:
    """부품 형상 특징과 최적 검사 설정의 관계 실험 (part shape features vs optimal configurations).

    1. part features (``isco.eval.part_features``) of every part, written before any evaluation;
    2. the fixed configuration grid (``FIXED_GRID``, 20 configurations) on every part;
    3. own optimum per part: the cross-application experiment's (budget 40, seed 0) where it
       exists, otherwise a search with the same settings;
    4. every optimum applied to every part (report split).
    Hypotheses and decision rules: pre-registration §12 (2026-10-26)."""
    from .part_features import FEATURES, part_descriptors

    d = exp_dir("feature_map", out)
    feats = {p: part_descriptors(p, set_id) for p in parts}
    write_json(d / "features.json", {"parts": parts, "features": feats, "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    s40 = read_json(Path(e1_dir or repo_root() / "runs" / "exp" / "e1") / "summary.json")["best_configs"]
    opt: dict[str, InspectionConfig] = {}
    for p in parts:
        if p in s40:
            opt[p] = InspectionConfig(**s40[p])
            continue
        ev = _ev(p, set_id, n_workers)
        try:
            opt[p] = InspectionConfig(**_best(_search(ev, d, f"search_{p}", "bo_warm", budget, seeds[0]), **CONSTRAINT)["config"])
        finally:
            ev.close()
    fixed: dict[str, dict[str, float]] = {}
    M = np.full((len(parts), len(parts)), np.nan)
    labels: list[str] = []
    for j, p in enumerate(parts):
        ev = _ev(p, set_id, n_workers)
        try:
            grid = fixed_configs(ev.space)
            labels = list(grid)
            fixed[p] = {lab: ev.evaluate(c, seeds[0])["report"]["fnr"] for lab, c in grid.items()}
            for i, q in enumerate(parts):
                M[i, j] = ev.evaluate(opt[q], seeds[0])["report"]["fnr"]
        finally:
            ev.close()
    rng = np.random.default_rng(seeds[0])
    n = len(parts)
    D = M - np.diag(M)[None, :]                                   # D[i, j]: loss on j of i's optimum
    target_loss = {p: float(np.mean(np.delete(D[:, j], j))) for j, p in enumerate(parts)}
    pairs = [(n_, el, rot) for n_, el, rot in FIXED_GRID if rot == "none"]
    rot_adv = {p: float(np.mean([fixed[p][f"n{n_}_el{int(el)}_none"] - fixed[p][f"n{n_}_el{int(el)}_r90x1"]
                                 for n_, el, _ in pairs])) for p in parts}
    spread = {p: float(np.std(list(fixed[p].values()))) for p in parts}
    f = lambda k: [feats[p][k] for p in parts]  # noqa: E731
    tests = {}
    tests["rotation"] = dict(zip(("rho", "p"), _perm_spearman(f("rotation_gain"), [rot_adv[p] for p in parts], n_perm, rng, +1)),
                             threshold=0.5, direction="+", feature="rotation_gain", outcome="rotation_advantage")
    tests["transfer_loss"] = dict(zip(("rho", "p"), _perm_spearman(f("defect_hard_share"), [target_loss[p] for p in parts], n_perm, rng, +1)),
                                  threshold=0.5, direction="+", feature="defect_hard_share", outcome="target_loss")
    tests["sensitivity"] = dict(zip(("rho", "p"), _perm_spearman(f("best_1view"), [spread[p] for p in parts], n_perm, rng, -1)),
                                threshold=-0.5, direction="-", feature="best_1view", outcome="fixed_grid_fnr_sd")
    Z = np.array([[feats[p][k] for k in FEATURES] for p in parts], float)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    dist = np.linalg.norm(Z[:, None] - Z[None], axis=2)
    sym = (D + D.T) / 2
    tests["feature_distance"] = dict(zip(("rho", "p"), _mantel(dist, sym, n_perm, rng)), threshold=0.0, direction="+",
                                     feature="standardised feature distance", outcome="symmetric transfer loss")
    names = list(tests)
    for name, padj in zip(names, _holm([tests[k]["p"] for k in names])):
        t = tests[name]
        t["p_holm"] = float(padj)
        strong = t["rho"] >= t["threshold"] if t["direction"] == "+" else t["rho"] <= t["threshold"]
        t["supported"] = bool(strong and padj < 0.05)
    # practical rule (leave one part out)
    vec = {p: np.array([feats[p][k] for k in FEATURES], float) for p in parts}
    loo = _loo_rule(vec, fixed, labels, k_nn)
    gap_own = np.array([fixed[p][loo[p]["rule"]] - M[j, j] for j, p in enumerate(parts)])
    gain_uni = np.array([fixed[p][loo[p]["universal"]] - fixed[p][loo[p]["rule"]] for p in parts])
    bs = np.array([gain_uni[rng.integers(n, size=n)].mean() for _ in range(n_perm)])
    rule = {"per_part": loo, "gap_to_own_optimum_mean": float(gap_own.mean()), "gain_over_universal_mean": float(gain_uni.mean()),
            "gain_over_universal_ci": [float(x) for x in np.quantile(bs, [0.025, 0.975])]}
    rule["supported"] = bool(rule["gap_to_own_optimum_mean"] <= 0.05 and rule["gain_over_universal_ci"][0] > 0)
    summ = {"parts": parts, "features": feats, "fixed_grid": fixed, "fixed_labels": labels,
            "optima": {p: c.model_dump() for p, c in opt.items()}, "cross_matrix": M.tolist(),
            "target_loss": target_loss, "rotation_advantage": rot_adv, "fixed_grid_fnr_sd": spread,
            "tests": tests, "practical_rule": rule, "n_perm": n_perm, "k_nn": k_nn}
    write_json(d / "summary.json", summ)
    figures.heatmap(D, [f"opt for {p}" for p in parts], [f"on {p}" for p in parts], d / "fig_feature_map_cross.png",
                    title="FNR(B | optimum of A) - FNR(B | own optimum)", label="DeltaFNR")
    return d
