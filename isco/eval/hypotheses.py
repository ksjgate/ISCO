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
from ..utils.io import read_jsonl, write_json
from ..utils.logging import get_logger
from .mvtec3d import e0_6, e5b  # noqa: F401  (CLI entry points)
from .sanity import e0_2, e0_3, e0_4, e0_5, exp_dir, reference_config  # noqa: F401  (CLI entry points)

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


def _kendall(a, b) -> float:
    from scipy.stats import kendalltau

    return float(kendalltau(a, b).statistic)


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
            b = recommend(rows, **CONSTRAINT)
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
                opt[proc] = recommend(rows, **cons)
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
        fr = np.array([r["report"]["fnr"] for r in real])
        tau = _kendall(fs, fr)
        overlap = len(set(np.argsort(fs)[:5]) & set(np.argsort(fr)[:5]))
        summ[pid] = {"kendall_tau": tau, "top5_overlap": overlap, "n_configs": len(cfgs),
                     "fnr_synthetic": fs.tolist(), "fnr_real": fr.tolist(),
                     "supported_for_part": bool(tau >= 0.5 and overlap >= 3)}
        log.info(f"E5 {pid}: Kendall tau {tau:.2f}, top-5 overlap {overlap}")
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
                b = recommend(rows, **CONSTRAINT)
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
