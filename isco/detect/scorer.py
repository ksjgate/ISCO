"""Operating-point metrics (plan "지표").

* ``threshold_at_fpr``: tau = empirical (1-FPR) quantile of the validation normal scores
  (numpy "linear" = Hyndman-Fan type 7); optional ``"gpd"`` tail fit (POT/SPOT style,
  protocol v2 refinement) for small N_val.
* ``fnr_at_fpr``: P(defect score <= tau).
* ``bootstrap_fnr``: joint bootstrap (resample val -> re-estimate tau, resample defects).
* ``paired_bootstrap_diff``: CI of FNR(a) - FNR(b) on the same instances (CRN).
* ``auroc``, ``pauroc`` (FPR <= max_fpr, McClish-standardised), ``aupro`` (point level,
  restricted to observed points, FPR integrated to 0.3 and normalised).
"""
from __future__ import annotations

import numpy as np


def threshold_at_fpr(val_scores: np.ndarray, fpr: float = 0.01, method: str = "type7") -> float:
    s = np.asarray(val_scores, dtype=np.float64)
    if len(s) == 0:
        return np.inf
    if method == "type7":
        return float(np.quantile(s, 1.0 - fpr, method="linear"))
    if method == "gpd":
        from scipy.stats import genpareto

        u = np.quantile(s, 0.8)
        exc = s[s > u] - u
        if len(exc) < 10:
            return float(np.quantile(s, 1.0 - fpr, method="linear"))
        c, loc, scale = genpareto.fit(exc, floc=0.0)
        p_u = len(exc) / len(s)
        return float(u + genpareto.ppf(1 - fpr / p_u, c, loc=0.0, scale=scale))
    raise ValueError(method)


def fnr_at_fpr(val_scores, defect_scores, fpr: float = 0.01, method: str = "type7") -> float:
    d = np.asarray(defect_scores)
    if len(d) == 0:
        return float("nan")
    tau = threshold_at_fpr(val_scores, fpr, method)
    return float(np.mean(d <= tau))


def bootstrap_fnr(val_scores, defect_scores, fpr: float = 0.01, B: int = 1000, seed: int = 0,
                  method: str = "type7") -> tuple[float, float, float]:
    v = np.asarray(val_scores)
    d = np.asarray(defect_scores)
    if len(d) == 0 or len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        vb = v[rng.integers(len(v), size=len(v))]
        db = d[rng.integers(len(d), size=len(d))]
        vals[b] = np.mean(db <= threshold_at_fpr(vb, fpr, method))
    return fnr_at_fpr(v, d, fpr, method), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def paired_bootstrap_diff(val_a, def_a, val_b, def_b, fpr: float = 0.01, B: int = 1000, seed: int = 0):
    """CI of FNR_a - FNR_b where a and b are scored on the same instances (same order)."""
    va, da, vb, db = map(np.asarray, (val_a, def_a, val_b, def_b))
    rng = np.random.default_rng(seed)
    diffs = np.empty(B)
    for i in range(B):
        iv = rng.integers(len(va), size=len(va))
        idd = rng.integers(len(da), size=len(da))
        diffs[i] = np.mean(da[idd] <= threshold_at_fpr(va[iv], fpr)) - np.mean(db[idd] <= threshold_at_fpr(vb[iv], fpr))
    point = fnr_at_fpr(va, da, fpr) - fnr_at_fpr(vb, db, fpr)
    return float(point), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def auroc(normal_scores, defect_scores) -> float:
    from sklearn.metrics import roc_auc_score

    n, d = np.asarray(normal_scores), np.asarray(defect_scores)
    if len(n) == 0 or len(d) == 0:
        return float("nan")
    y = np.r_[np.zeros(len(n)), np.ones(len(d))]
    return float(roc_auc_score(y, np.r_[n, d]))


def pauroc(normal_scores, defect_scores, max_fpr: float = 0.1) -> float:
    from sklearn.metrics import roc_auc_score

    n, d = np.asarray(normal_scores), np.asarray(defect_scores)
    if len(n) == 0 or len(d) == 0:
        return float("nan")
    y = np.r_[np.zeros(len(n)), np.ones(len(d))]
    return float(roc_auc_score(y, np.r_[n, d], max_fpr=max_fpr))


def aupro(defect_point_scores: list[np.ndarray], defect_point_masks: list[np.ndarray],
          normal_point_scores: np.ndarray, max_fpr: float = 0.3, n_thresholds: int = 400) -> float:
    """Area under the per-region-overlap curve (one region = the defect points of one
    instance; a point is flagged when its score >= t), FPR measured on normal points,
    integrated from 0 to ``max_fpr`` (linear interpolation at the end) and normalised.
    Thresholds are taken from the union of all scores, so ties are handled."""
    regions = [np.asarray(s)[m] for s, m in zip(defect_point_scores, defect_point_masks) if np.any(m)]
    ns = np.asarray(normal_point_scores, dtype=float)
    if not regions or len(ns) == 0:
        return float("nan")
    allv = np.unique(np.concatenate([ns] + regions))
    if len(allv) > n_thresholds:
        allv = np.unique(np.quantile(allv, np.linspace(0, 1, n_thresholds)))
    th = allv[::-1]                                         # descending
    fpr = np.array([np.mean(ns >= t) for t in th])
    pro = np.array([np.mean([np.mean(r >= t) for r in regions]) for t in th])
    fpr = np.r_[0.0, fpr]
    pro = np.r_[0.0, pro]
    # keep the curve up to max_fpr, interpolating the crossing point
    keep = fpr <= max_fpr
    f, p = fpr[keep], pro[keep]
    if (~keep).any():
        i = int(np.argmax(~keep))
        f0, f1, p0, p1 = fpr[i - 1], fpr[i], pro[i - 1], pro[i]
        pe = p0 + (p1 - p0) * (max_fpr - f0) / max(f1 - f0, 1e-12)
        f, p = np.r_[f, max_fpr], np.r_[p, pe]
    else:
        f, p = np.r_[f, max_fpr], np.r_[p, p[-1]]
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trap(p, f) / max_fpr)


def summarize(val_scores, defects: list[dict], fpr: float = 0.01, B: int = 500, seed: int = 0,
              vis_threshold: float = 0.1) -> dict:
    """Metrics over a list of defect records ``{score, type, difficulty, V}``."""
    v = np.asarray(val_scores)
    ds = np.asarray([d["score"] for d in defects])
    out: dict = {}
    f, lo, hi = bootstrap_fnr(v, ds, fpr, B, seed)
    out["fnr"] = f
    out["fnr_ci"] = [lo, hi]
    out["fnr_5"] = fnr_at_fpr(v, ds, 0.05)
    out["fnr_gpd"] = fnr_at_fpr(v, ds, fpr, "gpd")
    out["tau"] = threshold_at_fpr(v, fpr)
    out["auroc"] = auroc(v, ds)
    out["pauroc"] = pauroc(v, ds, 0.1)
    vis = np.asarray([d.get("V", np.nan) for d in defects], dtype=float)
    m = np.isfinite(vis) & (vis >= vis_threshold)
    out["fnr_vis"] = fnr_at_fpr(v, ds[m]) if m.any() else float("nan")
    out["n_vis"] = int(m.sum())
    for key in ("type", "difficulty"):
        groups = sorted({d.get(key) for d in defects if d.get(key) is not None})
        out[f"fnr_by_{key}"] = {g: fnr_at_fpr(v, ds[[d.get(key) == g for d in defects]]) for g in groups}
    out["n_defects"] = int(len(ds))
    out["n_val"] = int(len(v))
    return out
