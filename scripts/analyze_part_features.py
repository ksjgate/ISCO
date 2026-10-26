"""Exploratory: which part features go with large cross-application losses and with which optimal configs?

Descriptors come from the nominal-surface visibility tables (2,000 surface samples x 96 slots, the same
ray casts stage 1 uses) and the part spec; losses and optima from the cross-application experiment (E1).

    python scripts/analyze_part_features.py [data_root] [e1_summary] [out_json]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
E1 = Path(sys.argv[2] if len(sys.argv) > 2 else "runs/exp/e1/summary.json")
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else "runs/exp/e1/part_features.json")
P = ["tless_13", "tless_18", "tless_29", "tless_30", "mulsen_nut", "mulsen_flat_pad"]
N_AZ, ELS, DISTS = 12, (20, 40, 60, 80), (0.25, 0.40)


def slot(i_az, i_el, i_dist):
    return (i_dist * 4 + i_el) * 12 + i_az


def rot(k, deg):                                 # SlotGrid.rotate: part +theta == camera -theta
    i_dist, rem = divmod(k, 48)
    i_el, i_az = divmod(rem, 12)
    return slot((i_az - deg // 30) % N_AZ, i_el, i_dist)


def greedy(V, n):
    """Greedy coverage of the reachable samples with n views (V: views x samples)."""
    cov = np.zeros(V.shape[1], bool)
    for _ in range(n):
        gain = (V & ~cov).sum(1)
        cov |= V[int(gain.argmax())]
    return cov


def descriptors(pid):
    z = np.load(ROOT / "visibility" / pid / "mvp" / "sl_default" / "vis.npz")
    V = z["nominal_vis"]                         # 96 x 2000
    spec = json.loads((ROOT / "parts" / "processed" / pid / "part.json").read_text())
    ext = np.sort(np.asarray(spec["extents_m"]))[::-1]
    reach = V.any(0)
    R = V[:, reach]
    n_reach = R.shape[1]
    by_el = [R[[slot(a, e, d) for a in range(N_AZ) for d in range(2)]].any(0).mean() for e in range(4)]
    one = R.sum(1).max() / n_reach                                       # best single view
    rot1 = max((R[k] | R[rot(k, 90)]).sum() for k in range(96)) / n_reach  # one camera + 90 deg turn
    g = [greedy(R, n).mean() for n in (1, 2, 3, 6)]
    redund = R.sum(0).mean() / 96                                        # mean share of views seeing a point
    d = z["vis"]                                                         # defects x 96 x 200
    dvis = d.mean(2)                                                     # per-view visible share of each defect
    # search + report defects only: the bottom controls ("u_*") would put the same floor under every part
    real = np.array([not str(i).startswith("u_") for i in z["inst_ids"]])
    best_view = dvis[real].max(1)
    return {
        "height_ratio": float(ext[2] / ext[0]),                          # flatness: smallest / largest extent
        "reachable": float(reach.mean()),                                # surface seen from >= 1 of 96 slots
        "view_share": float(redund),                                     # how many of the 96 views see an average point
        "best_1view": float(one), "best_1view_r90": float(rot1),
        "rotation_gain": float(rot1 - one),
        "greedy_1_2_3_6": [float(x) for x in g],
        "views_to_90pct": int(next((n for n in range(1, 13) if greedy(R, n).mean() >= 0.9), 13)),
        "by_elevation": [float(x) for x in by_el],
        "top_minus_low": float(by_el[3] - by_el[0]),
        "defect_best_view_median": float(np.median(best_view)),
        # pre-registered definition (§13): the best single view sees less than half of the defect
        "defect_hard_share": float((best_view < 0.5).mean()),
        "defect_hard_share_lt30": float((best_view < 0.3).mean()),       # first (10-26 morning) threshold, controls excluded
    }


def config_features(cfg):
    els = [ELS[(k % 48) // 12] for k in cfg["slots"]]
    return {"n_cams": len(cfg["slots"]), "rotation": cfg["rotation"], "speed_level": cfg["speed_level"],
            "mean_elevation": float(np.mean(els)), "elevations": els,
            "distances": [DISTS[k // 48] for k in cfg["slots"]]}


S = json.loads(E1.read_text())
D = np.array(S["delta_matrix"], float)
loss_as_target = {p: float(np.nanmean(np.delete(D[:, j], j))) for j, p in enumerate(P)}   # others' optima on p
loss_as_source = {p: float(np.nanmean(np.delete(D[i], i))) for i, p in enumerate(P)}      # p's optimum on others
feat = {p: descriptors(p) for p in P}
cfg = {p: config_features(S["best_configs"][p]) for p in P}
res = {"parts": P, "features": feat, "best_config": cfg,
       "loss_as_target_pp": {p: 100 * v for p, v in loss_as_target.items()},
       "loss_as_source_pp": {p: 100 * v for p, v in loss_as_source.items()}}

keys = ["height_ratio", "reachable", "view_share", "best_1view", "rotation_gain", "views_to_90pct",
        "top_minus_low", "defect_best_view_median", "defect_hard_share"]
corr = {}
for k in keys:
    x = [feat[p][k] for p in P]
    corr[k] = {"vs_loss_as_target": float(spearmanr(x, [loss_as_target[p] for p in P])[0]),
               "vs_n_cams": float(spearmanr(x, [cfg[p]["n_cams"] for p in P])[0]),
               "vs_uses_rotation": float(spearmanr(x, [cfg[p]["rotation"] != "none" for p in P])[0])}
res["spearman_n6"] = corr

# pairwise: does a larger feature distance go with a larger transfer loss? (Mantel-style permutation)
Z = np.array([[feat[p][k] for k in keys] for p in P], float)
Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
dist = np.linalg.norm(Z[:, None] - Z[None], axis=2)
iu = [(i, j) for i in range(6) for j in range(6) if i != j]
x = np.array([dist[i, j] for i, j in iu])
y = np.array([D[i, j] for i, j in iu])
r_obs = spearmanr(x, y)[0]
rng = np.random.default_rng(0)
perm = []
for _ in range(20000):
    o = rng.permutation(6)
    perm.append(spearmanr([dist[o[i], o[j]] for i, j in iu], y)[0])
res["feature_distance_vs_loss"] = {"spearman": float(r_obs), "perm_p": float((np.array(perm) >= r_obs).mean())}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(f"{'part':16s} {'loss_tgt':>8s} {'loss_src':>8s} {'h/L':>5s} {'reach':>6s} {'1view':>6s} {'+rot':>6s} {'v90':>4s} "
      f"{'el20/40/60/80':>22s} {'hardDef':>7s}  best config")
for p in P:
    f, c = feat[p], cfg[p]
    print(f"{p:16s} {100 * loss_as_target[p]:+8.1f} {100 * loss_as_source[p]:+8.1f} {f['height_ratio']:5.2f} {f['reachable']:6.3f} "
          f"{f['best_1view']:6.3f} {f['rotation_gain']:+6.3f} {f['views_to_90pct']:4d} "
          f"{' '.join(f'{v:.2f}' for v in f['by_elevation']):>22s} {f['defect_hard_share']:7.2f}  "
          f"{c['n_cams']} cam el {c['elevations']} {c['rotation']} v{c['speed_level']}")
print("spearman (n=6):")
for k, v in corr.items():
    print(f"  {k:24s} loss_tgt {v['vs_loss_as_target']:+.2f}  n_cams {v['vs_n_cams']:+.2f}  rotation {v['vs_uses_rotation']:+.2f}")
print("feature distance vs transfer loss:", res["feature_distance_vs_loss"])
print(f"[part_features] {OUT}")
