"""Report / paper figures (matplotlib, PNG). Conventions (data-viz method): one y-axis
per chart, thin marks, recessive grid, legend for >= 2 series, categorical colours in
fixed order, ordinal blue ramp for ordered levels (validated 5-step ramp + marker shape
for camera count), diverging blue-gray-red for signed differences."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import style

#: validated 5-step ordinal ramp (light -> dark); camera counts >= 5 share the darkest
ORDINAL5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
NCAM_MARKER = {1: "o", 2: "s", 3: "^", 4: "D", 5: "P", 6: "X"}


def _ncam_color(n: int) -> str:
    return ORDINAL5[min(max(n, 1), 5) - 1]


def _save(fig, out: str | Path) -> str:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(out)


def pareto_plot(rows: list[dict], front_ids: set[str], out: str | Path, use_report: bool = True, title: str = "") -> str:
    """(a) FNR vs cycle time, (b) FNR vs camera count; Pareto configurations ringed."""
    style.apply()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    key = "objectives_report" if use_report else "objectives"
    uniq = {r["config_id"]: r for r in rows if r.get("stage") == 2 and r.get(key)}
    for r in uniq.values():
        o = r[key]
        n = o["n_cam"]
        on = r["config_id"] in front_ids
        kw = dict(color=_ncam_color(n), marker=NCAM_MARKER.get(n, "o"), s=46 if on else 24, zorder=3 if on else 2,
                  edgecolors=style.TEXT if on else style.SURFACE, linewidths=1.2 if on else 0.6)
        axes[0].scatter(o["cycle_time_s"], o["fnr"], **kw)
        axes[1].scatter(n + np.random.default_rng(hash(r["config_id"]) % 2**32).uniform(-0.15, 0.15), o["fnr"], **kw)
    front = sorted((uniq[i][key] for i in front_ids if i in uniq), key=lambda o: o["cycle_time_s"])
    if front:
        axes[0].step([o["cycle_time_s"] for o in front], [o["fnr"] for o in front], where="post", color=style.TEXT_2,
                     linewidth=1.0, zorder=1)
        for o in front:
            axes[0].annotate(f"{o['n_cam']} cam", (o["cycle_time_s"], o["fnr"]), textcoords="offset points",
                             xytext=(5, 4), fontsize=7, color=style.TEXT_2)
    axes[0].set_xlabel("cycle time per part (s)")
    axes[0].set_ylabel("FNR @ 1% FPR" + (" (held-out report set)" if use_report else " (search set)"))
    axes[1].set_xlabel("number of cameras")
    axes[1].set_xticks(range(1, 7))
    handles = [plt.Line2D([], [], marker=NCAM_MARKER[n], color=_ncam_color(n), linestyle="", markersize=6,
                          label=f"{n} cam") for n in range(1, 7)]
    handles.append(plt.Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=style.TEXT, linestyle="",
                              label="Pareto"))
    axes[1].legend(handles=handles, loc="upper right", ncol=2)
    for ax in axes:
        ax.set_ylim(-0.03, 1.03)
    if title:
        fig.suptitle(title, fontsize=11)
    return _save(fig, out)


def stage1_plot(cands: list[dict], out: str | Path, metric: str = "V_area") -> str:
    style.apply()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for n in range(1, 7):
        xs = [c["surrogate"][metric] for c in cands if c["n_cam"] == n]
        if xs:
            ax.scatter(np.full(len(xs), n) + np.random.default_rng(n).uniform(-0.25, 0.25, len(xs)), xs, s=6,
                       color=_ncam_color(n), alpha=0.6, linewidths=0)
    ax.set_xlabel("number of cameras")
    ax.set_ylabel(f"stage-1 surrogate {metric}")
    ax.set_xticks(range(1, 7))
    ax.set_title(f"Stage 1: {len(cands)} candidate configurations", loc="left")
    return _save(fig, out)


def per_type_bars(rec: dict, out: str | Path) -> str:
    """Per defect type: visibility (surrogate V) and detection rate (1 - FNR) of one config."""
    style.apply()
    import matplotlib.pyplot as plt

    defs = (rec.get("report_defects") or []) + (rec.get("search_defects") or [])
    types = sorted({d["type"] for d in defs})
    val = np.asarray(rec.get("val_scores") or [])
    from ..detect.scorer import threshold_at_fpr

    tau = threshold_at_fpr(val, 0.01) if len(val) else np.inf
    vis = [np.nanmean([d["V"] for d in defs if d["type"] == t]) for t in types]
    det = [np.mean([d["score"] > tau for d in defs if d["type"] == t]) for t in types]
    x = np.arange(len(types))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    w = 0.38
    b1 = ax.bar(x - w / 2 - 0.01, vis, w, color=style.CAT[0], label="visibility V (surrogate)")
    b2 = ax.bar(x + w / 2 + 0.01, det, w, color=style.CAT[1], label="detection rate @1% FPR")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center",
                        va="bottom", fontsize=7, color=style.TEXT_2, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, types)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("fraction")
    ax.legend(loc="upper left", ncol=2)
    return _save(fig, out)


def rig_plot(cfg: dict, grid, focus: np.ndarray, mesh_vertices: np.ndarray, mesh_faces: np.ndarray,
             out: str | Path, title: str = "") -> str:
    """3D layout: part mesh at the focus point and the configured cameras (rotation-stage
    virtual views drawn hollow)."""
    style.apply()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from ..schemas.config import InspectionConfig
    from ..visibility.surrogate import config_views

    c = InspectionConfig(**cfg)
    fig = plt.figure(figsize=(5.4, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    sub = mesh_faces[:: max(1, len(mesh_faces) // 4000)]
    ax.add_collection3d(Poly3DCollection(mesh_vertices[sub], facecolor="#c9c8c2", edgecolor="none", alpha=0.6))
    for cam_slot, cache_slot, th in config_views(c, grid):
        e = grid.eye(cache_slot, focus)
        real = th == 0.0
        col = style.CAT[0] if real else style.MUTED
        ax.scatter(*e, color=col if real else "none", edgecolors=col, s=40, depthshade=False)
        ax.plot(*np.stack([e, focus]).T, color=col, linewidth=0.8, linestyle="-" if real else ":")
    r = max(grid.distances_m) * 1.05
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(0, r)
    ax.set_box_aspect((1, 1, 0.5))
    ax.set_xlabel("x (belt, m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title or f"{c.n_cam} cameras, rotation {c.rotation}", fontsize=10)
    return _save(fig, out)


def vis_vs_fnr(rows: list[dict], out: str | Path, metric: str = "V_area", use_report: bool = True) -> tuple[str, float]:
    """Stage-1 surrogate vs stage-2 FNR over evaluated configurations (H2) + Spearman rho."""
    style.apply()
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    key = "report" if use_report else "search"
    uniq = {r["config_id"]: r for r in rows if r.get("stage") == 2 and r.get(key)}
    x = np.array([r["surrogate"][metric] for r in uniq.values()])
    y = np.array([r[key]["fnr"] for r in uniq.values()])
    n = np.array([r["n_cam"] for r in uniq.values()])
    rho = float(spearmanr(x, y).statistic) if (len(x) > 2 and np.std(x) > 0 and np.std(y) > 0) else float("nan")
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    for k in sorted(set(n)):
        m = n == k
        ax.scatter(x[m], y[m], color=_ncam_color(k), marker=NCAM_MARKER.get(k, "o"), s=26, edgecolors=style.SURFACE,
                   linewidths=0.6, label=f"{k} cam")
    ax.set_xlabel(f"stage-1 visibility surrogate ({metric})")
    ax.set_ylabel(f"FNR @ 1% FPR ({key} set)")
    ax.set_title(f"Spearman rho = {rho:.2f} (n = {len(x)})", loc="left")
    if len(set(n)) > 1:
        ax.legend(loc="upper right", ncol=2)
    return _save(fig, out), rho


def hv_curves(curves: dict[str, np.ndarray], out: str | Path, title: str = "") -> str:
    """Hypervolume vs number of stage-2 evaluations per search method (E8)."""
    style.apply()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for i, (name, c) in enumerate(curves.items()):
        c = np.asarray(c)
        if c.ndim == 1:
            ax.plot(np.arange(1, len(c) + 1), c, color=style.CAT[i], label=name)
        else:   # seeds x evals -> median + IQR
            med = np.median(c, axis=0)
            ax.plot(np.arange(1, c.shape[1] + 1), med, color=style.CAT[i], label=name)
            ax.fill_between(np.arange(1, c.shape[1] + 1), np.quantile(c, 0.25, axis=0), np.quantile(c, 0.75, axis=0),
                            color=style.CAT[i], alpha=0.15, linewidth=0)
    ax.set_xlabel("stage-2 evaluations")
    ax.set_ylabel("hypervolume")
    ax.legend(loc="lower right")
    if title:
        ax.set_title(title, loc="left")
    return _save(fig, out)


def heatmap(M: np.ndarray, rows: list[str], cols: list[str], out: str | Path, title: str = "",
            diverging: bool = True, fmt: str = "{:+.2f}", label: str = "") -> str:
    style.apply()
    import matplotlib.pyplot as plt

    M = np.asarray(M, dtype=float)
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(cols), 0.8 + 0.6 * len(rows)))
    if diverging:
        lim = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
        im = ax.imshow(M, cmap=style.diverging_cmap(), vmin=-lim, vmax=lim)
    else:
        im = ax.imshow(M, cmap=style.sequential_cmap())
    ax.grid(False)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7, color=style.TEXT)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(label, fontsize=8)
    if title:
        ax.set_title(title, loc="left")
    return _save(fig, out)


def lines_by_n(series: dict[str, dict[int, float]], out: str | Path, ylabel: str, title: str = "") -> str:
    """Best metric vs camera count for several methods (E3)."""
    style.apply()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for i, (name, d) in enumerate(series.items()):
        ks = sorted(d)
        ax.plot(ks, [d[k] for k in ks], color=style.CAT[i], marker="o", markersize=5, label=name)
    ax.set_xlabel("number of cameras")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, 7))
    ax.legend(loc="upper right")
    if title:
        ax.set_title(title, loc="left")
    return _save(fig, out)
