"""Mid-term demo assets (plan Sprint 1 item 7, "최소 데모"):

* FNR / cost table: 3 Fusion parts (S1 bar 22340, S2 plate 41229, C1 bracket 51876) x
  baseline rigs (top 1 cam, ring 3 cams @40 deg, ring 6 cams @40 deg)
* defect examples: one instance per defect type, embree depth render with the defect
  pixels highlighted
* visibility heat map: nominal surface samples coloured by how many cameras of a rig see
  them (the PoC coverage figure, now driven by the stage-1 tables)
* Isaac RGB render of a rig (if the worker is running; ``--rgb``)

    python scripts/make_demo_assets.py [--rgb] [--rig-run runs/demo_fusion_51876]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from isco import config_loader as C
from isco.datagen.parts import load_part, load_part_mesh
from isco.datagen.variants import load_manifest
from isco.geometry.mesh_io import LabeledMesh
from isco.optim.evaluator import ConfigEvaluator
from isco.render.occluder import Occluder
from isco.render.raycast_backend import RaycastRenderer
from isco.render.slot_cache import SlotCache
from isco.report import style
from isco.schemas.config import InspectionConfig
from isco.utils.io import repo_root, resolve_data_path, write_json
from isco.utils.seeds import rng_for
from isco.visibility.raycast import sample_faces
from isco.visibility.surrogate import VisibilityTable, config_views

PARTS = {"fusion_22340": "S1 bar", "fusion_41229": "S2 thin plate", "fusion_51876": "C1 bracket"}
BASELINES = ("top_1cam", "ring3_el40", "ring6_el40")


def baseline_table(out: Path, workers: int) -> list[dict]:
    rows = []
    for pid, name in PARTS.items():
        ev = ConfigEvaluator(pid, "mvp", C.detector("fpfh_bank"), C.process("inline_default"), C.noise("N2"),
                             n_workers=workers)
        try:
            base = ev.space.baselines()
            for b in BASELINES:
                rec = ev.evaluate(base[b])
                rows.append({"part": pid, "shape": name, "rig": b, "slots": rec["config"]["slots"], "n_cam": rec["n_cam"],
                             "cycle_time_s": rec["objectives"]["cycle_time_s"], "fnr_report": rec["report"]["fnr"],
                             "fnr_ci": rec["report"]["fnr_ci"], "auroc": rec["report"]["auroc"],
                             "V_area": rec["surrogate"]["V_area"], "fnr_by_type": rec["report"]["fnr_by_type"]})
        finally:
            ev.close()
    write_json(out / "baselines.json", rows)
    md = ["| part | rig | cameras | cycle (s) | FNR@1%FPR (95% CI) | AUROC | visibility V |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['part']} ({r['shape']}) | {r['rig']} | {r['n_cam']} | {r['cycle_time_s']:.1f} | "
                  f"{r['fnr_report']:.2f} ({r['fnr_ci'][0]:.2f}-{r['fnr_ci'][1]:.2f}) | {r['auroc']:.2f} | {r['V_area']:.2f} |")
    (out / "baselines.md").write_text("\n".join(md) + "\n")
    # grouped bar chart: FNR per part, one bar per rig
    style.apply()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    parts = list(PARTS)
    w = 0.26
    for i, b in enumerate(BASELINES):
        vals = [next(r["fnr_report"] for r in rows if r["part"] == p and r["rig"] == b) for p in parts]
        bars = ax.bar(np.arange(len(parts)) + (i - 1) * (w + 0.01), vals, w, color=style.CAT[i], label=b)
        for bb, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (bb.get_x() + bb.get_width() / 2, v), ha="center", va="bottom", fontsize=7,
                        color=style.TEXT_2, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(np.arange(len(parts)), [f"{p}\n{PARTS[p]}" for p in parts])
    ax.set_ylabel("FNR @ 1% FPR (held-out)")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", ncol=3)
    fig.savefig(out / "baselines.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return rows


def defect_examples(pid: str, out: Path) -> None:
    style.apply()
    import matplotlib.pyplot as plt

    man = load_manifest(pid, "mvp")
    cache = SlotCache.open(pid, "mvp")
    cam = cache.camera
    rr = RaycastRenderer(cam, Occluder.default())
    types = ["dent", "bulge", "hole", "cut", "crack", "warp"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
    for ax, t in zip(axes.ravel(), types):
        recs = [r for r in man["instances"] if r["type"] == t and r["role"] in ("report", "search")]
        r = max(recs, key=lambda x: (x.get("tier") == "L", (x.get("defect") or {}).get("k_visible") or 0))
        mesh = LabeledMesh.load(resolve_data_path(r["mesh"]))
        T = np.asarray(r["T_world_part"])
        # pick the slot that sees most of the defect
        best_k, best_n = 12, -1
        for k in range(0, cache.grid.n_slots, 3):
            vp = cache.views(r["inst_id"], [k])[k]
            n = int((vp.label >= 2).sum())
            if n > best_n:
                best_k, best_n = k, n
        T_wc = cache.T_world_cam(best_k)
        roi = tuple(next(s["roi"] for s in cache.meta["slots"] if s["slot_id"] == best_k))
        depth, lab = rr.render_full_depth(mesh, T, T_wc, roi)
        z = np.where(lab > 0, depth, np.nan)
        lo, hi = np.nanpercentile(z, 1), np.nanpercentile(z, 99)
        g = np.clip((hi - z) / max(hi - lo, 1e-9), 0, 1)
        rgb = np.dstack([g, g, g])
        rgb[np.isnan(z)] = 1.0
        rgb[lab >= 2] = 0.65 * rgb[lab >= 2] + 0.35 * np.array([0.89, 0.29, 0.28])
        ax.imshow(rgb)
        d = r["defect"]
        if t == "hole" and d.get("through"):
            desc = f"through, dia {d['diameter_m'] * 1e3:.1f} mm"
        elif t == "warp":
            desc = f"sagitta {abs(d['sagitta_m']) * 1e3:.2f} mm"
        else:
            desc = f"depth {abs(d.get('depth_m') or 0) * 1e3:.2f} mm"
        ax.set_title(f"{t} ({r.get('tier')}, {d.get('difficulty', r.get('difficulty'))}), {desc}", fontsize=9, loc="left")
        ax.set_axis_off()
    fig.suptitle(f"{pid}: synthetic defects (red = ground-truth defect pixels)", fontsize=11)
    fig.savefig(out / f"defect_examples_{pid}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def visibility_heatmap(pid: str, cfg: InspectionConfig, out: Path, tag: str) -> None:
    style.apply()
    import matplotlib.pyplot as plt

    vt = VisibilityTable.for_set(pid, "mvp")
    nom = load_part_mesh(pid, "fine")
    pts, _ = sample_faces(nom, np.ones(len(nom.faces), bool), 2000, rng_for("vis-nominal", pid))
    views = sorted({v for _, v, _ in config_views(cfg, vt.grid)})
    cnt = vt.nominal_ok[views].sum(axis=0)
    fig = plt.figure(figsize=(10, 4.4))
    from isco.report.figures import ORDINAL5

    cols = ["#d9d8d3", ORDINAL5[0], ORDINAL5[2], ORDINAL5[4]]
    for i, (el, az) in enumerate(((60, -60), (-60, -60))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        for c in range(4):
            m = np.minimum(cnt, 3) == c
            ax.scatter(*pts[m].T * 1e3, s=5, color=cols[c], depthshade=False,
                       label=["unseen", "1 view", "2 views", "3+ views"][c])
        ax.view_init(el, az)
        ax.set_box_aspect(np.ptp(pts, axis=0) + 1e-6)
        ax.set_axis_off()
        if i == 0:
            ax.legend(loc="upper left", fontsize=8, markerscale=3)
    fig.suptitle(f"{pid}: nominal-surface coverage of rig {cfg.slots} rotation {cfg.rotation} "
                 f"({(cnt > 0).mean() * 100:.0f}% seen; left: from above, right: from below)", fontsize=10)
    fig.savefig(out / f"coverage_{pid}_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def isaac_rgb(pid: str, cfg: InspectionConfig, out: Path) -> dict:
    from isco.render.queue import FileJobQueue
    from isco.schemas.config import CameraSpec
    from isco.schemas.job import CaptureRgbPayload, Job

    q = FileJobQueue(repo_root() / "queue")
    if not q.worker_alive():
        return {"status": "worker not running"}
    spec = load_part(pid)
    grid = C.grid()
    cams = [CameraSpec(azimuth_deg=grid.slot(k).azimuth_deg, elevation_deg=grid.slot(k).elevation_deg,
                       distance_m=grid.slot(k).distance_m, slot_id=k) for k in cfg.slots]
    job = Job.make("capture_rgb", CaptureRgbPayload(part_id=pid, mesh_path=str(resolve_data_path(spec.mesh_fine)),
                                                   cameras=cams, sensor=C.sensor(), focus_m=spec.focus_m,
                                                   output_dir=str(out / f"isaac_rgb_{pid}")))
    q.submit(job)
    return json.loads(q.wait([job.job_id], timeout_s=1200)[job.job_id].model_dump_json())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(repo_root() / "runs" / "demo_assets"))
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--rgb", action="store_true")
    ap.add_argument("--rig-run", default=str(repo_root() / "runs" / "demo_fusion_51876"))
    ap.add_argument("--skip-table", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.skip_table:
        baseline_table(out, a.workers)
    for pid in PARTS:
        defect_examples(pid, out)
    rec_p = Path(a.rig_run) / "report" / "recommended.json"
    grid = C.grid()
    ring3 = InspectionConfig(slots=[grid.nearest_slot(az, 40, 0.25) for az in (0, 120, 240)])
    for pid in PARTS:
        visibility_heatmap(pid, ring3, out, "ring3_el40")
    if rec_p.exists():
        rec = json.loads(rec_p.read_text())
        cfg = InspectionConfig(**rec["config"])
        visibility_heatmap("fusion_51876", cfg, out, "recommended")
        if a.rgb:
            write_json(out / "isaac_rgb.json", isaac_rgb("fusion_51876", cfg, out))
    print(f"demo assets in {out}")


if __name__ == "__main__":
    main()
