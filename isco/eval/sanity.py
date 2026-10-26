"""E0 sanity / calibration experiments (plan "실험 목록" E0.x).

* ``e0_2``  slot-cache approximation vs direct Isaac inline render (needs the worker):
            per-view Chamfer between the system's interpretation of a direct render
            (part physically shifted by the trigger error, rotated on the turntable) and
            the cache emulation (azimuth-shifted cached view + rigid perturbation);
            plus completeness-vs-surrogate agreement over random configurations.
* ``e0_3``  GT defect-mask sanity: labelled area vs nominal area per defect type.
* ``e0_4``  timing: cache build per view, stage-2 evaluation per configuration.
* ``e0_5``  severity calibration: reference configuration x noise N1/N2/N3 -> FNR by
            tier/type; target FNR(N2) in [0.2, 0.6] (protocol D3), recommended scale_k.
(E0.1 Isaac<->embree equivalence: ``scripts/e01_isaac_vs_embree.py``.)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .. import config_loader as C
from ..schemas.config import InspectionConfig
from ..utils.io import read_json, repo_root, write_json
from ..utils.logging import get_logger

log = get_logger(__name__)


def exp_dir(name: str, out: str | None = None) -> Path:
    d = Path(out) if out else repo_root() / "runs" / "exp" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def reference_config(grid=None, speed_level: int = 1) -> InspectionConfig:
    """Protocol reference: 3 cameras at elevation 40 deg, azimuth 0/120/240, 0.25 m."""
    grid = grid or C.grid()
    slots = [grid.nearest_slot(az, 40.0, 0.25) for az in (0.0, 120.0, 240.0)]
    return InspectionConfig(slots=slots, rotation="none", speed_level=speed_level, trigger_mode="simultaneous")


def _md_table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------------------- E0.2
def e0_2(parts: list[str], set_id: str = "mvp", n_configs: int = 20, n_instances: int = 4, seed: int = 0,
         out: str | None = None, **_) -> Path:
    """Cache approximation vs direct Isaac inline render (worker must be running)."""
    from scipy.spatial import cKDTree

    from ..datagen.parts import load_part
    from ..datagen.variants import load_manifest
    from ..geometry.transforms import apply_T, rot_z
    from ..render.pixels import ViewPixels
    from ..render.queue import FileJobQueue
    from ..render.slot_cache import SlotCache
    from ..schemas.config import CameraSpec
    from ..schemas.job import InstanceRef, Job, VerifyConfigPayload
    from ..utils.io import resolve_data_path
    from ..optim.space import SearchSpace
    from ..render.occluder import Occluder

    d = exp_dir("e0_2", out)
    q = FileJobQueue(repo_root() / "queue")
    for _ in range(300):                       # the worker needs ~10-20 s to come up
        if q.worker_alive():
            break
        time.sleep(1)
    if not q.worker_alive():
        raise RuntimeError("E0.2 needs the Isaac worker: `isco worker start` (or scripts/start_worker.sh)")
    process = C.process("inline_default")
    rows = []
    rng = np.random.default_rng(seed)
    for pid in parts:
        cache = SlotCache.open(pid, set_id)
        grid, cam = cache.grid, cache.camera
        spec = load_part(pid)
        man = load_manifest(pid, set_id)
        insts = [r for r in man["instances"] if r["role"] in ("val", "report")]
        insts = [insts[i] for i in rng.choice(len(insts), n_instances, replace=False)]
        space = SearchSpace(grid, np.asarray(spec.focus_m), Occluder.default())
        for ci in range(n_configs):
            cfg = space.random_config(rng)
            cams = [CameraSpec(azimuth_deg=grid.slot(k).azimuth_deg, elevation_deg=grid.slot(k).elevation_deg,
                               distance_m=grid.slot(k).distance_m, slot_id=k) for k in cfg.slots]
            odir = d / "direct" / pid / cfg.config_id
            payload = VerifyConfigPayload(part_id=pid, config=cfg, cameras=cams, process=process,
                                          instances=[InstanceRef(inst_id=r["inst_id"], mesh_path=str(resolve_data_path(r["mesh"])),
                                                                 T_world_part=r["T_world_part"]) for r in insts],
                                          sensor=cache.sensor, focus_m=spec.focus_m, output_dir=str(odir), seed=seed + ci)
            job = Job.make("verify_config", payload)
            q.submit(job)
            res = q.wait([job.job_id], timeout_s=3600)[job.job_id]
            if res.status != "ok":
                log.warning(f"verify_config failed: {res.error}")
                continue
            for r in insts:
                offs = read_json(odir / r["inst_id"] / "offsets.json")
                for v in offs["views"]:
                    s_idx, th, c = v["stage"], v["stage_deg"], v["cam"]
                    vp = ViewPixels.load(odir / r["inst_id"] / f"stage_{s_idx}_cam_{c}.npz")
                    T_wc = np.asarray(vp.meta["T_world_cam"])
                    P_direct = apply_T(T_wc, cam.unproject(vp.pix, vp.depth))
                    P_sys = P_direct @ rot_z(-th).T                     # system undoes the nominal turn
                    k_cache = grid.rotate(cfg.slots[c], th)
                    cv = cache.views(r["inst_id"], [k_cache])[k_cache]
                    Q = apply_T(cache.T_world_cam(k_cache), cam.unproject(cv.pix, cv.depth))
                    shift = rot_z(-th) @ (np.array([v["dx_m"], 0, 0]) + np.r_[v["reposition_xy_m"], 0.0])
                    Q_emul = Q @ rot_z(v["dyaw_deg"]).T + shift
                    if len(P_sys) < 50 or len(Q_emul) < 50:
                        continue
                    d1, _ = cKDTree(Q_emul).query(P_sys)
                    d2, _ = cKDTree(P_sys).query(Q_emul)
                    rows.append({"part": pid, "config_id": cfg.config_id, "inst": r["inst_id"], "stage_deg": th,
                                 "trigger": cfg.trigger_mode, "dx_mm": v["dx_m"] * 1e3,
                                 "chamfer_med_mm": float(0.5 * (np.median(d1) + np.median(d2)) * 1e3),
                                 "chamfer_p95_mm": float(0.5 * (np.quantile(d1, 0.95) + np.quantile(d2, 0.95)) * 1e3),
                                 "n_direct": int(len(P_sys)), "n_cache": int(len(Q_emul)),
                                 "px_ratio": float(len(P_sys) / max(len(Q_emul), 1))})
            log.info(f"E0.2 {pid} config {ci + 1}/{n_configs}: {len(rows)} views compared")
    write_json(d / "views.json", rows)
    med = np.array([r["chamfer_med_mm"] for r in rows])
    p95 = np.array([r["chamfer_p95_mm"] for r in rows])
    summ = {"n_views": len(rows), "chamfer_med_mm_median": float(np.median(med)) if len(med) else None,
            "chamfer_med_mm_max": float(med.max()) if len(med) else None,
            "chamfer_p95_mm_median": float(np.median(p95)) if len(p95) else None,
            "px_ratio_median": float(np.median([r["px_ratio"] for r in rows])) if rows else None,
            "sigma_z_N2_mm": 0.15, "pass": bool(len(med) and np.median(med) < 0.15)}
    write_json(d / "summary.json", summ)
    (d / "summary.md").write_text("# E0.2 slot-cache approximation vs direct Isaac inline render\n\n" +
                                  "\n".join(f"- {k}: {v}" for k, v in summ.items()) + "\n")
    return d


# ------------------------------------------------------------------------------- E0.3
def _nominal_area(meta: dict) -> float | None:
    t = meta.get("type")
    if t in ("dent", "bulge"):
        R = meta["diameter_m"] / 2.0
        # labelled where the (1-(r/R)^2)^2 profile exceeds 10 % of the height -> r < 0.827 R
        return float(np.pi * (0.8269 * R) ** 2 * meta.get("aspect", 1.0))
    if t == "hole" and not meta.get("through", False) and meta.get("depth_m"):
        D = meta["diameter_m"]
        return float(np.pi * D * meta["depth_m"] + np.pi * D * D / 4)
    if t == "crack":
        L, w, h = meta["length_m"], meta["width_m"], meta["depth_m"]
        return float(2 * L * h + 2 * w * h + L * w)
    return None


def e0_3(parts: list[str], set_id: str = "mvp", out: str | None = None, **_) -> Path:
    from ..datagen.variants import load_manifest
    from ..geometry.mesh_io import LabeledMesh
    from ..utils.io import resolve_data_path

    d = exp_dir("e0_3", out)
    rows = []
    for pid in parts:
        for r in load_manifest(pid, set_id)["instances"]:
            m = r.get("defect")
            if not m:
                continue
            nom = _nominal_area(m)
            if nom:
                row = {"part": pid, "type": m["type"], "difficulty": r.get("difficulty"),
                       "labelled_mm2": m["area_m2"] * 1e6, "nominal_mm2": nom * 1e6,
                       "rel_err": m["area_m2"] / nom - 1.0}
                if m["type"] in ("dent", "bulge"):
                    # the nominal ellipse is planar: compare the label's area projected onto
                    # the defect's tangent plane (a bump stretches the surface itself)
                    mesh = LabeledMesh.load(resolve_data_path(r["mesh"]))
                    tm = mesh.to_trimesh()
                    lab = mesh.face_label > 0
                    proj = float((tm.area_faces[lab] * np.abs(tm.face_normals[lab] @ np.asarray(m["normal"]))).sum())
                    row["rel_err"] = proj / nom - 1.0
                    row["rel_err_3d"] = m["area_m2"] / nom - 1.0
                rows.append(row)
    write_json(d / "rows.json", rows)
    types = sorted({r["type"] for r in rows})
    table = []
    for t in types:
        e = np.array([r["rel_err"] for r in rows if r["type"] == t])
        flat = np.array([r["rel_err"] for r in rows if r["type"] == t and r["difficulty"] == "easy"])
        table.append([t, len(e), float(np.median(e)), float(np.median(np.abs(e))),
                      float(np.median(np.abs(flat))) if len(flat) else float("nan")])
    md = "# E0.3 GT defect-mask sanity (labelled vs nominal area)\n\n" + _md_table(
        table, ["type", "n", "median rel. err", "median |rel. err|", "median |rel. err| (easy/flat)"])
    d3 = [r["rel_err_3d"] for r in rows if "rel_err_3d" in r]
    md += ("\n\nNominal areas assume a locally flat surface (dent/bulge: ellipse where the bump exceeds 10 % of its"
           " height, compared with the label projected onto the tangent plane; blind hole: wall + bottom; crack: slot"
           " walls + floor), so curved placements and slots running past an edge deviate by design."
           " Acceptance (plan): |error| < 10 % on flat placements."
           + (f" Dent/bulge on the deformed (3D) surface instead: median |error| {np.median(np.abs(d3)):.3f}." if d3 else "")
           + "\n")
    (d / "summary.md").write_text(md)
    write_json(d / "summary.json", {"table": table})
    return d


# ------------------------------------------------------------------------------- E0.4
def e0_4(parts: list[str], set_id: str = "mvp", n_workers: int = 20, out: str | None = None, **_) -> Path:
    from ..optim.evaluator import ConfigEvaluator
    from ..render.slot_cache import SlotCache

    d = exp_dir("e0_4", out)
    rows = []
    for pid in parts:
        cache = SlotCache.open(pid, set_id)
        st = [s for s in cache.meta.get("instance_stats", {}).values() if "render_s" in s]
        per_inst = float(np.mean([s["render_s"] for s in st])) if st else float("nan")
        ev = ConfigEvaluator(pid, set_id, C.detector("fpfh_bank"), C.process("inline_default"), C.noise("N2"),
                             n_workers=n_workers, use_cache=False)
        try:
            ref = reference_config(cache.grid)
            t0 = time.time()
            r1 = ev.evaluate(ref)
            t_first = time.time() - t0
            six = InspectionConfig(slots=ev.space.ring(6, 40.0, 0.25), rotation="r90x3", speed_level=1)
            t0 = time.time()
            r2 = ev.evaluate(six)
            t_six = time.time() - t0
            t0 = time.time()
            s = [ev.surrogate(ev.space.random_config(np.random.default_rng(i))) for i in range(200)]
            t_s1 = (time.time() - t0) / 200
        finally:
            ev.close()
        rows.append({"part": pid, "n_instances": len(st), "cache_build_s": cache.meta.get("build_s"),
                     "render_s_per_instance_96_views": per_inst, "render_ms_per_view": per_inst / cache.grid.n_slots * 1e3,
                     "stage2_s_ref_3cam_first": round(t_first, 1), "stage2_s_ref_timing": r1["timing"],
                     "stage2_s_6cam_x4stages": round(t_six, 1), "stage2_timing_6x4": r2["timing"],
                     "stage1_ms_per_config": round(t_s1 * 1e3, 3),
                     "cache_mean_px_per_view": cache.meta["pixel_counts"]["mean_per_view"]})
    write_json(d / "summary.json", rows)
    md = ["# E0.4 timing", "", "| part | instances | render ms/view (embree, 6 proc) | stage-2 ref (s) | stage-2 6 cam x 4 stages (s) | stage-1 ms/config |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['part']} | {r['n_instances']} | {r['render_ms_per_view']:.1f} | {r['stage2_s_ref_3cam_first']} | "
                  f"{r['stage2_s_6cam_x4stages']} | {r['stage1_ms_per_config']} |")
    md.append("\nPlan budget: stage 2 <= 60 s/config (cut list applies otherwise); stage 1 < 1 s/config.")
    (d / "summary.md").write_text("\n".join(md) + "\n")
    return d


# ------------------------------------------------------------------------------- E0.5
def e0_5(parts: list[str], set_id: str = "mvp", n_workers: int = 20, out: str | None = None,
         noises=("N1", "N2", "N3"), **_) -> Path:
    from ..detect import scorer
    from ..optim.evaluator import ConfigEvaluator

    d = exp_dir("e0_5", out)
    rows = []
    for pid in parts:
        for nz in noises:
            ev = ConfigEvaluator(pid, set_id, C.detector("fpfh_bank"), C.process("inline_default"), C.noise(nz),
                                 n_workers=n_workers)
            try:
                rec = ev.evaluate(reference_config(ev.grid))
            finally:
                ev.close()
            val = np.asarray(rec["val_scores"])
            defs = rec["search_defects"] + rec["report_defects"]
            fnr_all = scorer.fnr_at_fpr(val, [x["score"] for x in defs])
            tiers = {t: scorer.fnr_at_fpr(val, [x["score"] for x in defs if x.get("tier") == t]) for t in ("S", "M", "L")}
            types = {t: scorer.fnr_at_fpr(val, [x["score"] for x in defs if x["type"] == t]) for t in sorted({x["type"] for x in defs})}
            rows.append({"part": pid, "noise": nz, "fnr_pooled": fnr_all, "fnr_report": rec["report"]["fnr"],
                         "auroc_report": rec["report"]["auroc"], "fnr_by_tier": tiers, "fnr_by_type": types})
            log.info(f"E0.5 {pid} {nz}: FNR pooled {fnr_all:.2f}")
    main = [r for r in rows if r["noise"] == "N2"]
    pooled = float(np.mean([r["fnr_pooled"] for r in main])) if main else float("nan")
    rec_k = 1.0
    if pooled < 0.2:
        rec_k = 0.7
    elif pooled > 0.6:
        rec_k = 1.4
    summ = {"rows": rows, "mean_fnr_N2": pooled, "target": [0.2, 0.6], "scale_k_current": 1.0,
            "scale_k_recommended": rec_k, "in_target": bool(0.2 <= pooled <= 0.6)}
    write_json(d / "summary.json", summ)
    md = ["# E0.5 severity calibration (reference config: 3 cam, el 40, az 0/120/240, 0.25 m, 0.3 m/s)", "",
          "| part | noise | FNR pooled | FNR report | AUROC report | FNR S / M / L |", "|---|---|---|---|---|---|"]
    for r in rows:
        t = r["fnr_by_tier"]
        md.append(f"| {r['part']} | {r['noise']} | {r['fnr_pooled']:.2f} | {r['fnr_report']:.2f} | {r['auroc_report']:.2f} | "
                  f"{t['S']:.2f} / {t['M']:.2f} / {t['L']:.2f} |")
    md += ["", f"Mean pooled FNR at N2: **{pooled:.2f}** (target 0.2-0.6) -> scale_k "
                f"{'kept at 1.0 (frozen)' if summ['in_target'] else f'recommended {rec_k}'}."]
    (d / "summary.md").write_text("\n".join(md) + "\n")
    return d


# --------------------------------------------------------------------- result sanity
def result_sanity(parts: list[str], set_id: str = "mvp", n_workers: int = 20, out: str | None = None, **_) -> Path:
    """Plan verification item 7: (a) FNR roughly non-increasing with camera count along
    nested rings, (b) unreachable-control defects ~never detected (FNR ~ 1), (c) memory
    bank vs CAD-distance baseline on the same configurations."""
    from ..optim.evaluator import ConfigEvaluator

    d = exp_dir("result_sanity", out)
    rows = []
    for pid in parts:
        res = {"part": pid}
        for det in ("fpfh_bank", "cad_distance"):
            ev = ConfigEvaluator(pid, set_id, C.detector(det), C.process("inline_default"), C.noise("N2"), n_workers=n_workers)
            try:
                b = ev.space.baselines()
                seq = [b["top_1cam"], b["ring2_el60"], b["ring3_el40"], b["ring4_el40"], b["ring6_el40"]]
                recs = [ev.evaluate(c) for c in seq]
            finally:
                ev.close()
            fn = [r["report"]["fnr"] for r in recs]
            res[det] = {"n_cam": [r["n_cam"] for r in recs], "fnr_report": fn,
                        "auroc_report": [r["report"]["auroc"] for r in recs],
                        "non_increasing_steps": int(sum(fn[i + 1] <= fn[i] + 0.02 for i in range(len(fn) - 1))),
                        "unreachable_detected_frac": [r.get("unreachable", {}).get("detected_frac") for r in recs]}
        rows.append(res)
    write_json(d / "summary.json", rows)
    md = ["# Result sanity (plan verification 7)", "",
          "| part | detector | cameras | FNR report | AUROC | unreachable detected |", "|---|---|---|---|---|---|"]
    for r in rows:
        for det in ("fpfh_bank", "cad_distance"):
            x = r[det]
            md.append(f"| {r['part']} | {det} | {x['n_cam']} | {[round(v, 2) for v in x['fnr_report']]} | "
                      f"{[round(v, 2) for v in x['auroc_report']]} | {[None if v is None else round(v, 2) for v in x['unreachable_detected_frac']]} |")
    (d / "summary.md").write_text("\n".join(md) + "\n")
    return d
