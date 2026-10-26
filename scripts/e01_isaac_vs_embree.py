#!/usr/bin/env python
"""E0.1 - Isaac Sim vs embree depth-renderer equivalence (plan: depth diff < 0.02 mm,
px/mm within +-2 %).

Objects: an analytic box (30x20x10 mm, with a defect patch) and sphere (r = 15 mm, with a
defect cap), plus the processed Fusion parts at their nominal pose (``T_world_part = I``).
Each is rendered from ~10 grid slots (mixed azimuth / elevation / distance)

* by the Isaac worker through ``render_slots`` jobs (``python -m isco.sim.worker``), and
* by ``isco.render.raycast_backend.RaycastRenderer`` with the belt-plane occluder and with
  the exported conveyor occluder (``data/conveyor/occluder.npz``),

and compared on common pixels: |depth difference| (median / p99 / max), signed bias,
sub-pixel shift estimate, label agreement, pixel-set IoU, |cos| agreement, symmetric
Chamfer distance of the unprojected points, silhouette scale (Isaac vs embree) and, for
the sphere, the absolute px/mm against the pinhole theory.

    python scripts/e01_isaac_vs_embree.py --queue queue --out runs/e0_1 [--start-worker]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from isco.geometry.camera_model import SlotGrid  # noqa: E402
from isco.geometry.mesh_io import LabeledMesh, bounding_radius, focus_point  # noqa: E402
from isco.geometry.transforms import apply_T  # noqa: E402
from isco.render.occluder import Occluder  # noqa: E402
from isco.render.pixels import ViewPixels  # noqa: E402
from isco.render.queue import FileJobQueue  # noqa: E402
from isco.render.raycast_backend import RaycastRenderer  # noqa: E402
from isco.schemas.job import InstanceRef, Job, RenderSlotsPayload  # noqa: E402
from isco.schemas.sensor import SensorModel  # noqa: E402

FUSION = ["fusion_22340", "fusion_41229", "fusion_51876", "fusion_26768", "fusion_28283"]
#: (azimuth deg, elevation deg, distance m) -> grid indices
SLOT_SPECS = [(0, 20, 0.25), (30, 40, 0.40), (60, 60, 0.25), (90, 80, 0.40), (150, 20, 0.40),
              (180, 40, 0.25), (210, 60, 0.40), (270, 80, 0.25), (300, 20, 0.25), (330, 60, 0.40)]


def make_box() -> LabeledMesh:
    import trimesh

    box = trimesh.creation.box(extents=[0.03, 0.02, 0.01])
    box.apply_translation([0.0, 0.0, 0.005])
    v, f = trimesh.remesh.subdivide_to_size(box.vertices, box.faces, max_edge=0.001)
    tm = trimesh.Trimesh(v, f, process=True)
    c = tm.triangles_center
    lab = ((c[:, 2] > 0.0099) & (np.hypot(c[:, 0] - 0.006, c[:, 1] - 0.002) < 0.004)).astype(np.int32)
    return LabeledMesh(tm.vertices, tm.faces, lab, ["part", "dent:0"], {"kind": "analytic_box", "extents_m": [0.03, 0.02, 0.01]})


def make_sphere(r: float = 0.015, subdivisions: int = 6) -> LabeledMesh:
    import trimesh

    s = trimesh.creation.icosphere(subdivisions=subdivisions, radius=r)
    s.apply_translation([0.0, 0.0, r])
    c = s.triangles_center
    lab = (c[:, 2] > 2 * r - 0.003).astype(np.int32)       # top cap = "defect" (label test)
    return LabeledMesh(s.vertices, s.faces, lab, ["part", "cap:0"], {"kind": "analytic_sphere", "radius_m": r})


def dense(vp: ViewPixels, W: int, H: int, field: str = "depth") -> np.ndarray:
    img = np.full(W * H, np.nan, np.float64)
    img[vp.pix] = getattr(vp, field)
    return img.reshape(H, W)


def subpixel_shift(a: ViewPixels, b: ViewPixels, W: int, H: int) -> tuple[float, float, int]:
    """Least-squares (du, dv) such that depth_a(u, v) ~ depth_b(u + du, v + dv) on smooth
    interior pixels (a pure image-plane shift; ~0 if both use the same pixel centres)."""
    Db, Da = dense(b, W, H), dense(a, W, H)
    gv, gu = np.gradient(Db)
    d = Da - Db
    lap = np.abs(np.gradient(gu, axis=1)) + np.abs(np.gradient(gv, axis=0))
    ok = np.isfinite(d) & np.isfinite(gu) & np.isfinite(gv) & np.isfinite(lap)
    if ok.sum() < 100:
        return float("nan"), float("nan"), int(ok.sum())
    thr = np.nanpercentile(lap[ok], 90)
    ok &= lap <= thr                                          # drop creases / silhouettes
    A = np.stack([gu[ok], gv[ok]], axis=1)
    if np.linalg.matrix_rank(A) < 2:
        return float("nan"), float("nan"), int(ok.sum())
    x, *_ = np.linalg.lstsq(A, d[ok], rcond=None)
    return float(x[0]), float(x[1]), int(ok.sum())


def chamfer(pa: np.ndarray, pb: np.ndarray, n: int = 50000, seed: int = 0) -> float:
    """Symmetric mean nearest-neighbour distance; queries a subsample of one cloud against
    the *full* other cloud (subsampling both would measure the pixel spacing)."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    qa = pa[rng.choice(len(pa), n, replace=False)] if len(pa) > n else pa
    qb = pb[rng.choice(len(pb), n, replace=False)] if len(pb) > n else pb
    da, _ = cKDTree(pb).query(qa)
    db, _ = cKDTree(pa).query(qb)
    return float(0.5 * (da.mean() + db.mean()))


def edge_mask(D: np.ndarray, jump_m: float = 5e-4) -> np.ndarray:
    """Pixels on a depth discontinuity of the reference image: a 4-neighbour is missing
    (silhouette) or differs by more than ``jump_m`` (occlusion boundary)."""
    H, W = D.shape
    P = np.pad(D, 1, constant_values=np.nan)
    m = np.zeros((H, W), bool)
    for dv, du in ((0, 1), (2, 1), (1, 0), (1, 2)):
        N = P[dv:dv + H, du:du + W]
        m |= ~np.isfinite(N) | (np.abs(N - D) > jump_m)
    return m


def compare(a: ViewPixels, b: ViewPixels, cam, T_wc: np.ndarray) -> dict:
    W, H = cam.width, cam.height
    common, ia, ib = np.intersect1d(a.pix, b.pix, return_indices=True)
    union = len(np.union1d(a.pix, b.pix))
    dd = a.depth[ia].astype(np.float64) - b.depth[ib].astype(np.float64)
    ad = np.abs(dd)
    rel = dd / b.depth[ib].astype(np.float64)
    du, dv, n_fit = subpixel_shift(a, b, W, H)
    pa = apply_T(T_wc, cam.unproject(a.pix, a.depth))
    pb = apply_T(T_wc, cam.unproject(b.pix, b.depth))
    dc = np.abs(a.cos[ia].astype(np.float64) - b.cos[ib].astype(np.float64))
    out_mask = ad > 2e-5
    on_edge = edge_mask(dense(b, W, H)).ravel()[common[out_mask]] if out_mask.any() else np.zeros(0, bool)
    return {
        "n_outliers_gt_20um": int(out_mask.sum()), "outliers_on_depth_edges": int(on_edge.sum()),
        "depth_abs_p999_m": float(np.percentile(ad, 99.9)) if len(ad) else None,
        "depth_abs_max_interior_m": float(ad[~out_mask].max()) if (~out_mask).any() else None,
        "n_isaac": int(len(a)), "n_embree": int(len(b)), "n_common": int(len(common)),
        "iou": len(common) / max(union, 1),
        "depth_abs_median_m": float(np.median(ad)) if len(ad) else None,
        "depth_abs_p99_m": float(np.percentile(ad, 99)) if len(ad) else None,
        "depth_abs_max_m": float(ad.max()) if len(ad) else None,
        "depth_bias_m": float(dd.mean()) if len(dd) else None,
        "depth_rel_bias": float(rel.mean()) if len(rel) else None,
        "frac_depth_gt_20um": float(np.mean(ad > 2e-5)) if len(ad) else None,
        "label_agreement": float(np.mean(a.label[ia] == b.label[ib])) if len(ia) else None,
        "defect_px_isaac": int(np.sum(a.label >= 2)), "defect_px_embree": int(np.sum(b.label >= 2)),
        "cos_abs_median": float(np.median(dc)) if len(dc) else None,
        "cos_abs_p99": float(np.percentile(dc, 99)) if len(dc) else None,
        "shift_du_px": du, "shift_dv_px": dv, "shift_fit_n": n_fit,
        "chamfer_m": chamfer(pa, pb) if len(pa) and len(pb) else None,
        "scale_ratio_isaac_over_embree": float(np.sqrt(len(a) / max(len(b), 1))),
    }


def sphere_scale(vp: ViewPixels, r: float, d: float, fx: float) -> dict:
    r_px_meas = float(np.sqrt(len(vp) / np.pi))
    r_px_theory = fx * r / np.sqrt(d * d - r * r)
    return {"r_px_measured": r_px_meas, "r_px_theory": r_px_theory, "ratio": r_px_meas / r_px_theory,
            "px_per_mm_theory": fx / (d * 1e3), "px_per_mm_measured": fx / (d * 1e3) * r_px_meas / r_px_theory}


def agg(rows: list[dict], key: str, fn) -> float | None:
    v = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return float(fn(v)) if v else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=str(REPO / "queue"))
    ap.add_argument("--out", default=str(REPO / "runs" / "e0_1"))
    ap.add_argument("--parts", nargs="*", default=FUSION)
    ap.add_argument("--start-worker", action="store_true", help="start scripts/start_worker.sh if no worker is alive")
    ap.add_argument("--throughput-part", default="fusion_51876", help="also render all 96 slots of this part (timing)")
    ap.add_argument("--timeout", type=float, default=3600)
    args = ap.parse_args(argv)

    out = Path(args.out)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    q = FileJobQueue(args.queue)
    if not q.worker_alive(60):
        if not args.start_worker:
            print("no live worker; start one with scripts/start_worker.sh or pass --start-worker")
            return 2
        subprocess.run(["bash", str(REPO / "scripts" / "start_worker.sh"), args.queue], check=True)

    sensor = SensorModel()
    cam = sensor.camera()
    grid = SlotGrid()
    slots = [grid.nearest_slot(a, e, d) for a, e, d in SLOT_SPECS]

    objects: dict[str, tuple[LabeledMesh, str]] = {}
    for name, mesh in (("box", make_box()), ("sphere", make_sphere())):
        p = out / "meshes" / f"{name}.npz"
        mesh.save(p)
        objects[name] = (mesh, str(p))
    from isco.datagen.parts import part_dir

    for pid in args.parts:
        p = part_dir(pid) / "fine.npz"
        objects[pid] = (LabeledMesh.load(p), str(p))

    jobs = {}
    for name, (mesh, path) in objects.items():
        focus = focus_point(mesh)
        payload = RenderSlotsPayload(part_id=name, instances=[InstanceRef(inst_id=name, mesh_path=path,
                                                                          T_world_part=np.eye(4).tolist())],
                                     slot_ids=slots, grid=grid.to_dict(), sensor=sensor, focus_m=focus.tolist(),
                                     output_dir=str(out / "isaac"))
        jobs[name] = q.submit(Job.make("render_slots", payload))
    tp_job = None
    if args.throughput_part:
        mesh, path = objects.get(args.throughput_part) or (None, None)
        if mesh is not None:
            payload = RenderSlotsPayload(part_id=args.throughput_part,
                                         instances=[InstanceRef(inst_id=f"{args.throughput_part}_all96", mesh_path=path,
                                                                T_world_part=np.eye(4).tolist())],
                                         slot_ids=list(range(grid.n_slots)), grid=grid.to_dict(), sensor=sensor,
                                         focus_m=focus_point(mesh).tolist(), output_dir=str(out / "isaac_throughput"))
            tp_job = q.submit(Job.make("render_slots", payload))
    t0 = time.time()
    results = q.wait(list(jobs.values()) + ([tp_job] if tp_job else []), timeout_s=args.timeout)
    print(f"isaac jobs finished in {time.time() - t0:.1f}s")
    for name, jid in jobs.items():
        r = results[jid]
        if r.status != "ok":
            print(f"{name}: job failed\n{r.error}")
            return 1

    occluders = {"belt_plane": Occluder.belt_plane()}
    occ_path = REPO / "data" / "conveyor" / "occluder.npz"
    if occ_path.exists():
        occluders["exported"] = Occluder.load(occ_path)
    rows: list[dict] = []
    timing = {}
    for name, (mesh, path) in objects.items():
        focus = focus_point(mesh)
        rad = bounding_radius(mesh, focus)
        timing[name] = results[jobs[name]].outputs
        for occ_name, occ in occluders.items():
            rr = RaycastRenderer(cam, occ)
            poses = [grid.T_world_cam(k, focus) for k in slots]
            views = [(T, cam.roi_for_sphere(T, focus, rad * 1.05)) for T in poses]
            t1 = time.perf_counter()
            ref = rr.render(mesh, np.eye(4), views)
            t_embree = time.perf_counter() - t1
            for k, T, b in zip(slots, poses, ref):
                a = ViewPixels.load(out / "isaac" / name / f"slot_{k:03d}.npz")
                row = {"object": name, "occluder": occ_name, "slot": int(k), "azimuth_deg": grid.slot(k).azimuth_deg,
                       "elevation_deg": grid.slot(k).elevation_deg, "distance_m": grid.slot(k).distance_m,
                       "verify": a.meta.get("verify"), "embree_s_per_view": t_embree / len(slots)}
                row.update(compare(a, b, cam, T))
                if name == "sphere":
                    row["sphere_isaac"] = sphere_scale(a, mesh.meta["radius_m"], grid.slot(k).distance_m, cam.fx)
                    row["sphere_embree"] = sphere_scale(b, mesh.meta["radius_m"], grid.slot(k).distance_m, cam.fx)
                rows.append(row)
        print(f"compared {name}")

    summary: dict = {"sensor": sensor.model_dump(mode="json"), "slots": slots, "slot_specs": SLOT_SPECS,
                     "occluders": list(occluders), "per_object": {}, "isaac_timing": timing}
    for occ_name in occluders:
        for name in objects:
            rs = [r for r in rows if r["object"] == name and r["occluder"] == occ_name]
            s = {
                "n_views": len(rs), "pixels_isaac": int(sum(r["n_isaac"] for r in rs)),
                "iou_min": agg(rs, "iou", np.min), "iou_mean": agg(rs, "iou", np.mean),
                "depth_abs_median_um": agg(rs, "depth_abs_median_m", np.median) * 1e6,
                "depth_abs_p99_um": agg(rs, "depth_abs_p99_m", np.max) * 1e6,
                "depth_abs_max_um": agg(rs, "depth_abs_max_m", np.max) * 1e6,
                "depth_bias_um": agg(rs, "depth_bias_m", np.mean) * 1e6,
                "depth_rel_bias_ppm": agg(rs, "depth_rel_bias", np.mean) * 1e6,
                "frac_depth_gt_20um_max": agg(rs, "frac_depth_gt_20um", np.max),
                "n_outliers_gt_20um": int(sum(r["n_outliers_gt_20um"] for r in rs)),
                "outliers_on_depth_edges": int(sum(r["outliers_on_depth_edges"] for r in rs)),
                "depth_abs_p999_um_max": agg(rs, "depth_abs_p999_m", np.max) * 1e6,
                "label_agreement_min": agg(rs, "label_agreement", np.min),
                "cos_abs_median": agg(rs, "cos_abs_median", np.median),
                "cos_abs_p99_max": agg(rs, "cos_abs_p99", np.max),
                "chamfer_um_max": agg(rs, "chamfer_m", np.max) * 1e6,
                "chamfer_um_mean": agg(rs, "chamfer_m", np.mean) * 1e6,
                "shift_du_px_mean": agg(rs, "shift_du_px", np.mean), "shift_dv_px_mean": agg(rs, "shift_dv_px", np.mean),
                "scale_dev_pct_max": agg(rs, "scale_ratio_isaac_over_embree", lambda v: np.max(np.abs(np.array(v) - 1)) * 100),
                "verify_ok": all((r.get("verify") or {}).get("ok", True) for r in rs),
            }
            if name == "sphere":
                si = [r["sphere_isaac"]["ratio"] for r in rs]
                se = [r["sphere_embree"]["ratio"] for r in rs]
                s["sphere_px_scale_dev_pct_isaac_max"] = float(np.max(np.abs(np.array(si) - 1)) * 100)
                s["sphere_px_scale_dev_pct_embree_max"] = float(np.max(np.abs(np.array(se) - 1)) * 100)
                s["sphere_px_per_mm"] = [{"d": r["distance_m"], "theory": r["sphere_isaac"]["px_per_mm_theory"],
                                          "isaac": r["sphere_isaac"]["px_per_mm_measured"],
                                          "embree": r["sphere_embree"]["px_per_mm_measured"]} for r in rs]
            summary["per_object"].setdefault(occ_name, {})[name] = s
    main_occ = "exported" if "exported" in occluders else "belt_plane"
    allr = [r for r in rows if r["occluder"] == main_occ]
    summary["overall"] = {
        "occluder": main_occ, "n_views": len(allr),
        "depth_abs_median_um": agg(allr, "depth_abs_median_m", np.median) * 1e6,
        "depth_abs_p99_um_max": agg(allr, "depth_abs_p99_m", np.max) * 1e6,
        "depth_abs_max_um": agg(allr, "depth_abs_max_m", np.max) * 1e6,
        "depth_abs_max_interior_um": agg(allr, "depth_abs_max_interior_m", np.max) * 1e6,
        "n_common_pixels": int(sum(r["n_common"] for r in allr)),
        "n_outliers_gt_20um": int(sum(r["n_outliers_gt_20um"] for r in allr)),
        "outliers_on_depth_edges": int(sum(r["outliers_on_depth_edges"] for r in allr)),
        "chamfer_um_max": agg(allr, "chamfer_m", np.max) * 1e6,
        "iou_min": agg(allr, "iou", np.min), "label_agreement_min": agg(allr, "label_agreement", np.min),
        "scale_dev_pct_max": agg(allr, "scale_ratio_isaac_over_embree", lambda v: np.max(np.abs(np.array(v) - 1)) * 100),
        "accept_depth_lt_20um": bool(agg(allr, "depth_abs_p99_m", np.max) < 2e-5),
        "accept_chamfer_lt_20um": bool(agg(allr, "chamfer_m", np.max) < 2e-5),
        "accept_px_scale_within_2pct": bool(summary["per_object"][main_occ]["sphere"]["sphere_px_scale_dev_pct_isaac_max"] < 2.0),
    }
    if tp_job:
        summary["throughput_all96"] = results[tp_job].outputs
    (out / "e01_rows.json").write_text(json.dumps(rows, indent=1, default=float))
    (out / "e01_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    md = [f"# E0.1 Isaac vs embree ({len(slots)} slots/object, sensor {sensor.width}x{sensor.height}, "
          f"hfov {sensor.hfov_deg} deg, nominal pose)", ""]
    for occ_name in occluders:
        md += [f"## embree occluder: {occ_name}", "",
               "| object | views | Isaac px | IoU min | depth abs med (um) | p99 max (um) | p99.9 max (um) | max (um) | "
               ">20um px (on depth edges) | bias (um) | label agree min | cos abs med | Chamfer max (um) | "
               "shift du,dv (px) | scale dev max (%) |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name in objects:
            s = summary["per_object"][occ_name][name]
            md.append(f"| {name} | {s['n_views']} | {s['pixels_isaac']} | {s['iou_min']:.5f} | {s['depth_abs_median_um']:.3f} | "
                      f"{s['depth_abs_p99_um']:.3f} | {s['depth_abs_p999_um_max']:.2f} | {s['depth_abs_max_um']:.1f} | "
                      f"{s['n_outliers_gt_20um']} ({s['outliers_on_depth_edges']}) | {s['depth_bias_um']:+.3f} | "
                      f"{s['label_agreement_min']:.5f} | {s['cos_abs_median']:.1e} | {s['chamfer_um_max']:.3f} | "
                      f"{s['shift_du_px_mean']:+.4f},{s['shift_dv_px_mean']:+.4f} | {s['scale_dev_pct_max']:.3f} |")
        md.append("")
    sp = summary["per_object"][main_occ]["sphere"]
    md += ["## Sphere px/mm vs pinhole theory", "", "| distance (m) | theory px/mm | Isaac px/mm | embree px/mm |", "|---|---|---|---|"]
    for e in sp["sphere_px_per_mm"]:
        md.append(f"| {e['d']} | {e['theory']:.3f} | {e['isaac']:.3f} | {e['embree']:.3f} |")
    md += ["", f"max |scale deviation| sphere: Isaac {sp['sphere_px_scale_dev_pct_isaac_max']:.3f} %, "
               f"embree {sp['sphere_px_scale_dev_pct_embree_max']:.3f} % (pixelisation of the silhouette area)", "",
           "## Overall / acceptance", "", "```", json.dumps(summary["overall"], indent=1), "```"]
    if tp_job:
        md += ["", "## Isaac render_slots throughput (1 instance x 96 slots)", "", "```",
               json.dumps(summary["throughput_all96"], indent=1), "```"]
    (out / "e01_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
