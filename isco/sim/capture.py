"""Capturing ``ViewPixels`` from the rig + the PoC multi-view capture (race fixed).

Freshness
---------
Plain ``world.render()`` / ``app.update()`` returns annotator data that lags the USD
state by a *variable* 1-3 frames (measured: after moving the part, the depth changed on
the 1st, 2nd or 3rd update). ``render_fresh()`` therefore renders with
``omni.replicator.core.orchestrator.step(delta_time=0.0)``, which blocks until the
SDG pipeline has produced the frame for the current stage state and does not advance
the timeline (no physics step). ``verify_pose`` is the independent safety net: the
part pixels are unprojected and compared with the expected part surface; a stale or
mis-posed frame shows up as a large point-to-surface residual (warning > 1 mm).

Pixel extraction (matches ``isco.render.raycast_backend``)
-------------------------------------------------------
* depth = ``distance_to_image_plane`` (camera z), kept where the semantic label is
  part/defect;
* label: class ``part`` -> 1, class ``defect`` with ``defect_id`` j -> 1 + j;
* cos = |n . ray| with the world-space ``normals`` annotator and the ray through the
  pixel centre ``(u + 0.5, v + 0.5)``.

This module imports Isaac lazily; the pure helpers (metrics, plots) work without it.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np

from ..geometry.camera_model import PinholeCamera
from ..geometry.mesh_io import LabeledMesh
from ..geometry.transforms import apply_T, inv_T
from ..render.pixels import ViewPixels

VERIFY_TOL_M = 1.0e-3


# ----------------------------------------------------------------------------- rendering
def render_fresh(n: int = 1, rt_subframes: int = 1) -> float:
    """Render ``n`` frames that are guaranteed to reflect the current stage state."""
    import omni.replicator.core as rep

    t0 = time.perf_counter()
    for _ in range(max(1, int(n))):
        rep.orchestrator.step(rt_subframes=int(rt_subframes), delta_time=0.0, pause_timeline=False)
    return time.perf_counter() - t0


def semantic_label_lut(id_to_labels: dict) -> np.ndarray:
    """Semantic id -> ISCO pixel label (0 background, 1 part, 1 + j defect j)."""
    if not id_to_labels:
        return np.zeros(1, np.uint8)
    max_id = max(int(k) for k in id_to_labels)
    lut = np.zeros(max_id + 1, np.uint8)
    for k, lab in id_to_labels.items():
        classes = [c.strip() for c in str(lab.get("class", "")).split(",")]
        if "defect" in classes:
            try:
                j = int(str(lab.get("defect_id", "1")).split(",")[0])
            except ValueError:
                j = 1
            lut[int(k)] = np.uint8(min(255, 1 + max(j, 1)))
        elif "part" in classes:
            lut[int(k)] = 1
    return lut


class StaleFrameError(RuntimeError):
    pass


def view_pixels_from_frame(depth: np.ndarray, sem_data: np.ndarray, id_to_labels: dict, normals: np.ndarray | None,
                           pinhole: PinholeCamera, T_world_cam: np.ndarray, meta: dict | None = None) -> ViewPixels:
    """Pure conversion of one camera frame into ``ViewPixels`` (part/defect pixels only)."""
    H, W = pinhole.height, pinhole.width
    depth = np.asarray(depth)
    sem_data = np.asarray(sem_data)
    if depth.shape != (H, W) or sem_data.shape[:2] != (H, W):
        raise StaleFrameError(f"unexpected frame shapes depth={depth.shape} sem={sem_data.shape}, want {(H, W)}")
    lut = semantic_label_lut(id_to_labels)
    ids = sem_data.reshape(H, W).astype(np.int64)
    ids = np.where(ids < len(lut), ids, 0)
    label = lut[ids]
    mask = (label > 0) & np.isfinite(depth) & (depth > 0)
    vv, uu = np.nonzero(mask)
    pix = (vv.astype(np.int64) * W + uu.astype(np.int64))
    d = depth[vv, uu].astype(np.float32)
    lab = label[vv, uu].astype(np.uint8)
    if normals is not None and len(pix):
        n = np.asarray(normals)[vv, uu, :3].astype(np.float64)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        r = pinhole.ray_dirs_cam(pix)
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        r_w = r @ np.asarray(T_world_cam)[:3, :3].T
        cos = np.abs(np.einsum("ij,ij->i", n, r_w)).astype(np.float32)
    else:
        cos = np.full(len(pix), np.nan, np.float32)
    return ViewPixels(pix, d, lab, cos, dict(meta or {}))


def read_view(cam, meta: dict | None = None) -> ViewPixels:
    """``ViewPixels`` of a rig camera's latest frame (call ``render_fresh`` first)."""
    if cam.T_world_cam is None:
        raise RuntimeError(f"camera {cam.index} has no pose")
    depth = cam.get_data("distance_to_image_plane")
    sem = cam.get_data("semantic_segmentation")
    nrm = cam.get_data("normals") if cam.has("normals") else None
    if sem is None or depth is None or np.asarray(depth).size == 0:
        raise StaleFrameError(f"camera {cam.index}: no annotator data yet")
    return view_pixels_from_frame(depth, sem["data"], sem["info"].get("idToLabels", {}), nrm, cam.pinhole,
                                  cam.T_world_cam, meta)


# ----------------------------------------------------------------------------- verify_pose
@dataclass
class SurfaceIndex:
    """Dense surface samples of a part (rest frame) for point-to-surface residuals and
    the expected silhouette."""

    points: np.ndarray
    tree: object
    spacing_m: float = 0.2e-3

    @classmethod
    def from_mesh(cls, mesh: LabeledMesh, spacing_m: float = 0.2e-3, seed: int = 0,
                  min_n: int = 20000, max_n: int = 400000) -> "SurfaceIndex":
        import trimesh
        from scipy.spatial import cKDTree

        tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
        n = int(np.clip(tm.area / spacing_m ** 2, min_n, max_n))
        pts, _ = trimesh.sample.sample_surface(tm, n, seed=seed)
        pts = np.concatenate([np.asarray(pts), np.asarray(mesh.vertices)], axis=0)
        eff_spacing = float(np.sqrt(tm.area / max(n, 1)))
        return cls(pts, cKDTree(pts), eff_spacing)


#: fraction of observed part pixels allowed outside the expected (dilated) silhouette
SILHOUETTE_TOL = 0.005


def expected_silhouette_outside(vp: ViewPixels, pinhole: PinholeCamera, T_world_cam: np.ndarray,
                                T_world_part: np.ndarray, index: SurfaceIndex) -> float:
    """Fraction of observed part pixels that fall outside the expected silhouette.

    The expected silhouette is the splat of the dense surface samples (expected pose,
    no occlusion test needed for a silhouette), dilated by the sample spacing in pixels.
    This catches in-plane shifts that a point-to-surface residual cannot see (planar
    faces sliding onto themselves)."""
    from scipy.ndimage import binary_dilation

    W, H = pinhole.width, pinhole.height
    p_cam = apply_T(inv_T(np.asarray(T_world_cam)) @ np.asarray(T_world_part), index.points)
    u, v, z = pinhole.project(p_cam)
    ok = pinhole.in_image(u, v, z)
    if not ok.any():
        return 1.0
    ui, vi = np.floor(u[ok]).astype(np.int64), np.floor(v[ok]).astype(np.int64)
    u0, u1, v0, v1 = ui.min(), ui.max() + 1, vi.min(), vi.max() + 1
    r = int(np.ceil(index.spacing_m / pinhole.footprint(max(float(z[ok].min()), 1e-6)))) + 1
    pad = r + 1
    u0, v0 = max(u0 - pad, 0), max(v0 - pad, 0)
    u1, v1 = min(u1 + pad, W), min(v1 + pad, H)
    mask = np.zeros((v1 - v0, u1 - u0), bool)
    mask[vi - v0, ui - u0] = True
    mask = binary_dilation(mask, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
    ou, ov = pinhole.pix_to_uv(vp.pix)
    inside = (ou >= u0) & (ou < u1) & (ov >= v0) & (ov < v1)
    hit = np.zeros(len(vp), bool)
    hit[inside] = mask[ov[inside] - v0, ou[inside] - u0]
    return float(1.0 - hit.mean())


def verify_pose(vp: ViewPixels, pinhole: PinholeCamera, T_world_cam: np.ndarray, T_world_part: np.ndarray,
                index: SurfaceIndex, tol_m: float = VERIFY_TOL_M, max_points: int = 5000, seed: int = 0,
                silhouette_tol: float = SILHOUETTE_TOL) -> dict:
    """Check that a rendered view shows the part at the expected pose.

    1. unproject the part pixels and measure the point-to-surface residual against the
       expected pose (``median_m``, ``p95_m``) and the norm of the mean residual vector
       (``centroid_offset_m``: a rigid shift of the rendered part shows up here);
    2. ``outside_frac``: fraction of part pixels outside the expected silhouette.

    ``ok`` = median and centroid offset < ``tol_m`` (1 mm) and ``outside_frac`` <
    ``silhouette_tol``. Empty views are ``ok`` with ``n = 0`` (nothing to check)."""
    if len(vp) == 0:
        return {"n": 0, "median_m": None, "p95_m": None, "centroid_offset_m": None, "outside_frac": None, "ok": True}
    idx = np.arange(len(vp))
    if len(idx) > max_points:
        idx = np.random.default_rng(seed).choice(idx, max_points, replace=False)
    pts_cam = pinhole.unproject(vp.pix[idx], vp.depth[idx])
    pts_part = apply_T(inv_T(np.asarray(T_world_part)) @ np.asarray(T_world_cam), pts_cam)
    dist, nn = index.tree.query(pts_part)
    resid = pts_part - index.points[nn]
    med = float(np.median(dist))
    cen = float(np.linalg.norm(resid.mean(axis=0)))
    out = expected_silhouette_outside(vp, pinhole, T_world_cam, T_world_part, index)
    return {"n": int(len(idx)), "median_m": med, "p95_m": float(np.percentile(dist, 95)), "centroid_offset_m": cen,
            "outside_frac": out, "ok": bool(med < tol_m and cen < tol_m and out < silhouette_tol)}


def capture_views(cams, metas: list[dict] | None = None, warmup: int = 1, verify: dict | None = None,
                  max_retries: int = 3, rt_subframes: int = 1, log=None) -> tuple[list[ViewPixels], list[str], dict]:
    """Render once (``warmup`` fresh frames) and read ``ViewPixels`` from ``cams``.

    ``verify`` = ``{"index": SurfaceIndex, "T_world_part": 4x4}`` (or a list of per-camera
    ``T_world_part`` under ``"T_world_part_per_cam"``) enables ``verify_pose`` for every
    camera; failing frames are re-rendered up to ``max_retries`` times and reported in
    the returned warnings. Returns ``(views, warnings, timing)``."""
    warnings: list[str] = []
    timing = {"render_s": 0.0, "read_s": 0.0, "verify_s": 0.0, "renders": 0}
    timing["render_s"] += render_fresh(warmup, rt_subframes)
    timing["renders"] += max(1, warmup)
    views: list[ViewPixels | None] = [None] * len(cams)
    pending = list(range(len(cams)))
    for attempt in range(max_retries + 1):
        retry = []
        for i in pending:
            cam = cams[i]
            meta = dict(metas[i]) if metas else {}
            t0 = time.perf_counter()
            try:
                vp = read_view(cam, meta)
            except StaleFrameError as e:
                timing["read_s"] += time.perf_counter() - t0
                retry.append(i)
                if attempt == max_retries:
                    warnings.append(str(e))
                    views[i] = ViewPixels.empty(meta)
                continue
            timing["read_s"] += time.perf_counter() - t0
            if verify is not None:
                t1 = time.perf_counter()
                T_wp = verify["T_world_part_per_cam"][i] if "T_world_part_per_cam" in verify else verify["T_world_part"]
                chk = verify_pose(vp, cam.pinhole, cam.T_world_cam, T_wp, verify["index"],
                                  tol_m=verify.get("tol_m", VERIFY_TOL_M))
                timing["verify_s"] += time.perf_counter() - t1
                vp.meta["verify"] = chk
                vp.meta["verify_attempts"] = attempt + 1
                if not chk["ok"] and attempt < max_retries:
                    retry.append(i)
                    continue
                if not chk["ok"]:
                    warnings.append(f"camera {cam.index}: verify_pose failed after {attempt + 1} renders "
                                    f"(median {chk['median_m'] * 1e3:.3f} mm, centroid {chk['centroid_offset_m'] * 1e3:.3f} mm, "
                                    f"outside silhouette {chk['outside_frac'] * 100:.2f} %)")
            views[i] = vp
        if not retry:
            break
        pending = retry
        if attempt < max_retries:
            timing["render_s"] += render_fresh(1, rt_subframes)
            timing["renders"] += 1
            if log:
                log(f"re-rendering for cameras {[cams[i].index for i in retry]} (attempt {attempt + 2})")
    return [v if v is not None else ViewPixels.empty() for v in views], warnings, timing


def view_points_world(vp: ViewPixels, pinhole: PinholeCamera, T_world_cam: np.ndarray) -> np.ndarray:
    return apply_T(np.asarray(T_world_cam), pinhole.unproject(vp.pix, vp.depth))


# ----------------------------------------------------------------------------- metrics (pure)
from ..eval.coverage import compare_reconstruction, point_mesh_distance, visualize_coverage  # noqa: E402,F401  (moved; re-exported)


@dataclass
class LegacyCaptureOptions:
    """The PoC ``--capture-views`` parameters (``create_conveyor.py`` flags)."""

    part_size: float = 0.04
    speed: float = 0.3
    parts_dir: str | None = None
    views_part: str | None = None
    views_count: int = 3
    views_elevation: float = 30.0
    views_radius_factor: float = 45.0
    views_focal_length: float = 24.0
    views_width: int = 1024
    views_height: int = 1024
    warmup_steps: int = 10
    seed: int | None = None


def _pinhole_from_isaac_camera(camera) -> PinholeCamera:
    w, h = camera.get_resolution()
    hfov = np.degrees(2.0 * np.arctan(camera.get_horizontal_aperture() / (2.0 * camera.get_focal_length())))
    return PinholeCamera(int(w), int(h), float(hfov))


def legacy_capture_views(scene, part_files, opts: LegacyCaptureOptions, out_dir: str, race_fix: bool = True,
                         aabb_filter: bool = False, log=print) -> dict:
    """The PoC ``run_capture_views`` on top of ``isco.sim``.

    One part is spawned at the belt entry and carried to the focus point (40 % along the
    belt) by the running belt, then ``views_count`` cameras on a ring capture it.

    ``race_fix=False`` reproduces the original PoC logic exactly (ADR-0008): the GT pose
    is read *before* the camera warm-up while the belt keeps moving the part, and belt
    points are removed with an AABB filter around the stale GT pose.

    ``race_fix=True`` (default): the part is frozen (belt surface velocity 0 and the body
    switched to kinematic) before the warm-up, frames are rendered fresh
    (``render_fresh``), the GT pose is read at capture time, and part points come from the
    semantic mask + own unprojection of ``distance_to_image_plane`` (``aabb_filter=True``
    keeps the PoC pointcloud-annotator + AABB filter as a fallback).

    Writes the PoC outputs (view PNGs, merged point cloud .npy/.ply,
    ground_truth_mesh.ply, coverage.png, summary.txt) plus ``summary.json``."""
    import imageio
    from isaacsim.core.utils.semantics import add_labels
    from pxr import Gf, UsdGeom, UsdPhysics

    from ..geometry.mesh_io import load_and_scale_mesh
    from .part_prim import get_world_T, load_part_mesh_stl
    from .rig import add_inspection_cameras
    from .scene import PART_COLORS

    import random

    world, stage, info = scene.world, scene.stage, scene.info
    rng = random.Random(opts.seed)
    t_start = time.perf_counter()
    whole = info["whole_aabb"]
    entry_x = whole[0] + 0.15
    focus_x = whole[0] + (whole[3] - whole[0]) * 0.4
    entry_pos = np.array([entry_x, 0.0, info["belt_top_z"] + opts.part_size / 2.0 + 0.02])
    parts_dir = opts.parts_dir
    stl_path = os.path.join(parts_dir, opts.views_part) if opts.views_part else rng.choice(part_files)
    local_verts, local_faces = load_and_scale_mesh(stl_path, opts.part_size)

    part_path = "/World/Parts/InspectedPart"
    load_part_mesh_stl(stage, part_path, stl_path, opts.part_size, PART_COLORS[0])
    prim = stage.GetPrimAtPath(part_path)
    if race_fix:
        add_labels(prim, ["part"], instance_name="class")
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*entry_pos.tolist()))
    xformable.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

    physics_fps = round(1.0 / world.get_physics_dt())
    travel_time = max(0.1, (focus_x - entry_x) / max(opts.speed, 1e-3))
    travel_steps = int(round(travel_time * physics_fps))
    scene.set_belt_speed(opts.speed)
    world.reset()
    for _ in range(travel_steps):
        world.step(render=False)

    if race_fix:
        # ADR-0008: stop everything that could move the part before any rendering step
        scene.set_belt_speed(0.0)
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)
        for _ in range(2):
            world.step(render=False)

    T_pre = get_world_T(prim)                       # pose before the camera warm-up (PoC GT)
    gt_pre = apply_T(T_pre, local_verts)
    part_center = gt_pre.mean(axis=0)

    cameras = add_inspection_cameras(center=part_center, num_cameras=opts.views_count, part_size=opts.part_size,
                                     radius_factor=opts.views_radius_factor, elevation_deg=opts.views_elevation,
                                     resolution=(opts.views_width, opts.views_height),
                                     focal_length=opts.views_focal_length, pointcloud=(not race_fix) or aabb_filter)
    if race_fix:
        for cam in cameras:
            cam.attach_annotator("distance_to_image_plane")
            cam.attach_annotator("semantic_segmentation", colorize=False, semanticTypes=["class"])

    for _ in range(opts.warmup_steps):        # PoC render warm-up (moves the part if not frozen!)
        world.step(render=True)
    if race_fix:
        render_fresh(2)

    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    saved, per_camera_points, cam_meta = [], [], []
    pad = opts.part_size * 0.12
    bbox_min, bbox_max = gt_pre.min(axis=0) - pad, gt_pre.max(axis=0) + pad

    for i, camera in enumerate(cameras):
        frame = camera.get_rgba()
        retries = 0
        while (frame is None or frame.size == 0) and retries < 30:
            world.step(render=True)
            frame = camera.get_rgba()
            retries += 1
        if frame is None or frame.size == 0:
            log(f"[capture] warning: camera {i}: no RGB frame")
            per_camera_points.append(np.empty((0, 3)))
            continue
        out_path = os.path.join(out_dir, f"view_{i}.png")
        imageio.imwrite(out_path, np.asarray(frame[:, :, :3], dtype=np.uint8))
        saved.append(out_path)
        if race_fix and not aabb_filter:
            pin = _pinhole_from_isaac_camera(camera)
            V = np.asarray(camera.get_view_matrix_ros(), dtype=float)
            T_wc = np.linalg.inv(V)
            depth = np.asarray(camera.get_depth())
            sem = camera._custom_annotators["semantic_segmentation"].get_data()
            vp = view_pixels_from_frame(depth, sem["data"], sem["info"].get("idToLabels", {}), None, pin, T_wc)
            pc = view_points_world(vp, pin, T_wc)
            cam_meta.append({"fx": pin.fx, "hfov_deg": pin.hfov_deg, "n_part_pixels": len(vp)})
        else:
            pc = np.asarray(camera.get_pointcloud(world_frame=True))
            if pc.size > 0:
                inside = np.all((pc >= bbox_min) & (pc <= bbox_max), axis=1)
                pc = pc[inside]
            cam_meta.append({"n_points_aabb": int(len(pc))})
        per_camera_points.append(np.asarray(pc, dtype=float).reshape(-1, 3))
        log(f"[capture] camera {i}: {len(pc)} points on/near the part")

    T_cap = get_world_T(prim)                       # pose at capture time
    drift_mm = float(np.linalg.norm(T_cap[:3, 3] - T_pre[:3, 3]) * 1e3)
    gt_verts_world = apply_T(T_cap, local_verts) if race_fix else gt_pre
    merged_points = (np.concatenate([p for p in per_camera_points if len(p)], axis=0)
                     if any(len(p) for p in per_camera_points) else np.empty((0, 3)))
    threshold = opts.part_size * 0.05
    comparison = compare_reconstruction(gt_verts_world, local_faces, merged_points, threshold, seed=0)
    log(f"[capture] merged points: {len(merged_points)} (per camera {[len(p) for p in per_camera_points]})\n"
        f"[capture] completeness@{threshold * 1e3:.1f}mm {comparison['completeness'] * 100:.1f}% "
        f"(mean {comparison['mean_completeness_dist'] * 1e3:.2f} mm), accuracy@{threshold * 1e3:.1f}mm "
        f"{comparison['accuracy'] * 100:.1f}% (mean {comparison['mean_accuracy_dist'] * 1e3:.2f} mm)\n"
        f"[capture] completeness@0.2mm {comparison.get('completeness_fine', 0) * 100:.1f}%, "
        f"accuracy@0.2mm (exact) {comparison.get('accuracy_exact_fine', 0) * 100:.2f}%; "
        f"part drift between GT read and capture: {drift_mm:.2f} mm")

    pc_path = os.path.join(out_dir, "merged_pointcloud.npy")
    np.save(pc_path, merged_points)
    saved.append(pc_path)
    try:
        import trimesh

        ply_path = os.path.join(out_dir, "merged_pointcloud.ply")
        trimesh.points.PointCloud(merged_points).export(ply_path)
        saved.append(ply_path)
        gt_ply_path = os.path.join(out_dir, "ground_truth_mesh.ply")
        trimesh.Trimesh(vertices=gt_verts_world, faces=local_faces, process=False).export(gt_ply_path)
        saved.append(gt_ply_path)
    except Exception as e:  # pragma: no cover
        log(f"[capture] warning: ply export failed ({e})")
    coverage_img_path = os.path.join(out_dir, "coverage.png")
    try:
        visualize_coverage(gt_verts_world, local_faces, per_camera_points, comparison, threshold, coverage_img_path)
        saved.append(coverage_img_path)
    except Exception as e:  # pragma: no cover
        log(f"[capture] warning: coverage plot failed ({e})")

    summary = {
        "part": os.path.basename(stl_path),
        "mode": "race_fixed" if race_fix else "legacy_poc_race",
        "point_source": ("pointcloud_annotator+aabb" if (not race_fix or aabb_filter)
                         else "distance_to_image_plane+semantic_mask (own unprojection)"),
        "num_cameras": len(cameras),
        "points_per_camera": [int(len(p)) for p in per_camera_points],
        "merged_points": int(len(merged_points)),
        "match_threshold_m": threshold,
        "completeness_ratio": comparison["completeness"],
        "completeness_mean_dist_m": comparison["mean_completeness_dist"],
        "accuracy_ratio": comparison["accuracy"],
        "accuracy_mean_dist_m": comparison["mean_accuracy_dist"],
        "completeness@2mm": comparison["completeness"],
        "accuracy@2mm": comparison["accuracy"],
        "accuracy_exact@2mm": comparison.get("accuracy_exact", 0.0),
        "completeness@0.2mm": comparison.get("completeness_fine", 0.0),
        "accuracy@0.2mm": comparison.get("accuracy_exact_fine", 0.0),
        "accuracy_exact_median_m": comparison.get("accuracy_exact_median_m"),
        "accuracy_exact_p99_m": comparison.get("accuracy_exact_p99_m"),
        "part_drift_between_gt_read_and_capture_mm": drift_mm,
        "T_stage_part_gt": (T_cap if race_fix else T_pre).tolist(),
        "T_stage_part_at_capture": T_cap.tolist(),
        "camera_ring": {"elevation_deg": opts.views_elevation, "radius_m": opts.part_size * opts.views_radius_factor,
                        "focal_length_stage_units": opts.views_focal_length,
                        "resolution": [opts.views_width, opts.views_height]},
        "cameras": cam_meta,
        "belt_speed_mps": opts.speed,
        "wall_s": round(time.perf_counter() - t_start, 2),
    }
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        for k in ("part", "num_cameras", "points_per_camera", "merged_points", "match_threshold_m",
                  "completeness_ratio", "completeness_mean_dist_m", "accuracy_ratio", "accuracy_mean_dist_m"):
            f.write(f"{k}: {summary[k]}\n")
        f.write(f"mode: {summary['mode']}\n")
        f.write(f"completeness@0.2mm: {summary['completeness@0.2mm']}\n")
        f.write(f"accuracy@0.2mm: {summary['accuracy@0.2mm']}\n")
        f.write(f"part_drift_mm: {drift_mm}\n")
    saved.append(summary_path)
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    saved.append(json_path)
    return {"saved": saved, "summary": summary}
