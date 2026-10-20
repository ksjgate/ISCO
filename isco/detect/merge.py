"""Compose a configuration's merged point cloud for one instance from the slot cache.

Steps (plan "2단계 검출 파이프라인", ADR-0006):

1. pick the cached views of the configuration (rotation stage theta -> slot rotated by
   theta, exact for the part);
2. per view: parametric sensor noise (``datagen.sensor_noise``) and unprojection to the
   ISCO world frame with the slot's ``T_world_cam``;
3. process perturbations (the inspection system does not know them):
   * inline capture (no rotation stage): along-belt trigger error, common to all views
     (``simultaneous``) or independent per view (``staggered``), sigma = v * sigma_t;
     exposure motion blur over ``v * t_acq`` (``sensor_noise.apply_motion_blur``);
   * rotation stage: the part is stopped on the turntable (no blur / trigger error);
     every stage theta > 0 gets a repositioning error (translation, yaw);
   * static mode (E4 baseline): none of the above;
4. merge, voxel-average (label = max in voxel), statistical outlier removal (flying
   pixels; standard 3D-AD preprocessing), and register the merged cloud to the
   nominal template (point-to-plane ICP, ``registration="icp"``; ``"oracle"`` uses the
   ground-truth pose; ``"none"`` keeps the nominal frame).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..datagen.sensor_noise import apply_motion_blur, apply_sensor_noise
from ..geometry.transforms import apply_T, inv_T, make_T, rot_z
from ..render.slot_cache import SlotCache
from ..schemas.config import InspectionConfig
from ..schemas.process import ProcessParams
from ..schemas.sensor import NoiseModel
from ..visibility.surrogate import config_views


@dataclass
class MergedCloud:
    xyz: np.ndarray        # (N,3) float32, part (template) frame after registration
    label: np.ndarray      # (N,) uint8 (1 part, >=2 defect)
    cam_dir: np.ndarray    # (N,3) float32 unit vectors point->camera (normal orientation)
    n_raw: int             # points before voxelisation
    n_views: int
    reg_fitness: float = 1.0
    reg_rmse: float = 0.0
    reg_T: np.ndarray | None = None
    rep_view: np.ndarray | None = None     # (N,) index into ``views`` of a representative pixel
    rep_pix: np.ndarray | None = None      # (N,) global pixel index of that pixel
    views: list | None = None              # [(ViewPixels after noise, cache slot)] when keep_views


def voxel_average(xyz: np.ndarray, label: np.ndarray, cam_dir: np.ndarray, voxel: float, return_rep: bool = False):
    """Average points per voxel (label = max). ``return_rep`` also returns, per voxel, the
    index of one source point (used to look up image features of that pixel)."""
    key = np.floor(xyz / voxel).astype(np.int64)
    key -= key.min(axis=0)
    dims = key.max(axis=0) + 1
    lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    uniq, first, inv = np.unique(lin, return_index=True, return_inverse=True)
    n = len(uniq)
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    out = np.empty((n, 3))
    cd = np.empty((n, 3))
    for i in range(3):
        out[:, i] = np.bincount(inv, weights=xyz[:, i], minlength=n) / cnt
        cd[:, i] = np.bincount(inv, weights=cam_dir[:, i], minlength=n)
    cd /= np.maximum(np.linalg.norm(cd, axis=1, keepdims=True), 1e-12)
    lab = np.zeros(n, np.uint8)
    np.maximum.at(lab, inv, label)
    if return_rep:
        return out.astype(np.float32), lab, cd.astype(np.float32), first
    return out.astype(np.float32), lab, cd.astype(np.float32)


def statistical_outlier_mask(xyz: np.ndarray, nb: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """open3d statistical outlier removal -> boolean keep mask."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
    _, idx = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std_ratio)
    keep = np.zeros(len(xyz), bool)
    keep[np.asarray(idx, dtype=np.int64)] = True
    return keep


class Registrar:
    """Point-to-plane ICP of a merged cloud onto the nominal template surface."""

    def __init__(self, template_pts: np.ndarray, template_nrm: np.ndarray):
        import open3d as o3d

        self.o3d = o3d
        self.tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(template_pts.astype(np.float64)))
        self.tgt.normals = o3d.utility.Vector3dVector(template_nrm.astype(np.float64))

    def register(self, xyz: np.ndarray) -> tuple[np.ndarray, float, float]:
        o3d = self.o3d
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
        reg = o3d.pipelines.registration
        T = np.eye(4)
        for dmax, it, est in ((6e-3, 30, reg.TransformationEstimationPointToPoint()),
                              (1.5e-3, 40, reg.TransformationEstimationPointToPlane()),
                              (0.6e-3, 30, reg.TransformationEstimationPointToPlane())):
            r = reg.registration_icp(src, self.tgt, dmax, T, est, reg.ICPConvergenceCriteria(max_iteration=it))
            T = r.transformation
        return np.asarray(T), float(r.fitness), float(r.inlier_rmse)


def compose_instance(cache: SlotCache, inst: dict, cfg: InspectionConfig, process: ProcessParams, noise: NoiseModel,
                     seed: int, voxel: float, registrar: Registrar | None,
                     registration: str = "icp", sor: tuple[int, float] | None = (20, 2.0),
                     keep_views: bool = False) -> MergedCloud:
    """Random streams are keyed so that a view shared by two configurations receives
    the *same* sensor-noise realisation (common random numbers for paired comparisons)."""
    from ..utils.seeds import rng_for

    grid, cam = cache.grid, cache.camera
    rng = rng_for(seed, inst["inst_id"], "process", cfg.config_id)
    views = config_views(cfg, grid)
    vps = cache.views(inst["inst_id"], sorted({v for _, v, _ in views}))
    static = process.mode == "static"
    rotation_stage = cfg.rotation != "none"
    v = 0.0 if (static or rotation_stage) else process.speed(cfg.speed_level)
    sig_x = v * process.trigger_sigma_s
    common_dx = rng.normal(0.0, sig_x) if (cfg.trigger_mode == "simultaneous" and sig_x > 0) else 0.0
    stage_T = {}
    for th in cfg.stages_deg:
        if th == 0.0 or static:
            stage_T[th] = np.eye(4)
        else:
            t = np.r_[rng.normal(0.0, process.rot_reposition_sigma_m, 2), 0.0]
            stage_T[th] = make_T(rot_z(rng.normal(0.0, process.rot_yaw_sigma_deg)), t)
    xs, ls, ds, vids, pids = [], [], [], [], []
    kept_views = []
    n_raw = 0
    for cam_slot, cache_slot, th in views:
        vp = vps[cache_slot]
        T_wc = cache.T_world_cam(cache_slot)
        belt_dir = rot_z(-th) @ np.array([1.0, 0.0, 0.0])      # along-belt direction in the stage-0 frame
        if v > 0 and noise.t_acq_s > 0:
            vp = apply_motion_blur(vp, cam, T_wc, belt_dir, v * noise.t_acq_s)
        if not noise.is_noiseless:
            vp = apply_sensor_noise(vp, cam, T_wc, noise, rng_for(seed, inst["inst_id"], "noise", cache_slot, th, noise.name))
        if len(vp) == 0:
            continue
        pw = apply_T(T_wc, cam.unproject(vp.pix, vp.depth))
        # camera direction for normal orientation (in the cached, stage-0 frame)
        cdir = T_wc[:3, 3][None, :] - pw
        # trigger error: the part is dx further along the belt than the system assumes
        shift = np.zeros(3)
        if v > 0:
            dx = common_dx if cfg.trigger_mode == "simultaneous" else rng.normal(0.0, sig_x)
            shift += dx * belt_dir
        pw = apply_T(stage_T[th], pw + shift)
        if cfg.trigger_offset_m != 0.0:
            # deliberate offset: the part sits offset_m downstream when imaged; points
            # that leave the image are lost (viewpoint change itself neglected, E0.2)
            pc = apply_T(inv_T(T_wc), pw + cfg.trigger_offset_m * belt_dir)
            u, vv, z = cam.project(pc)
            keep = cam.in_image(u, vv, z)
            pw, cdir, vp = pw[keep], cdir[keep], vp.subset(keep)
        xs.append(pw)
        ls.append(vp.label)
        ds.append(cdir)
        if keep_views:
            vids.append(np.full(len(pw), len(kept_views), np.int16))
            pids.append(vp.pix.astype(np.int64))
            kept_views.append((vp, cache_slot))
        n_raw += len(pw)
    if not xs:
        z3 = np.zeros((0, 3), np.float32)
        return MergedCloud(z3, np.zeros(0, np.uint8), z3, 0, len(views), 0.0, 0.0, np.eye(4))
    xyz = np.concatenate(xs)
    lab = np.concatenate(ls)
    cdir = np.concatenate(ds)
    cdir /= np.maximum(np.linalg.norm(cdir, axis=1, keepdims=True), 1e-12)
    xyz, lab, cdir, rep = voxel_average(xyz, lab, cdir, voxel, return_rep=True)
    rep_view = np.concatenate(vids)[rep] if keep_views else None
    rep_pix = np.concatenate(pids)[rep] if keep_views else None
    if sor is not None and len(xyz) > sor[0] + 1:
        keep = statistical_outlier_mask(xyz, int(sor[0]), float(sor[1]))
        xyz, lab, cdir = xyz[keep], lab[keep], cdir[keep]
        if keep_views:
            rep_view, rep_pix = rep_view[keep], rep_pix[keep]
    fit, rmse, T = 1.0, 0.0, np.eye(4)
    if registration == "oracle":
        T = inv_T(np.asarray(inst["T_world_part"]))
    elif registration == "icp" and registrar is not None and len(xyz) > 50:
        T, fit, rmse = registrar.register(xyz)
    if registration != "none":
        xyz = apply_T(T, xyz).astype(np.float32)
        cdir = (cdir @ T[:3, :3].T).astype(np.float32)
    return MergedCloud(xyz, lab, cdir, n_raw, len(views), fit, rmse, T, rep_view, rep_pix,
                       kept_views if keep_views else None)
