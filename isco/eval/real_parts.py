"""Real-data parts and real-defect variant sets (plan: T-LESS + MulSen-AD part set,
Real3D-AD secondary; H3 / E5).

* **T-LESS** (CC BY 4.0): CAD models -> parts ``tless_XX`` (normalised to 40 mm).
* **MulSen-AD** (MIT): category scans are STL meshes in scanner coordinates (mm).
  ``prepare_mulsen`` turns ``train/0.stl`` (hole-filled) into the part ``mulsen_<cat>``;
  ``build_mulsen_real_set`` registers every normal train scan and every *geometric*
  defective test scan to that template (FPFH-RANSAC + ICP), maps them into the part's
  rest frame (same uniform scale), labels defect faces from the GT points (exact
  vertex matches) and writes a variant set (``set_id="real"``) with the same manifest
  format as the synthetic sets -> real part-to-part variation + real defects for H3.
* **Real3D-AD** (CC BY 4.0): 360-degree training prototypes (PCD) -> Poisson meshes ->
  parts ``real3d_<cat>`` (synthetic defects only: the test scans are one-sided).
"""
from __future__ import annotations

import glob
import os
import time
from pathlib import Path

import numpy as np
import trimesh

from ..datagen.parts import data_root, load_part, load_part_mesh, part_dir, prepare_part
from ..geometry.mesh_io import LabeledMesh
from ..schemas.part import PartSpec
from ..utils.io import data_rel, write_json
from ..utils.logging import get_logger

log = get_logger(__name__)

TLESS_DEFAULT = [13, 18, 29, 30]          # 2 self-occluding (13, 18) + 2 simple (29, 30)
MULSEN_GEOMETRIC = {"broken", "foreign_body", "scratch", "hole", "crack", "bent", "squeeze", "open", "crease",
                    "broken_outside", "substandard"}   # excludes colour / label / *_inside defects


def datasets_root() -> Path:
    return data_root() / "datasets"


# ------------------------------------------------------------------------- T-LESS
def prepare_tless(ids: list[int] | None = None, overwrite: bool = False) -> list[PartSpec]:
    root = datasets_root() / "tless" / "raw" / "extracted" / "models_cad"
    out = []
    for i in ids or TLESS_DEFAULT:
        out.append(prepare_part(f"tless_{i:02d}", root / f"obj_{i:06d}.ply", "tless", overwrite=overwrite))
    return out


# ---------------------------------------------------------------------- MulSen-AD
def mulsen_root() -> Path:
    return datasets_root() / "mulsen_ad" / "raw" / "extracted" / "MulSen_AD"


def repair_mesh(tm: trimesh.Trimesh, max_hole_edges: int = 400) -> trimesh.Trimesh:
    """Close small scan holes / fix non-manifold edges (pymeshlab), keep the largest
    component. Units are left unchanged."""
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(tm.vertices, float), np.asarray(tm.faces, np.int32)))
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_repair_non_manifold_edges()
    try:
        ms.meshing_close_holes(maxholesize=max_hole_edges)
    except Exception:  # pragma: no cover
        pass
    m = ms.current_mesh()
    out = trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=True)
    parts = out.split(only_watertight=False)
    if len(parts) > 1:
        out = max(parts, key=lambda p: len(p.faces))
    return out


def prepare_mulsen(categories: list[str], overwrite: bool = False) -> list[PartSpec]:
    out = []
    for cat in categories:
        src = mulsen_root() / cat / "Pointcloud" / "train" / "0.stl"
        tm = trimesh.load(str(src), force="mesh")
        tm = repair_mesh(tm)
        spec = prepare_part(f"mulsen_{cat}", src, "mulsen_ad", overwrite=overwrite, mesh=tm)
        out.append(spec)
    return out


def _o3d_pcd(v: np.ndarray, voxel: float):
    import open3d as o3d

    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(v, np.float64)))
    p = p.voxel_down_sample(voxel)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30))
    f = o3d.pipelines.registration.compute_fpfh_feature(p, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    return p, f


def register_scan(src_v: np.ndarray, tgt_v: np.ndarray, voxel: float, seed: int = 0, n_restarts: int = 3) -> tuple[np.ndarray, float, float]:
    """Global (FPFH + RANSAC) then point-to-plane ICP registration of a scan onto the
    template, both in scanner units. Returns (T_tgt_src, fitness, inlier RMSE)."""
    import open3d as o3d

    reg = o3d.pipelines.registration
    s, fs = _o3d_pcd(src_v, voxel)
    t, ft = _o3d_pcd(tgt_v, voxel)
    best = (np.eye(4), -1.0, np.inf)
    for r in range(n_restarts):
        o3d.utility.random.seed(seed + r)
        g = reg.registration_ransac_based_on_feature_matching(
            s, t, fs, ft, True, voxel * 1.5, reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(voxel * 1.5)],
            reg.RANSACConvergenceCriteria(100000, 0.999))
        icp = reg.registration_icp(s, t, voxel * 1.0, g.transformation, reg.TransformationEstimationPointToPlane(),
                                   reg.ICPConvergenceCriteria(max_iteration=60))
        icp = reg.registration_icp(s, t, voxel * 0.4, icp.transformation, reg.TransformationEstimationPointToPlane(),
                                   reg.ICPConvergenceCriteria(max_iteration=60))
        if icp.fitness > best[1] + 1e-6 or (abs(icp.fitness - best[1]) < 1e-6 and icp.inlier_rmse < best[2]):
            best = (np.asarray(icp.transformation), float(icp.fitness), float(icp.inlier_rmse))
        if best[1] > 0.9:
            break
    return best


def _gt_points(cat: str, dtype: str, idx: str) -> np.ndarray | None:
    p = mulsen_root() / cat / "Pointcloud" / "GT" / dtype / f"{idx}.txt"
    if not p.exists():
        return None
    return np.loadtxt(p, delimiter=",", comments="#", ndmin=2)


def _face_labels_from_gt(tm: trimesh.Trimesh, gt: np.ndarray | None, tol_mm: float = 1e-3) -> np.ndarray:
    lab = np.zeros(len(tm.faces), np.int32)
    if gt is None or len(gt) == 0:
        return lab
    from scipy.spatial import cKDTree

    d, i = cKDTree(np.asarray(tm.vertices)).query(gt)
    vmask = np.zeros(len(tm.vertices), bool)
    vmask[i[d < tol_mm]] = True
    lab[vmask[tm.faces].sum(axis=1) >= 2] = 1
    return lab


def build_mulsen_real_set(cat: str, set_id: str = "real", n_bank: int = 40, max_val: int = 100, seed: int = 0,
                         min_fitness: float = 0.6, overwrite: bool = False) -> dict:
    """Variant set from *real* MulSen-AD scans (normals: train; defects: geometric tests)."""
    from ..datagen.defects import visible_direction_count, difficulty_of
    from ..datagen.normal_variation import PoseParams, sample_pose
    from ..datagen.variants import set_dir
    from ..utils.hashing import content_hash
    from ..utils.seeds import rng_for

    pid = f"mulsen_{cat}"
    spec = load_part(pid)
    out = set_dir(pid, set_id)
    if (out / "manifest.json").exists() and not overwrite:
        from ..utils.io import read_json

        return read_json(out / "manifest.json")
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    T_src_rest = np.asarray(spec.rest_info["T_source_to_rest"])
    tmpl = trimesh.load(str(mulsen_root() / cat / "Pointcloud" / "train" / "0.stl"), force="mesh")
    tmpl_v = np.asarray(tmpl.vertices)
    voxel = float(max(tmpl.extents) / 60.0)
    train = sorted(glob.glob(str(mulsen_root() / cat / "Pointcloud" / "train" / "*.stl")),
                   key=lambda p: int(Path(p).stem))
    tests = []
    for d in sorted(os.listdir(mulsen_root() / cat / "Pointcloud" / "test")):
        if d in MULSEN_GEOMETRIC:
            tests += [(d, p) for p in sorted(glob.glob(str(mulsen_root() / cat / "Pointcloud" / "test" / d / "*.stl")),
                                             key=lambda p: int(Path(p).stem))]
    pose_p = PoseParams()
    rng_split = np.random.default_rng(seed)
    records, rejected = [], []
    t0 = time.time()

    def add(path: str, role: str, dtype: str, gt: np.ndarray | None, k: int):
        tm = trimesh.load(path, force="mesh")
        if dtype != "normal":
            lab = _face_labels_from_gt(tm, gt)
            if lab.sum() == 0:
                rejected.append({"path": path, "reason": "no GT faces"})
                return
        else:
            lab = np.zeros(len(tm.faces), np.int32)
        T, fit, rmse = register_scan(np.asarray(tm.vertices), tmpl_v, voxel, seed=k)
        if fit < min_fitness:
            rejected.append({"path": path, "reason": f"registration fitness {fit:.2f}"})
            return
        v = trimesh.transform_points(np.asarray(tm.vertices), T_src_rest @ T)
        v[:, 2] -= v[:, 2].min()
        lm = LabeledMesh(v, np.asarray(tm.faces), lab, ["part"] + ([f"{dtype}:0"] if dtype != "normal" else []),
                         {"source_scan": path, "reg_fitness": fit, "reg_rmse_mm": rmse})
        inst_id = f"{'n' if dtype == 'normal' else 'd'}_{role}_{dtype}_{Path(path).stem}"
        rng = rng_for(pid, set_id, inst_id)
        Tp, pinfo = sample_pose(rng, pose_p, lm, None)
        lm.save(out / "meshes" / f"{inst_id}.npz")
        rec = {"inst_id": inst_id, "role": role, "type": dtype, "tier": None, "mesh": data_rel(out / "meshes" / f"{inst_id}.npz"),
               "T_world_part": Tp.tolist(), "pose": pinfo, "variation": {"real_scan": True}, "n_faces": int(len(tm.faces)),
               "reg_fitness": fit, "reg_rmse_mm": rmse}
        if dtype != "normal":
            fm = lab > 0
            tmr = lm.to_trimesh()
            c = tmr.triangles_center[fm].mean(axis=0)
            n = tmr.face_normals[fm].mean(axis=0)
            n /= max(np.linalg.norm(n), 1e-12)
            kv = int(visible_direction_count(lm, c[None], n[None])[0])
            rec["difficulty"] = str(difficulty_of(np.array([kv]))[0])
            rec["defect"] = {"type": dtype, "center_m": c.tolist(), "normal": n.tolist(), "k_visible": kv,
                             "area_m2": float(tmr.area_faces[fm].sum()), "n_defect_faces": int(fm.sum()), "real": True}
        records.append(rec)

    for j, p in enumerate(train[1: 1 + n_bank + max_val]):
        add(p, "bank" if j < n_bank else "val", "normal", None, j)
    order = rng_split.permutation(len(tests))
    for j, ti in enumerate(order):
        d, p = tests[ti]
        add(p, "search" if j % 2 == 0 else "report", d, _gt_points(cat, d, Path(p).stem), 1000 + j)
    man = {"part_id": pid, "set_id": set_id, "spec": {"real": True, "source": "MulSen-AD", "n_bank": n_bank},
           "spec_hash": content_hash({"real": cat, "n_bank": n_bank, "max_val": max_val, "seed": seed}),
           "template": spec.mesh_fine, "instances": records, "rejected": rejected,
           "build_s": round(time.time() - t0, 1),
           "counts": {r: sum(1 for x in records if x["role"] == r) for r in ("bank", "val", "search", "report", "unreachable")}}
    write_json(out / "manifest.json", man)
    log.info(f"real set {pid}/{set_id}: {man['counts']} rejected={len(rejected)} in {man['build_s']}s")
    return man


# ---------------------------------------------------------------------- Real3D-AD
def real3d_root() -> Path:
    return datasets_root() / "real3d_ad" / "raw" / "extracted" / "Real3D-AD-PCD"


def poisson_from_pcd(path: str | Path, depth: int = 8, density_quantile: float = 0.0, voxel_div: float = 150.0):
    """Screened-Poisson mesh of a 360-degree prototype scan. The cloud is voxel-thinned to
    ``max_extent / voxel_div`` first: consistent normal orientation on the raw ~1M points
    took > 10 min, on ~25-40k points it takes ~1 s (E0 log). No low-density trimming by
    default: the prototypes are complete 360-degree scans, and trimming 0.5-2 % opened
    holes (non-watertight -> boolean defects impossible) without removing artefacts."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    ext = float(max(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())))
    pcd = pcd.voxel_down_sample(ext / voxel_div)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=ext / 40.0, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(12)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, density_quantile))
    tm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=True)
    parts = tm.split(only_watertight=False)
    if len(parts) > 1:
        tm = max(parts, key=lambda p: len(p.faces))
    return repair_mesh(tm)


def prepare_real3d(categories: list[str], overwrite: bool = False) -> list[PartSpec]:
    out = []
    for cat in categories:
        protos = sorted(glob.glob(str(real3d_root() / cat / "train" / "*.pcd")))
        tm = poisson_from_pcd(protos[0])
        out.append(prepare_part(f"real3d_{cat}", protos[0], "real3d_ad", overwrite=overwrite, mesh=tm))
    return out
