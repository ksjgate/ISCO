"""MVTec 3D-AD anchor (single top-view structured light; no meshes): E0.6 detector
reproduction and E5b sensor-gap / noise-model calibration.

Data: put the extracted categories under ``data/datasets/mvtec3d/raw/<category>/``
(``train/good/xyz/*.tiff``, ``test/<defect>/xyz/*.tiff``, ``test/<defect>/gt/*.png``).
The dataset needs a registration form, so ISCO does not download it.

* ``load_xyz``          organized xyz TIFF -> (H, W, 3) float32 metres-or-millimetres as stored
* ``remove_background`` RANSAC plane on the border strip (BTF preprocessing)
* ``sensor_statistics`` plane-residual sigma vs depth, dropout vs incidence, edge
  (flying-pixel) fraction -> fits the NoiseModel parameters (sigma0, theta_c)
* ``e0_6``              BTF-style FPFH memory bank on a category -> image AUROC
                        (public BTF FPFH values for comparison: I-AUROC ~0.78-0.86)
* ``e5b``               simulated top view of the ``dowel`` stand-in vs real dowel
                        statistics (noise level that matches -> pre-registered N_main)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils.io import data_root, write_json
from ..utils.logging import get_logger

log = get_logger(__name__)


def root() -> Path:
    return data_root() / "datasets" / "mvtec3d" / "raw"


def available(category: str = "dowel") -> bool:
    return (root() / category / "train").exists()


def load_xyz(path: str | Path) -> np.ndarray:
    import tifffile

    a = tifffile.imread(str(path)).astype(np.float32)
    return a


def valid_points(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(xyz).all(axis=2) & (np.abs(xyz).sum(axis=2) > 0)
    return xyz[m], m


def remove_background(xyz: np.ndarray, border: int = 10, thr: float = 0.005) -> np.ndarray:
    """Mask of foreground pixels: distance to the RANSAC plane fitted on the border strip
    exceeds ``thr`` (units of the file, MVTec stores metres)."""
    import open3d as o3d

    H, W, _ = xyz.shape
    strip = np.zeros((H, W), bool)
    strip[:border], strip[-border:], strip[:, :border], strip[:, -border:] = True, True, True, True
    pts, valid = valid_points(xyz)
    b = xyz[strip & valid.reshape(H, W)] if valid.ndim == 2 else xyz[strip]
    b = b[np.isfinite(b).all(axis=1) & (np.abs(b).sum(axis=1) > 0)]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(b.astype(np.float64)))
    model, _ = pcd.segment_plane(thr / 4, 3, 1000)
    a, bb, c, dd = model
    dist = np.abs(xyz @ np.array([a, bb, c]) + dd)
    return (dist > thr) & np.isfinite(xyz).all(axis=2) & (np.abs(xyz).sum(axis=2) > 0)


def sensor_statistics(paths: list[Path], max_files: int = 20) -> dict:
    """Noise statistics from good samples: residual of local planes (sigma) vs depth,
    invalid-pixel fraction vs incidence (dropout), and depth-edge fraction."""
    import open3d as o3d

    sig, depth, inc_hist_valid, inc_hist_all = [], [], np.zeros(18), np.zeros(18)
    for p in paths[:max_files]:
        xyz = load_xyz(p)
        H, W, _ = xyz.shape
        pts, m = valid_points(xyz)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
        n = np.asarray(pcd.normals)
        view = -pts / np.linalg.norm(pts, axis=1, keepdims=True)
        inc = np.degrees(np.arccos(np.clip(np.abs((n * view).sum(axis=1)), 0, 1)))
        inc_hist_valid += np.histogram(inc, bins=18, range=(0, 90))[0]
        # local plane residuals on 5x5 neighbourhoods
        tree = o3d.geometry.KDTreeFlann(pcd)
        rng = np.random.default_rng(0)
        for i in rng.choice(len(pts), min(2000, len(pts)), replace=False):
            _, idx, _ = tree.search_knn_vector_3d(pcd.points[i], 25)
            q = pts[np.asarray(idx)]
            q = q - q.mean(axis=0)
            s = np.linalg.svd(q, compute_uv=False)
            sig.append(s[-1] / np.sqrt(len(q)))
            depth.append(float(np.linalg.norm(pts[i])))
        inc_hist_all += np.histogram(inc, bins=18, range=(0, 90))[0]
    sig, depth = np.asarray(sig), np.asarray(depth)
    return {"sigma_median": float(np.median(sig)), "sigma_p90": float(np.quantile(sig, 0.9)),
            "depth_median": float(np.median(depth)), "incidence_hist_valid": inc_hist_valid.tolist(),
            "n_files": min(len(paths), max_files)}


def e0_6(categories=("dowel", "cable_gland"), n_train: int = 60, out: str | None = None, **_) -> Path:
    """BTF-style FPFH memory bank on MVTec 3D-AD (3D only) -> image AUROC per category."""
    from ..detect import scorer
    from ..detect.features import fpfh_multiscale
    from ..detect.memory_bank import MemoryBank
    from .sanity import exp_dir

    d = exp_dir("e0_6", out)
    res = {}
    for cat in categories:
        if not available(cat):
            res[cat] = {"status": "missing - download MVTec 3D-AD (registration form) into " + str(root())}
            continue
        import open3d as o3d

        def feats(p):
            xyz = load_xyz(p)
            fg = remove_background(xyz)
            pts = xyz[fg]
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64))).voxel_down_sample(0.0006)
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.002, max_nn=30))
            pcd.orient_normals_towards_camera_location(np.zeros(3))
            P = np.asarray(pcd.points)
            return fpfh_multiscale(P, np.asarray(pcd.normals), [0.002, 0.004])

        train = sorted((root() / cat / "train" / "good" / "xyz").glob("*.tiff"))[:n_train]
        bank = MemoryBank(coreset_ratio=0.1, coreset_max=20000).fit([feats(p) for p in train])
        normal, anomal = [], []
        for dd in sorted((root() / cat / "test").iterdir()):
            for p in sorted((dd / "xyz").glob("*.tiff")):
                s = float(bank.score(feats(p)).max())
                (normal if dd.name == "good" else anomal).append(s)
        res[cat] = {"image_auroc": scorer.auroc(normal, anomal), "n_good": len(normal), "n_anomalous": len(anomal)}
    write_json(d / "summary.json", res)
    return d


def e5b(out: str | None = None, **_) -> Path:
    """Real MVTec dowel statistics vs ISCO noise levels (sigma at the matching depth)."""
    from .. import config_loader as C
    from .sanity import exp_dir

    d = exp_dir("e5b", out)
    if not available("dowel"):
        write_json(d / "summary.json", {"status": "missing MVTec 3D-AD dowel data", "path": str(root() / "dowel")})
        return d
    stats = sensor_statistics(sorted((root() / "dowel" / "train" / "good" / "xyz").glob("*.tiff")))
    levels = {}
    for nz in ("N1", "N2", "N3"):
        m = C.noise(nz)
        levels[nz] = m.sigma0_m * (stats["depth_median"] / m.z0_m) ** 2
    best = min(levels, key=lambda k: abs(levels[k] - stats["sigma_median"]))
    write_json(d / "summary.json", {"mvtec_dowel": stats, "isco_sigma_at_that_depth": levels, "closest_level": best})
    return d
