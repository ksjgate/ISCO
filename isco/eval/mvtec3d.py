"""MVTec 3D-AD anchor: one fixed top view of a structured-light camera (Zivid), no meshes.

* **E0.6 - detector reproduction.** The BTF 3D-only baseline (FPFH + PatchCore; Horwitz &
  Hoshen 2022, official ``3D-ADS`` code) is re-implemented step by step - border-RANSAC
  plane removal (5 mm), DBSCAN cleaning, nearest resize to 224², FPFH with 30-/100-NN
  (radii 0.1 / 0.25 m reach every point), 3x3 average + adaptive 28x28 patch pooling,
  10 % greedy k-center coreset from index 0, re-weighted max patch distance - and its image
  AUROC / AUPRO are compared with the published values. ISCO's own point detector
  (multi-scale FPFH 1/2/4 mm on a 0.4 mm voxel cloud, 96-NN smoothing, max) runs on the
  same data as a second row (no xyz term: single views are not registered to a CAD).
* **E5b - sensor gap.** The MVTec ``dowel`` is a ribbed plastic wall plug, not a turned
  wooden dowel, so a procedural stand-in cannot match it. Instead every real scan is
  re-rendered through ISCO's sensor model: its object pixels become a 2.5D mesh, the
  support becomes the fitted ideal plane (the "belt"), and the scene is ray cast with the
  real camera (pinhole fitted to the organized xyz) and noise levels N0-N3 (as defined and
  at their native working distance). Geometry and
  defects are the real ones, so every difference between the real and the re-rendered
  results comes from the sensor model. Sensor statistics (local noise on the part and on
  the support, dropout vs incidence, outliers) and detection results (same detector, same
  bank / threshold / defect split) are computed identically for both.

Data: ``data/datasets/mvtec3d/raw/<category>/{train,validation,test}/<type>/{xyz,gt,rgb}``
(registration form; ISCO does not download it).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..utils.io import data_root, write_json
from ..utils.logging import get_logger

log = get_logger(__name__)

#: published 3D-only FPFH results (M3DM, arXiv 2303.00601, Tables 1-2, transcribing BTF;
#: BTF README "preprocessed" row: cable gland I-AUROC 0.534, dowel 0.883)
BTF_PUBLISHED = {"dowel": {"image_auroc": 0.883, "aupro": 0.892},
                 "cable_gland": {"image_auroc": 0.551, "aupro": 0.879}}
BTF_SIZE = 224


def root() -> Path:
    return data_root() / "datasets" / "mvtec3d" / "raw"


def available(category: str = "dowel") -> bool:
    return (root() / category / "train").exists()


@dataclass(frozen=True)
class Sample:
    path: Path
    split: str          # train / validation / test
    dtype: str          # "good" or the defect type

    @property
    def gt_path(self) -> Path | None:
        p = self.path.parent.parent / "gt" / (self.path.stem + ".png")
        return p if p.exists() else None

    @property
    def key(self) -> str:
        return f"{self.split}/{self.dtype}/{self.path.stem}"


def samples(category: str, split: str) -> list[Sample]:
    base = root() / category / split
    out = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        out += [Sample(p, split, d.name) for p in sorted((d / "xyz").glob("*.tiff"))]
    return out


def load_xyz(path: str | Path) -> np.ndarray:
    """Organized (H, W, 3) float32 point cloud in metres (camera frame); 0 = invalid."""
    import tifffile

    a = tifffile.imread(str(path)).astype(np.float32)
    a[~np.isfinite(a).all(axis=2)] = 0.0
    return a


def load_gt(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, bool)
    from PIL import Image

    return np.asarray(Image.open(path)) > 0


def valid_mask(xyz: np.ndarray) -> np.ndarray:
    return np.all(xyz != 0, axis=-1)


# ======================================================================================
# BTF preprocessing and features (verbatim logic of 3D-ADS utils/preprocessing.py and
# feature_extractors/{fpfh_features,features}.py)
# ======================================================================================
def btf_plane(xyz: np.ndarray) -> np.ndarray:
    """Plane (a, b, c, d) from RANSAC on the 10-pixel image border (threshold 4 mm)."""
    import open3d as o3d

    e = np.concatenate([xyz[:10].reshape(-1, 3), xyz[-10:].reshape(-1, 3),
                        xyz[:, :10].reshape(-1, 3), xyz[:, -10:].reshape(-1, 3)])
    e = e[np.all(e != 0, axis=1)]
    o3d.utility.random.seed(0)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(e.astype(np.float64)))
    model, _ = pc.segment_plane(distance_threshold=0.004, ransac_n=50, num_iterations=1000)
    return np.asarray(model, dtype=np.float64)


def btf_preprocess(xyz: np.ndarray) -> np.ndarray:
    """Plane removal (points within 5 mm of the border plane), zero-padding to a square
    multiple of 100, largest DBSCAN cluster (eps 6 mm, 30 points)."""
    import open3d as o3d

    pc = xyz.reshape(-1, 3).copy()
    plane = btf_plane(xyz)
    dist = np.abs(np.c_[pc, np.ones(len(pc))] @ plane)
    pc[dist < 0.005] = 0
    H, W, _ = xyz.shape
    side = max(int(np.ceil(H / 100.0)) * 100, int(np.ceil(W / 100.0)) * 100)
    a, b = (side - H) // 2, (side - W) // 2
    org = np.pad(pc.reshape(H, W, 3), ((a, side - H - a), (b, side - W - b), (0, 0)))
    flat = org.reshape(-1, 3)
    nz = np.nonzero(np.all(flat != 0, axis=1))[0]
    if len(nz) == 0:
        return org
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(flat[nz].astype(np.float64)))
    labels = np.asarray(o3d_pc.cluster_dbscan(eps=0.006, min_points=30, print_progress=False))
    ids, sizes = np.unique(labels, return_counts=True)
    flat[nz[labels != ids[np.argmax(sizes)]]] = 0
    return flat.reshape(org.shape)


def resize_nearest(img: np.ndarray, size: int = BTF_SIZE) -> np.ndarray:
    """torch ``interpolate(mode='nearest')``: source index floor(i * in / out)."""
    H, W = img.shape[:2]
    ri = np.floor(np.arange(size) * (H / size)).astype(int)
    ci = np.floor(np.arange(size) * (W / size)).astype(int)
    return img[ri][:, ci]


def btf_fpfh_map(xyz224: np.ndarray, voxel_size: float = 0.05) -> np.ndarray:
    """(S, S, 33) FPFH map; zero (background) pixels keep a zero feature."""
    import open3d as o3d

    S = xyz224.shape[0]
    flat = xyz224.reshape(-1, 3)
    nz = np.nonzero(np.all(flat != 0, axis=1))[0]
    full = np.zeros((len(flat), 33), np.float32)
    if len(nz) > 0:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(flat[nz].astype(np.float64)))
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        f = o3d.pipelines.registration.compute_fpfh_feature(
            pc, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        full[nz] = np.asarray(f.data).T
    return full.reshape(S, S, 33)


def btf_patches(fmap: np.ndarray) -> np.ndarray:
    """AvgPool2d(3, stride 1) then AdaptiveAvgPool2d(28, 28) -> (784, 33)."""
    import torch

    t = torch.from_numpy(np.ascontiguousarray(fmap.transpose(2, 0, 1)))[None].float()
    t = torch.nn.AdaptiveAvgPool2d((28, 28))(torch.nn.AvgPool2d(3, stride=1)(t))
    return t[0].reshape(t.shape[1], -1).T.contiguous().numpy()


def btf_features(xyz: np.ndarray) -> np.ndarray:
    return btf_patches(btf_fpfh_map(resize_nearest(btf_preprocess(xyz))))


class BTFPatchCore:
    """Memory bank of the BTF FPFH baseline: greedy k-center coreset (10 %, first pick =
    index 0, half precision; BTF's sparse random projection fails for 33-d features and
    falls back to the raw features) and the re-weighted image score
    ``s = (1 - exp(s*/D) / sum_k exp(|m_test - m_k|/D)) * s*`` with the 2 neighbours of
    the nearest bank patch (D = sqrt(33))."""

    def __init__(self, f_coreset: float = 0.1, n_reweight: int = 3, device: str = "cuda"):
        import torch

        self.f = f_coreset
        self.k = n_reweight
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.lib = None

    def fit(self, patches: list[np.ndarray]) -> "BTFPatchCore":
        import torch

        lib = torch.from_numpy(np.concatenate(patches)).float()
        n = int(self.f * len(lib))
        z = lib.to(self.device)
        z = z.half() if self.device.type == "cuda" else z
        mind = torch.linalg.norm(z - z[0:1], dim=1, keepdim=True)
        idx = [0]
        last = z[0:1]
        for _ in range(n - 1):
            mind = torch.minimum(torch.linalg.norm(z - last, dim=1, keepdim=True), mind)
            j = int(torch.argmax(mind))
            last = z[j:j + 1]
            mind[j] = 0
            idx.append(j)
        self.lib = lib[torch.tensor(idx)].to(self.device)
        return self

    def score(self, patch: np.ndarray) -> tuple[float, np.ndarray]:
        import torch

        p = torch.from_numpy(patch).float().to(self.device)
        dist = torch.cdist(p, self.lib)
        min_val, min_idx = torch.min(dist, dim=1)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)
        m_test = p[s_idx].unsqueeze(0)
        m_star = self.lib[min_idx[s_idx]].unsqueeze(0)
        _, nn_idx = torch.topk(torch.cdist(m_star, self.lib), k=self.k, largest=False)
        m_star_knn = torch.linalg.norm(m_test - self.lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(float(p.shape[1])))
        w = 1 - (torch.exp(s_star / D) / torch.sum(torch.exp(m_star_knn / D)))
        side = int(round(np.sqrt(len(patch))))
        return float(w * s_star), min_val.reshape(side, side).cpu().numpy()


def btf_pixel_map(patch_map: np.ndarray, size: int = BTF_SIZE) -> np.ndarray:
    """Bilinear upsampling to 224² and BTF's 8-bit Gaussian blur (PIL radius 4)."""
    import torch
    from PIL import Image, ImageFilter

    t = torch.nn.functional.interpolate(torch.from_numpy(patch_map)[None, None].float(), size=(size, size),
                                        mode="bilinear")[0, 0].numpy()
    mx = float(t.max())
    if mx <= 0:
        return t
    img = Image.fromarray((t / mx * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=4))
    return np.asarray(img, dtype=np.float32) / 255.0 * mx


def aupro_components(maps: list[np.ndarray], gts: list[np.ndarray], max_fpr: float = 0.3) -> float:
    """MVTec AU-PRO: regions = connected components of the ground truth, FPR over all
    ground-truth-free pixels of all test images, integrated to ``max_fpr``."""
    from scipy.ndimage import label

    from ..detect.scorer import aupro

    reg_scores, reg_masks = [], []
    normal = np.concatenate([m[~g] for m, g in zip(maps, gts)])
    for m, g in zip(maps, gts):
        lab, n = label(g)
        for k in range(1, n + 1):
            reg_scores.append(m.ravel())
            reg_masks.append((lab == k).ravel())
    return aupro(reg_scores, reg_masks, normal, max_fpr=max_fpr)


# ======================================================================================
# ISCO's point detector on a single real view (secondary row of E0.6, second detector of E5b)
# ======================================================================================
ISCO_VOXEL, ISCO_NORMAL_R, ISCO_RADII, ISCO_SMOOTH_K = 0.4e-3, 1.0e-3, [1.0e-3, 2.0e-3, 4.0e-3], 96


def isco_points(xyz: np.ndarray) -> np.ndarray:
    """BTF foreground -> 0.4 mm voxel grid -> statistical outlier removal (20, 2.0)."""
    import open3d as o3d

    fg = btf_preprocess(xyz).reshape(-1, 3)
    fg = fg[np.all(fg != 0, axis=1)]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fg.astype(np.float64))).voxel_down_sample(ISCO_VOXEL)
    if len(pc.points) > 30:
        pc, _ = pc.remove_statistical_outlier(20, 2.0)
    return np.asarray(pc.points)


def isco_features(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(point features float16, points float32) of ISCO's detector (no xyz term)."""
    from ..detect.features import point_features

    P = isco_points(xyz)
    if len(P) < 10:
        return np.zeros((0, 33), np.float16), np.zeros((0, 3), np.float32)
    cam_dir = -P / np.linalg.norm(P, axis=1, keepdims=True)
    f = point_features(P, cam_dir, ISCO_NORMAL_R, ISCO_RADII, 0.0)
    return f.astype(np.float16), P.astype(np.float32)


def isco_image_score(bank, feats: np.ndarray, P: np.ndarray) -> float:
    """Max over points of the 96-NN mean of the memory-bank distances."""
    from ..detect.features import knn_indices

    if len(feats) == 0:
        return 0.0
    s = bank.score(feats.astype(np.float32))
    return float(s[knn_indices(P, ISCO_SMOOTH_K)].mean(axis=1).max())


# ======================================================================================
# E5b: the real camera, the support plane, 2.5D scan meshes and re-rendering
# ======================================================================================
@dataclass(frozen=True)
class FittedCamera:
    fx: float
    fy: float
    cx: float           # principal point in crop pixel coordinates (pixel centre = +0.5)
    cy: float
    width: int
    height: int
    rms_px: float

    def embed(self):
        """ISCO PinholeCamera (principal point at the image centre, square pixels) whose
        sub-ROI reproduces this crop: (camera, roi, principal-point error in px)."""
        from ..geometry.camera_model import PinholeCamera

        f = 0.5 * (self.fx + self.fy)
        ou = max(0, int(np.ceil(self.width - 2 * self.cx)))
        ov = max(0, int(np.ceil(self.height - 2 * self.cy)))
        Wf, Hf = int(round(2 * (self.cx + ou))), int(round(2 * (self.cy + ov)))
        Wf, Hf = max(Wf, ou + self.width), max(Hf, ov + self.height)
        cam = PinholeCamera(Wf, Hf, float(np.degrees(2 * np.arctan(Wf / 2.0 / f))))
        # largest displacement (px) over the crop between the fitted and the embedded pinhole
        uu, vv = np.meshgrid([0.5, self.width - 0.5], [0.5, self.height - 0.5])
        x, y = (uu - self.cx) / self.fx, (vv - self.cy) / self.fy
        err = max(np.abs(cam.fx * x + Wf / 2.0 - ou - uu).max(), np.abs(cam.fy * y + Hf / 2.0 - ov - vv).max())
        return cam, (ou, ov, ou + self.width, ov + self.height), float(err)


def fit_camera(paths: list[Path]) -> FittedCamera:
    """Least-squares pinhole ``u + 0.5 = fx x/z + cx`` over the valid pixels of ``paths``."""
    us, vs, xs, ys = [], [], [], []
    H = W = 0
    for p in paths:
        a = load_xyz(p)
        H, W = a.shape[:2]
        m = valid_mask(a)
        v, u = np.nonzero(m)
        P = a[m].astype(np.float64)
        us.append(u + 0.5), vs.append(v + 0.5), xs.append(P[:, 0] / P[:, 2]), ys.append(P[:, 1] / P[:, 2])
    u, v, x, y = map(np.concatenate, (us, vs, xs, ys))
    (fx, cx), (fy, cy) = np.polyfit(x, u, 1), np.polyfit(y, v, 1)
    rms = float(np.sqrt(np.mean(np.r_[u - (fx * x + cx), v - (fy * y + cy)] ** 2)))
    return FittedCamera(float(fx), float(fy), float(cx), float(cy), W, H, rms)


def support_frame(xyz: np.ndarray) -> np.ndarray:
    """T_world_cam with the world z = 0 plane on the support (BTF border plane refined by
    least squares on its inliers), +z towards the camera."""
    a, b, c, d = btf_plane(xyz)
    n = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    e = np.concatenate([xyz[:10].reshape(-1, 3), xyz[-10:].reshape(-1, 3), xyz[:, :10].reshape(-1, 3),
                        xyz[:, -10:].reshape(-1, 3)]).astype(np.float64)
    e = e[np.all(e != 0, axis=1)]
    p0 = -n * d / np.linalg.norm([a, b, c])
    e = e[np.abs((e - p0) @ n) < 0.004]
    c0 = e.mean(axis=0)
    n = np.linalg.svd(e - c0)[2][-1]
    if n @ (-c0) < 0:
        n = -n
    x = np.array([1.0, 0.0, 0.0]) - n[0] * n
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(n, x), n])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ c0
    return T


def height_above_support(xyz: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    h = xyz.reshape(-1, 3).astype(np.float64) @ T_world_cam[2, :3] + T_world_cam[2, 3]
    h = h.reshape(xyz.shape[:2])
    h[~valid_mask(xyz)] = np.nan
    return h


def object_mask(xyz: np.ndarray, T_world_cam: np.ndarray, min_height: float = 0.5e-3) -> np.ndarray:
    """Pixels of the object: above the support and in the largest 8-connected region."""
    from scipy.ndimage import label

    h = height_above_support(xyz, T_world_cam)
    m = np.nan_to_num(h, nan=-1.0) > min_height
    lab, n = label(m, structure=np.ones((3, 3)))
    if n == 0:
        return m
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)


def scan_mesh(xyz: np.ndarray, mask: np.ndarray, T_world_cam: np.ndarray, max_jump: float = 3.0e-3):
    """2.5D mesh (world frame) of the organized points under ``mask``: two triangles per
    2x2 pixel block whose four corners are valid and within ``max_jump`` in depth. 3 mm
    keeps the steep flanks of the MVTec objects (1 mm dropped them: 12 % of the pixels
    around the dowel re-rendered > 1 mm off, 2.7 % with 3 mm; 5 mm gives the same)."""
    from ..geometry.mesh_io import LabeledMesh
    from ..geometry.transforms import apply_T

    H, W, _ = xyz.shape
    ok = mask & valid_mask(xyz)
    z = xyz[..., 2]
    blk = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    zz = np.stack([z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]])
    blk &= (zz.max(axis=0) - zz.min(axis=0)) < max_jump
    i, j = np.nonzero(blk)
    idx = np.full((H, W), -1, np.int64)
    used = np.zeros((H, W), bool)
    for di, dj in ((0, 0), (0, 1), (1, 0), (1, 1)):
        used[i + di, j + dj] = True
    idx[used] = np.arange(int(used.sum()))
    a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
    faces = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])
    return LabeledMesh(apply_T(T_world_cam, xyz[used].astype(np.float64)), faces)


def rerender(xyz: np.ndarray, cam, roi, T_world_cam: np.ndarray, noise, rng: np.random.Generator,
             mesh=None) -> np.ndarray:
    """Organized (H, W, 3) camera-frame cloud of the scan's object (2.5D mesh) on the ideal
    support plane, seen by ``cam`` (ROI = the real crop) through ``noise``; 0 = invalid."""
    from ..datagen.sensor_noise import _belt_depth, apply_sensor_noise
    from ..render.pixels import ViewPixels
    from ..render.raycast_backend import RaycastRenderer

    if mesh is None:
        mesh = scan_mesh(xyz, object_mask(xyz, T_world_cam), T_world_cam)
    part = RaycastRenderer(cam, n_threads=1).render(mesh, np.eye(4), [(T_world_cam, roi)])[0]
    pix_all = cam.roi_pixels(roi)
    belt = np.setdiff1d(pix_all, part.pix)
    zb = _belt_depth(cam, T_world_cam, belt)
    d = cam.ray_dirs_cam(belt)
    cosb = np.abs((d / np.linalg.norm(d, axis=1, keepdims=True)) @ T_world_cam[2, :3])
    ok = np.isfinite(zb)
    vp = ViewPixels(np.r_[part.pix, belt[ok]], np.r_[part.depth, zb[ok]].astype(np.float32),
                    np.r_[part.label, np.zeros(int(ok.sum()), np.uint8)], np.r_[part.cos, cosb[ok]].astype(np.float32))
    vp = apply_sensor_noise(vp, cam, T_world_cam, noise, rng, belt_z=0.0)
    P = cam.unproject(vp.pix, vp.depth.astype(np.float64))
    u, v = cam.pix_to_uv(vp.pix)
    out = np.zeros((roi[3] - roi[1], roi[2] - roi[0], 3), np.float32)
    out[v - roi[1], u - roi[0]] = P
    return out


def ideal_render(xyz, cam, roi, T_world_cam, mesh=None) -> tuple[np.ndarray, np.ndarray]:
    """Noise-free re-render (no filters): (organized cloud, |cos| incidence image, nan off the object)."""
    from ..render.raycast_backend import RaycastRenderer
    from ..schemas.sensor import NoiseModel

    if mesh is None:
        mesh = scan_mesh(xyz, object_mask(xyz, T_world_cam), T_world_cam)
    nz = NoiseModel(name="ideal", sigma0_m=0.0, dropout_p0=0.0, dropout_theta_c_deg=90.0, flying_p=0.0, edge_filter_m=0.0)
    img = rerender(xyz, cam, roi, T_world_cam, nz, np.random.default_rng(0), mesh)
    part = RaycastRenderer(cam, n_threads=1).render(mesh, np.eye(4), [(T_world_cam, roi)])[0]
    u, v = cam.pix_to_uv(part.pix)
    cos = np.full(img.shape[:2], np.nan, np.float32)
    cos[v - roi[1], u - roi[0]] = part.cos
    return img, cos


# --- sensor statistics (identical code for real and re-rendered clouds) ------------------
def _local_residuals(xyz: np.ndarray, sel: np.ndarray, win: int, quadratic: bool, rng, n: int = 400):
    """Median RMS residual of local plane (or quadric) fits in fully valid ``win``² windows
    centred in ``sel``, and the lag-1 autocorrelation of the residuals."""
    H, W, _ = xyz.shape
    m = valid_mask(xyz)
    ii, jj = np.nonzero(sel)
    if len(ii) == 0:
        return float("nan"), float("nan")
    res, rms = [], []
    for k in rng.permutation(len(ii))[: n * 4]:
        i, j = ii[k] - win // 2, jj[k] - win // 2
        if i < 0 or j < 0 or i + win > H or j + win > W or not m[i:i + win, j:j + win].all() \
                or not sel[i:i + win, j:j + win].all():
            continue
        q = xyz[i:i + win, j:j + win].reshape(-1, 3).astype(np.float64)
        q = q - q.mean(axis=0)
        loc = q @ np.linalg.svd(q)[2].T
        X, zl = loc[:, :2], loc[:, 2]
        A = np.c_[np.ones(len(X)), X, X ** 2, X[:, :1] * X[:, 1:]] if quadratic else np.c_[np.ones(len(X)), X]
        r = zl - A @ np.linalg.lstsq(A, zl, rcond=None)[0]
        rms.append(np.sqrt((r ** 2).sum() / (len(r) - A.shape[1])))
        res.append(r.reshape(win, win))
        if len(rms) >= n:
            break
    if not rms:
        return float("nan"), float("nan")
    R = np.asarray(res)
    return float(np.median(rms)), float(np.mean(R[:, :, 1:] * R[:, :, :-1]) / max(np.mean(R ** 2), 1e-30))


def scan_statistics(xyz: np.ndarray, ref: np.ndarray, ref_cos: np.ndarray, obj: np.ndarray,
                    T_world_cam: np.ndarray, seed: int = 0) -> dict:
    """``ref``/``ref_cos``: noise-free re-render and its incidence (the geometry);
    ``obj``: object pixels of the real scan."""
    from scipy.ndimage import binary_dilation

    rng = np.random.default_rng(seed)
    m = valid_mask(xyz)
    h_ref = height_above_support(ref, T_world_cam)
    on_obj = np.isfinite(ref_cos)
    top = on_obj & (np.nan_to_num(h_ref, nan=-1) > np.nanpercentile(h_ref[on_obj], 99) - 1.0e-3) & (ref_cos > np.cos(np.radians(35)))
    support = (~binary_dilation(on_obj | obj, iterations=8)) & valid_mask(ref)
    s_top, ac_top = _local_residuals(xyz, top & m, 5, True, rng)
    s_sup, ac_sup = _local_residuals(xyz, support & m, 7, False, rng)
    inc = np.degrees(np.arccos(np.clip(ref_cos, 0, 1)))
    bins = np.arange(0, 91, 10)
    valid_vs_inc = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = on_obj & (inc >= lo) & (inc < hi)
        valid_vs_inc.append(float(m[b].mean()) if b.sum() >= 20 else float("nan"))
    both = m & valid_mask(ref)
    near_obj = binary_dilation(on_obj, iterations=3)
    dz = np.abs(xyz[..., 2] - ref[..., 2])
    return {"sigma_part_mm": s_top * 1e3, "ac1_part": ac_top, "sigma_support_mm": s_sup * 1e3, "ac1_support": ac_sup,
            "valid_object": float(m[on_obj].mean()), "valid_support": float(m[support].mean()),
            "valid_vs_incidence": valid_vs_inc,
            "outlier_frac_near_object": float((dz[both & near_obj] > 1.0e-3).mean()) if (both & near_obj).any() else float("nan")}


# ======================================================================================
# workers
# ======================================================================================
_G: dict = {}


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _init(threads: int = 1) -> None:
    import torch

    from ..utils.seeds import pin_worker

    pin_worker(threads)    # open3d's TBB pools follow the affinity mask, not OMP_NUM_THREADS
    torch.set_num_threads(threads)
    _G.clear()


def _btf_job(path: str) -> np.ndarray:
    return btf_features(load_xyz(path))


def _isco_job(path: str):
    return isco_features(load_xyz(path))


def level_noise(level: str, scene_depth_m: float):
    """``"N2"``: the level as defined (sigma0 at z0 = 0.25 m, scaled by (z/z0)^2 to the scene);
    ``"N2@native"``: the same level with z0 = the scene depth, i.e. the sensor's noise at its
    own working distance transplanted to the MVTec scene."""
    from .. import config_loader as C

    name, _, mode = level.partition("@")
    nz = C.noise(name)
    if mode == "native":
        nz = nz.model_copy(update={"name": level, "z0_m": float(scene_depth_m)})
    elif mode:
        raise ValueError(level)
    return nz


def _e5b_job(task: tuple) -> dict:
    """One real scan under every condition: real + re-rendered levels -> features + stats."""
    path, cam_d, levels, seed, idx = task
    cam_fit = FittedCamera(**cam_d)
    cam, roi, _ = cam_fit.embed()
    xyz = load_xyz(path)
    T = support_frame(xyz)
    obj = object_mask(xyz, T)
    mesh = scan_mesh(xyz, obj, T)
    ref, ref_cos = ideal_render(xyz, cam, roi, T, mesh)
    out = {}
    clouds = {"real": xyz}
    depth = float(np.median(xyz[obj][:, 2]))
    for li, lv in enumerate(levels):
        clouds[lv] = rerender(xyz, cam, roi, T, level_noise(lv, depth), np.random.default_rng([seed, li, idx]), mesh)
    for name, c in clouds.items():
        f_isco, nbr = isco_features(c)
        out[name] = {"btf": btf_features(c), "isco": (f_isco, nbr),
                     "stats": scan_statistics(c, ref, ref_cos, obj, T, seed=idx)}
    out["_ref_n0_err_mm"] = float(np.nanmedian(np.abs(ref[..., 2] - xyz[..., 2])[valid_mask(ref) & valid_mask(xyz) & obj]) * 1e3)
    return out


def _pool(n_workers: int):
    """Spawned single-threaded workers (the thread variables are set while spawning, so
    BLAS / OpenMP in the children read them at import)."""
    import multiprocessing as mp

    import torch

    from ..utils.seeds import core_order_env

    torch.set_num_threads(2)      # the parent only runs the GPU bank and bookkeeping
    old = {k: os.environ.get(k) for k in _THREAD_VARS}
    os.environ.update({k: "1" for k in _THREAD_VARS})
    try:
        with core_order_env():
            pool = mp.get_context("spawn").Pool(n_workers, initializer=_init, initargs=(1,))
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return pool


# ======================================================================================
# experiments
# ======================================================================================
def _score_isco(bank_feats: list, items: list):
    from ..detect.memory_bank import MemoryBank

    bank = MemoryBank(coreset_ratio=0.03, coreset_max=10000).fit([f.astype(np.float32) for f, _ in bank_feats if len(f)])
    return [isco_image_score(bank, f, P) for f, P in items]


def e0_6(categories=("dowel", "cable_gland"), n_workers: int = 4, out: str | None = None, **_) -> Path:
    """BTF FPFH reproduction (bank = all train goods, AUROC on the full test split) and
    ISCO's point detector on the same split."""
    from ..detect import scorer
    from .sanity import exp_dir

    d = exp_dir("e0_6", out)
    res = {}
    for cat in categories:
        if not available(cat):
            res[cat] = {"status": "missing - download MVTec 3D-AD (registration form) into " + str(root())}
            continue
        train, test = samples(cat, "train"), samples(cat, "test")
        with _pool(n_workers) as pool:
            tr = pool.map(_btf_job, [str(s.path) for s in train], chunksize=4)
            te = pool.map(_btf_job, [str(s.path) for s in test], chunksize=4)
            tr_i = pool.map(_isco_job, [str(s.path) for s in train], chunksize=4)
            te_i = pool.map(_isco_job, [str(s.path) for s in test], chunksize=4)
        bank = BTFPatchCore().fit(tr)
        shape = load_xyz(train[0].path).shape[:2]
        scores, maps, gts = [], [], []
        for s, p in zip(test, te):
            sc, pm = bank.score(p)
            scores.append(sc)
            maps.append(btf_pixel_map(pm))
            gts.append(resize_nearest(load_gt(s.gt_path, shape)))
        good = np.array([s.dtype == "good" for s in test])
        sc = np.asarray(scores)
        i_scores = np.asarray(_score_isco(tr_i, te_i))
        types = sorted({s.dtype for s in test} - {"good"})
        per_type = {t: scorer.auroc(sc[good], sc[np.array([s.dtype == t for s in test])]) for t in types}
        res[cat] = {
            "btf_fpfh": {"image_auroc": scorer.auroc(sc[good], sc[~good]),
                         "aupro_0.3": aupro_components(maps, gts),
                         "image_auroc_by_type": per_type,
                         "published": BTF_PUBLISHED.get(cat),
                         "bank_patches": int(bank.lib.shape[0])},
            "isco_fpfh": {"image_auroc": scorer.auroc(i_scores[good], i_scores[~good]),
                          "image_auroc_by_type": {t: scorer.auroc(i_scores[good], i_scores[np.array([s.dtype == t for s in test])])
                                                  for t in types}},
            "n_train": len(train), "n_test_good": int(good.sum()), "n_test_anomalous": int((~good).sum()),
        }
        pub = BTF_PUBLISHED[cat]["image_auroc"]
        res[cat]["btf_fpfh"]["delta_vs_published"] = res[cat]["btf_fpfh"]["image_auroc"] - pub
        res[cat]["btf_fpfh"]["within_0.02"] = abs(res[cat]["btf_fpfh"]["delta_vs_published"]) <= 0.02
        log.info(f"E0.6 {cat}: BTF I-AUROC {res[cat]['btf_fpfh']['image_auroc']:.3f} (published {pub}), "
                 f"ISCO {res[cat]['isco_fpfh']['image_auroc']:.3f}")
    write_json(d / "summary.json", res)
    _write_md(d / "summary.md", "E0.6 detector reproduction on MVTec 3D-AD", res)
    return d


def e5b(categories=("dowel", "cable_gland"),
        levels=("N0", "N1", "N2", "N3", "N1@native", "N2@native", "N3@native"), n_workers: int = 4, seed: int = 0,
        bank_max: int = 200, B: int = 1000, out: str | None = None, **_) -> Path:
    """Sensor gap: real scans vs the same scans re-rendered through ISCO's sensor model.

    Levels: ``"N2"`` as defined (sigma0 at 0.25 m scaled by (z/0.25)^2 to the MVTec scene
    at 0.55-0.66 m) and ``"N2@native"`` (sigma0 at the scene depth: the sensor at its own
    working distance). Split per category (identical for every condition): bank = the first
    ``min(bank_max, n_train - 60)`` train goods; threshold normals = the remaining train,
    all validation and the test goods; defects = all test defects. FNR @ 1 % / 5 % FPR
    (joint bootstrap), AUROC on the test split, FNR by type. Decision (plan): a level
    reproduces the real sensor if |FNR_real - FNR_level| <= 0.15 in >= 2/3 of the types.
    """
    from ..detect import scorer
    from .sanity import exp_dir

    d = exp_dir("e5b", out)
    res = {}
    for cat in categories:
        if not available(cat):
            res[cat] = {"status": "missing MVTec 3D-AD " + cat, "path": str(root() / cat)}
            continue
        train, val, test = samples(cat, "train"), samples(cat, "validation"), samples(cat, "test")
        n_bank = min(bank_max, len(train) - 60)
        bank_s, thr_s = train[:n_bank], train[n_bank:] + val + [s for s in test if s.dtype == "good"]
        def_s = [s for s in test if s.dtype != "good"]
        allS = bank_s + thr_s + def_s
        cam_fit = fit_camera([s.path for s in train[::12]])
        cam, roi, emb_err = cam_fit.embed()
        log.info(f"E5b {cat}: camera fx {cam_fit.fx:.1f} fy {cam_fit.fy:.1f} c ({cam_fit.cx:.1f}, {cam_fit.cy:.1f}) "
                 f"rms {cam_fit.rms_px:.2f} px; embedded {cam.width}x{cam.height}, roi {roi}, err {emb_err:.2f} px")
        tasks = [(str(s.path), cam_fit.__dict__, tuple(levels), seed, i) for i, s in enumerate(allS)]
        with _pool(n_workers) as pool:
            rows = pool.map(_e5b_job, tasks, chunksize=2)
        conds = ["real"] + list(levels)
        nb, nt = len(bank_s), len(thr_s)
        types = sorted({s.dtype for s in def_s})
        dtype = np.array([s.dtype for s in def_s])
        test_good = np.array([s.split == "test" for s in thr_s])
        r_cat = {"camera": {**cam_fit.__dict__, "embedded": [cam.width, cam.height, cam.hfov_deg], "roi": roi,
                            "embed_err_px": emb_err},
                 "split": {"bank": nb, "threshold_normals": nt, "defects": len(def_s),
                           "defects_by_type": {t: int((dtype == t).sum()) for t in types}},
                 "n0_vs_real_depth_median_mm": float(np.median([r["_ref_n0_err_mm"] for r in rows])),
                 "stats": {}, "detection": {}}
        for c in conds:
            st = [r[c]["stats"] for r in rows]
            agg = {}
            for k in st[0]:
                if k == "valid_vs_incidence":
                    agg[k] = np.nanmean(np.array([s[k] for s in st], float), axis=0).round(4).tolist()
                else:
                    agg[k] = float(np.nanmedian([s[k] for s in st]))
            r_cat["stats"][c] = agg
            det = {}
            # BTF detector
            bank = BTFPatchCore().fit([r[c]["btf"] for r in rows[:nb]])
            sc = np.array([bank.score(r[c]["btf"])[0] for r in rows[nb:]])
            # ISCO detector
            si = np.array(_score_isco([r[c]["isco"] for r in rows[:nb]], [r[c]["isco"] for r in rows[nb:]]))
            for name, s in (("btf_fpfh", sc), ("isco_fpfh", si)):
                v, dd = s[:nt], s[nt:]
                f1, lo1, hi1 = scorer.bootstrap_fnr(v, dd, 0.01, B=B, seed=seed)
                f5, lo5, hi5 = scorer.bootstrap_fnr(v, dd, 0.05, B=B, seed=seed)
                det[name] = {"fnr_1": f1, "fnr_1_ci": [lo1, hi1], "fnr_5": f5, "fnr_5_ci": [lo5, hi5],
                             "auroc_test": scorer.auroc(v[test_good], dd), "auroc_all_normals": scorer.auroc(v, dd),
                             "fnr_1_by_type": {t: scorer.fnr_at_fpr(v, dd[dtype == t], 0.01) for t in types},
                             "fnr_5_by_type": {t: scorer.fnr_at_fpr(v, dd[dtype == t], 0.05) for t in types}}
            r_cat["detection"][c] = det
        # decision per level and detector
        dec = {}
        for name in ("btf_fpfh", "isco_fpfh"):
            real = r_cat["detection"]["real"][name]
            dec[name] = {}
            for lv in levels:
                sim = r_cat["detection"][lv][name]
                diffs = {t: sim["fnr_1_by_type"][t] - real["fnr_1_by_type"][t] for t in types}
                n_ok = sum(abs(x) <= 0.15 for x in diffs.values())
                dec[name][lv] = {"delta_fnr_1_by_type": diffs, "types_within_0.15": n_ok,
                                 "pass": n_ok >= int(np.ceil(2 * len(types) / 3)),
                                 "mean_abs_delta": float(np.mean(np.abs(list(diffs.values())))),
                                 "delta_auroc_test": sim["auroc_test"] - real["auroc_test"]}
            dec[name]["closest_level"] = min(levels, key=lambda lv: dec[name][lv]["mean_abs_delta"])
        r_cat["decision"] = dec
        sig_real = r_cat["stats"]["real"]["sigma_part_mm"]
        r_cat["closest_level_by_sigma_part"] = min(
            levels, key=lambda lv: abs(np.nan_to_num(r_cat["stats"][lv]["sigma_part_mm"], nan=1e9) - sig_real))
        res[cat] = r_cat
        log.info(f"E5b {cat}: real sigma_part {sig_real:.4f} mm; decisions "
                 + ", ".join(f"{n}: closest {dec[n]['closest_level']}" for n in dec))
        write_json(d / "summary.json", res)
    write_json(d / "summary.json", res)
    _write_md(d / "summary.md", "E5b sensor gap (real MVTec 3D-AD scans vs ISCO re-renders)", res)
    return d


def _write_md(path: Path, title: str, res: dict) -> None:
    import json

    from ..utils.io import atomic_write_text

    atomic_write_text(path, f"# {title}\n\n```json\n{json.dumps(res, indent=1, default=str)}\n```\n")
