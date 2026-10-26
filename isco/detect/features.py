"""Point features for the memory bank.

v1 (ADR-0005): multi-scale FPFH (open3d) on the voxelised merged cloud, histograms
scaled to sum 1 per sub-histogram, concatenated with part-frame coordinates (mm)
times ``xyz_weight`` so the bank is also position-aware (catches "material at a place
where normal parts have none").
"""
from __future__ import annotations

import numpy as np


def estimate_normals(xyz: np.ndarray, cam_dir: np.ndarray, radius: float, max_nn: int = 30) -> np.ndarray:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    n = np.asarray(pcd.normals)
    flip = (n * cam_dir).sum(axis=1) < 0
    n[flip] *= -1
    return n.astype(np.float32)


def fpfh_multiscale(xyz: np.ndarray, normals: np.ndarray, radii: list[float], max_nn: int = 100) -> np.ndarray:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    feats = []
    for r in radii:
        f = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=max_nn))
        feats.append(np.asarray(f.data).T / 100.0)        # three 11-bin sub-histograms, each sums to 1
    return np.concatenate(feats, axis=1).astype(np.float32)


def point_features(xyz: np.ndarray, cam_dir: np.ndarray, normal_radius: float, fpfh_radii: list[float],
                   xyz_weight: float, max_nn: int = 100) -> np.ndarray:
    n = estimate_normals(xyz, cam_dir, normal_radius)
    f = fpfh_multiscale(xyz, n, fpfh_radii, max_nn)
    if xyz_weight > 0:
        f = np.concatenate([f, (xyz * 1e3 * xyz_weight).astype(np.float32)], axis=1)
    return f


def knn_indices(xyz: np.ndarray, k: int) -> np.ndarray:
    from scipy.spatial import cKDTree

    k = min(k, len(xyz))
    if k <= 1:
        return np.arange(len(xyz))[:, None]
    _, idx = cKDTree(xyz).query(xyz, k=k)
    return idx.astype(np.int32)


# --------------------------------------------------------------------------------------
# v2 backbone (E6): DINOv2 features of rendered normal-map views, back-projected to points
# (CPMF / ISMP style, no training). Images are made in the workers; the network runs on
# the GPU in the main process.
# --------------------------------------------------------------------------------------
def view_normal_image(vp, cam, size: int = 224, pad: int = 4):
    """Camera-frame normal map of one view as an RGB uint8 image (square crop around the
    part, resized to ``size``), plus the crop (u0, v0, side) for pixel -> image mapping."""
    import cv2

    u, v = cam.pix_to_uv(vp.pix)
    cu, cv_ = (u.min() + u.max()) / 2.0, (v.min() + v.max()) / 2.0
    side = int(max(u.max() - u.min(), v.max() - v.min()) + 1 + 2 * pad)
    u0, v0 = int(np.floor(cu - side / 2)), int(np.floor(cv_ - side / 2))
    depth = np.full((side, side), np.nan, np.float64)
    depth[v - v0, u - u0] = vp.depth
    gu, gv = np.meshgrid(np.arange(u0, u0 + side), np.arange(v0, v0 + side))
    X = (gu + 0.5 - cam.cx) / cam.fx * depth
    Y = (gv + 0.5 - cam.cy) / cam.fy * depth
    P = np.stack([X, Y, depth], axis=2)
    du = np.zeros_like(P)
    dv = np.zeros_like(P)
    du[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dv[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(du, dv)
    nn = np.linalg.norm(n, axis=2, keepdims=True)
    ok = np.isfinite(nn[..., 0]) & (nn[..., 0] > 0)
    n = np.where(ok[..., None], n / np.where(nn > 0, nn, 1), 0.0)
    n[ok & (n[..., 2] > 0)] *= -1                      # face the camera
    img = np.where(ok[..., None], (n + 1.0) * 127.5, 0.0).astype(np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img, (u0, v0, side)


def pixel_to_image_coords(pix: np.ndarray, cam, crop, size: int = 224) -> np.ndarray:
    u0, v0, side = crop
    u, v = cam.pix_to_uv(pix)
    s = size / side
    return np.stack([(u - u0 + 0.5) * s, (v - v0 + 0.5) * s], axis=1).astype(np.float32)


class DinoFeatures:
    """DINOv2 patch features sampled at point image coordinates (GPU, fp16)."""

    def __init__(self, model: str = "facebook/dinov2-small", device: str = "cuda", size: int = 224):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model).to(self.dev).eval()
        self.size = size
        self.patch = 14
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)

    def __call__(self, imgs: np.ndarray, view_idx: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """imgs (V,H,W,3) uint8; per point: view index and (x, y) image coords -> (N, C)."""
        torch = self.torch
        x = torch.from_numpy(imgs).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / self.std
        with torch.no_grad(), torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.dev.type == "cuda"):
            tok = self.model(pixel_values=x).last_hidden_state[:, 1:, :]
        g = self.size // self.patch
        fmap = tok.float().reshape(len(imgs), g, g, -1).permute(0, 3, 1, 2)       # V,C,g,g
        # normalised sampling grid in [-1, 1] (align_corners=False convention)
        xy = torch.from_numpy(coords).to(self.dev) / self.size * 2.0 - 1.0
        out = torch.empty(len(coords), fmap.shape[1], device=self.dev)
        vi = torch.from_numpy(view_idx.astype(np.int64)).to(self.dev)
        for v in range(len(imgs)):
            m = vi == v
            if m.any():
                grid = xy[m].view(1, -1, 1, 2)
                out[m] = torch.nn.functional.grid_sample(fmap[v:v + 1], grid, align_corners=False).view(fmap.shape[1], -1).T
        out = torch.nn.functional.normalize(out, dim=1)
        return out.cpu().numpy().astype(np.float32)
