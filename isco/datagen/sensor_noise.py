"""Parametric structured-light depth noise, applied per view before unprojection.

Isaac Sim 5.1 renders noise-free geometric depth (ADR-0001/0003), so realism comes from
this documented post-process (levels N0-N3, N2-MS in ``configs/noise``; main level to be
fixed by the MVTec 3D-AD fit, E5b):

1. **flying pixels** - a part pixel next to a depth jump larger than ``flying_jump_m``
   (to another part pixel or to the belt behind it) is, with probability ``flying_p``,
   moved a random fraction of the jump towards the far side;
2. **incidence dropout** - ``p = p0 + (1 - p0) * sigmoid((theta - theta_c) / width)``,
   always dropped beyond 85 deg;
3. **axial Gaussian noise** - ``sigma = sigma0 * (z / z0)^2 * (1 + g * (1 - cos theta))``;
4. optional quantisation;
5. **on-board edge filter** - pixels next to a measured depth jump larger than
   ``edge_filter_m`` (to a part pixel or the belt) are invalidated, as structured-light
   cameras do by default. Without it the flying pixels of oblique views form veils
   reaching towards the belt that survive outlier removal, and their number (and the
   image-level maximum score) grows with every added camera.

Motion blur (``apply_motion_blur``) models a global-shutter structured-light camera
imaging a part that moves ``L = v * t_acq`` during the exposure: each pixel reports the
*average* depth along the motion path (image-space box filter along the projected belt
direction), with the mean shift ``L/2`` removed (encoder-compensated). Flat surfaces are
unaffected; features narrower than ``L`` are smeared - so faster belts lose small
defects (e.g. 0.2-0.5 mm cracks at 0.6 m/s with 2 ms exposure). A pixel whose exposure
crosses a depth jump larger than the sensor's jump threshold is invalidated (inconsistent
code) instead of averaging two surfaces into a point that floats off the part.
"""
from __future__ import annotations

import numpy as np

from ..geometry.camera_model import PinholeCamera
from ..render.pixels import ViewPixels
from ..schemas.sensor import NoiseModel


def _belt_depth(cam: PinholeCamera, T_world_cam: np.ndarray, pix: np.ndarray, belt_z: float = 0.0) -> np.ndarray:
    d_cam = cam.ray_dirs_cam(pix)                    # z = 1
    d_w = d_cam @ T_world_cam[:3, :3].T
    oz = T_world_cam[2, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (belt_z - oz) / d_w[:, 2]
    return np.where(t > 0, t, np.inf)


def flying_pixel_jump(vp: ViewPixels, cam: PinholeCamera, T_world_cam: np.ndarray, belt_z: float = 0.0,
                      absolute: bool = False) -> np.ndarray:
    """Largest positive depth step from each part pixel to its 4-neighbours (m);
    ``absolute``: largest step in either direction."""
    u, v = cam.pix_to_uv(vp.pix)
    if len(u) == 0:
        return np.zeros(0)
    u0, v0 = u.min() - 1, v.min() - 1
    W, H = u.max() - u0 + 2, v.max() - v0 + 2
    img = np.full((H, W), np.nan)
    img[v - v0, u - u0] = vp.depth
    jump = np.zeros(len(u))
    for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nu, nv = u + du, v + dv
        zn = img[nv - v0, nu - u0]
        bg = np.isnan(zn)
        if bg.any():
            nu_b = np.clip(nu[bg], 0, cam.width - 1)
            nv_b = np.clip(nv[bg], 0, cam.height - 1)
            zn = zn.copy()
            zn[bg] = _belt_depth(cam, T_world_cam, nv_b.astype(np.int64) * cam.width + nu_b, belt_z)
        step = zn - vp.depth
        jump = np.maximum(jump, np.nan_to_num(np.abs(step) if absolute else step, nan=0.0, posinf=0.0))
    return jump


def apply_sensor_noise(vp: ViewPixels, cam: PinholeCamera, T_world_cam: np.ndarray, noise: NoiseModel,
                       rng: np.random.Generator, belt_z: float = 0.0) -> ViewPixels:
    if len(vp) == 0:
        return vp
    z = vp.depth.astype(np.float64).copy()
    cos = np.clip(vp.cos.astype(np.float64), 1e-6, 1.0)
    theta = np.degrees(np.arccos(cos))
    keep = np.ones(len(z), bool)
    # 1. flying pixels
    if noise.flying_p > 0:
        jump = flying_pixel_jump(vp, cam, T_world_cam, belt_z)
        edge = (jump > noise.flying_jump_m) & np.isfinite(jump)
        fly = edge & (rng.uniform(size=len(z)) < noise.flying_p)
        z[fly] += rng.uniform(0.0, 1.0, size=int(fly.sum())) * np.minimum(jump[fly], 0.05)
    # 2. incidence dropout
    if noise.dropout_p0 > 0 or noise.dropout_theta_c_deg < 90:
        p = noise.dropout_p0 + (1 - noise.dropout_p0) / (1 + np.exp(-(theta - noise.dropout_theta_c_deg) / max(noise.dropout_width_deg, 1e-6)))
        keep &= rng.uniform(size=len(z)) >= p
        keep &= theta < 85.0
    # 3. axial noise
    if noise.sigma0_m > 0:
        sigma = noise.sigma0_m * (z / noise.z0_m) ** 2 * (1 + noise.incidence_gain * (1 - cos))
        z = z + rng.normal(size=len(z)) * sigma
    # 4. quantisation
    if noise.quant_m > 0:
        z = np.round(z / noise.quant_m) * noise.quant_m
    # 5. on-board edge filter, on the measured depth image (before dropout holes)
    if noise.edge_filter_m > 0:
        meas = ViewPixels(vp.pix, z.astype(np.float32), vp.label, vp.cos, {})
        keep &= flying_pixel_jump(meas, cam, T_world_cam, belt_z, absolute=True) <= noise.edge_filter_m
    out = ViewPixels(vp.pix[keep], z[keep].astype(np.float32), vp.label[keep], vp.cos[keep], dict(vp.meta))
    return out


def apply_motion_blur(vp: ViewPixels, cam: PinholeCamera, T_world_cam: np.ndarray, belt_dir_world: np.ndarray,
                      blur_len_m: float, n_samples: int = 9, jump_m: float | None = None) -> ViewPixels:
    """Exposure-averaged depth for a part translating ``blur_len_m`` along ``belt_dir_world``;
    pixels whose (motion-compensated) depth jumps by more than ``jump_m`` between
    consecutive exposure samples are invalidated."""
    if len(vp) == 0 or blur_len_m <= 0:
        return vp
    u, v = cam.pix_to_uv(vp.pix)
    pc = cam.unproject(vp.pix, vp.depth)
    b_cam = T_world_cam[:3, :3].T @ np.asarray(belt_dir_world, dtype=float)
    P = pc.mean(axis=0)
    u0, v0, _ = cam.project(P[None])
    u1, v1, _ = cam.project((P + b_cam * 1e-3)[None])
    du_mm, dv_mm = float(u1[0] - u0[0]), float(v1[0] - v0[0])
    dz_mm = float(b_cam[2]) * 1e-3
    ua, va = u.min() - 2, v.min() - 2
    Wd, Hd = u.max() - ua + 3, v.max() - va + 3
    img = np.full((Hd, Wd), np.nan)
    img[v - va, u - ua] = vp.depth
    acc = np.zeros(len(u))
    cnt = np.zeros(len(u))
    prev = np.full(len(u), np.nan)
    max_jump = np.zeros(len(u))
    for i in range(n_samples):
        s_mm = (blur_len_m * 1e3) * ((i + 0.5) / n_samples - 0.5)       # centred: mean shift removed
        su = np.rint(u - s_mm * du_mm).astype(np.int64) - ua
        sv = np.rint(v - s_mm * dv_mm).astype(np.int64) - va
        ok = (su >= 0) & (su < Wd) & (sv >= 0) & (sv < Hd)
        z = np.full(len(u), np.nan)
        z[ok] = img[sv[ok], su[ok]]
        good = np.isfinite(z)
        zc = z + s_mm * dz_mm
        both = good & np.isfinite(prev)
        max_jump[both] = np.maximum(max_jump[both], np.abs(zc[both] - prev[both]))
        prev[good] = zc[good]
        acc[good] += zc[good]
        cnt[good] += 1
    keep = cnt >= 0.5 * n_samples           # mixed fg/bg pixels are invalidated by the sensor
    if jump_m is not None:
        keep &= max_jump <= jump_m          # ... and so are pixels straddling an internal jump
    out = vp.subset(keep)
    out.depth = (acc[keep] / cnt[keep]).astype(np.float32)
    return out
