"""Pinhole camera model and the discrete camera-slot grid.

Pixel convention (shared by the embree renderer, the Isaac capture and the
unprojection): pixel ``(u, v)`` (column, row) covers ``[u, u+1) x [v, v+1)`` and its
centre is ``(u + 0.5, v + 0.5)``; the principal point is ``(W/2, H/2)`` in these
continuous coordinates; ``fx = fy = (W/2) / tan(hfov/2)`` (square pixels). A global
pixel index is ``pix = v * W + u``. Depth is the *distance to the image plane*
(camera-frame z), matching Isaac Sim's ``distance_to_image_plane`` annotator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from .transforms import apply_T, inv_T, look_at_opencv


@dataclass(frozen=True)
class PinholeCamera:
    width: int = 1024
    height: int = 1024
    hfov_deg: float = 20.0

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / np.tan(np.radians(self.hfov_deg) / 2.0)

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def isaac_apertures(self, focal_length_mm: float = 24.0) -> tuple[float, float, float]:
        """(focal_length_mm, horizontal_aperture_mm, vertical_aperture_mm) for USD cameras."""
        h_ap = 2.0 * focal_length_mm * np.tan(np.radians(self.hfov_deg) / 2.0)
        return focal_length_mm, h_ap, h_ap * self.height / self.width

    def footprint(self, depth: np.ndarray | float) -> np.ndarray | float:
        """Pixel footprint (metres per pixel) on a fronto-parallel surface at ``depth``."""
        return np.asarray(depth) / self.fx

    # -- projection ---------------------------------------------------------
    def project(self, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Continuous pixel coords (u, v) and depth z of camera-frame points."""
        z = pts_cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * pts_cam[:, 0] / z + self.cx
            v = self.fy * pts_cam[:, 1] / z + self.cy
        return u, v, z

    def in_image(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (z > 0) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)

    def pix_to_uv(self, pix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pix = np.asarray(pix, dtype=np.int64)
        return pix % self.width, pix // self.width

    def ray_dirs_cam(self, pix: np.ndarray) -> np.ndarray:
        """Unnormalised camera-frame ray directions with z = 1 through pixel centres."""
        u, v = self.pix_to_uv(pix)
        x = (u + 0.5 - self.cx) / self.fx
        y = (v + 0.5 - self.cy) / self.fy
        return np.stack([x, y, np.ones_like(x, dtype=float)], axis=1)

    def unproject(self, pix: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Camera-frame points from pixel indices and z-depth."""
        return self.ray_dirs_cam(pix) * np.asarray(depth, dtype=float)[:, None]

    def roi_for_sphere(self, T_world_cam: np.ndarray, center_world: np.ndarray, radius: float,
                       margin_px: int = 4) -> tuple[int, int, int, int]:
        """Pixel ROI ``(u0, v0, u1, v1)`` (half-open) covering a world-space sphere."""
        c = apply_T(inv_T(T_world_cam), np.asarray(center_world, dtype=float)[None])[0]
        # conservative: project the sphere's bounding box corners
        offs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float) * radius
        pts = c[None] + offs
        pts[:, 2] = np.maximum(pts[:, 2], 1e-3)
        u, v, _ = self.project(pts)
        u0 = int(np.clip(np.floor(u.min()) - margin_px, 0, self.width))
        u1 = int(np.clip(np.ceil(u.max()) + margin_px, 0, self.width))
        v0 = int(np.clip(np.floor(v.min()) - margin_px, 0, self.height))
        v1 = int(np.clip(np.ceil(v.max()) + margin_px, 0, self.height))
        return u0, v0, u1, v1

    def roi_pixels(self, roi: tuple[int, int, int, int]) -> np.ndarray:
        u0, v0, u1, v1 = roi
        uu, vv = np.meshgrid(np.arange(u0, u1), np.arange(v0, v1))
        return (vv.ravel().astype(np.int64) * self.width + uu.ravel()).astype(np.int64)


@dataclass(frozen=True)
class Slot:
    slot_id: int
    i_az: int
    i_el: int
    i_dist: int
    azimuth_deg: float
    elevation_deg: float
    distance_m: float

    @property
    def cell(self) -> tuple[int, int]:
        """(azimuth, elevation) cell; two cameras may not share a cell."""
        return (self.i_az, self.i_el)


@dataclass(frozen=True)
class SlotGrid:
    """Discretised camera hemisphere around the focus point (ADR-0003).

    Slot id ordering: ``k = (i_dist * n_el + i_el) * n_az + i_az``. Azimuth 0 places the
    camera on the +X (downstream) side; azimuth grows counter-clockwise seen from +Z.
    """

    azimuths_deg: tuple[float, ...] = tuple(float(a) for a in range(0, 360, 30))
    elevations_deg: tuple[float, ...] = (20.0, 40.0, 60.0, 80.0)
    distances_m: tuple[float, ...] = (0.25, 0.40)

    @property
    def n_az(self) -> int:
        return len(self.azimuths_deg)

    @property
    def n_el(self) -> int:
        return len(self.elevations_deg)

    @property
    def n_dist(self) -> int:
        return len(self.distances_m)

    @property
    def n_slots(self) -> int:
        return self.n_az * self.n_el * self.n_dist

    @property
    def az_step_deg(self) -> float:
        return 360.0 / self.n_az

    def slot_id(self, i_az: int, i_el: int, i_dist: int) -> int:
        return (i_dist * self.n_el + i_el) * self.n_az + (i_az % self.n_az)

    def slot(self, k: int) -> Slot:
        if not 0 <= k < self.n_slots:
            raise IndexError(k)
        i_az = k % self.n_az
        i_el = (k // self.n_az) % self.n_el
        i_dist = k // (self.n_az * self.n_el)
        return Slot(k, i_az, i_el, i_dist, self.azimuths_deg[i_az], self.elevations_deg[i_el], self.distances_m[i_dist])

    def slots(self) -> list[Slot]:
        return [self.slot(k) for k in range(self.n_slots)]

    def nearest_slot(self, azimuth_deg: float, elevation_deg: float, distance_m: float) -> int:
        az = np.asarray(self.azimuths_deg)
        d_az = np.abs(((az - azimuth_deg) + 180.0) % 360.0 - 180.0)
        i_az = int(np.argmin(d_az))
        i_el = int(np.argmin(np.abs(np.asarray(self.elevations_deg) - elevation_deg)))
        i_d = int(np.argmin(np.abs(np.asarray(self.distances_m) - distance_m)))
        return self.slot_id(i_az, i_el, i_d)

    def rotate(self, k: int, part_rotation_deg: float) -> int:
        """Slot equivalent to slot ``k`` after the part is rotated by ``part_rotation_deg``
        about the vertical axis: rotating the part by +theta is the same as rotating the
        camera by -theta about the part (the belt plane is invariant)."""
        s = self.slot(k)
        steps = part_rotation_deg / self.az_step_deg
        if abs(steps - round(steps)) > 1e-6:
            raise ValueError(f"rotation {part_rotation_deg} deg is not a multiple of the azimuth step")
        return self.slot_id((s.i_az - int(round(steps))) % self.n_az, s.i_el, s.i_dist)

    def eye(self, k: int, focus: np.ndarray) -> np.ndarray:
        s = self.slot(k)
        az, el = np.radians(s.azimuth_deg), np.radians(s.elevation_deg)
        return np.asarray(focus, dtype=float) + s.distance_m * np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

    def T_world_cam(self, k: int, focus: np.ndarray) -> np.ndarray:
        return look_at_opencv(self.eye(k, focus), np.asarray(focus, dtype=float))

    def to_dict(self) -> dict:
        return {"azimuths_deg": list(self.azimuths_deg), "elevations_deg": list(self.elevations_deg),
                "distances_m": list(self.distances_m)}

    @classmethod
    def from_dict(cls, d: dict) -> "SlotGrid":
        return cls(tuple(float(a) for a in d["azimuths_deg"]), tuple(float(e) for e in d["elevations_deg"]),
                   tuple(float(x) for x in d["distances_m"]))


def camera_pose_spherical(focus: np.ndarray, azimuth_deg: float, elevation_deg: float, distance_m: float) -> np.ndarray:
    """``T_world_cam`` for an arbitrary (not necessarily on-grid) camera."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    eye = np.asarray(focus, dtype=float) + distance_m * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return look_at_opencv(eye, np.asarray(focus, dtype=float))
