"""Static scene geometry that can block the view of the part (belt, conveyor frame).

Default: the belt top as a large rectangle at z = 0 (ISCO world). When the Isaac worker
has exported the real conveyor meshes (``isco sim export-occluder`` ->
``data/conveyor/occluder.npz``) those are used instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Occluder:
    vertices: np.ndarray            # (V,3) world
    faces: np.ndarray               # (F,3)
    source: str = "belt_plane"
    belt_half_width_m: float = 0.25
    meta: dict | None = None

    @classmethod
    def belt_plane(cls, half_length_m: float = 1.0, half_width_m: float = 0.25, z: float = 0.0) -> "Occluder":
        L, W = half_length_m, half_width_m
        v = np.array([[-L, -W, z], [L, -W, z], [L, W, z], [-L, W, z]], dtype=float)
        f = np.array([[0, 1, 2], [0, 2, 3]])
        return cls(v, f, "belt_plane", W, {"z": z})

    @classmethod
    def load(cls, path: str | Path) -> "Occluder":
        import json

        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"])) if "meta" in z else {}
            return cls(z["vertices"].astype(np.float64), z["faces"].astype(np.int64), "isaac_export",
                       float(meta.get("belt_half_width_m", 0.25)), meta)

    def save(self, path: str | Path) -> None:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vertices=self.vertices.astype(np.float32), faces=self.faces.astype(np.int32),
                            meta=np.array(json.dumps({**(self.meta or {}), "belt_half_width_m": self.belt_half_width_m})))

    @classmethod
    def default(cls, data_root: str | Path | None = None) -> "Occluder":
        from ..utils.io import data_root as _dr

        p = Path(data_root or _dr()) / "conveyor" / "occluder.npz"
        if p.exists():
            return cls.load(p)
        return cls.belt_plane()

    def point_inside_solid_bbox(self, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """True for points inside the occluder's bounding box *below the belt top*
        (cameras must not sit inside the conveyor body)."""
        lo, hi = self.vertices.min(axis=0) - margin, self.vertices.max(axis=0) + margin
        inside = np.all((pts >= lo) & (pts <= hi), axis=1)
        return inside & (pts[:, 2] < 0.0)
