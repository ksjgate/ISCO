"""ViewPixels: the per-view sparse depth record shared by every renderer.

Only pixels that hit the *part* are kept (belt / conveyor pixels are dropped; occlusion
by them is still honoured because those rays hit the occluder first).

Fields
------
pix   int64  global pixel index ``v * W + u``
depth float32 distance to the image plane (m)
label uint8  1 = nominal part surface, 2+ = defect (``face_label + 1``)
cos   float32 |cos| of the angle between surface normal and viewing ray
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ViewPixels:
    pix: np.ndarray
    depth: np.ndarray
    label: np.ndarray
    cos: np.ndarray
    meta: dict = field(default_factory=dict)

    @classmethod
    def empty(cls, meta: dict | None = None) -> "ViewPixels":
        return cls(np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.uint8), np.zeros(0, np.float32),
                   dict(meta or {}))

    def __len__(self) -> int:
        return int(len(self.pix))

    def subset(self, mask: np.ndarray) -> "ViewPixels":
        return ViewPixels(self.pix[mask], self.depth[mask], self.label[mask], self.cos[mask], dict(self.meta))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, pix=self.pix.astype(np.int64), depth=self.depth.astype(np.float32),
                            label=self.label.astype(np.uint8), cos=self.cos.astype(np.float32),
                            meta=np.array(json.dumps(self.meta)))

    @classmethod
    def load(cls, path: str | Path) -> "ViewPixels":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["pix"], z["depth"], z["label"], z["cos"], json.loads(str(z["meta"])))


# -- compact quantised storage used by the slot cache ---------------------------
DEPTH_Q = 1.0e-5   # 10 micrometre depth quantum (<< sensor noise)


def quantise(depth: np.ndarray, z0: float) -> np.ndarray:
    q = np.round((np.asarray(depth, dtype=np.float64) - z0) / DEPTH_Q)
    if q.size and (q.min() < 1 or q.max() > 65535):
        raise ValueError(f"depth outside quantisation range: z0={z0}, min={depth.min()}, max={depth.max()}")
    return q.astype(np.uint16)


def dequantise(q: np.ndarray, z0: float) -> np.ndarray:
    return (np.asarray(q, dtype=np.float64) * DEPTH_Q + z0).astype(np.float32)
