"""Pixel / point label conventions shared by renderers and the detector."""
from __future__ import annotations

import numpy as np

#: no part hit (background, belt, conveyor frame)
BACKGROUND = 0
#: nominal part surface (face_label == 0)
PART = 1
#: first defect instance (face_label == 1); multi-defect meshes use PART + face_label
DEFECT = 2


def pixel_label_from_face_label(face_label: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(face_label, dtype=np.int64) + 1, 0, 255).astype(np.uint8)


def is_defect(label: np.ndarray) -> np.ndarray:
    return np.asarray(label) >= DEFECT
