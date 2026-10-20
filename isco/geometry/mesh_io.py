"""Labeled meshes and part preparation.

A ``LabeledMesh`` stores a triangle mesh plus a per-face label:
``face_label == 0`` is nominal part surface, ``face_label == j >= 1`` belongs to defect
instance ``j`` (``label_names[j]`` e.g. ``"dent:0"``). Rendered pixel labels use
``pixel_label = face_label + 1`` so that 0 means "no part" (see ``labels.py``).

Files are ``.npz`` written with ``allow_pickle=False``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from .transforms import make_T, rot_z


@dataclass
class LabeledMesh:
    vertices: np.ndarray
    faces: np.ndarray
    face_label: np.ndarray | None = None
    label_names: list[str] = field(default_factory=lambda: ["part"])
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.face_label is None:
            self.face_label = np.zeros(len(self.faces), dtype=np.int32)
        self.face_label = np.asarray(self.face_label, dtype=np.int32)
        assert len(self.face_label) == len(self.faces)

    # -- conversions ----------------------------------------------------------
    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh, face_label=None, label_names=None, meta=None) -> "LabeledMesh":
        return cls(np.array(mesh.vertices), np.array(mesh.faces), face_label,
                   list(label_names) if label_names else ["part"], dict(meta or {}))

    def to_trimesh(self, process: bool = False) -> trimesh.Trimesh:
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=process)

    def copy(self) -> "LabeledMesh":
        return LabeledMesh(self.vertices.copy(), self.faces.copy(), self.face_label.copy(),
                           list(self.label_names), json.loads(json.dumps(self.meta)))

    @property
    def n_defects(self) -> int:
        return int(self.face_label.max()) if len(self.face_label) else 0

    def defect_mask(self) -> np.ndarray:
        return self.face_label > 0

    # -- io -------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            vertices=self.vertices.astype(np.float32),
            faces=self.faces.astype(np.int32),
            face_label=self.face_label.astype(np.int32),
            label_names=np.array(json.dumps(self.label_names)),
            meta=np.array(json.dumps(self.meta)),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "LabeledMesh":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["vertices"].astype(np.float64), z["faces"].astype(np.int64), z["face_label"],
                       json.loads(str(z["label_names"])), json.loads(str(z["meta"])))


def load_mesh_any(path: str | Path) -> trimesh.Trimesh:
    """Load STL/PLY/OBJ as a single Trimesh with merged vertices."""
    mesh = trimesh.load(str(path), force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):  # pragma: no cover
        raise ValueError(f"not a triangle mesh: {path}")
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh


def load_and_scale_mesh(stl_path: str | Path, target_max_dim: float) -> tuple[np.ndarray, np.ndarray]:
    """PoC helper (moved from ``create_conveyor.py``): centre the bbox at the origin and
    scale so the longest bbox side equals ``target_max_dim`` metres."""
    mesh = trimesh.load(str(stl_path), force="mesh")
    verts = np.array(mesh.vertices, dtype=float)
    bbox_min, bbox_max = verts.min(axis=0), verts.max(axis=0)
    verts -= (bbox_min + bbox_max) / 2.0
    extent = float((bbox_max - bbox_min).max())
    scale = target_max_dim / extent if extent > 0 else 1.0
    verts *= scale
    return verts, np.array(mesh.faces, dtype=int)


def _principal_yaw_deg(mesh: trimesh.Trimesh) -> float:
    """Yaw that aligns the long side of the minimum-area bounding rectangle of the part's
    footprint (xy projection of its convex hull) with +X (belt direction)."""
    from trimesh.bounds import oriented_bounds_2D

    xy = np.asarray(mesh.convex_hull.vertices)[:, :2] if len(mesh.faces) > 3 else np.asarray(mesh.vertices)[:, :2]
    T2, ext = oriented_bounds_2D(xy)
    ang = np.degrees(np.arctan2(T2[1, 0], T2[0, 0]))       # rotation that axis-aligns the rectangle
    if ext[1] > ext[0]:
        ang += 90.0
    return float(((ang + 90.0) % 180.0) - 90.0)


def to_rest_frame(mesh: trimesh.Trimesh, target_max_dim: float | None, stable_pose_index: int = 0,
                  align_major_axis: bool = True) -> tuple[trimesh.Trimesh, dict]:
    """Scale a part and place it in its rest frame (see ``transforms`` docstring).

    1. pick the resting pose with the lowest centre of mass among
       ``trimesh.poses.compute_stable_poses`` (``stable_pose_index`` = rank);
    2. rotate about +Z so the major horizontal axis is along +X (guided feeding,
       ADR-0007);
    3. uniform scale so the longest side of the *rest-frame* bbox equals
       ``target_max_dim`` (if given);
    4. translate so the bbox centre is at x = y = 0 and the lowest point at z = 0.
    Returns the transformed mesh and a dict describing the applied transform.
    """
    m = mesh.copy()
    info: dict = {"source_extents": [float(x) for x in m.extents]}
    try:
        transforms, probs = trimesh.poses.compute_stable_poses(m, n_samples=1, threshold=0.0)
        # Rank resting poses by centre-of-mass height (lowest = how a part lies on a
        # belt). Ranking by the per-facet probability alone fails for faceted round
        # parts: each side facet of a 32-gon rod gets ~3 % while an end cap gets more,
        # so "most probable facet" would stand a rod on its end.
        hull = m.convex_hull
        com = hull.center_mass if hull.is_volume else m.vertices.mean(axis=0)
        com_z = []
        for T in transforms:
            Tm = np.asarray(T)
            v = m.vertices @ Tm[:3, :3].T + Tm[:3, 3]
            c = Tm[:3, :3] @ com + Tm[:3, 3]
            com_z.append(c[2] - v[:, 2].min())
        probs = np.asarray(probs)
        order = np.lexsort((-probs, np.round(np.asarray(com_z), 6)))
        idx = int(order[min(stable_pose_index, len(order) - 1)])
        T_stable = np.asarray(transforms[idx])
        info["stable_pose_probability"] = float(probs[idx])
        info["stable_pose_com_height_m"] = float(com_z[idx])
        info["n_stable_poses"] = int(len(probs))
        info["stable_pose_rule"] = "lowest centre of mass (ties: higher probability)"
    except Exception as e:  # pragma: no cover - non-watertight fallbacks
        T_stable = np.eye(4)
        info["stable_pose_error"] = str(e)
    m.apply_transform(T_stable)
    T_total = T_stable
    if align_major_axis:
        yaw = _principal_yaw_deg(m)
        Tz = make_T(rot_z(yaw))
        m.apply_transform(Tz)
        T_total = Tz @ T_total
        info["align_yaw_deg"] = yaw
    # scale *in the rest frame*: the longest side of the resting part's bounding box
    # equals target_max_dim (scaling the source AABB instead made rotated parts smaller
    # or larger than the target, e.g. 35 mm for a tilted box - unit test)
    scale = 1.0
    if target_max_dim is not None:
        scale = float(target_max_dim / max(m.extents.max(), 1e-12))
        m.apply_scale(scale)
        T_total = np.diag([scale, scale, scale, 1.0]) @ T_total
    info["scale"] = scale
    lo, hi = m.bounds
    t = np.array([-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]])
    m.apply_translation(t)
    T_total = make_T(None, t) @ T_total
    info["T_source_to_rest"] = T_total.tolist()
    info["rest_extents"] = [float(x) for x in m.extents]
    return m, info


def focus_point(mesh_rest: trimesh.Trimesh | LabeledMesh) -> np.ndarray:
    """World focus point = bbox centre of the nominally placed part."""
    v = mesh_rest.vertices
    lo, hi = v.min(axis=0), v.max(axis=0)
    return np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0])


def bounding_radius(mesh_rest: trimesh.Trimesh | LabeledMesh, center: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(mesh_rest.vertices) - center, axis=1).max())
