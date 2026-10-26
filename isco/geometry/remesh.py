"""Isotropic remeshing with face-label propagation.

The PoC CAD bodies have only 172-540 faces, far coarser than the 0.2-1 mm defects we
synthesise, so every part is remeshed to ~0.25 mm edges before displacement defects
(ADR: plan "데이터 생성기"). Labels are transferred by closest-point lookup on the
labeled input mesh, which is robust to the remesher's re-triangulation.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .mesh_io import LabeledMesh

_MM = 1000.0  # remesh in millimetres to keep pymeshlab's absolute tolerances sane


def closest_face_ids(src: LabeledMesh, query_pts: np.ndarray) -> np.ndarray:
    """Index of the closest face of ``src`` for each query point (open3d, exact)."""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(src.vertices.astype(np.float32)),
                        o3d.core.Tensor(src.faces.astype(np.uint32)))
    ans = scene.compute_closest_points(o3d.core.Tensor(np.asarray(query_pts, dtype=np.float32)))
    return ans["primitive_ids"].numpy().astype(np.int64)


def transfer_face_labels(src: LabeledMesh, dst_vertices: np.ndarray, dst_faces: np.ndarray) -> np.ndarray:
    centroids = dst_vertices[dst_faces].mean(axis=1)
    return src.face_label[closest_face_ids(src, centroids)]


def remesh_isotropic(mesh: LabeledMesh, target_edge_m: float = 2.5e-4, feature_deg: float = 30.0,
                     iterations: int = 8) -> LabeledMesh:
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices * _MM, mesh.faces.astype(np.int32)))
    ms.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PureValue(target_edge_m * _MM), featuredeg=feature_deg, iterations=iterations)
    cm = ms.current_mesh()
    v = np.asarray(cm.vertex_matrix(), dtype=np.float64) / _MM
    f = np.asarray(cm.face_matrix(), dtype=np.int64)
    out = LabeledMesh(v, f, None, list(mesh.label_names), dict(mesh.meta))
    if mesh.n_defects > 0:
        out.face_label = transfer_face_labels(mesh, v, f).astype(np.int32)
    out.meta["remesh_target_edge_m"] = target_edge_m
    return out


def mesh_stats(mesh: LabeledMesh) -> dict:
    tm = mesh.to_trimesh()
    el = tm.edges_unique_length
    return {
        "n_vertices": int(len(mesh.vertices)), "n_faces": int(len(mesh.faces)),
        "watertight": bool(tm.is_watertight), "edge_mean_m": float(el.mean()) if len(el) else 0.0,
        "edge_max_m": float(el.max()) if len(el) else 0.0, "area_m2": float(tm.area),
        "volume_m3": float(tm.volume) if tm.is_watertight else None,
        "n_defect_faces": int((mesh.face_label > 0).sum()),
    }
