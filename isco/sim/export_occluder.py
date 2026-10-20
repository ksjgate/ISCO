"""Export the conveyor's render geometry near the focus point as an embree occluder.

Every visible ``UsdGeom.Mesh`` under the conveyor prim (body/frame, belt, decal) is
triangulated, transformed to the stage with its composed local-to-world matrix and then
to the ISCO world frame (``scene.frame``). Triangles whose AABB intersects the cube
``[-r, r]^3`` around the ISCO origin are kept (a conservative superset of the sphere of
radius ``r``). The result is written with ``isco.render.occluder.Occluder.save`` and a
``conveyor_meta.json`` (belt height/width, conveyor AABB, ISCO origin, asset, focus-x
rule) is written next to it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..geometry.transforms import apply_T


def triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Fan-triangulate USD polygons (``faceVertexCounts`` / ``faceVertexIndices``)."""
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    tris = []
    for c in np.unique(counts):
        if c < 3:
            continue
        s = starts[counts == c]
        for k in range(1, c - 1):
            tris.append(np.stack([indices[s], indices[s + k], indices[s + k + 1]], axis=1))
    return np.concatenate(tris, axis=0) if tris else np.zeros((0, 3), np.int64)


def surface_z_at_xy(V: np.ndarray, F: np.ndarray, xy: np.ndarray) -> float | None:
    """Highest z of the triangles whose xy-projection contains ``xy`` (None if none)."""
    tri = V[F]
    a, b, c = tri[:, 0, :2], tri[:, 1, :2], tri[:, 2, :2]
    v0, v1, v2 = c - a, b - a, np.asarray(xy, float)[None] - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    ok = np.abs(den) > 1e-18
    den = np.where(ok, den, 1.0)
    u, v = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
    m = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    if not m.any():
        return None
    z = tri[m, 0, 2] + u[m] * (tri[m, 2, 2] - tri[m, 0, 2]) + v[m] * (tri[m, 1, 2] - tri[m, 0, 2])
    return float(z.max())


def collect_meshes(stage, root_path: str) -> list[dict]:
    """Visible render meshes under ``root_path``: ``[{path, V_stage, F}]``."""
    from pxr import Usd, UsdGeom

    from .part_prim import get_world_T

    out = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if img.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not idx:
            continue
        V = apply_T(get_world_T(prim), np.asarray(pts, dtype=np.float64))
        F = triangulate(np.asarray(counts), np.asarray(idx))
        out.append({"path": prim.GetPath().pathString, "V_stage": V, "F": F})
    return out


def crop_triangles(V: np.ndarray, F: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep triangles whose AABB intersects ``[-radius, radius]^3``; compact vertices."""
    if len(F) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    tri = V[F]
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    keep = np.all((hi >= -radius) & (lo <= radius), axis=1)
    F = F[keep]
    used, inv = np.unique(F.ravel(), return_inverse=True)
    return V[used], inv.reshape(-1, 3)


def export_occluder(scene, output_path: str | Path, radius_m: float = 1.0, meta_path: str | Path | None = None) -> dict:
    from ..render.occluder import Occluder

    frame = scene.frame
    meshes = collect_meshes(scene.stage, scene.conveyor_path)
    Vs, Fs, per_mesh = [], [], []
    off = 0
    for m in meshes:
        V_world = frame.to_world_pts(m["V_stage"])
        V, F = crop_triangles(V_world, m["F"], radius_m)
        per_mesh.append({"path": m["path"], "n_tri_total": int(len(m["F"])), "n_tri_kept": int(len(F))})
        if len(F) == 0:
            continue
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    V = np.concatenate(Vs, axis=0) if Vs else np.zeros((0, 3))
    F = np.concatenate(Fs, axis=0) if Fs else np.zeros((0, 3), np.int64)

    smeta = scene.meta()
    # belt surface height at the ISCO origin (the plane z = 0 assumption of the embree path)
    belt_info = {}
    for m in meshes:
        if m["path"].startswith(scene.belt_path):
            z0 = surface_z_at_xy(frame.to_world_pts(m["V_stage"]), m["F"], np.zeros(2))
            if z0 is not None:
                belt_info["belt_surface_z_at_origin_m"] = z0
    above = V[:, 2] > 1e-4
    lo, hi = (V.min(axis=0), V.max(axis=0)) if len(V) else (np.zeros(3), np.zeros(3))
    occ_meta = {
        "source": "isaac_export", "asset_name": scene.asset_name, "radius_m": radius_m,
        "crop": "triangle AABB intersects [-r, r]^3 around the ISCO origin",
        "n_vertices": int(len(V)), "n_faces": int(len(F)), "extent_world_min": lo.tolist(),
        "extent_world_max": hi.tolist(), "max_height_above_belt_m": float(V[above, 2].max()) if above.any() else 0.0,
        "min_abs_y_of_geometry_above_belt_m": float(np.abs(V[above, 1]).min()) if above.any() else None,
        "per_mesh": per_mesh, "isco_origin_stage": frame.origin_stage.tolist(), **belt_info,
    }
    occ = Occluder(V, F, "isaac_export", belt_half_width_m=smeta["belt_width_m"] / 2.0, meta=occ_meta)
    output_path = Path(output_path)
    occ.save(output_path)
    meta_path = Path(meta_path) if meta_path else output_path.parent / "conveyor_meta.json"
    conveyor_meta = {**smeta, "occluder": str(output_path), "occluder_meta": occ_meta}
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(conveyor_meta, indent=2))
    return {"occluder": str(output_path), "conveyor_meta": str(meta_path), **{k: occ_meta[k] for k in (
        "n_vertices", "n_faces", "extent_world_min", "extent_world_max", "max_height_above_belt_m",
        "min_abs_y_of_geometry_above_belt_m")}, **belt_info}
