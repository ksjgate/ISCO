"""Geometric defect generator (plan "결함 사전", ADR-0005/0007).

Types and mechanisms (sizes in the dictionary YAML, depths scaled by the E0.5
calibration factor ``scale_k``):

=========  =====================================================  =========================
type       mechanism                                              reference
=========  =====================================================  =========================
dent       vertex displacement along -n, smooth (1-(r/R)^2)^2     AF3AD kernels (MIT)
           bump, anisotropic elliptic footprint
bulge      same along +n
hole       boolean subtraction of a cylinder along -n (50 %       Looking3D primitive
           through)                                               booleans
cut        boolean subtraction of a box at a sharp convex edge    "
           (notch / chip)
crack      boolean subtraction of a thin box slot (0.2-0.5 mm)    "
warp       global bend (sagitta) +/- twist about the long axis    —
=========  =====================================================  =========================

Booleans are applied with manifold3d directly on the fine (remeshed) mesh. This
deviates from the original plan (boolean on the coarse CAD, then remesh) because a
0.25 mm isotropic remesh would erase 0.2-0.5 mm cracks; manifold3d is robust on the
~100k-face meshes (≈20 ms per boolean) and tracks which triangles come from the tool,
which gives exact defect face labels.

Placement is stratified by *difficulty*: for candidate surface points we count how
many of 41 upper-hemisphere directions see the point (unoccluded by the part or the
belt, incidence < 75 deg): easy k>=20, medium 6-19, hard 1-5, unreachable k=0
(belt-contact faces; used only as a control group).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from ..geometry.mesh_io import LabeledMesh
from ..geometry.transforms import make_T

DEFECT_TYPES = ("dent", "bulge", "hole", "cut", "crack", "warp")
LOCAL_TYPES = ("dent", "bulge", "hole", "cut", "crack")
DIFFICULTY_BINS = ("easy", "medium", "hard", "unreachable")


class DefectDictionary(BaseModel):
    """Defect-type dictionary (sizes in metres). ``scale_k`` multiplies depths/heights
    (E0.5 severity calibration, frozen afterwards)."""

    scale_k: float = 1.0
    types: list[str] = list(DEFECT_TYPES)
    dent_diameter_m: tuple[float, float] = (2e-3, 6e-3)
    dent_depth_m: tuple[float, float] = (0.2e-3, 1.0e-3)
    dent_aspect: tuple[float, float] = (1.0, 2.0)
    bulge_diameter_m: tuple[float, float] = (2e-3, 6e-3)
    bulge_height_m: tuple[float, float] = (0.2e-3, 1.0e-3)
    hole_diameter_m: tuple[float, float] = (1e-3, 4e-3)
    hole_depth_m: tuple[float, float] = (1e-3, 3e-3)
    hole_through_prob: float = 0.5
    cut_length_m: tuple[float, float] = (2e-3, 8e-3)
    cut_depth_m: tuple[float, float] = (0.5e-3, 2e-3)
    crack_width_m: tuple[float, float] = (0.2e-3, 0.5e-3)
    crack_length_m: tuple[float, float] = (3e-3, 10e-3)
    crack_depth_m: tuple[float, float] = (0.5e-3, 2e-3)
    warp_sagitta_m: tuple[float, float] = (0.3e-3, 1.5e-3)
    warp_twist_deg: tuple[float, float] = (1.0, 3.0)
    warp_twist_prob: float = 0.3
    difficulty_mix: dict[str, float] = {"easy": 0.4, "medium": 0.4, "hard": 0.2}
    label_threshold_m: float = 0.02e-3
    max_tries: int = 6


@dataclass
class Candidates:
    points: np.ndarray       # (N,3) rest frame
    normals: np.ndarray      # (N,3)
    face_ids: np.ndarray     # (N,)
    k_visible: np.ndarray    # (N,) visible hemisphere directions (0..41)
    difficulty: np.ndarray   # (N,) str


def hemisphere_directions(n: int = 41, min_elevation_deg: float = 5.0) -> np.ndarray:
    """Fibonacci directions on the upper hemisphere (z > sin(min_elevation))."""
    zmin = np.sin(np.radians(min_elevation_deg))
    i = np.arange(n) + 0.5
    z = zmin + (1 - zmin) * (i / n)
    phi = i * np.pi * (3 - np.sqrt(5))
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def difficulty_of(k: np.ndarray) -> np.ndarray:
    out = np.full(len(k), "easy", dtype=object)
    out[k < 20] = "medium"
    out[k <= 5] = "hard"
    out[k == 0] = "unreachable"
    return out


def _scene(mesh: LabeledMesh, with_belt: bool = True):
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(mesh.vertices.astype(np.float32)), o3d.core.Tensor(mesh.faces.astype(np.uint32)))
    if with_belt:
        L = 1.0
        v = np.array([[-L, -L, 0], [L, -L, 0], [L, L, 0], [-L, L, 0]], np.float32)
        scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(np.array([[0, 1, 2], [0, 2, 3]], np.uint32)))
    return scene


def visible_direction_count(mesh: LabeledMesh, points: np.ndarray, normals: np.ndarray,
                            dirs: np.ndarray | None = None, max_incidence_deg: float = 75.0) -> np.ndarray:
    import open3d as o3d

    dirs = hemisphere_directions() if dirs is None else dirs
    scene = _scene(mesh)
    cos_ok = (normals @ dirs.T) > np.cos(np.radians(max_incidence_deg))     # (N, D)
    origins = points + normals * 2e-5
    N, D = cos_ok.shape
    o = np.repeat(origins, D, axis=0)
    d = np.tile(dirs, (N, 1))
    ans = scene.cast_rays(o3d.core.Tensor(np.concatenate([o, d], 1).astype(np.float32)))
    free = ~np.isfinite(ans["t_hit"].numpy()).reshape(N, D)
    return (free & cos_ok).sum(axis=1)


def surface_candidates(mesh: LabeledMesh, n: int = 6000, seed: int = 0) -> Candidates:
    import trimesh

    tm = mesh.to_trimesh()
    pts, fid = trimesh.sample.sample_surface(tm, n, seed=seed)
    nrm = np.asarray(tm.face_normals)[fid]
    k = visible_direction_count(mesh, pts, nrm)
    return Candidates(pts, nrm, fid, k, difficulty_of(k))


@dataclass
class EdgeCandidates:
    points: np.ndarray
    edge_dir: np.ndarray
    n1: np.ndarray
    n2: np.ndarray
    k_visible: np.ndarray
    difficulty: np.ndarray


def edge_candidates(mesh: LabeledMesh, min_angle_deg: float = 60.0, n: int = 3000, seed: int = 0) -> EdgeCandidates:
    """Points on sharp convex edges (for notches/chips)."""
    tm = mesh.to_trimesh()
    ang = tm.face_adjacency_angles
    sel = (ang > np.radians(min_angle_deg)) & tm.face_adjacency_convex
    if not sel.any():
        z = np.zeros((0, 3))
        return EdgeCandidates(z, z, z, z, np.zeros(0, int), np.zeros(0, object))
    edges = tm.face_adjacency_edges[sel]
    pairs = tm.face_adjacency[sel]
    a, b = tm.vertices[edges[:, 0]], tm.vertices[edges[:, 1]]
    length = np.linalg.norm(b - a, axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(edges), size=n, p=length / length.sum())
    t = rng.uniform(0, 1, size=n)[:, None]
    pts = a[idx] + t * (b[idx] - a[idx])
    e = (b[idx] - a[idx]) / np.maximum(length[idx][:, None], 1e-12)
    n1 = np.asarray(tm.face_normals)[pairs[idx, 0]]
    n2 = np.asarray(tm.face_normals)[pairs[idx, 1]]
    bis = n1 + n2
    bis /= np.maximum(np.linalg.norm(bis, axis=1, keepdims=True), 1e-12)
    k = visible_direction_count(mesh, pts, bis)
    return EdgeCandidates(pts, e, n1, n2, k, difficulty_of(k))


def _log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def tier_value(rng: np.random.Generator, lo: float, hi: float, tier: str) -> float:
    """Log-uniform draw inside the S/M/L third (in log space) of [lo, hi]."""
    edges = np.exp(np.linspace(np.log(lo), np.log(hi), 4))
    i = {"S": 0, "M": 1, "L": 2}[tier]
    return _log_uniform(rng, edges[i], edges[i + 1])


def _tangent_frame(n: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    ang = rng.uniform(0, np.pi)
    return np.cos(ang) * t1 + np.sin(ang) * t2, -np.sin(ang) * t1 + np.cos(ang) * t2


def _closest_on(mesh: LabeledMesh, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    scene = _scene(mesh, with_belt=False)
    ans = scene.compute_closest_points(o3d.core.Tensor(p[None].astype(np.float32)))
    q = ans["points"].numpy()[0].astype(np.float64)
    fid = int(ans["primitive_ids"].numpy()[0])
    tm = mesh.to_trimesh()
    return q, np.asarray(tm.face_normals)[fid]


# --------------------------------------------------------------------------------------
# displacement defects
# --------------------------------------------------------------------------------------
def apply_bump(mesh: LabeledMesh, center: np.ndarray, normal: np.ndarray, radius: float, height: float,
               aspect: float, t1: np.ndarray, t2: np.ndarray, label_thr: float, name: str) -> tuple[LabeledMesh, dict]:
    """Signed bump (height < 0 = dent) with an elliptic (1-(r/R)^2)^2 profile."""
    out = mesh.copy()
    vn = np.asarray(out.to_trimesh().vertex_normals)
    d = out.vertices - center
    r = np.sqrt((d @ t1 / aspect) ** 2 + (d @ t2) ** 2 + (d @ normal) ** 2 * 0.25)
    same_side = vn @ normal > 0.3
    w = np.where((r < radius) & same_side, (1 - (r / radius) ** 2) ** 2, 0.0)
    disp = height * w
    out.vertices = out.vertices + normal[None, :] * disp[:, None]
    moved = np.abs(disp) > max(label_thr, 0.1 * abs(height))
    fmask = moved[out.faces].any(axis=1)
    j = out.n_defects + 1
    out.face_label = np.where(fmask & (out.face_label == 0), j, out.face_label).astype(np.int32)
    out.label_names = list(out.label_names) + [name]
    return out, {"n_moved_vertices": int(moved.sum()), "n_defect_faces": int(fmask.sum())}


def apply_warp(mesh: LabeledMesh, sagitta: float, twist_deg: float, label_thr: float, name: str) -> tuple[LabeledMesh, dict]:
    out = mesh.copy()
    v0 = out.vertices.copy()
    x = out.vertices[:, 0]
    L = max(x.max() - x.min(), 1e-9)
    xc = 0.5 * (x.max() + x.min())
    u = (x - xc) / (L / 2)
    out.vertices[:, 2] += sagitta * (1 - u ** 2)
    if twist_deg != 0.0:
        zc = 0.5 * (out.vertices[:, 2].max() + out.vertices[:, 2].min())
        phi = np.radians(twist_deg) * u
        y, z = out.vertices[:, 1].copy(), out.vertices[:, 2] - zc
        out.vertices[:, 1] = np.cos(phi) * y - np.sin(phi) * z
        out.vertices[:, 2] = np.sin(phi) * y + np.cos(phi) * z + zc
    out.vertices[:, 2] -= out.vertices[:, 2].min()
    disp = np.linalg.norm(out.vertices - v0 - (out.vertices - v0).mean(axis=0), axis=1)
    thr = max(label_thr, 0.25 * abs(sagitta))
    fmask = (disp[out.faces].mean(axis=1) > thr)
    j = out.n_defects + 1
    out.face_label = np.where(fmask & (out.face_label == 0), j, out.face_label).astype(np.int32)
    out.label_names = list(out.label_names) + [name]
    return out, {"n_defect_faces": int(fmask.sum())}


# --------------------------------------------------------------------------------------
# boolean defects (manifold3d)
# --------------------------------------------------------------------------------------
def _to_manifold(v: np.ndarray, f: np.ndarray):
    import manifold3d as m3

    return m3.Manifold(m3.Mesh(vert_properties=np.ascontiguousarray(v, dtype=np.float32),
                               tri_verts=np.ascontiguousarray(f, dtype=np.uint32)))


def boolean_subtract(mesh: LabeledMesh, tool_v: np.ndarray, tool_f: np.ndarray, name: str) -> tuple[LabeledMesh | None, dict]:
    """``mesh - tool``; triangles created by the tool get a new defect label."""
    A = _to_manifold(mesh.vertices, mesh.faces).as_original()
    B = _to_manifold(tool_v, tool_f).as_original()
    idA, idB = A.original_id(), B.original_id()
    R = A - B
    if R.is_empty():
        return None, {"error": "empty result"}
    m = R.to_mesh()
    v = np.asarray(m.vert_properties)[:, :3].astype(np.float64)
    f = np.asarray(m.tri_verts).astype(np.int64)
    run_idx = np.asarray(m.run_index).astype(np.int64)
    run_oid = np.asarray(m.run_original_id).astype(np.int64)
    tri_oid = np.zeros(len(f), np.int64)
    for i, oid in enumerate(run_oid):
        tri_oid[run_idx[i] // 3: run_idx[i + 1] // 3] = oid
    from_tool = tri_oid == idB
    # carry existing labels of A's triangles by closest face on the input
    from ..geometry.remesh import transfer_face_labels

    base_lab = transfer_face_labels(mesh, v, f) if mesh.n_defects > 0 else np.zeros(len(f), np.int32)
    j = mesh.n_defects + 1
    lab = np.where(from_tool, j, base_lab).astype(np.int32)
    out = LabeledMesh(v, f, lab, list(mesh.label_names) + [name], dict(mesh.meta))
    vol_before = mesh.to_trimesh().volume
    vol_after = out.to_trimesh().volume
    return out, {"n_defect_faces": int(from_tool.sum()), "removed_volume_m3": float(vol_before - vol_after),
                 "tool_original_id": int(idB), "part_original_id": int(idA)}


def _box(center: np.ndarray, R: np.ndarray, size: np.ndarray):
    import trimesh

    b = trimesh.creation.box(extents=size)
    b.apply_transform(make_T(R, center))
    return np.asarray(b.vertices), np.asarray(b.faces)


def _cylinder(start: np.ndarray, axis: np.ndarray, radius: float, height: float, sections: int = 48):
    import trimesh

    c = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    # trimesh cylinder is centred at origin along +Z
    z = axis / np.linalg.norm(axis)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    c.apply_transform(make_T(R, start + z * height / 2))
    return np.asarray(c.vertices), np.asarray(c.faces)


# --------------------------------------------------------------------------------------
class DefectGenerator:
    """Creates one defect on a (normal-varied) part instance.

    The candidate locations are computed once on the nominal fine mesh; each instance
    snaps the chosen location onto its own (slightly varied) surface.
    """

    def __init__(self, nominal: LabeledMesh, dictionary: DefectDictionary, seed: int = 0, n_candidates: int = 6000):
        self.nominal = nominal
        self.d = dictionary
        self.cands = surface_candidates(nominal, n_candidates, seed=seed)
        self.edges = edge_candidates(nominal, n=max(500, n_candidates // 2), seed=seed + 1)

    def difficulty_table(self) -> dict:
        out = {}
        for b in DIFFICULTY_BINS:
            out[b] = {"surface": int((self.cands.difficulty == b).sum()), "edge": int((self.edges.difficulty == b).sum())}
        return out

    #: fallback order when a part has no candidate in the requested difficulty bin
    #: (e.g. flat parts have no "hard" points: everything is either seen from many
    #: directions or not at all). The actual bin is recorded in the defect meta.
    FALLBACK = {"hard": ("hard", "medium", "easy"), "medium": ("medium", "easy", "hard"),
                "easy": ("easy", "medium", "hard"), "unreachable": ("unreachable",)}

    def _pick(self, rng, pool_difficulty: np.ndarray, difficulty: str) -> tuple[int, str] | tuple[None, None]:
        for b in self.FALLBACK.get(difficulty, (difficulty,)):
            idx = np.flatnonzero(pool_difficulty == b)
            if len(idx):
                return int(rng.choice(idx)), b
        return None, None

    def make(self, base: LabeledMesh, dtype: str, difficulty: str, tier: str, rng: np.random.Generator) -> tuple[LabeledMesh, dict]:
        d = self.d
        k = d.scale_k
        name = f"{dtype}:0"
        last_err = None
        for attempt in range(d.max_tries):
            meta: dict = {"type": dtype, "tier": tier, "difficulty": difficulty, "attempt": attempt, "scale_k": k}
            if dtype == "warp":
                sag = tier_value(rng, *d.warp_sagitta_m, tier) * k * rng.choice([-1.0, 1.0])
                tw = rng.uniform(*d.warp_twist_deg) * rng.choice([-1.0, 1.0]) if rng.uniform() < d.warp_twist_prob else 0.0
                out, info = apply_warp(base, sag, tw, d.label_threshold_m, name)
                meta.update({"sagitta_m": sag, "twist_deg": tw, "depth_m": abs(sag), "difficulty": "global"}, **info)
                return out, meta
            if dtype == "cut":
                i, actual = self._pick(rng, self.edges.difficulty, difficulty)
                if i is None:
                    raise ValueError(f"no edge candidates with difficulty {difficulty}")
                meta["difficulty"] = actual
                meta["difficulty_requested"] = difficulty
                p, e = self.edges.points[i], self.edges.edge_dir[i]
                bis = self.edges.n1[i] + self.edges.n2[i]
                bis /= np.linalg.norm(bis)
                depth = tier_value(rng, *d.cut_depth_m, tier) * k
                length = rng.uniform(*d.cut_length_m)
                y = np.cross(bis, e)
                R = np.stack([e, y, bis], axis=1)
                # a box centred on the edge removes a wedge of ~depth below the edge
                tv, tf = _box(p, R, np.array([length, 2.5 * depth, 2.0 * depth]))
                meta.update({"center_m": p.tolist(), "normal": bis.tolist(), "length_m": length, "depth_m": depth,
                             "k_visible": int(self.edges.k_visible[i])})
            else:
                i, actual = self._pick(rng, self.cands.difficulty, difficulty)
                if i is None:
                    raise ValueError(f"no surface candidates with difficulty {difficulty}")
                meta["difficulty"] = actual
                meta["difficulty_requested"] = difficulty
                p, n = _closest_on(base, self.cands.points[i])
                meta.update({"center_m": p.tolist(), "normal": n.tolist(), "k_visible": int(self.cands.k_visible[i])})
                t1, t2 = _tangent_frame(n, rng)
                if dtype in ("dent", "bulge"):
                    rng_d = d.dent_diameter_m if dtype == "dent" else d.bulge_diameter_m
                    rng_h = d.dent_depth_m if dtype == "dent" else d.bulge_height_m
                    diam = rng.uniform(*rng_d)
                    h = tier_value(rng, *rng_h, tier) * k
                    aspect = rng.uniform(*d.dent_aspect)
                    sign = -1.0 if dtype == "dent" else 1.0
                    out, info = apply_bump(base, p, n, diam / 2, sign * h, aspect, t1, t2, d.label_threshold_m, name)
                    meta.update({"diameter_m": diam, "depth_m": h, "aspect": aspect}, **info)
                    if info["n_defect_faces"] > 0:
                        return out, meta
                    last_err = "bump moved no faces"
                    continue
                if dtype == "hole":
                    diam = rng.uniform(*d.hole_diameter_m)
                    through = rng.uniform() < d.hole_through_prob
                    depth = 0.1 if through else tier_value(rng, *d.hole_depth_m, tier) * k
                    tv, tf = _cylinder(p + n * 0.5e-3, -n, diam / 2, depth + 0.5e-3)
                    meta.update({"diameter_m": diam, "depth_m": (None if through else depth), "through": through})
                elif dtype == "crack":
                    w = rng.uniform(*d.crack_width_m)
                    length = rng.uniform(*d.crack_length_m)
                    depth = tier_value(rng, *d.crack_depth_m, tier) * k
                    R = np.stack([t1, t2, n], axis=1)
                    tv, tf = _box(p + n * (0.25e-3 - depth / 2), R, np.array([length, w, depth + 0.5e-3]))
                    meta.update({"width_m": w, "length_m": length, "depth_m": depth})
                else:
                    raise ValueError(dtype)
            out, info = boolean_subtract(base, tv, tf, name)
            if out is not None and info.get("n_defect_faces", 0) > 0 and info.get("removed_volume_m3", 0) > 1e-12:
                meta.update(info)
                return out, meta
            last_err = info.get("error", "tool did not intersect the part")
        raise RuntimeError(f"could not place {dtype} ({difficulty}) after {d.max_tries} tries: {last_err}")


def defect_area(mesh: LabeledMesh) -> float:
    tm = mesh.to_trimesh()
    return float(tm.area_faces[mesh.face_label > 0].sum())


def load_dictionary(path: str | Path | None) -> DefectDictionary:
    if path is None:
        return DefectDictionary()
    from ..utils.io import read_yaml

    return DefectDictionary.model_validate(read_yaml(path) or {})
