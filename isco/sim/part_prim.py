"""Parts as USD prims: labelled render meshes (kinematic) and rigid bodies (settling).

``PartPrim`` turns a ``LabeledMesh`` (rest frame) into one ``UsdGeom.Mesh`` per label
under a single Xform root:

    /World/Part              Xform  (translate + orient ops = T_stage_part)
        part                 Mesh   faces with face_label == 0, semantics class="part"
        defect_001           Mesh   faces with face_label == 1, class="defect", defect_id="1"
        ...

Pixel labels follow ``isco.geometry.labels``: part -> 1, defect j -> 1 + j. Face normals
are authored ``faceVarying`` (flat shading) so the ``normals`` annotator returns the
geometric triangle normal, like the embree renderer.

The root is purely kinematic (no physics) - ``set_pose`` places it exactly; meshes are
swapped between items without touching the rest of the stage.
"""
from __future__ import annotations

import numpy as np

from ..geometry.mesh_io import LabeledMesh
from ..geometry.transforms import quat_wxyz_from_R

PART_COLOR = (0.75, 0.55, 0.15)
DEFECT_COLOR = (0.85, 0.10, 0.10)


def _vt_arrays(V: np.ndarray, F: np.ndarray, flat_normals: bool):
    from pxr import Vt

    V = np.ascontiguousarray(V, dtype=np.float32)
    F = np.ascontiguousarray(F, dtype=np.int32)
    pts = Vt.Vec3fArray.FromNumpy(V)
    counts = Vt.IntArray.FromNumpy(np.full(len(F), 3, dtype=np.int32))
    idx = Vt.IntArray.FromNumpy(F.ravel())
    nrm = None
    if flat_normals and len(F):
        a, b, c = V[F[:, 0]].astype(np.float64), V[F[:, 1]].astype(np.float64), V[F[:, 2]].astype(np.float64)
        n = np.cross(b - a, c - a)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
        nrm = Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(np.repeat(n, 3, axis=0), dtype=np.float32))
    return pts, counts, idx, nrm


def define_mesh(stage, path: str, V: np.ndarray, F: np.ndarray, color=PART_COLOR, flat_normals: bool = True):
    """Author a triangle ``UsdGeom.Mesh`` from numpy arrays (no subdivision)."""
    from pxr import Gf, UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, path)
    pts, counts, idx, nrm = _vt_arrays(V, F, flat_normals)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    if nrm is not None:
        mesh.CreateNormalsAttr(nrm)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*[float(c) for c in color])])
    if len(V):
        lo, hi = V.min(axis=0), V.max(axis=0)
        mesh.CreateExtentAttr([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())])
    return mesh


def set_xform_pose(prim, T_stage: np.ndarray) -> None:
    """Set a prim's local transform to the rigid ``T_stage`` via translate + orient ops
    (double precision), creating the ops if needed (parent assumed at identity)."""
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    names = [op.GetOpName() for op in ops]
    if names != ["xformOp:translate", "xformOp:orient"] or ops[0].GetPrecision() != UsdGeom.XformOp.PrecisionDouble \
            or ops[1].GetPrecision() != UsdGeom.XformOp.PrecisionDouble:
        xf.ClearXformOpOrder()
        for n in ("xformOp:translate", "xformOp:orient", "xformOp:scale", "xformOp:transform"):
            if prim.HasAttribute(n):
                prim.RemoveProperty(n)
        t_op = xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        r_op = xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    else:
        t_op, r_op = ops
    T_stage = np.asarray(T_stage, dtype=float)
    w, x, y, z = quat_wxyz_from_R(T_stage[:3, :3])
    t_op.Set(Gf.Vec3d(*T_stage[:3, 3].tolist()))
    r_op.Set(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))


def get_world_T(prim) -> np.ndarray:
    """4x4 local-to-world transform of a prim (stage frame, column-vector convention)."""
    from pxr import Usd, UsdGeom

    M = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(M, dtype=float).T  # Gf is row-vector convention


class PartPrim:
    """A kinematic, semantically labelled part built from a ``LabeledMesh``."""

    def __init__(self, stage, frame, root_path: str = "/World/Part", part_color=PART_COLOR,
                 defect_color=DEFECT_COLOR):
        from pxr import UsdGeom

        self.stage = stage
        self.frame = frame
        self.root_path = root_path
        self.part_color = part_color
        self.defect_color = defect_color
        UsdGeom.Xform.Define(stage, root_path)
        self.root = stage.GetPrimAtPath(root_path)
        self.mesh: LabeledMesh | None = None
        self.mesh_key: str | None = None
        self.children: list[str] = []
        self.T_world_part = np.eye(4)
        set_xform_pose(self.root, self.frame.to_stage_T(np.eye(4)))

    # -- geometry ---------------------------------------------------------------------
    def clear(self) -> None:
        """Hide all label meshes (they are kept for reuse, see ``load``)."""
        for path in self.children:
            self._set_child_geometry(path, None, None)
        self.mesh = None
        self.mesh_key = None

    def _set_child_geometry(self, path: str, V: np.ndarray | None, F: np.ndarray | None) -> None:
        from pxr import Gf, UsdGeom

        mesh = UsdGeom.Mesh(self.stage.GetPrimAtPath(path))
        vis = UsdGeom.Imageable(mesh.GetPrim()).GetVisibilityAttr()
        if V is None or len(F) == 0:
            # hide only: RTX ignores an update to *empty* arrays and then keeps rendering
            # the stale mesh (even when made invisible afterwards) - never empty a mesh
            vis.Set(UsdGeom.Tokens.invisible)
            return
        pts, counts, idx, nrm = _vt_arrays(V, F, True)
        mesh.GetPointsAttr().Set(pts)
        mesh.GetFaceVertexCountsAttr().Set(counts)
        mesh.GetFaceVertexIndicesAttr().Set(idx)
        mesh.GetNormalsAttr().Set(nrm)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        lo, hi = np.asarray(V).min(axis=0), np.asarray(V).max(axis=0)
        mesh.GetExtentAttr().Set([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())])
        vis.Set(UsdGeom.Tokens.inherited)

    def load(self, mesh: LabeledMesh, key: str | None = None) -> None:
        """Show ``mesh`` (no-op if ``key`` is unchanged).

        The per-label ``UsdGeom.Mesh`` prims are created once and then *updated in place*
        (points / topology / normals) and hidden when unused. Deleting and re-creating
        labelled prims at the same paths makes Isaac 5.1's semantic ids and the
        annotator's ``idToLabels`` disagree (observed: part/defect labels swapped for the
        second mesh loaded), so prims are never removed."""
        from isaacsim.core.utils.semantics import add_labels

        if key is not None and key == self.mesh_key and self.mesh is not None:
            return
        labels = np.asarray(mesh.face_label)
        present = set(int(j) for j in np.unique(labels))
        for j in sorted(present):
            sel = labels == j
            F_all = mesh.faces[sel]
            used, inv = np.unique(F_all.ravel(), return_inverse=True)
            V = mesh.vertices[used]
            F = inv.reshape(-1, 3)
            name = "part" if j == 0 else f"defect_{j:03d}"
            path = f"{self.root_path}/{name}"
            if path not in self.children:
                define_mesh(self.stage, path, V, F, self.part_color if j == 0 else self.defect_color)
                prim = self.stage.GetPrimAtPath(path)
                if j == 0:
                    add_labels(prim, ["part"], instance_name="class")
                else:
                    add_labels(prim, ["defect"], instance_name="class")
                    add_labels(prim, [str(j)], instance_name="defect_id")
                self.children.append(path)
            else:
                self._set_child_geometry(path, V, F)
        for path in self.children:
            name = path.rsplit("/", 1)[-1]
            j = 0 if name == "part" else int(name.split("_")[-1])
            if j not in present:
                self._set_child_geometry(path, None, None)
        self.mesh = mesh
        self.mesh_key = key

    # -- pose -------------------------------------------------------------------------
    def set_pose(self, T_world_part: np.ndarray) -> None:
        """Place the part at ``T_world_part`` (ISCO world frame, rigid)."""
        T = np.asarray(T_world_part, dtype=float)
        self.T_world_part = T.copy()
        set_xform_pose(self.root, self.frame.to_stage_T(T))

    def pose_world(self) -> np.ndarray:
        """Read the composed pose back from USD (ISCO world)."""
        return self.frame.to_world_T(get_world_T(self.root))

    def set_visible(self, visible: bool) -> None:
        from pxr import UsdGeom

        UsdGeom.Imageable(self.root).GetVisibilityAttr().Set(
            UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible)


# ----------------------------------------------------------------------------- PoC parts
def load_part_mesh_stl(stage, prim_path, stl_path, target_max_dim, color, mass=0.02):
    """PoC ``load_part_mesh``: an STL body as ``UsdGeom.Mesh`` + rigid body.

    The STL is centred on its bbox and scaled so its longest side is ``target_max_dim``
    (``isco.geometry.mesh_io.load_and_scale_mesh``). convexHull collision because PhysX
    does not support non-convex dynamic triangle meshes."""
    from pxr import UsdPhysics

    from ..geometry.mesh_io import load_and_scale_mesh
    from .scene import apply_physics_material

    verts, faces = load_and_scale_mesh(stl_path, target_max_dim)
    define_mesh(stage, prim_path, verts, faces, color, flat_normals=False)
    prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    apply_physics_material(stage, prim_path, static_friction=0.8, dynamic_friction=0.7)
    return prim_path


def add_rigid_part(stage, prim_path: str, mesh: LabeledMesh, mass: float = 0.02, static_friction: float = 0.8,
                   dynamic_friction: float = 0.7, hull_vertex_limit: int = 255, color=PART_COLOR,
                   true_inertia: bool = True) -> dict:
    """A dynamic rigid body for physics settling.

    Collision = convex hull of the part (exact for contact with a flat belt: a rigid body
    resting on a plane only touches it at convex-hull points). Mass ``mass`` with the
    centre of mass / inertia of the *solid* part (uniform density, trimesh) when the mesh
    is watertight, otherwise PhysX derives them from the hull. Returns physics metadata.
    """
    import trimesh
    from pxr import Gf, PhysxSchema, UsdPhysics

    from .scene import apply_physics_material

    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    hull = tm.convex_hull
    define_mesh(stage, prim_path, hull.vertices, hull.faces, color, flat_normals=False)
    prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
    PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim).CreateHullVertexLimitAttr(int(hull_vertex_limit))
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass))
    meta = {"collision": "convexHull", "hull_vertices": int(len(hull.vertices)), "hull_vertex_limit": hull_vertex_limit,
            "mass_kg": mass, "static_friction": static_friction, "dynamic_friction": dynamic_friction}
    if true_inertia and tm.is_watertight and tm.volume > 0:
        tm.density = float(mass / tm.volume)
        com = np.asarray(tm.center_mass, dtype=float)
        I = np.asarray(tm.moment_inertia, dtype=float)       # about the COM, mesh axes
        w, Vec = np.linalg.eigh(I)
        if np.linalg.det(Vec) < 0:
            Vec[:, 0] *= -1
        q = quat_wxyz_from_R(Vec)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*com.tolist()))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(max(x, 1e-12)) for x in w]))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))
        meta.update(inertia="solid mesh (trimesh, uniform density)", com_m=com.tolist(), principal_inertia=w.tolist())
    else:
        meta.update(inertia="PhysX from collision hull")
    apply_physics_material(stage, prim_path, static_friction=static_friction, dynamic_friction=dynamic_friction)
    return meta
