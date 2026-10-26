"""Per-slot visibility of defect surface samples (precomputed once per variant set).

For each defective instance j we draw ``n`` area-weighted samples on its defect faces
(and, once, on the nominal part for ``coverage_normal``) and for every slot k record:
``vis`` (in image, front-facing and unoccluded by the part/belt/conveyor), ``cos``
(incidence), ``z`` (depth). Configuration metrics are then pure array lookups
(``surrogate.py``), well below the plan's 1 s/config budget.

Output ``data/visibility/<part>/<set>/<sensor_id>/vis.npz``.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from ..geometry.camera_model import SlotGrid
from ..geometry.mesh_io import LabeledMesh
from ..geometry.transforms import apply_R, apply_T
from ..render.occluder import Occluder
from ..schemas.sensor import SensorModel, VisibilityParams
from ..utils.io import data_root, resolve_data_path, write_json
from ..utils.logging import get_logger
from ..utils.seeds import core_order_env, rng_for

log = get_logger(__name__)


def vis_dir(part_id: str, set_id: str, sensor_id: str) -> Path:
    return data_root() / "visibility" / part_id / set_id / sensor_id


def sample_faces(mesh: LabeledMesh, face_mask: np.ndarray, n: int, rng: np.random.Generator):
    tm = mesh.to_trimesh()
    areas = np.asarray(tm.area_faces) * face_mask
    if areas.sum() <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    fid = rng.choice(len(areas), size=n, p=areas / areas.sum())
    tri = mesh.vertices[mesh.faces[fid]]
    r1, r2 = rng.uniform(size=(2, n))
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * tri[:, 0] + (s * (1 - r2))[:, None] * tri[:, 1] + (s * r2)[:, None] * tri[:, 2]
    return pts, np.asarray(tm.face_normals)[fid]


def slot_visibility(mesh: LabeledMesh, T_world_part: np.ndarray, pts_part: np.ndarray, nrm_part: np.ndarray,
                    cam_T: list[np.ndarray], camera, occluder: Occluder, eps: float = 1e-4):
    """Returns vis (K,N) bool, cos (K,N) float32, z (K,N) float32 for K cameras."""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(apply_T(T_world_part, mesh.vertices).astype(np.float32)),
                        o3d.core.Tensor(mesh.faces.astype(np.uint32)))
    scene.add_triangles(o3d.core.Tensor(occluder.vertices.astype(np.float32)), o3d.core.Tensor(occluder.faces.astype(np.uint32)))
    P = apply_T(T_world_part, pts_part)
    Nn = apply_R(T_world_part, nrm_part)
    K, N = len(cam_T), len(P)
    vis = np.zeros((K, N), bool)
    cos = np.zeros((K, N), np.float32)
    zz = np.zeros((K, N), np.float32)
    rays, meta = [], []
    for k, T in enumerate(cam_T):
        c = T[:3, 3]
        d = P - c
        dist = np.linalg.norm(d, axis=1)
        dn = d / dist[:, None]
        pc = apply_T(np.linalg.inv(T), P)
        u, v, z = camera.project(pc)
        inimg = camera.in_image(u, v, z)
        cs = -(Nn * dn).sum(axis=1)                     # >0 when the face looks at the camera
        rays.append(np.concatenate([np.broadcast_to(c, d.shape), dn], axis=1))
        meta.append((dist, inimg, cs, z))
    ans = scene.cast_rays(o3d.core.Tensor(np.concatenate(rays).astype(np.float32)))
    t = ans["t_hit"].numpy().reshape(K, N)
    for k, (dist, inimg, cs, z) in enumerate(meta):
        unocc = t[k] >= dist - eps
        vis[k] = inimg & (cs > 0) & unocc
        cos[k] = np.clip(cs, 0, 1)
        zz[k] = z
    return vis, cos, zz


_W: dict = {}


def _init(cfg: dict) -> None:
    from ..utils.seeds import pin_worker

    pin_worker(max(1, (__import__("os").cpu_count() or 8) // 8))
    _W.update(cfg)
    _W["camera"] = SensorModel.model_validate(cfg["sensor"]).camera()
    _W["occ"] = Occluder.load(cfg["occluder_path"]) if cfg.get("occluder_path") else Occluder.belt_plane()
    grid = SlotGrid.from_dict(cfg["grid"])
    _W["cam_T"] = [grid.T_world_cam(k, np.asarray(cfg["focus"])) for k in range(grid.n_slots)]


def _one(rec: dict):
    mesh = LabeledMesh.load(resolve_data_path(rec["mesh"]))
    rng = rng_for("vis", rec["inst_id"])
    pts, nrm = sample_faces(mesh, mesh.face_label > 0, _W["n_samples"], rng)
    vis, cos, z = slot_visibility(mesh, np.asarray(rec["T_world_part"]), pts, nrm, _W["cam_T"], _W["camera"], _W["occ"])
    return rec["inst_id"], vis, cos.astype(np.float16), z.astype(np.float16)


def _vis_cfg(part_id: str, sensor: SensorModel, grid: SlotGrid, n_samples: int) -> dict:
    from ..datagen.parts import load_part

    occ_p = data_root() / "conveyor" / "occluder.npz"
    return {"sensor": sensor.model_dump(mode="json"), "grid": grid.to_dict(), "focus": load_part(part_id).focus_m,
            "n_samples": n_samples, "occluder_path": str(occ_p) if occ_p.exists() else None}


def nominal_visibility(part_id: str, sensor: SensorModel | None = None, grid: SlotGrid | None = None,
                       n_samples: int = 2000):
    """Slot visibility of area-weighted samples of the nominal part surface at the nominal
    pose (no variant set needed): vis (K,N) bool, cos (K,N), z (K,N)."""
    from ..datagen.parts import load_part_mesh

    sensor, grid = sensor or SensorModel(), grid or SlotGrid()
    _init(_vis_cfg(part_id, sensor, grid, n_samples))
    nom = load_part_mesh(part_id, "fine")
    pts, nrm = sample_faces(nom, np.ones(len(nom.faces), bool), n_samples, rng_for("vis-nominal", part_id))
    return slot_visibility(nom, np.eye(4), pts, nrm, _W["cam_T"], _W["camera"], _W["occ"])


def build_visibility(part_id: str, set_id: str, sensor: SensorModel | None = None, grid: SlotGrid | None = None,
                     params: VisibilityParams | None = None, n_workers: int = 8, overwrite: bool = False) -> Path:
    from ..datagen.variants import load_manifest, set_dir

    sensor = sensor or SensorModel()
    grid = grid or SlotGrid()
    params = params or VisibilityParams()
    from ..render.slot_cache import instance_digest
    from ..utils.hashing import content_hash
    from ..utils.io import read_json

    out = vis_dir(part_id, set_id, sensor.sensor_id)
    man = load_manifest(part_id, set_id)
    # reused only while the defect instances (meshes, labels, poses) are unchanged
    digest = content_hash({r["inst_id"]: instance_digest(r["mesh"], r["T_world_part"]) for r in man["instances"]
                           if r["role"] in ("search", "report", "unreachable")})
    if (out / "vis.npz").exists() and not overwrite and (out / "vis.json").exists():
        meta = read_json(out / "vis.json")
        known = meta.get("defects_digest")
        if known is None and (out / "vis.npz").stat().st_mtime >= (set_dir(part_id, set_id) / "manifest.json").stat().st_mtime:
            write_json(out / "vis.json", {**meta, "defects_digest": digest})    # older table, newer than its manifest
            known = digest
        if known == digest:
            return out / "vis.npz"
    cfg = _vis_cfg(part_id, sensor, grid, params.n_samples_per_defect)
    recs = [r for r in man["instances"] if r["role"] in ("search", "report", "unreachable")]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with core_order_env(), ctx.Pool(n_workers, initializer=_init, initargs=(cfg,)) as pool:
        res = pool.map(_one, recs)
    ids = [r[0] for r in res]
    vis = np.stack([r[1] for r in res])           # (J, K, N)
    cos = np.stack([r[2] for r in res])
    z = np.stack([r[3] for r in res])
    nv, nc, nz = nominal_visibility(part_id, sensor, grid)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "vis.npz", inst_ids=np.array(ids), vis=vis, cos=cos, z=z,
                        nominal_vis=nv, nominal_cos=nc.astype(np.float16), nominal_z=nz.astype(np.float16))
    write_json(out / "vis.json", {"part_id": part_id, "set_id": set_id, "grid": grid.to_dict(),
                                  "sensor": sensor.model_dump(mode="json"), "params": params.model_dump(),
                                  "n_instances": len(ids), "build_s": round(time.time() - t0, 1),
                                  "occluder": cfg["occluder_path"] or "belt_plane", "defects_digest": digest})
    log.info(f"visibility {part_id}/{set_id}: {len(ids)} defect instances x {grid.n_slots} slots in {time.time() - t0:.1f}s")
    return out / "vis.npz"
