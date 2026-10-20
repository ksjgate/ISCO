"""Render-once / compose-many slot cache (ADR-0003, D1).

Every instance of a variant set is rendered once from every slot of the
``SlotGrid`` (96 by default) at its sampled resting pose. A configuration is then
evaluated by *selecting* cached views (rotation stages = azimuth-shifted slots), so
evaluation cost is independent of rendering.

Layout ``data/cache/<part_id>/<set_id>/<sensor_id>/``::

    cache.json            grid, sensor, focus, per-slot T_world_cam / ROI / z0, stats
    views/<inst_id>.npz   per slot k: p<k> uint32 pixel idx, d<k> uint16 depth quanta,
                          l<k> uint8 label, c<k> uint8 round(255*cos)
"""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from ..geometry.camera_model import PinholeCamera, SlotGrid
from ..geometry.mesh_io import LabeledMesh
from ..geometry.transforms import apply_T
from ..schemas.sensor import SensorModel
from ..utils.io import data_root, read_json, resolve_data_path, write_json
from ..utils.logging import get_logger
from .occluder import Occluder
from .pixels import ViewPixels, dequantise, quantise

log = get_logger(__name__)

Z0_BACKOFF_M = 0.15   # depth quantisation origin = slot distance - this


def cache_dir(part_id: str, set_id: str, sensor_id: str) -> Path:
    return data_root() / "cache" / part_id / set_id / sensor_id


def slot_rois(cam: PinholeCamera, grid: SlotGrid, focus: np.ndarray, extents: np.ndarray,
              margin_xy: float = 7e-3, margin_z: float = 2e-3, yaw_random: bool = False) -> list[tuple[int, int, int, int]]:
    """Per-slot pixel ROI covering the part's nominal bbox grown by the pose-jitter margin."""
    ex, ey, ez = extents
    if yaw_random:
        r = 0.5 * np.hypot(ex, ey)
        ex = ey = 2 * r
    xs = (-ex / 2 - margin_xy, ex / 2 + margin_xy)
    ys = (-ey / 2 - margin_xy, ey / 2 + margin_xy)
    zs = (-margin_z, ez + margin_z)
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    rois = []
    for k in range(grid.n_slots):
        T = grid.T_world_cam(k, focus)
        pc = apply_T(np.linalg.inv(T), corners)
        u, v, z = cam.project(pc)
        u0 = int(np.clip(np.floor(u.min()) - 4, 0, cam.width))
        u1 = int(np.clip(np.ceil(u.max()) + 4, 0, cam.width))
        v0 = int(np.clip(np.floor(v.min()) - 4, 0, cam.height))
        v1 = int(np.clip(np.ceil(v.max()) + 4, 0, cam.height))
        rois.append((u0, v0, u1, v1))
    return rois


class SlotCache:
    """Read access to a built cache."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = read_json(self.root / "cache.json")
        self.sensor = SensorModel.model_validate(self.meta["sensor"])
        self.camera = self.sensor.camera()
        self.grid = SlotGrid.from_dict(self.meta["grid"])
        self.focus = np.asarray(self.meta["focus_m"])
        self._T = {int(s["slot_id"]): np.asarray(s["T_world_cam"]) for s in self.meta["slots"]}
        self._z0 = {int(s["slot_id"]): float(s["z0"]) for s in self.meta["slots"]}

    @classmethod
    def open(cls, part_id: str, set_id: str, sensor_id: str = "sl_default") -> "SlotCache":
        return cls(cache_dir(part_id, set_id, sensor_id))

    def T_world_cam(self, slot: int) -> np.ndarray:
        return self._T[slot]

    def views(self, inst_id: str, slot_ids: list[int]) -> dict[int, ViewPixels]:
        out = {}
        with np.load(self.root / "views" / f"{inst_id}.npz", allow_pickle=False) as z:
            for k in slot_ids:
                out[k] = ViewPixels(z[f"p{k:03d}"].astype(np.int64), dequantise(z[f"d{k:03d}"], self._z0[k]),
                                    z[f"l{k:03d}"], z[f"c{k:03d}"].astype(np.float32) / 255.0,
                                    {"slot_id": k, "inst_id": inst_id})
        return out

    def pixel_counts(self) -> dict:
        return self.meta.get("pixel_counts", {})


# -- building -----------------------------------------------------------------------
_W: dict = {}


def _init(meta_json: str, occluder_path: str | None, n_threads: int) -> None:
    import json

    meta = json.loads(meta_json)
    sensor = SensorModel.model_validate(meta["sensor"])
    occ = Occluder.load(occluder_path) if occluder_path else Occluder.belt_plane()
    from .raycast_backend import RaycastRenderer

    _W.update(meta=meta, renderer=RaycastRenderer(sensor.camera(), occ, n_threads=n_threads))


def _render_instance(job: tuple[str, str, list, str]) -> dict:
    inst_id, mesh_path, T_list, out_path = job
    meta = _W["meta"]
    mesh = LabeledMesh.load(resolve_data_path(mesh_path))
    T = np.asarray(T_list)
    views = [(np.asarray(s["T_world_cam"]), tuple(s["roi"])) for s in meta["slots"]]
    t0 = time.time()
    vps = _W["renderer"].render(mesh, T, views)
    arrays = {}
    counts, trunc = [], 0
    W = meta["sensor"]["width"]
    for s, vp in zip(meta["slots"], vps):
        k = int(s["slot_id"])
        u0, v0, u1, v1 = s["roi"]
        u, v = vp.pix % W, vp.pix // W
        if len(vp) and ((u <= u0).any() or (u >= u1 - 1).any() or (v <= v0).any() or (v >= v1 - 1).any()):
            trunc += 1
        arrays[f"p{k:03d}"] = vp.pix.astype(np.uint32)
        arrays[f"d{k:03d}"] = quantise(vp.depth, float(s["z0"]))
        arrays[f"l{k:03d}"] = vp.label.astype(np.uint8)
        arrays[f"c{k:03d}"] = np.round(np.clip(vp.cos, 0, 1) * 255).astype(np.uint8)
        counts.append(len(vp))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(out_path).with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(out_path)
    return {"inst_id": inst_id, "render_s": round(time.time() - t0, 3), "n_px": counts, "truncated_views": trunc}


def build_slot_cache(part_id: str, set_id: str, sensor: SensorModel | None = None, grid: SlotGrid | None = None,
                     n_workers: int = 6, overwrite: bool = False, occluder_path: str | None = None) -> SlotCache:
    from ..datagen.parts import load_part
    from ..datagen.variants import load_manifest

    sensor = sensor or SensorModel()
    grid = grid or SlotGrid()
    root = cache_dir(part_id, set_id, sensor.sensor_id)
    man = load_manifest(part_id, set_id)
    spec = load_part(part_id)
    if occluder_path is None:
        default_occ = data_root() / "conveyor" / "occluder.npz"
        occluder_path = str(default_occ) if default_occ.exists() else None
    cam = sensor.camera()
    focus = np.asarray(spec.focus_m)
    yaw_random = man["spec"].get("pose", {}).get("yaw_mode", "guided") == "random"   # real sets have no pose spec
    rois = slot_rois(cam, grid, focus, np.asarray(spec.extents_m), yaw_random=yaw_random)
    slots = []
    for k in range(grid.n_slots):
        s = grid.slot(k)
        slots.append({"slot_id": k, "azimuth_deg": s.azimuth_deg, "elevation_deg": s.elevation_deg,
                      "distance_m": s.distance_m, "T_world_cam": grid.T_world_cam(k, focus).tolist(),
                      "roi": list(rois[k]), "z0": s.distance_m - Z0_BACKOFF_M})
    meta = {"part_id": part_id, "set_id": set_id, "sensor": sensor.model_dump(mode="json"), "grid": grid.to_dict(),
            "focus_m": focus.tolist(), "slots": slots, "backend": "embree(open3d)",
            "occluder": occluder_path or "belt_plane", "manifest_hash": man["spec_hash"]}
    jobs = []
    for r in man["instances"]:
        out = root / "views" / f"{r['inst_id']}.npz"
        if out.exists() and not overwrite:
            continue
        jobs.append((r["inst_id"], r["mesh"], r["T_world_part"], str(out)))
    t0 = time.time()
    results = []
    if jobs:
        import json

        n_threads = max(1, (mp.cpu_count() // n_workers))
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers, initializer=_init, initargs=(json.dumps(meta), occluder_path, n_threads)) as pool:
            for i, res in enumerate(pool.imap_unordered(_render_instance, jobs)):
                results.append(res)
                if (i + 1) % 25 == 0:
                    log.info(f"slot cache {part_id}/{set_id}: {i + 1}/{len(jobs)} instances")
    old = read_json(root / "cache.json") if (root / "cache.json").exists() else {}
    stats = {**old.get("instance_stats", {}), **{r["inst_id"]: r for r in results}}
    meta["instance_stats"] = stats
    meta["build_s"] = round(time.time() - t0, 1)
    n_px = np.array([s["n_px"] for s in stats.values()]) if stats else np.zeros((0, grid.n_slots))
    meta["pixel_counts"] = {"mean_per_view": float(n_px.mean()) if n_px.size else 0.0,
                            "per_slot_mean": n_px.mean(axis=0).round(1).tolist() if n_px.size else []}
    meta["truncated_views_total"] = int(sum(s["truncated_views"] for s in stats.values()))
    write_json(root / "cache.json", meta)
    log.info(f"slot cache {part_id}/{set_id}: {len(results)} rendered in {meta['build_s']}s, "
             f"mean {meta['pixel_counts']['mean_per_view']:.0f} px/view, truncated views {meta['truncated_views_total']}")
    return SlotCache(root)
