"""Long-lived Isaac Sim render worker over the filesystem job queue (ADR-0002/0003).

    python -m isco.sim.worker --queue queue --headless      # or scripts/start_worker.sh

Start-up: launch ``SimulationApp`` once, build the scene once (ground, lights, official
conveyor, measured ISCO<->stage frame), create the camera rig (6 inspection cameras +
overview, parked) and the kinematic part prim, ``FileJobQueue.recover()``.

Loop: honour ``control/shutdown`` -> ``claim_next`` -> dispatch on ``job_type`` ->
``complete(JobResult)``. A background thread refreshes ``worker.heartbeat`` every few
seconds (also during long jobs). Exceptions become a ``failed`` result with the
traceback and the loop continues.

Job types (payload models in ``isco.schemas.job``): ``settle``, ``render_slots``,
``export_occluder``, ``verify_config``, ``capture_rgb``, ``ping``, ``shutdown``.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..geometry.camera_model import SlotGrid, camera_pose_spherical
from ..geometry.mesh_io import LabeledMesh
from ..geometry.transforms import make_T, rot_z
from ..render.pixels import ViewPixels
from ..render.queue import FileJobQueue
from ..schemas.job import Job, JobResult
from ..schemas.sensor import SensorModel
from ..utils.io import repo_root, write_json


def _log(msg: str) -> None:
    print(f"[isco.worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def _parse_payload(job: Job):
    """``Job.parsed_payload()`` except for payload-less job types: ``isco.schemas.job.PAYLOADS``
    maps ``ping``/``shutdown`` to bare ``pydantic.BaseModel``, which pydantic 2.11 refuses
    to instantiate - those get the raw dict."""
    from pydantic import BaseModel

    from ..schemas.job import PAYLOADS

    if PAYLOADS.get(job.job_type) is BaseModel:
        return dict(job.payload or {})
    return job.parsed_payload()


def _chunks(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def rotation_stage_pose(T_base: np.ndarray, theta_deg: float, reposition_xy=(0.0, 0.0), dyaw_deg: float = 0.0) -> np.ndarray:
    """Part pose after a rotation stage turned it by ``theta_deg`` (+ ``dyaw_deg`` error)
    about the turntable axis - the vertical axis through the ISCO world origin (the
    focus point), fixed in the cell - then shifted by the repositioning error
    ``reposition_xy`` (ISCO world). The part keeps its lateral placement error relative
    to the axis, which makes the rotated view *exactly* the cached view of the
    azimuth-shifted slot (ADR-0003; coordinator decision 2026-10-20, replacing the
    first version that rotated about the part's own origin)."""
    T_base = np.asarray(T_base, dtype=float)
    R = rot_z(theta_deg + dyaw_deg)
    shift = np.array([reposition_xy[0], reposition_xy[1], 0.0])
    return make_T(R, shift) @ T_base


def frustum_mesh(T_world_cam: np.ndarray, size: float = 0.03, aspect: float = 1.0):
    """Small pyramid (apex at the camera centre, base along +z_cv) for overview images."""
    apex = np.zeros(3)
    h = size * 0.5
    base = np.array([[-h, -h / aspect, size], [h, -h / aspect, size], [h, h / aspect, size], [-h, h / aspect, size]])
    V = np.vstack([apex, base]) @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
    F = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 3, 2], [1, 4, 3]])
    return V, F


class Worker:
    def __init__(self, queue_root: str | Path, headless: bool = True, conveyor_asset: str = "ConveyorBelt_A06",
                 n_cams: int = 6, poll_s: float = 0.5, heartbeat_s: float = 5.0, exit_when_idle: bool = False,
                 mesh_cache_size: int = 6):
        self.queue = FileJobQueue(queue_root)
        self.headless = headless
        self.conveyor_asset = conveyor_asset
        self.n_cams = n_cams
        self.poll_s = poll_s
        self.heartbeat_s = heartbeat_s
        self.exit_when_idle = exit_when_idle
        self.mesh_cache: OrderedDict[str, tuple] = OrderedDict()
        self.mesh_cache_size = mesh_cache_size
        self.status = {"state": "starting", "job_id": None, "job_type": None, "n_done": 0, "n_failed": 0}
        self._hb_stop = threading.Event()
        self.scene = None
        self.rig = None
        self.part = None
        self.app_info: dict = {}

    # -- lifecycle ------------------------------------------------------------------------
    def heartbeat(self, **extra) -> None:
        try:
            self.queue.heartbeat(**self.status, **self.app_info, **extra)
        except Exception:  # pragma: no cover - never let the heartbeat kill a job
            pass

    def _hb_loop(self) -> None:
        while not self._hb_stop.wait(self.heartbeat_s):
            self.heartbeat()

    def start(self) -> None:
        from . import app as isco_app

        self.heartbeat()
        threading.Thread(target=self._hb_loop, name="isco-heartbeat", daemon=True).start()
        t0 = time.perf_counter()
        isco_app.launch(headless=self.headless)
        t_app = time.perf_counter() - t0
        from .part_prim import PartPrim
        from .rig import CameraRig
        from .scene import IscoScene

        t1 = time.perf_counter()
        self.scene = IscoScene.build(conveyor_asset=self.conveyor_asset, speed=0.0)
        self.rig = CameraRig(self.scene.frame, SensorModel(), n_cams=self.n_cams, overview=True)
        self.part = PartPrim(self.scene.stage, self.scene.frame)
        t_scene = time.perf_counter() - t1
        n_rec = self.queue.recover()
        if self.queue.shutdown_requested():
            _log("clearing a stale control/shutdown request")
            self.queue.clear_shutdown()
        self.app_info = {"isaac_version": isco_app.isaac_version(), "host": socket.gethostname(),
                         "startup_s": round(t_app, 2), "scene_build_s": round(t_scene, 2),
                         "isco_origin_stage": self.scene.frame.origin_stage.tolist()}
        self.status["state"] = "idle"
        self.heartbeat()
        _log(f"ready: app {t_app:.1f}s, scene+rig {t_scene:.1f}s, origin_stage="
             f"{np.round(self.scene.frame.origin_stage, 5).tolist()}, recovered {n_rec} job(s)")

    def run(self) -> None:
        self.start()
        try:
            while True:
                if self.queue.shutdown_requested():
                    self.queue.clear_shutdown()
                    _log("shutdown requested")
                    break
                job = self.queue.claim_next()
                if job is None:
                    if self.exit_when_idle:
                        _log("queue empty, exiting (--exit-when-idle)")
                        break
                    time.sleep(self.poll_s)
                    continue
                result = self.handle(job)
                self.queue.complete(result)
                if job.job_type == "shutdown":
                    break
        finally:
            self.status.update(state="stopped", job_id=None, job_type=None)
            self.heartbeat()
            self._hb_stop.set()

    # -- dispatch ---------------------------------------------------------------------------
    def handle(self, job: Job) -> JobResult:
        self.status.update(state="busy", job_id=job.job_id, job_type=job.job_type)
        self.heartbeat()
        t0 = time.perf_counter()
        _log(f"job {job.job_id} ({job.job_type}) started")
        warnings: list[str] = []
        try:
            payload = _parse_payload(job)
            handler = getattr(self, f"job_{job.job_type}")
            if job.job_type == "settle":
                outputs = handler(payload, warnings, raw=job.payload)
            else:
                outputs = handler(payload, warnings)
            res = JobResult(job_id=job.job_id, job_type=job.job_type, status="ok", outputs=outputs or {},
                            warnings=warnings[:200], wall_s=round(time.perf_counter() - t0, 3),
                            worker=self.worker_info())
            self.status["n_done"] += 1
        except Exception as e:
            tb = traceback.format_exc()
            _log(f"job {job.job_id} FAILED: {e}\n{tb}")
            res = JobResult(job_id=job.job_id, job_type=job.job_type, status="failed", warnings=warnings[:200],
                            error=f"{type(e).__name__}: {e}\n{tb}", wall_s=round(time.perf_counter() - t0, 3),
                            worker=self.worker_info())
            self.status["n_failed"] += 1
            self._recover_scene()
        self.status.update(state="idle", job_id=None, job_type=None)
        _log(f"job {job.job_id} ({job.job_type}) {res.status} in {res.wall_s:.1f}s"
             + (f", {len(res.warnings)} warning(s)" if res.warnings else ""))
        return res

    def worker_info(self) -> dict:
        return {"pid": os.getpid(), **self.app_info}

    def _recover_scene(self) -> None:
        try:
            self.rig.park_all()
            self.part.set_visible(True)
        except Exception:  # pragma: no cover
            pass

    # -- helpers ----------------------------------------------------------------------------
    def load_mesh(self, mesh_path: str):
        from .capture import SurfaceIndex

        key = str(resolve(mesh_path))
        if key in self.mesh_cache:
            self.mesh_cache.move_to_end(key)
            return self.mesh_cache[key]
        mesh = LabeledMesh.load(key)
        entry = (key, mesh, SurfaceIndex.from_mesh(mesh))
        self.mesh_cache[key] = entry
        while len(self.mesh_cache) > self.mesh_cache_size:
            self.mesh_cache.popitem(last=False)
        return entry

    def _view_meta(self, **kw) -> dict:
        pin = self.rig.pinhole
        return {"backend": "isaac", "width": pin.width, "height": pin.height, "hfov_deg": pin.hfov_deg,
                "K": pin.K.tolist(), **kw}

    # -- job handlers -------------------------------------------------------------------------
    def job_ping(self, payload, warnings) -> dict:
        return {"pong": True, "time": time.time(), "origin_stage": self.scene.frame.origin_stage.tolist(),
                "n_cams": self.rig.n_cams, "sensor": self.rig.sensor.model_dump(mode="json")}

    def job_shutdown(self, payload, warnings) -> dict:
        return {"shutdown": True}

    def job_export_occluder(self, p, warnings) -> dict:
        from .export_occluder import export_occluder

        if p.conveyor_asset != self.scene.asset_name:
            warnings.append(f"worker scene uses {self.scene.asset_name}, payload asked {p.conveyor_asset}")
        return export_occluder(self.scene, resolve(p.output_path), radius_m=p.radius_m)

    #: optional settle keys read from the raw job payload (not part of SettlePayload)
    SETTLE_EXTRAS = ("tilt_sigma_deg", "ang_vel_sigma_dps", "physics_dt")

    def job_settle(self, p, warnings, raw: dict | None = None) -> dict:
        from .process import settle_part

        extras = {k: float(v) for k, v in (raw or {}).items() if k in self.SETTLE_EXTRAS and v is not None}
        key, mesh, _ = self.load_mesh(p.mesh_path)
        self.part.set_visible(False)
        try:
            res = settle_part(self.scene, mesh, speeds_mps=p.speeds_mps, n_drops=p.n_drops,
                              drop_height_m=p.drop_height_m, lateral_range_m=p.lateral_range_m,
                              yaw_sigma_deg=p.yaw_sigma_deg, settle_time_s=p.settle_time_s, seed=p.seed,
                              heartbeat=self.heartbeat, log=_log, part_id=p.part_id, **extras)
        finally:
            self.part.set_visible(True)
        res["mesh_path"] = p.mesh_path
        out = resolve(p.output_path)
        write_json(out, res)
        summary = {k: {kk: v["summary"].get(kk) for kk in ("n", "flipped_fraction", "settled_fraction")}
                   for k, v in res["speeds"].items()}
        for k, v in res["speeds"].items():
            if v["summary"].get("settled_fraction", 1.0) < 1.0:
                warnings.append(f"speed {k}: {1 - v['summary']['settled_fraction']:.0%} of drops not at rest "
                                f"after {p.settle_time_s}s")
        return {"output_path": str(out), "summary": summary, "n_physics_steps": res["n_physics_steps"],
                "settle_wall_s": res["wall_s"]}

    def job_render_slots(self, p, warnings) -> dict:
        from .capture import capture_views

        self.rig.configure(p.sensor)
        grid = SlotGrid.from_dict(p.grid)
        focus = np.asarray(p.focus_m, dtype=float)
        out_root = resolve(p.output_dir)
        n_views, n_fail, worst = 0, 0, 0.0
        timing = {"render_s": 0.0, "read_s": 0.0, "verify_s": 0.0, "save_s": 0.0, "load_s": 0.0, "renders": 0}
        t_all = time.perf_counter()
        if p.save_rgb:
            self.rig.enable_rgb()
        from concurrent.futures import ThreadPoolExecutor

        # optional background npz compression; measured slower (4.3 vs 8.3 views/s) because the
        # threads compete with Kit's main loop, hence off by default
        n_threads = int(os.environ.get("ISCO_SAVE_THREADS", "0"))
        pool = ThreadPoolExecutor(max_workers=n_threads) if n_threads > 0 else None
        futures = []
        for inst in p.instances:
            t0 = time.perf_counter()
            key, mesh, index = self.load_mesh(inst.mesh_path)
            self.part.load(mesh, key=key)
            T_wp = np.asarray(inst.T_world_part, dtype=float)
            self.part.set_pose(T_wp)
            timing["load_s"] += time.perf_counter() - t0
            for batch in _chunks(p.slot_ids, self.rig.n_cams):
                poses = [grid.T_world_cam(int(k), focus) for k in batch]
                cams = self.rig.place(poses)
                metas = [self._view_meta(slot_id=int(k), inst_id=inst.inst_id, part_id=p.part_id,
                                         T_world_cam=T.tolist(), T_world_part=T_wp.tolist(),
                                         sensor_id=p.sensor.sensor_id, focus_m=focus.tolist())
                         for k, T in zip(batch, poses)]
                views, warns, tm = capture_views(cams, metas, warmup=1, rt_subframes=4 if p.save_rgb else 1,
                                                 verify={"index": index, "T_world_part": T_wp}, log=_log)
                warnings.extend(f"{inst.inst_id}: {w}" for w in warns)
                for k_ in ("render_s", "read_s", "verify_s", "renders"):
                    timing[k_] += tm[k_]
                t1 = time.perf_counter()
                d = out_root / inst.inst_id
                for k, vp, cam in zip(batch, views, cams):
                    if pool is not None:
                        futures.append(pool.submit(vp.save, d / f"slot_{int(k):03d}.npz"))
                    else:
                        vp.save(d / f"slot_{int(k):03d}.npz")
                    v = vp.meta.get("verify", {})
                    if v and not v.get("ok", True):
                        n_fail += 1
                    if v and v.get("median_m") is not None:
                        worst = max(worst, float(v["median_m"]))
                    if p.save_rgb:
                        self._save_rgb(cam, d / f"slot_{int(k):03d}.png")
                timing["save_s"] += time.perf_counter() - t1
                n_views += len(batch)
            self.heartbeat()
        self.rig.park_all()
        t1 = time.perf_counter()
        for f in futures:
            f.result()
        if pool is not None:
            pool.shutdown()
        timing["save_wait_s"] = time.perf_counter() - t1
        wall = time.perf_counter() - t_all
        return {"output_dir": str(out_root), "n_views": n_views, "n_instances": len(p.instances),
                "verify_failures": n_fail, "verify_worst_median_mm": worst * 1e3,
                "timing_s": {k: round(v, 3) for k, v in timing.items()}, "wall_s": round(wall, 3),
                "views_per_s": round(n_views / max(wall, 1e-9), 2)}

    def _save_rgb(self, cam, path: Path) -> None:
        import imageio

        rgba = cam.get_data("rgb")
        if rgba is None or np.asarray(rgba).size == 0:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(path, np.asarray(rgba)[:, :, :3].astype(np.uint8))

    def job_verify_config(self, p, warnings) -> dict:
        from .capture import capture_views

        self.rig.configure(p.sensor)
        cfg, proc = p.config, p.process
        focus = np.asarray(p.focus_m, dtype=float)
        v = float(proc.speed(cfg.speed_level))
        sigma_t = float(proc.trigger_sigma_s)
        cams_T = [camera_pose_spherical(focus, c.azimuth_deg, c.elevation_deg, c.distance_m) for c in p.cameras]
        if len(cams_T) > self.rig.n_cams:
            raise ValueError(f"config has {len(cams_T)} cameras, rig has {self.rig.n_cams}")
        rng = np.random.default_rng(p.seed)
        out_root = resolve(p.output_dir)
        n_views, n_fail = 0, 0
        all_offsets = []
        t_all = time.perf_counter()
        for inst in p.instances:
            key, mesh, index = self.load_mesh(inst.mesh_path)
            self.part.load(mesh, key=key)
            T_base = np.asarray(inst.T_world_part, dtype=float)
            inst_offsets = []
            for s, theta in enumerate(cfg.stages_deg):
                if s > 0:
                    rep_xy = rng.normal(0.0, proc.rot_reposition_sigma_m, 2)
                    dyaw = float(rng.normal(0.0, proc.rot_yaw_sigma_deg))
                else:
                    rep_xy, dyaw = np.zeros(2), 0.0
                T_stage = rotation_stage_pose(T_base, theta, rep_xy, dyaw)
                if cfg.trigger_mode == "simultaneous":
                    t_trig = float(rng.normal(0.0, sigma_t)) if sigma_t > 0 else 0.0
                    groups = [(list(range(len(cams_T))), t_trig)]
                else:
                    groups = [([c], float(rng.normal(0.0, sigma_t)) if sigma_t > 0 else 0.0) for c in range(len(cams_T))]
                for cam_ids, t_trig in groups:
                    dx = float(cfg.trigger_offset_m + v * t_trig)
                    T_wp = make_T(None, [dx, 0.0, 0.0]) @ T_stage
                    self.part.set_pose(T_wp)
                    cams = self.rig.place([cams_T[c] for c in cam_ids])
                    metas = []
                    for c in cam_ids:
                        spec = p.cameras[c]
                        metas.append(self._view_meta(inst_id=inst.inst_id, part_id=p.part_id, stage=s,
                                                     stage_deg=float(theta), cam=c, slot_id=spec.slot_id,
                                                     T_world_cam=cams_T[c].tolist(), T_world_part=T_wp.tolist(),
                                                     dx_m=dx, t_trigger_s=t_trig, speed_mps=v,
                                                     reposition_xy_m=np.asarray(rep_xy).tolist(), dyaw_deg=dyaw,
                                                     config_id=cfg.config_id, sensor_id=p.sensor.sensor_id))
                    views, warns, _ = capture_views(cams, metas, warmup=1,
                                                    verify={"index": index, "T_world_part": T_wp}, log=_log)
                    warnings.extend(f"{inst.inst_id}: {w}" for w in warns)
                    for c, vp in zip(cam_ids, views):
                        vp.save(out_root / inst.inst_id / f"stage_{s}_cam_{c}.npz")
                        if not vp.meta.get("verify", {}).get("ok", True):
                            n_fail += 1
                        n_views += 1
                        inst_offsets.append({"stage": s, "stage_deg": float(theta), "cam": c, "dx_m": dx,
                                             "t_trigger_s": t_trig, "reposition_xy_m": np.asarray(rep_xy).tolist(),
                                             "dyaw_deg": dyaw, "T_world_part": T_wp.tolist()})
            write_json(out_root / inst.inst_id / "offsets.json",
                       {"inst_id": inst.inst_id, "config": cfg.model_dump(mode="json"), "config_id": cfg.config_id,
                        "speed_mps": v, "trigger_sigma_s": sigma_t, "T_world_part_base": T_base.tolist(),
                        "views": inst_offsets})
            all_offsets.append(inst.inst_id)
            self.heartbeat()
        self.rig.park_all()
        self.part.set_pose(np.eye(4))
        wall = time.perf_counter() - t_all
        return {"output_dir": str(out_root), "n_views": n_views, "instances": all_offsets, "verify_failures": n_fail,
                "config_id": cfg.config_id, "wall_s": round(wall, 3)}

    def job_capture_rgb(self, p, warnings) -> dict:
        import imageio

        from .capture import render_fresh
        from .part_prim import define_mesh

        self.rig.configure(p.sensor)
        focus = np.asarray(p.focus_m, dtype=float)
        key, mesh, _ = self.load_mesh(p.mesh_path)
        self.part.load(mesh, key=key)
        self.part.set_pose(np.eye(4))
        out = resolve(p.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cams_T = [camera_pose_spherical(focus, c.azimuth_deg, c.elevation_deg, c.distance_m) for c in p.cameras]
        files = []
        for start in range(0, len(cams_T), self.rig.n_cams):
            batch = list(range(start, min(start + self.rig.n_cams, len(cams_T))))
            cams = self.rig.place([cams_T[i] for i in batch])
            self.rig.enable_rgb(cams)
            render_fresh(2, rt_subframes=16)
            for i, cam in zip(batch, cams):
                f = out / f"cam_{i}.png"
                self._save_rgb(cam, f)
                files.append(str(f))
        self.rig.park_all()
        if p.overview and self.rig.overview is not None:
            stage = self.scene.stage
            marker_root = "/World/RigMarkers"
            for i, T in enumerate(cams_T):
                V, F = frustum_mesh(T, size=0.04)
                define_mesh(stage, f"{marker_root}/cam_{i}", self.scene.frame.to_stage_pts(V), F, (0.1, 0.4, 0.9))
            eye = focus + np.array([0.75, -0.85, 0.65])
            ov = self.rig.place_overview(eye, focus)
            render_fresh(2, rt_subframes=16)
            f = out / "overview.png"
            self._save_rgb(ov, f)
            files.append(str(f))
            self.rig.park(ov)
            stage.RemovePrim(marker_root)
        return {"output_dir": str(out), "files": files}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m isco.sim.worker", description="ISCO Isaac Sim render worker")
    ap.add_argument("--queue", default="queue", help="queue root directory (default: ./queue)")
    ap.add_argument("--headless", dest="headless", action="store_true", default=True, help="run headless (default)")
    ap.add_argument("--gui", dest="headless", action="store_false", help="open the Isaac Sim GUI")
    ap.add_argument("--conveyor-asset", default="ConveyorBelt_A06")
    ap.add_argument("--n-cams", type=int, default=6)
    ap.add_argument("--poll-s", type=float, default=0.5)
    ap.add_argument("--exit-when-idle", action="store_true", help="exit once the queue is empty (batch mode)")
    args = ap.parse_args(argv)
    w = Worker(args.queue, headless=args.headless, conveyor_asset=args.conveyor_asset, n_cams=args.n_cams,
               poll_s=args.poll_s, exit_when_idle=args.exit_when_idle)
    w.run()
    from . import app as isco_app

    _log("closing SimulationApp")
    sys.stdout.flush()
    isco_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
