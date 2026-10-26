"""Isaac Sim footage of the mid-term demo video (``scripts/midterm/video.py`` adds captions).

Shots (1280x720, 30 fps) in ``runs/midterm_reports/isaac/``:

* ``line.mp4``      - the conveyor cell; normal parts of the six products flow to the inspection point;
* ``lineup.mp4``    - the six products side by side (``lineup.json``: label anchors per frame);
* ``defect_<type>.mp4`` - orbit around one defect of each type (T-LESS 29);
* ``slots.mp4``     - the 96 candidate camera positions around the inspection point;
* ``turntable.mp4`` - the rotation stage turning a part 90 degrees three times;
* ``part_<id>.mp4`` - per product: its recommended rig (camera frusta), belt speed and rotation
  stage; the demo defect part stops/rotates/is captured (``part_<id>.json``: capture frames);
* ``capture_<id>/`` - what the rig cameras record for the demo defect part (RGB, depth, labels),
  per rotation stage.

    python scripts/midterm/isaac_shots.py [--only part_tless_29 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PARTS, SPEEDS, TYPES, WORK, best_record, save_json, showcase  # noqa: E402

FPS = 30
PITCH = 0.33
T_ROT = {"none": 0.0, "r90x1": 0.9, "r180x1": 1.1, "r90x3": 1.7}
STAGES = {"none": [0.0], "r90x1": [0.0, 90.0], "r180x1": [0.0, 180.0], "r90x3": [0.0, 90.0, 180.0, 270.0]}


def smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def project(ov, p_world: np.ndarray) -> list[float] | None:
    """Pixel of a world point in the overview camera (None if behind it)."""
    T = ov.T_world_cam
    pc = T[:3, :3].T @ (np.asarray(p_world, float) - T[:3, 3])
    if pc[2] <= 1e-6:
        return None
    K = ov.pinhole.K
    return [float(K[0, 0] * pc[0] / pc[2] + K[0, 2]), float(K[1, 1] * pc[1] / pc[2] + K[1, 2])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="shot names to render (default: all)")
    ap.add_argument("--subframes", type=int, default=4)
    a = ap.parse_args()
    want = (lambda s: True) if not a.only else (lambda s: s in a.only)

    from isco.sim import app as isco_app

    isco_app.launch(headless=True)
    import imageio
    import trimesh
    from pxr import UsdGeom

    from isco.datagen.parts import load_part
    from isco.datagen.variants import load_manifest
    from isco.geometry.camera_model import SlotGrid
    from isco.geometry.mesh_io import LabeledMesh
    from isco.geometry.transforms import make_T
    from isco.schemas.sensor import SensorModel
    from isco.sim.capture import render_fresh
    from isco.sim.part_prim import PartPrim, define_mesh
    from isco.sim.rig import CameraRig, set_pinhole_intrinsics
    from isco.sim.scene import IscoScene
    from isco.sim.worker import frustum_mesh, rotation_stage_pose
    from isco.utils.io import resolve_data_path

    out = WORK / "isaac"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    grid = SlotGrid()
    shows = showcase()
    mans = {p: load_manifest(p, "mvp") for p in PARTS}
    by_id = {p: {r["inst_id"]: r for r in m["instances"]} for p, m in mans.items()}
    focus_of = {p: np.asarray(load_part(p).focus_m, dtype=float) for p in PARTS}   # where the rig cameras aim
    origin = np.zeros(3)                       # the inspection point on the belt (ISCO world origin)
    focus = origin + np.array([0.0, 0.0, 0.012])

    scene = IscoScene.build(speed=0.0)
    stage = scene.stage
    rig = CameraRig(scene.frame, SensorModel(), n_cams=3, overview=True)
    ov = rig.overview
    set_pinhole_intrinsics(ov.camera, ov.pinhole, 24.0, 0.005, 50.0)   # close-ups come within 5 cm

    def show(prims, on: bool) -> None:
        for p in prims:
            UsdGeom.Imageable(p).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if on else UsdGeom.Tokens.invisible)

    def frusta(name: str, Ts: list, size: float, colors) -> list:
        prims = []
        for i, T in enumerate(Ts):
            V, F = frustum_mesh(T, size=size)
            p = f"/World/{name}/f_{i}"
            define_mesh(stage, p, scene.frame.to_stage_pts(V), F, colors[i] if isinstance(colors, list) else colors)
            prims.append(stage.GetPrimAtPath(p))
        show(prims, False)
        return prims

    # turntable disc at the inspection point (visual only)
    disc = trimesh.creation.cylinder(radius=0.05, height=0.0014, sections=72)
    disc.apply_translation(origin + np.array([0.0, 0.0, 0.0002]))      # top 0.9 mm above the belt
    define_mesh(stage, "/World/Turntable", scene.frame.to_stage_pts(np.asarray(disc.vertices)),
                np.asarray(disc.faces), (0.30, 0.33, 0.37), flat_normals=False)
    turntable = [stage.GetPrimAtPath("/World/Turntable")]
    show(turntable, False)

    def loaded(pid: str, iid: str, path: str) -> tuple:
        pp = PartPrim(stage, scene.frame, root_path=path)
        r = by_id[pid][iid]
        pp.load(LabeledMesh.load(str(resolve_data_path(r["mesh"]))), key=f"{pid}/{iid}")
        pp.set_visible(False)
        return pp, np.asarray(r["T_world_part"], dtype=float)

    def normal_id(pid: str, k: int = 0) -> str:
        return [r["inst_id"] for r in mans[pid]["instances"] if r["role"] == "val"][k]

    def grab() -> np.ndarray:
        render_fresh(1, rt_subframes=a.subframes)
        return np.asarray(ov.get_data("rgb"))[:, :, :3].astype(np.uint8)

    def writer(name: str):
        return imageio.get_writer(str(out / f"{name}.mp4"), fps=FPS, codec="libx264", quality=9, macro_block_size=8)

    def log(msg: str) -> None:
        print(f"[isaac_shots] {msg} ({time.time() - t_start:.0f}s)", flush=True)

    for _ in range(3):   # renderer warm-up (asset streaming, shader compilation)
        rig.place_overview(focus + np.array([0.6, -0.7, 0.5]), focus)
        render_fresh(2, rt_subframes=8)

    normals = {p: loaded(p, normal_id(p), f"/World/N/{p}") for p in PARTS}

    # -- the line: normal parts of all six products flow at 0.3 m/s ----------------------
    if want("line"):
        flow = [normals[p] for p in PARTS]
        x_lo, x_hi, span = -0.78, 1.15, PITCH * len(flow)
        n = 7 * FPS
        eye0, tgt0 = np.array([1.75, -1.75, 1.15]), np.array([0.15, 0.0, 0.0])
        eye1, tgt1 = focus + np.array([0.48, -0.58, 0.38]), focus
        w = writer("line")
        for f in range(n):
            u = smooth(f / (n - 1))
            for j, (pp, T0) in enumerate(flow):
                x = (-0.5 + PITCH * j + 0.3 * f / FPS - x_lo) % span + x_lo
                pp.set_visible(x <= x_hi)
                pp.set_pose(make_T(None, [x, 0.0, 0.0]) @ T0)
            rig.place_overview(eye0 + (eye1 - eye0) * u, tgt0 + (tgt1 - tgt0) * u)
            w.append_data(grab())
        w.close()
        for pp, _ in flow:
            pp.set_visible(False)
        log("line")

    # -- the six products side by side --------------------------------------------------
    if want("lineup"):
        xs = focus[0] + 0.078 * (np.arange(len(PARTS)) - (len(PARTS) - 1) / 2)
        for (pp, T0), x in zip((normals[p] for p in PARTS), xs):
            pp.set_pose(make_T(None, [x, 0.0, 0.0]) @ T0)
            pp.set_visible(True)
        n = 7 * FPS
        w = writer("lineup")
        anchors = []
        for f in range(n):
            u = smooth(f / (n - 1))
            xc = xs[0] - 0.02 + (xs[-1] - xs[0] + 0.04) * u
            eye = np.array([xc + 0.04, focus[1] - 0.21, focus[2] + 0.13])
            tgt = np.array([xc, focus[1], focus[2] + 0.012])
            rig.place_overview(eye, tgt)
            w.append_data(grab())
            anchors.append([project(ov, np.array([x, 0.0, 2 * focus_of[p][2] + 0.006])) for x, p in zip(xs, PARTS)])
        w.close()
        save_json(out / "lineup.json", {"parts": PARTS, "anchors": anchors})
        for p in PARTS:
            normals[p][0].set_visible(False)
        log("lineup")

    # -- one still per product (report / slides) -----------------------------------------
    if want("thumbs"):
        (out / "thumbs").mkdir(exist_ok=True)
        for p in PARTS:
            pp, T0 = normals[p]
            pp.set_pose(T0)
            pp.set_visible(True)
            h = 2 * focus_of[p][2]
            rig.place_overview(np.array([0.055, -0.085, h + 0.06]), np.array([0.0, 0.0, h * 0.45]))
            render_fresh(2, rt_subframes=16)
            imageio.imwrite(out / "thumbs" / f"{p}.png", grab())
            pp.set_visible(False)
        log("thumbs")

    # -- one defect of each type (T-LESS 29) --------------------------------------------
    hero_pid = "tless_29"
    picks = {}
    for t in TYPES:
        cands = [r for r in mans[hero_pid]["instances"] if r["role"] == "report" and r["type"] == t]
        tier = {"L": 0, "M": 1, "S": 2}
        picks[t] = sorted(cands, key=lambda r: (float(np.asarray((r["defect"] or {}).get("normal", [0, 0, 1]))[2]) < 0.8,
                                                tier.get(r.get("tier"), 3)))[0]
    if any(want(f"defect_{t}") for t in TYPES):
        hero = PartPrim(stage, scene.frame, root_path="/World/Hero")
        meta = {}
        for t in TYPES:
            if not want(f"defect_{t}"):
                continue
            r = picks[t]
            hero.load(LabeledMesh.load(str(resolve_data_path(r["mesh"]))), key=r["inst_id"])
            T0 = np.asarray(r["T_world_part"], dtype=float)
            hero.set_pose(T0)
            hero.set_visible(True)
            d = r["defect"] or {}
            if "center_m" in d:
                c = T0[:3, :3] @ np.asarray(d["center_m"]) + T0[:3, 3]
                nrm = T0[:3, :3] @ np.asarray(d["normal"])
                dist = 0.055
            else:                                   # warp bends the whole part
                c, nrm, dist = focus_of[hero_pid].copy(), np.array([0.0, -0.55, 0.85]), 0.085
            nrm = nrm / np.linalg.norm(nrm)
            t1 = np.cross(nrm, [0.0, 0.0, 1.0])
            t1 = t1 / np.linalg.norm(t1) if np.linalg.norm(t1) > 1e-6 else np.array([1.0, 0.0, 0.0])
            t2 = np.cross(nrm, t1)
            n = int(2.2 * FPS)
            w = writer(f"defect_{t}")
            for f in range(n):
                phi = np.radians(-40 + 80 * smooth(f / (n - 1)))
                v = nrm + 0.75 * (np.cos(phi) * t1 + np.sin(phi) * t2)
                v[2] = max(v[2], 0.35)
                rig.place_overview(c + dist * v / np.linalg.norm(v), c)
                w.append_data(grab())
            w.close()
            meta[t] = {"inst_id": r["inst_id"], "tier": r.get("tier"), "defect": d}
        hero.set_visible(False)
        save_json(out / "defects.json", meta)
        log("defect close-ups")

    # -- the 96 candidate camera positions -----------------------------------------------
    if want("slots"):
        Ts = [grid.T_world_cam(k, focus_of["tless_13"]) for k in range(grid.n_slots)]
        cols = [(0.16, 0.47, 0.84) if grid.slot(k).distance_m < 0.3 else (0.92, 0.41, 0.20) for k in range(grid.n_slots)]
        prims = frusta("Slots", Ts, 0.02, cols)
        show(prims, True)
        pp, T0 = normals["tless_13"]
        pp.set_pose(T0)
        pp.set_visible(True)
        n = 7 * FPS
        w = writer("slots")
        for f in range(n):
            az = np.radians(-70 + 110 * smooth(f / (n - 1)))
            el = np.radians(38 - 10 * smooth(f / (n - 1)))
            eye = focus + 1.25 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            rig.place_overview(eye, focus + np.array([0.0, 0.0, 0.12]))
            w.append_data(grab())
        w.close()
        show(prims, False)
        pp.set_visible(False)
        log("slots")

    # -- the rotation stage: 90 degrees three times --------------------------------------
    if want("turntable"):
        show(turntable, True)
        pp, T0 = normals["tless_13"]
        pp.set_visible(True)
        seq = []                                  # (seconds, angle start, angle end)
        for k in range(3):
            seq += [(0.6, 90.0 * k, 90.0 * k), (0.9, 90.0 * k, 90.0 * (k + 1))]
        seq += [(0.8, 270.0, 270.0)]
        w = writer("turntable")
        flashes, fr = [], 0
        eye = focus + np.array([0.20, -0.26, 0.20])
        for sec, a0, a1 in seq:
            n = int(round(sec * FPS))
            if a0 == a1:
                flashes.append(fr + n // 2)
            for f in range(n):
                ang = a0 + (a1 - a0) * smooth(f / max(n - 1, 1))
                pp.set_pose(rotation_stage_pose(T0, ang))
                rig.place_overview(eye, focus + np.array([0.0, 0.0, 0.01]))
                w.append_data(grab())
                fr += 1
        w.close()
        save_json(out / "turntable.json", {"flash_frames": flashes})
        pp.set_visible(False)
        show(turntable, False)
        log("turntable")

    # -- per product: its recommended cell ------------------------------------------------
    rig_colors = [(0.16, 0.47, 0.84), (0.11, 0.69, 0.48), (0.93, 0.63, 0.0)]
    for pid in PARTS:
        name = f"part_{pid}"
        if not want(name):
            continue
        cfg = best_record(pid)["config"]
        slots, rot, v = cfg["slots"], cfg["rotation"], SPEEDS[cfg["speed_level"]]
        cams_T = [grid.T_world_cam(k, focus_of[pid]) for k in slots]
        markers = frusta(f"Rig_{pid}", cams_T, 0.035, rig_colors[: len(slots)])
        show(markers, True)
        demo_id = shows[pid]["inst_id"]
        flow = [normals[pid], loaded(pid, demo_id, f"/World/D/{pid}"), loaded(pid, normal_id(pid, 1), f"/World/N2/{pid}")]
        stages = STAGES[rot]
        if rot != "none":
            show(turntable, True)
        # timeline of the demo part (index 1): approach, stop + capture per stage, leave
        t_app = 2.2
        t_stop = 0.0 if rot == "none" else 0.45 * len(stages) + T_ROT[rot]
        t_end = t_app + t_stop + (0.9 if rot == "none" else 0.4)
        n = int(round((t_end + 1.4) * FPS))
        az = np.radians(-58.0)
        el = np.radians(46.0)
        view_dir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

        def eye_target(t: float, x_demo: float) -> tuple:
            """Wide while the part approaches, close while it is captured, wide again after."""
            u = smooth((t - (t_app - 1.0)) / 0.9) * (1.0 - smooth((t - t_end) / 1.0))
            d = 0.95 + (0.40 - 0.95) * u
            tgt = focus + np.array([float(np.clip(x_demo, -0.12, 0.12)) * u, 0.0, 0.05 - 0.035 * u])
            return tgt + d * view_dir, tgt

        caps, w = [], writer(name)
        for f in range(n):
            t = f / FPS
            if t < t_app:
                x1, ang = -v * (t_app - t), 0.0
            elif t < t_app + t_stop:
                x1 = 0.0
                tau = t - t_app
                k = min(int(tau // (0.45 + (T_ROT[rot] / max(len(stages) - 1, 1)))), len(stages) - 1)
                seg = 0.45 + T_ROT[rot] / max(len(stages) - 1, 1)
                within = tau - k * seg
                ang = stages[k] if within < 0.45 or k == len(stages) - 1 else \
                    stages[k] + (stages[k + 1] - stages[k]) * smooth((within - 0.45) / (seg - 0.45))
            else:
                x1, ang = v * (t - t_app - t_stop), stages[-1]
            for j, (pp, T0) in enumerate(flow):
                x = x1 + PITCH * (j - 1)
                pp.set_visible(-0.75 < x < 1.15)
                T = make_T(None, [x, 0.0, 0.0]) @ T0
                pp.set_pose(rotation_stage_pose(T, ang) if j == 1 else T)
            rig.place_overview(*eye_target(t, x1))
            w.append_data(grab())
        w.close()
        # capture frames (for the flash): rotation configs capture at each stop, inline
        # configs as the part crosses the inspection point
        if rot == "none":
            caps = [int(round(t_app * FPS))]
        else:
            seg = 0.45 + T_ROT[rot] / max(len(stages) - 1, 1)
            caps = [int(round((t_app + k * seg + 0.2) * FPS)) for k in range(len(stages))]
        save_json(out / f"{name}.json", {"config": cfg, "speed_mps": v, "capture_frames": caps, "inst_id": demo_id,
                                          "n_frames": n})
        # what the rig records for the demo part at each rotation stage
        show(markers, False)
        show(turntable, False)
        for j, (pp, _) in enumerate(flow):
            pp.set_visible(j == 1)
        cams = rig.place(cams_T)
        rig.enable_rgb(cams)
        cdir = out / f"capture_{pid}"
        cdir.mkdir(exist_ok=True)
        views = []
        for si, ang in enumerate(stages):
            flow[1][0].set_pose(rotation_stage_pose(flow[1][1], ang))
            render_fresh(3, rt_subframes=16)
            for i, (c, k) in enumerate(zip(cams, slots)):
                imageio.imwrite(cdir / f"s{si}_cam{i}_rgb.png", np.asarray(c.get_data("rgb"))[:, :, :3].astype(np.uint8))
                np.save(cdir / f"s{si}_cam{i}_depth.npy", np.asarray(c.get_data("distance_to_image_plane"), np.float32))
                sem = c.get_data("semantic_segmentation")
                if isinstance(sem, dict) and "data" in sem:
                    np.save(cdir / f"s{si}_cam{i}_sem.npy", np.asarray(sem["data"]))
                    save_json(cdir / f"s{si}_cam{i}_sem_labels.json", sem.get("info", {}).get("idToLabels", {}))
                s = grid.slot(k)
                views.append({"stage": si, "angle_deg": ang, "cam": i, "slot": k, "azimuth_deg": s.azimuth_deg,
                              "elevation_deg": s.elevation_deg, "distance_m": s.distance_m})
        save_json(cdir / "capture.json", {"part": pid, "inst_id": demo_id, "config": cfg, "views": views})
        rig.park_all()
        for pp, _ in flow:
            pp.set_visible(False)
        log(name)

    isco_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
