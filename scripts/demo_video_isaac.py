"""Isaac Sim footage for the demo video (``scripts/make_demo_video.py`` adds captions).

Shots, written to ``runs/demo_video/isaac/`` (1280x720, 30 fps):

* ``line.mp4``     - the conveyor cell: parts of the variant set (defect faces red) flow at
  belt speed through the inspection point, the recommended rig drawn as camera frusta;
  the view moves from the whole line to the inspection point;
* ``inspect.mp4``  - close view of one defective part passing the cameras;
* ``capture/``     - what the rig cameras record at the trigger: RGB, depth
  (``distance_to_image_plane``) and the semantic label image;
* ``defect_<type>.mp4`` - orbit around one defect of each type.

    python scripts/demo_video_isaac.py [--part fusion_51876] [--run runs/demo_fusion_51876]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FPS = 30
TYPES = ("dent", "bulge", "hole", "cut", "crack", "warp")


def smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def pick_instances(man: dict) -> dict:
    """One held-out defect instance per type, preferring large defects on upward faces."""
    out = {}
    for t in TYPES:
        cands = [r for r in man["instances"] if r["role"] == "report" and r["type"] == t]
        tier = {"L": 0, "M": 1, "S": 2}

        def key(r):
            d = r["defect"] or {}
            up = float(np.asarray(d.get("normal", [0, 0, 1]))[2])
            return (up < 0.9, tier.get(r.get("tier"), 3), -float(d.get("depth_m") or 0.0))
        out[t] = sorted(cands, key=key)[0]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="fusion_51876")
    ap.add_argument("--set-id", default="mvp")
    ap.add_argument("--run", default=str(REPO / "runs" / "demo_fusion_51876"))
    ap.add_argument("--out", default=str(REPO / "runs" / "demo_video" / "isaac"))
    ap.add_argument("--speed", type=float, default=0.3, help="belt speed shown in the flow shots (m/s)")
    ap.add_argument("--subframes", type=int, default=4)
    a = ap.parse_args()

    from isco.sim import app as isco_app

    isco_app.launch(headless=True)
    import imageio
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
    from isco.sim.worker import frustum_mesh
    from isco.utils.io import resolve_data_path, write_json

    out = Path(a.out)
    (out / "capture").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rec = json.loads((Path(a.run) / "report" / "recommended.json").read_text())
    slots = rec["config"]["slots"]
    spec = load_part(a.part)
    focus = np.asarray(spec.focus_m, dtype=float)
    man = load_manifest(a.part, a.set_id)
    by_id = {r["inst_id"]: r for r in man["instances"]}
    picks = pick_instances(man)

    scene = IscoScene.build(speed=0.0)
    stage = scene.stage
    rig = CameraRig(scene.frame, SensorModel(), n_cams=len(slots), overview=True)
    ov = rig.overview
    set_pinhole_intrinsics(ov.camera, ov.pinhole, 24.0, 0.005, 50.0)   # close-ups come within 5 cm
    grid = SlotGrid()
    cams_T = [grid.T_world_cam(k, focus) for k in slots]

    # recommended rig as camera frusta (hidden while the rig cameras record)
    markers = []
    for i, T in enumerate(cams_T):
        V, F = frustum_mesh(T, size=0.035)
        p = f"/World/RigMarkers/cam_{i}"
        define_mesh(stage, p, scene.frame.to_stage_pts(V), F, (0.10, 0.40, 0.90))
        markers.append(stage.GetPrimAtPath(p))

    def show_markers(on: bool) -> None:
        for m in markers:
            UsdGeom.Imageable(m).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if on else UsdGeom.Tokens.invisible)

    # the flow: a normal part and five defective ones, one pitch (0.33 m) apart
    flow_ids = [next(r["inst_id"] for r in man["instances"] if r["role"] == "val"),
                picks["hole"]["inst_id"], picks["dent"]["inst_id"], picks["cut"]["inst_id"],
                picks["bulge"]["inst_id"], picks["crack"]["inst_id"]]
    flow = []
    for j, iid in enumerate(flow_ids):
        pp = PartPrim(stage, scene.frame, root_path=f"/World/Flow/P{j}")
        pp.load(LabeledMesh.load(str(resolve_data_path(by_id[iid]["mesh"]))))
        flow.append((pp, np.asarray(by_id[iid]["T_world_part"], dtype=float)))
    x_lo, x_hi = -0.78, 1.15
    span = 0.33 * len(flow)

    def place_flow(t: float, x_phase: float = 0.0, only: int | None = None) -> None:
        for j, (pp, T0) in enumerate(flow):
            x = (x_phase + 0.33 * j + a.speed * t - x_lo) % span + x_lo
            vis = x <= x_hi and (only is None or only == j)
            pp.set_visible(vis)
            pp.set_pose(make_T(None, [x, 0.0, 0.0]) @ T0)

    def grab() -> np.ndarray:
        render_fresh(1, rt_subframes=a.subframes)
        rgba = ov.get_data("rgb")
        return np.asarray(rgba)[:, :, :3].astype(np.uint8)

    def writer(name: str):
        return imageio.get_writer(str(out / f"{name}.mp4"), fps=FPS, codec="libx264", quality=9,
                                  macro_block_size=8)

    for _ in range(3):  # renderer warm-up (asset streaming, shader compilation)
        rig.place_overview(focus + np.array([0.6, -0.7, 0.5]), focus)
        render_fresh(2, rt_subframes=8)

    # -- shot 1: the line ----------------------------------------------------------------
    n = 8 * FPS
    eye0, tgt0 = np.array([1.75, -1.75, 1.15]), np.array([0.15, 0.0, 0.0])
    eye1, tgt1 = focus + np.array([0.42, -0.50, 0.33]), focus
    w = writer("line")
    for f in range(n):
        u = smooth(f / (n - 1))
        place_flow(f / FPS, x_phase=-0.33 * 1.5)
        rig.place_overview(eye0 + (eye1 - eye0) * u, tgt0 + (tgt1 - tgt0) * u)
        w.append_data(grab())
    w.close()
    print(f"[demo_video_isaac] line.mp4 {n} frames ({time.time() - t_start:.0f}s)", flush=True)

    # -- shot 2: one defective part passes the cameras; the rig records at x = 0 ---------
    j_star = 1                                            # the hole part
    t_cross = 1.6
    phase = -(0.33 * j_star) - a.speed * t_cross          # part j_star at x = 0 when t = t_cross
    n = int(3.2 * FPS)
    eye = focus + np.array([0.05, -0.60, 0.42])           # behind the cameras: all three and the part in view
    w = writer("inspect")
    for f in range(n):
        place_flow(f / FPS, x_phase=phase)
        rig.place_overview(eye, focus + np.array([0.0, 0.0, 0.06]))
        w.append_data(grab())
    w.close()
    place_flow(t_cross, x_phase=phase, only=j_star)
    show_markers(False)
    cams = rig.place(cams_T)
    rig.enable_rgb(cams)
    render_fresh(3, rt_subframes=16)
    cap = {"slots": slots, "inst_id": flow_ids[j_star], "views": []}
    for i, (c, k) in enumerate(zip(cams, slots)):
        rgb = np.asarray(c.get_data("rgb"))[:, :, :3].astype(np.uint8)
        depth = np.asarray(c.get_data("distance_to_image_plane"), dtype=np.float32)
        sem = c.get_data("semantic_segmentation")
        imageio.imwrite(out / "capture" / f"cam_{i}_rgb.png", rgb)
        np.save(out / "capture" / f"cam_{i}_depth.npy", depth)
        if isinstance(sem, dict) and "data" in sem:
            np.save(out / "capture" / f"cam_{i}_sem.npy", np.asarray(sem["data"]))
            write_json(out / "capture" / f"cam_{i}_sem_labels.json", sem.get("info", {}).get("idToLabels", {}))
        s = grid.slot(k)
        cap["views"].append({"cam": i, "slot": k, "azimuth_deg": s.azimuth_deg, "elevation_deg": s.elevation_deg,
                             "distance_m": s.distance_m})
    write_json(out / "capture" / "capture.json", cap)
    rig.park_all()
    show_markers(True)
    print(f"[demo_video_isaac] inspect.mp4 + capture ({time.time() - t_start:.0f}s)", flush=True)

    # -- shot 3: orbit around one defect of each type ------------------------------------
    for pp, _ in flow:
        pp.set_visible(False)
    show_markers(False)
    hero = PartPrim(stage, scene.frame, root_path="/World/Hero")
    meta = {}
    for t in TYPES:
        r = picks[t]
        hero.load(LabeledMesh.load(str(resolve_data_path(r["mesh"]))), key=r["inst_id"])
        T0 = np.asarray(r["T_world_part"], dtype=float)
        hero.set_pose(T0)
        d = r["defect"] or {}
        if "center_m" in d:
            c = T0[:3, :3] @ np.asarray(d["center_m"]) + T0[:3, 3]
            nrm = T0[:3, :3] @ np.asarray(d["normal"])
            dist = 0.055
        else:                                             # warp: the whole part bends
            c, nrm, dist = focus.copy(), np.array([0.0, -0.55, 0.85]), 0.085
        nrm = nrm / np.linalg.norm(nrm)
        t1 = np.cross(nrm, [0.0, 0.0, 1.0])
        if np.linalg.norm(t1) < 1e-6:
            t1 = np.array([1.0, 0.0, 0.0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(nrm, t1)
        n = int(2.4 * FPS)
        w = writer(f"defect_{t}")
        for f in range(n):
            phi = np.radians(-40 + 80 * smooth(f / (n - 1)))
            v = nrm + 0.75 * (np.cos(phi) * t1 + np.sin(phi) * t2)
            v[2] = max(v[2], 0.35)
            eye = c + dist * v / np.linalg.norm(v)
            rig.place_overview(eye, c)
            w.append_data(grab())
        w.close()
        meta[t] = {"inst_id": r["inst_id"], "tier": r.get("tier"), "defect": d}
    write_json(out / "defects.json", meta)
    print(f"[demo_video_isaac] defect close-ups done ({time.time() - t_start:.0f}s)", flush=True)
    isco_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
