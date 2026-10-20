"""Physics settling on the running belt (ADR-0006): where does a part come to rest?

A part (rest-frame ``LabeledMesh``) is dropped from ``drop_height_m`` (lowest point
above the belt) onto the belt running at speed ``v`` with a random lateral offset
U(-lateral_range, lateral_range) and yaw N(0, yaw_sigma); the simulation runs until the
part moves with the belt (|v_lin - (v,0,0)| < 2 mm/s and |omega| < 0.03 rad/s for
``rest_steps`` consecutive steps, or its pose relative to the belt changes < 0.3 mm /
0.25 deg over three 1/12 s windows) or ``settle_time_s`` elapses. The final pose is
expressed relative to the nominal rest pose of its drop lane (``T_world_part = I``
shifted to the lane centre, along-belt position discarded):

* ``dz`` - height of the rest-frame origin above the belt top (m)
* ``droll``, ``dpitch``, ``dyaw`` - intrinsic Z-Y-X Euler angles (deg) of the part
  rotation, i.e. ``R = Rz(dyaw) Ry(dpitch) Rx(droll)`` = ``transforms.pose_from_params``
  (roll/pitch about the part's own yawed axes)
* ``dy`` - lateral offset from the lane centre (m, includes the initial offset)
* ``tilt_deg`` - angle between the part's +Z and world +Z; ``flipped = tilt > 20 deg``
  (a different face down; yaw is excluded because guided feeding fixes yaw separately)

Several drops run in parallel in separate lanes across the belt (independent bodies,
12 cm apart) to amortise physics stepping. Collision = convex hull (exact for resting
on a plane), mass 0.02 kg (PoC) with solid-body inertia, friction 0.8 / 0.7 on both
part and belt (PoC).
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.spatial.transform import Rotation

from ..geometry.mesh_io import LabeledMesh
from ..geometry.transforms import R_from_quat_wxyz, make_T, quat_wxyz_from_R, rot_x, rot_y, rot_z

LANES_Y = (-0.30, -0.18, -0.06, 0.06, 0.18, 0.30)
FLIP_TILT_DEG = 20.0
REST_LIN_TOL = 2.0e-3
REST_ANG_TOL = 0.03
#: pose-stability rest test (residual PhysX micro-rocking of faceted round parts keeps
#: |omega| above REST_ANG_TOL although the pose no longer changes)
POSE_CHECK_STEPS = 20
POSE_STABLE_CHECKS = 3
POSE_TOL_M = 0.3e-3
POSE_TOL_DEG = 0.25
KEYS = ("dz", "droll", "dpitch", "dyaw", "dy")


def pose_relative_to_lane(T_world: np.ndarray, lane_y: float) -> dict:
    T_world = np.asarray(T_world, dtype=float)
    R = T_world[:3, :3]
    yaw, pitch, roll = Rotation.from_matrix(R).as_euler("ZYX", degrees=True)
    tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))
    rot = float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())))
    return {"dz": float(T_world[2, 3]), "droll": float(roll), "dpitch": float(pitch), "dyaw": float(yaw),
            "dy": float(T_world[1, 3] - lane_y), "x_world": float(T_world[0, 3]), "tilt_deg": tilt,
            "rot_from_nominal_deg": rot, "flipped": bool(tilt > FLIP_TILT_DEG)}


def summarize(samples: list[dict]) -> dict:
    out: dict = {"n": len(samples)}
    if not samples:
        return out
    flipped = np.array([s["flipped"] for s in samples], bool)
    settled = np.array([s["settled"] for s in samples], bool)
    out["flipped_fraction"] = float(flipped.mean())
    out["settled_fraction"] = float(settled.mean())
    out["fell_off"] = int(sum(bool(s.get("fell_off")) for s in samples))
    out["t_settle_mean_s"] = float(np.mean([s["t_settle_s"] for s in samples]))
    for subset, mask in (("all", np.ones(len(samples), bool)), ("not_flipped", ~flipped)):
        d = {}
        for k in KEYS + ("tilt_deg",):
            v = np.array([s[k] for s in samples], float)[mask]
            d[k] = {"mean": float(v.mean()) if len(v) else None, "std": float(v.std(ddof=1)) if len(v) > 1 else None,
                    "min": float(v.min()) if len(v) else None, "max": float(v.max()) if len(v) else None}
        out[subset] = d
    return out


def settle_part(scene, mesh: LabeledMesh, speeds_mps=(0.1, 0.3, 0.6), n_drops: int = 100, drop_height_m: float = 0.02,
                lateral_range_m: float = 0.005, yaw_sigma_deg: float = 3.0, settle_time_s: float = 1.5, seed: int = 0,
                physics_dt: float = 1.0 / 240.0, lanes_y=LANES_Y, rest_steps: int = 12, min_time_s: float = 0.25,
                mass: float = 0.02, tilt_sigma_deg: float = 0.0, ang_vel_sigma_dps: float = 0.0,
                heartbeat=None, log=None, part_id: str = "") -> dict:
    """Drop ``n_drops`` copies per speed and return per-speed raw samples + summary.

    ``tilt_sigma_deg`` / ``ang_vel_sigma_dps`` (default 0 = the ``SettlePayload``
    model) add an initial roll/pitch ~ N(0, tilt) and angular velocity ~ N(0, w) per axis
    (feeder-chute disturbance, sensitivity runs); the part is lifted so its lowest point
    is still ``drop_height_m`` above the belt."""
    from isaacsim.core.prims import RigidPrim

    from .part_prim import add_rigid_part

    world, stage, frame = scene.world, scene.stage, scene.frame
    t_job = time.perf_counter()
    root = "/World/Settle"
    if stage.GetPrimAtPath(root).IsValid():
        stage.RemovePrim(root)
    n_lanes = len(lanes_y)
    phys_meta = None
    import trimesh

    hull_v = np.asarray(trimesh.Trimesh(mesh.vertices, mesh.faces, process=False).convex_hull.vertices)
    for i in range(n_lanes):
        phys_meta = add_rigid_part(stage, f"{root}/lane_{i:02d}", mesh, mass=mass)
    # belt x range in ISCO world: start the drops 0.3 m after the belt entry
    belt = np.asarray(scene.info["belt_aabb"], float)
    x_drop = float(belt[0] + 0.3 - frame.origin_stage[0])
    # belt recirculation: a lane that reaches x_wrap is moved back by wrap_dx with its
    # pose (except x) and velocity unchanged - the belt is translation invariant along x,
    # so arbitrarily long settle times fit on the 2 m belt
    x_wrap = float(belt[3] - frame.origin_stage[0] - 0.25)
    wrap_dx = x_wrap - x_drop
    if wrap_dx < 0.3:
        raise ValueError("belt too short for settling")

    old_dt, old_rdt = world.get_physics_dt(), world.get_rendering_dt()
    old_speed = scene.belt_speed()
    world.set_simulation_dt(physics_dt=physics_dt, rendering_dt=old_rdt)
    world.reset()
    view = RigidPrim(f"{root}/lane_.*", name=f"isco_settle_{int(time.time() * 1e3)}", reset_xform_properties=False)
    view.initialize()
    rng = np.random.default_rng(seed)
    max_steps = int(math.ceil(settle_time_s / physics_dt))
    min_steps = int(math.ceil(min_time_s / physics_dt))
    results: dict = {}
    n_steps_total = 0
    try:
        for speed in speeds_mps:
            scene.set_belt_speed(speed)
            samples: list[dict] = []
            n_rounds = int(math.ceil(n_drops / n_lanes))
            t_speed = time.perf_counter()
            for r in range(n_rounds):
                yaw0 = rng.normal(0.0, yaw_sigma_deg, n_lanes)
                dy0 = rng.uniform(-lateral_range_m, lateral_range_m, n_lanes)
                tilt0 = rng.normal(0.0, tilt_sigma_deg, (n_lanes, 2)) if tilt_sigma_deg > 0 else np.zeros((n_lanes, 2))
                w0 = (np.radians(rng.normal(0.0, ang_vel_sigma_dps, (n_lanes, 3))) if ang_vel_sigma_dps > 0
                      else np.zeros((n_lanes, 3)))
                pos, quat = [], []
                for i in range(n_lanes):
                    R0 = rot_z(yaw0[i]) @ rot_y(tilt0[i, 1]) @ rot_x(tilt0[i, 0])
                    z0 = drop_height_m - float((hull_v @ R0.T)[:, 2].min())
                    T0 = make_T(R0, [x_drop, lanes_y[i] + dy0[i], z0])
                    Ts = frame.to_stage_T(T0)
                    pos.append(Ts[:3, 3])
                    quat.append(quat_wxyz_from_R(Ts[:3, :3]))
                view.set_world_poses(positions=np.array(pos), orientations=np.array(quat))
                view.set_velocities(np.concatenate([np.zeros((n_lanes, 3)), w0], axis=1))
                rest_count = np.zeros(n_lanes, int)
                t_rest = np.full(n_lanes, np.nan)
                steps = 0
                n_wraps = 0
                prev = None
                stable_count = np.zeros(n_lanes, int)
                pose_rest = np.zeros(n_lanes, bool)
                for steps in range(1, max_steps + 1):
                    world.step(render=False)
                    if steps % POSE_CHECK_STEPS == 0:
                        Pw, Qw = view.get_world_poses()
                        Pw, Qw = np.asarray(Pw, float), np.asarray(Qw, float)
                        if prev is not None:   # pose-stability criterion (belt motion removed)
                            dp = Pw - prev[0] - np.array([speed * POSE_CHECK_STEPS * physics_dt, 0.0, 0.0])
                            dq = np.abs(np.einsum("ij,ij->i", Qw, prev[1]))
                            drot = np.degrees(2.0 * np.arccos(np.clip(dq, -1.0, 1.0)))
                            ok = (np.linalg.norm(dp, axis=1) < POSE_TOL_M) & (drot < POSE_TOL_DEG) & (steps >= min_steps)
                            stable_count = np.where(ok, stable_count + 1, 0)
                            pose_rest = stable_count >= POSE_STABLE_CHECKS
                        far = frame.to_world_pts(Pw)[:, 0] > x_wrap
                        if far.any():
                            v_keep = np.asarray(view.get_velocities(), float)
                            Pw[far, 0] -= wrap_dx
                            view.set_world_poses(positions=Pw, orientations=Qw)
                            view.set_velocities(v_keep)
                            n_wraps += int(far.sum())
                        prev = (Pw, Qw)
                    vel = np.asarray(view.get_velocities(), float)
                    lin = np.linalg.norm(vel[:, :3] - np.array([speed, 0.0, 0.0]), axis=1)
                    ang = np.linalg.norm(vel[:, 3:], axis=1)
                    at_rest = (lin < REST_LIN_TOL) & (ang < REST_ANG_TOL) & (steps >= min_steps)
                    rest_count = np.where(at_rest, rest_count + 1, 0)
                    vel_rest = rest_count >= rest_steps
                    newly = (vel_rest | pose_rest) & np.isnan(t_rest)
                    t_rest[newly] = np.where(vel_rest[newly], (steps - rest_steps + 1) * physics_dt,
                                             (steps - POSE_CHECK_STEPS * POSE_STABLE_CHECKS) * physics_dt)
                    if np.all(~np.isnan(t_rest)):
                        break
                n_steps_total += steps
                vel = np.asarray(view.get_velocities(), float)
                v_rel = np.linalg.norm(vel[:, :3] - np.array([speed, 0.0, 0.0]), axis=1)
                w_fin = np.linalg.norm(vel[:, 3:], axis=1)
                P, Q = view.get_world_poses()
                P, Q = np.asarray(P, float), np.asarray(Q, float)
                for i in range(n_lanes):
                    if len(samples) >= n_drops:
                        break
                    T_stage = make_T(R_from_quat_wxyz(Q[i]), P[i])
                    T_world = frame.to_world_T(T_stage)
                    s = pose_relative_to_lane(T_world, lanes_y[i])
                    s.update(drop=len(samples), lane=i, yaw0_deg=float(yaw0[i]), dy0=float(dy0[i]),
                             roll0_deg=float(tilt0[i, 0]), pitch0_deg=float(tilt0[i, 1]),
                             dyaw_drift=float(s["dyaw"] - yaw0[i]), dy_drift=float(s["dy"] - dy0[i]),
                             settled=bool(not np.isnan(t_rest[i])),
                             t_settle_s=float(t_rest[i]) if not np.isnan(t_rest[i]) else float(steps * physics_dt),
                             fell_off=bool(T_world[2, 3] < -0.05), v_rel_final_mps=float(v_rel[i]),
                             w_final_rps=float(w_fin[i]), belt_wraps_in_round=n_wraps,
                             T_world_part=T_world.tolist())
                    samples.append(s)
                if heartbeat:
                    heartbeat()
            results[f"{speed:g}"] = {"speed_mps": float(speed), "samples": samples, "summary": summarize(samples),
                                     "wall_s": round(time.perf_counter() - t_speed, 2)}
            if log:
                sm = results[f"{speed:g}"]["summary"]
                log(f"[settle] {part_id} v={speed:g}: n={sm['n']} flipped={sm['flipped_fraction']:.2f} "
                    f"settled={sm['settled_fraction']:.2f} dz={sm['all']['dz']['mean'] * 1e3:.3f}+-"
                    f"{(sm['all']['dz']['std'] or 0) * 1e3:.3f} mm dyaw={sm['all']['dyaw']['mean']:.2f}+-"
                    f"{sm['all']['dyaw']['std'] or 0:.2f} deg ({results[f'{speed:g}']['wall_s']} s)")
    finally:
        scene.set_belt_speed(old_speed)
        world.set_simulation_dt(physics_dt=old_dt, rendering_dt=old_rdt)
        if stage.GetPrimAtPath(root).IsValid():
            stage.RemovePrim(root)
        world.reset()
    return {
        "part_id": part_id,
        "convention": {
            "frame": "ISCO world (belt top z=0, +X belt travel); pose relative to the nominal rest pose "
                     "(T_world_part = I) at the drop-lane centre, along-belt x discarded",
            "angles": "intrinsic ZYX: R = Rz(dyaw) Ry(dpitch) Rx(droll) (transforms.pose_from_params), degrees",
            "units": "dz, dy in metres; angles in degrees",
            "flipped": f"tilt (angle of part +Z to world +Z) > {FLIP_TILT_DEG} deg",
            "rest": (f"(|v - v_belt| < {REST_LIN_TOL} m/s and |omega| < {REST_ANG_TOL} rad/s for {rest_steps} steps) or "
                     f"(pose change, belt motion removed, < {POSE_TOL_M * 1e3} mm and < {POSE_TOL_DEG} deg over "
                     f"{POSE_STABLE_CHECKS} consecutive {POSE_CHECK_STEPS}-step windows)"),
        },
        "params": {"speeds_mps": list(map(float, speeds_mps)), "n_drops": n_drops, "drop_height_m": drop_height_m,
                   "lateral_range_m": lateral_range_m, "yaw_sigma_deg": yaw_sigma_deg,
                   "settle_time_s": settle_time_s, "seed": seed, "physics_dt": physics_dt, "lanes_y": list(lanes_y),
                   "tilt_sigma_deg": tilt_sigma_deg, "ang_vel_sigma_dps": ang_vel_sigma_dps,
                   "belt_recirculation": {"x_wrap_world_m": x_wrap, "wrap_dx_m": wrap_dx},
                   "x_drop_world_m": x_drop},
        "physics": phys_meta,
        "speeds": results,
        "n_physics_steps": n_steps_total,
        "wall_s": round(time.perf_counter() - t_job, 2),
    }
