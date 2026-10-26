"""Scene construction, ISCO<->stage frame, camera intrinsics/extrinsics, look-at equivalence."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.isaac


def test_scene_build_and_measure(scene):
    info = scene.info
    whole, belt = np.asarray(info["whole_aabb"]), np.asarray(info["belt_aabb"])
    # official ConveyorBelt_A06: 2 m long along +X, ~0.9 m belt, belt top well above the ground
    assert whole[3] - whole[0] == pytest.approx(2.0, abs=0.05)
    assert belt[4] - belt[1] == pytest.approx(0.9, abs=0.05)
    assert info["belt_top_z"] == pytest.approx(belt[5])
    assert 0.5 < info["belt_top_z"] < whole[5]
    o = scene.frame.origin_stage
    assert o[0] == pytest.approx(whole[0] + 0.4 * (whole[3] - whole[0]))
    assert o[1] == pytest.approx(0.5 * (belt[1] + belt[4]))
    assert o[2] == pytest.approx(info["belt_top_z"])
    T = np.eye(4)
    T[:3, 3] = [0.01, -0.02, 0.003]
    assert np.allclose(scene.frame.to_world_T(scene.frame.to_stage_T(T)), T)
    m = scene.meta()
    for k in ("belt_top_z_stage", "belt_width_m", "conveyor_aabb_world", "isco_origin_stage", "focus_x_rule"):
        assert k in m
    # belt top is z = 0 in ISCO world
    assert scene.meta()["belt_aabb_world"][5] == pytest.approx(0.0, abs=1e-9)


def test_intrinsics_match_pinhole(rig):
    K_model = rig.pinhole.K
    for i in range(rig.n_cams):
        K_isaac = rig.isaac_K(i)
        rel = np.abs(K_isaac - K_model) / np.maximum(np.abs(K_model), 1.0)
        assert rel.max() < 1e-6, (i, K_isaac, K_model)


def test_extrinsics_match_T_world_cam(rig):
    from isco.geometry.camera_model import SlotGrid

    grid = SlotGrid()
    focus = np.array([0.0, 0.0, 0.005])
    slots = [0, 13, 26, 39, 50, 95]
    poses = [grid.T_world_cam(k, focus) for k in slots]
    rig.place(poses)
    for i, T in enumerate(poses):
        T_isaac = rig.isaac_T_world_cam(i)
        assert np.allclose(T_isaac[:3, :3], T[:3, :3], atol=1e-6)
        assert np.allclose(T_isaac[:3, 3], T[:3, 3], atol=1e-6)
        # get_world_pose(camera_axes="ros") is the same OpenCV-axes pose
        pos, q = rig.cams[i].camera.get_world_pose(camera_axes="ros")
        from isco.geometry.transforms import R_from_quat_wxyz

        assert np.allclose(R_from_quat_wxyz(np.asarray(q)), T[:3, :3], atol=1e-5)
        assert np.allclose(np.asarray(pos) - rig.frame.origin_stage, T[:3, 3], atol=1e-6)
    rig.park_all()


def test_look_at_gf_free_equals_poc(sim_app):
    from isco.geometry.transforms import look_at_quat_wxyz_usd
    from isco.sim.rig import look_at_quat_wxyz_gf

    rng = np.random.default_rng(0)
    for _ in range(50):
        target = rng.normal(size=3)
        d = rng.normal(size=3)
        d[2] = abs(d[2]) * 0.5 + 0.05      # avoid looking exactly along +-Z (PoC version is singular there)
        eye = target + d
        q1 = look_at_quat_wxyz_gf(eye, target)
        q2 = look_at_quat_wxyz_usd(eye, target)
        assert min(np.abs(q1 - q2).max(), np.abs(q1 + q2).max()) < 1e-6, (q1, q2)
