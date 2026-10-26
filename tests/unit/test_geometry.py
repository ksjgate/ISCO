import numpy as np
import pytest
import trimesh

from isco.geometry.camera_model import PinholeCamera, SlotGrid, camera_pose_spherical
from isco.geometry.mesh_io import LabeledMesh, focus_point, to_rest_frame
from isco.geometry.remesh import remesh_isotropic
from isco.geometry.transforms import (R_from_quat_wxyz, apply_T, inv_T, look_at_opencv, look_at_quat_wxyz_usd,
                                      quat_wxyz_from_R)


def test_intrinsics_from_hfov():
    cam = PinholeCamera(1024, 1024, 20.0)
    assert cam.fx == pytest.approx(512 / np.tan(np.radians(10)))
    assert cam.cx == 512 and cam.K[1, 2] == 512
    f, ha, va = cam.isaac_apertures(24.0)
    assert 2 * np.degrees(np.arctan(ha / 2 / f)) == pytest.approx(20.0)


def test_project_unproject_identity(rng):
    cam = PinholeCamera(640, 480, 30.0)
    pix = rng.integers(0, 640 * 480, 1000)
    z = rng.uniform(0.2, 0.5, 1000)
    pc = cam.unproject(pix, z)
    u, v, zz = cam.project(pc)
    uu, vv = cam.pix_to_uv(pix)
    assert np.allclose(u, uu + 0.5) and np.allclose(v, vv + 0.5) and np.allclose(zz, z)


def test_look_at_is_rotation_and_points_at_target():
    T = look_at_opencv(np.array([0.3, -0.1, 0.2]), np.zeros(3))
    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12) and np.linalg.det(R) == pytest.approx(1.0)
    fwd = R[:, 2]
    assert np.allclose(fwd, -T[:3, 3] / np.linalg.norm(T[:3, 3]))
    q = quat_wxyz_from_R(R)
    assert np.allclose(R_from_quat_wxyz(q), R, atol=1e-12)


def test_gf_free_look_at_matches_poc_construction():
    """The PoC built rows [x, y, z] with z = eye - target, x = up x z, y = z x x (USD axes)."""
    eye, tgt, up = np.array([0.5, 0.2, 0.4]), np.array([0.0, 0.0, 0.05]), np.array([0.0, 0.0, 1.0])
    z = eye - tgt
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q_poc = quat_wxyz_from_R(np.stack([x, y, z], axis=1))
    q = look_at_quat_wxyz_usd(eye, tgt)
    assert min(np.abs(q - q_poc).max(), np.abs(q + q_poc).max()) < 1e-9


def test_slot_grid_layout_and_rotation():
    g = SlotGrid()
    assert g.n_slots == 96
    s = g.slot(g.slot_id(3, 1, 0))
    assert (s.azimuth_deg, s.elevation_deg, s.distance_m) == (90.0, 40.0, 0.25)
    k = g.slot_id(5, 2, 1)
    assert g.slot(g.rotate(k, 90)).i_az == 2 and g.rotate(g.rotate(k, 180), 180) == k
    with pytest.raises(ValueError):
        g.rotate(k, 45)
    F = np.array([0, 0, 0.01])
    T = g.T_world_cam(k, F)
    assert np.linalg.norm(g.eye(k, F) - F) == pytest.approx(0.40)
    assert np.allclose(T, camera_pose_spherical(F, g.slot(k).azimuth_deg, g.slot(k).elevation_deg, 0.40))
    assert g.nearest_slot(91, 38, 0.3) == g.slot_id(3, 1, 0)


def test_labeled_mesh_roundtrip(tmp_path, box_mesh):
    box_mesh.face_label[:2] = 1
    box_mesh.label_names.append("dent:0")
    p = tmp_path / "m.npz"
    box_mesh.save(p)
    m = LabeledMesh.load(p)
    assert np.allclose(m.vertices, box_mesh.vertices, atol=1e-7) and m.n_defects == 1 and m.label_names[1] == "dent:0"


def test_rest_frame_lowest_com_and_normalisation():
    b = trimesh.creation.box(extents=[1.0, 4.0, 2.0])
    b.apply_transform(trimesh.transformations.rotation_matrix(0.7, [1, 1, 0]))
    rest, info = to_rest_frame(b, 0.04)
    ext = np.sort(rest.extents)
    assert ext[-1] == pytest.approx(0.04, rel=1e-6)
    assert rest.extents[2] == pytest.approx(0.01, rel=1e-6)          # lies on its largest face
    assert rest.bounds[0, 2] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(focus_point(rest)[:2], 0, atol=1e-12)
    assert rest.extents[0] >= rest.extents[1]                          # major axis along the belt


def test_remesh_preserves_shape_and_labels(box_mesh):
    top = box_mesh.vertices[box_mesh.faces].mean(axis=1)[:, 2] > 0.0099
    box_mesh.face_label = top.astype(np.int32)
    box_mesh.label_names.append("top:0")
    r = remesh_isotropic(box_mesh, 1e-3)
    assert len(r.faces) > 10 * len(box_mesh.faces)
    tm = r.to_trimesh()
    assert tm.is_watertight and tm.volume == pytest.approx(box_mesh.to_trimesh().volume, rel=1e-3)
    zc = r.vertices[r.faces].mean(axis=1)[:, 2]
    assert np.all(zc[r.face_label == 1] > 0.0099) and (r.face_label == 1).sum() > 0
