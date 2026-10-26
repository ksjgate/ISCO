"""Capture correctness: own unprojection vs the pointcloud annotator, semantic labels,
verify_pose, freshness after a move and the rotation-stage pose."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.isaac

FOCUS = np.array([0.0, 0.0, 0.005])


def _slot_pose(k):
    from isco.geometry.camera_model import SlotGrid

    return SlotGrid().T_world_cam(k, FOCUS)


def test_unprojection_matches_pointcloud_annotator(rig, placed_box):
    from scipy.spatial import cKDTree

    from isco.sim.capture import capture_views, view_points_world

    T = _slot_pose(26)                          # az 60, el 60, d 0.25
    cam = rig.place([T])[0]
    cam.camera.add_pointcloud_to_frame(include_unlabelled=False)
    try:
        views, _, _ = capture_views([cam], warmup=2)
        vp = views[0]
        own = rig.frame.to_stage_pts(view_points_world(vp, cam.pinhole, T))
        pc = cam.get_data("pointcloud")
        ann = np.asarray(pc["data"], dtype=float).reshape(-1, 3)
    finally:
        cam.camera.remove_pointcloud_from_frame()
        rig.park_all()
    assert len(vp) > 10000
    assert abs(len(ann) - len(own)) <= 0.01 * len(own)
    d_own, _ = cKDTree(ann).query(own)
    d_ann, _ = cKDTree(own).query(ann)
    assert np.percentile(d_own, 99.9) < 1e-4, np.percentile(d_own, [50, 99, 99.9, 100])
    assert np.median(d_own) < 1e-5
    assert np.percentile(d_ann, 99.9) < 1e-4


def test_semantic_labels_part_and_defect(rig, placed_box, defect_box):
    from isco.geometry.transforms import apply_T, inv_T
    from isco.sim.capture import capture_views

    T = _slot_pose(87)                          # az 90, el 80 (near top-down), d 0.40
    cam = rig.place([T])[0]
    views, warns, _ = capture_views([cam])      # retries if the annotator graph was just rebuilt
    vp = views[0]
    rig.park_all()
    assert not warns
    labels = set(np.unique(vp.label).tolist())
    assert labels == {1, 2}, labels
    # the projected defect centre carries label 2, the projected part corner region label 1
    c = np.asarray(defect_box.meta["defect_center"])
    u, v, _ = cam.pinhole.project(apply_T(inv_T(T), c[None]))
    pix = int(np.floor(v[0])) * cam.pinhole.width + int(np.floor(u[0]))
    i = np.searchsorted(vp.pix, pix)
    assert vp.pix[i] == pix and vp.label[i] == 2
    n_def = int((vp.label == 2).sum())
    # defect patch area / top area ~ pi*4^2 / (30*20) = 8.4 %; tolerate tessellation + perspective
    frac = n_def / len(vp)
    assert 0.03 < frac < 0.15, frac
    assert np.all(np.isfinite(vp.cos)) and vp.cos.min() >= 0 and vp.cos.max() <= 1.0 + 1e-6


def test_verify_pose_detects_wrong_pose(rig, placed_box, defect_box_index):
    from isco.geometry.transforms import make_T
    from isco.sim.capture import capture_views

    cams = rig.place([_slot_pose(0), _slot_pose(50)])
    views, warns, _ = capture_views(cams, verify={"index": defect_box_index, "T_world_part": np.eye(4)})
    for vp in views:
        chk = vp.meta["verify"]
        assert chk["ok"] and chk["median_m"] < 1e-4 and chk["centroid_offset_m"] < 1e-4, chk
    assert not warns
    wrong = make_T(None, [0.003, 0.0, 0.0])
    views, warns, _ = capture_views(cams, verify={"index": defect_box_index, "T_world_part": wrong}, max_retries=1)
    rig.park_all()
    assert all(not vp.meta["verify"]["ok"] for vp in views), [vp.meta["verify"] for vp in views]
    assert len(warns) == 2
    # a 3 mm in-plane shift of a box seen from a low elevation is (mostly) caught by the silhouette test
    assert max(vp.meta["verify"]["outside_frac"] for vp in views) > 0.01


def test_fresh_frame_after_move(rig, placed_box, defect_box_index):
    """After moving the part, a single render_fresh() must already show the new pose."""
    from isco.geometry.transforms import make_T, rot_z
    from isco.sim.capture import capture_views

    cams = rig.place([_slot_pose(13), _slot_pose(42)])
    rng = np.random.default_rng(1)
    for _ in range(4):
        T = make_T(rot_z(rng.uniform(-30, 30)), [rng.uniform(-0.004, 0.004), rng.uniform(-0.004, 0.004), 0.0])
        placed_box.set_pose(T)
        views, warns, timing = capture_views(cams, warmup=1, max_retries=0,
                                             verify={"index": defect_box_index, "T_world_part": T})
        assert timing["renders"] == 1
        assert not warns, warns
        assert all(vp.meta["verify"]["ok"] for vp in views)
    placed_box.set_pose(np.eye(4))
    rig.park_all()


@pytest.mark.parametrize("dy", [0.0, 0.004])
def test_rotation_stage_equals_rotated_slot(rig, placed_box, dy):
    """Rotating the part by +90 deg about the turntable axis (vertical axis through the
    ISCO origin / focus point) renders the same depth as the unrotated part seen from
    SlotGrid.rotate(k, 90) - also when the part sits off-axis (lateral placement error)."""
    from isco.geometry.camera_model import SlotGrid
    from isco.geometry.transforms import make_T
    from isco.sim.capture import read_view, render_fresh
    from isco.sim.worker import rotation_stage_pose

    grid = SlotGrid()
    T0 = make_T(None, [0.0, dy, 0.0])
    T_rot = rotation_stage_pose(T0, 90.0)
    assert np.allclose(T_rot[:3, 3], [-dy, 0.0, 0.0]) and np.allclose(T_rot[:3, :3] @ [1, 0, 0], [0, 1, 0])
    k = grid.slot_id(1, 1, 0)                       # az 30, el 40, d 0.25
    k_rot = grid.rotate(k, 90.0)                    # az -60 == 300
    assert grid.slot(k_rot).azimuth_deg == 300.0
    cam = rig.place([grid.T_world_cam(k, FOCUS)])[0]
    placed_box.set_pose(T_rot)
    render_fresh(1)
    a = read_view(cam)
    cam = rig.place([grid.T_world_cam(k_rot, FOCUS)])[0]
    placed_box.set_pose(T0)
    render_fresh(1)
    b = read_view(cam)
    rig.park_all()
    common, ia, ib = np.intersect1d(a.pix, b.pix, return_indices=True)
    iou = len(common) / len(np.union1d(a.pix, b.pix))
    assert iou > 0.999, iou
    dd = np.abs(a.depth[ia].astype(float) - b.depth[ib].astype(float))
    assert np.percentile(dd, 99) < 1e-5 and dd.max() < 5e-5, (np.percentile(dd, 99), dd.max())
    assert np.mean(a.label[ia] == b.label[ib]) > 0.999


def test_view_pixels_match_embree(rig, placed_box, defect_box):
    """Quick E0.1 check on the defect box: Isaac and the embree renderer agree."""
    from isco.render.occluder import Occluder
    from isco.render.raycast_backend import RaycastRenderer
    from isco.sim.capture import capture_views

    slots = [0, 37, 74]
    poses = [_slot_pose(k) for k in slots]
    cams = rig.place(poses)
    views, _, _ = capture_views(cams)
    rig.park_all()
    rr = RaycastRenderer(rig.pinhole, Occluder.belt_plane())
    full = (0, 0, rig.pinhole.width, rig.pinhole.height)
    ref = rr.render(defect_box, np.eye(4), [(T, full) for T in poses])
    for a, b in zip(views, ref):
        common, ia, ib = np.intersect1d(a.pix, b.pix, return_indices=True)
        assert len(common) / len(np.union1d(a.pix, b.pix)) > 0.999
        dd = np.abs(a.depth[ia].astype(float) - b.depth[ib].astype(float))
        assert np.median(dd) < 2e-6 and dd.max() < 2e-5
        assert np.mean(a.label[ia] == b.label[ib]) > 0.999
        assert np.median(np.abs(a.cos[ia] - b.cos[ib])) < 1e-3


def test_mesh_swaps_keep_labels_consistent(rig, part, defect_box):
    """Regression: re-creating labelled prims made Isaac's semantic ids and idToLabels
    disagree (labels swapped for the 2nd mesh) and emptied meshes kept rendering stale
    geometry. PartPrim now updates persistent prims in place and only hides unused ones."""
    import trimesh

    from isco.geometry.mesh_io import LabeledMesh, focus_point
    from isco.render.occluder import Occluder
    from isco.render.raycast_backend import RaycastRenderer
    from isco.sim.capture import capture_views

    s = trimesh.creation.icosphere(subdivisions=5, radius=0.012)
    s.apply_translation([0.0, 0.0, 0.012])
    sphere = LabeledMesh(s.vertices, s.faces, (s.triangles_center[:, 2] > 0.021).astype(np.int32))
    plain = LabeledMesh(defect_box.vertices, defect_box.faces)            # no defect
    rr = RaycastRenderer(rig.pinhole, Occluder.belt_plane())
    full = (0, 0, rig.pinhole.width, rig.pinhole.height)
    from isco.geometry.camera_model import SlotGrid

    for i, (name, mesh) in enumerate([("box", defect_box), ("sphere", sphere), ("plain", plain), ("sphere", sphere),
                                      ("box", defect_box)]):
        part.load(mesh, key=f"swap_{i}_{name}")
        part.set_pose(np.eye(4))
        T = SlotGrid().T_world_cam(26, focus_point(mesh))
        views, warns, _ = capture_views(rig.place([T]))
        ref = rr.render(mesh, np.eye(4), [(T, full)])[0]
        a = views[0]
        common, ia, ib = np.intersect1d(a.pix, ref.pix, return_indices=True)
        assert len(common) / len(np.union1d(a.pix, ref.pix)) > 0.999, name
        assert np.mean(a.label[ia] == ref.label[ib]) > 0.999, (name, np.bincount(a.label), np.bincount(ref.label))
    rig.park_all()
