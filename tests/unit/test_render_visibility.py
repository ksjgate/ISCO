import numpy as np
import pytest
import trimesh

from isco.geometry.camera_model import PinholeCamera, SlotGrid
from isco.geometry.mesh_io import LabeledMesh, bounding_radius, focus_point
from isco.geometry.transforms import apply_T, look_at_opencv
from isco.render.occluder import Occluder
from isco.render.pixels import dequantise, quantise
from isco.render.raycast_backend import RaycastRenderer
from isco.visibility.raycast import slot_visibility


def test_raycast_unprojection_lies_on_surface(box_mesh):
    cam = PinholeCamera(256, 256, 20.0)
    g = SlotGrid()
    F = focus_point(box_mesh)
    views = []
    for k in (12, 40, 77):
        T = g.T_world_cam(k, F)
        views.append((T, cam.roi_for_sphere(T, F, bounding_radius(box_mesh, F))))
    vps = RaycastRenderer(cam, Occluder.belt_plane()).render(box_mesh, np.eye(4), views)
    tm = box_mesh.to_trimesh()
    for (T, _), vp in zip(views, vps):
        assert len(vp) > 500
        pw = apply_T(T, cam.unproject(vp.pix, vp.depth))
        assert np.abs(trimesh.proximity.signed_distance(tm, pw[::20])).max() < 1e-6
        assert np.all(vp.label == 1) and np.all((vp.cos > 0) & (vp.cos <= 1.0001))


def test_depth_quantisation_roundtrip():
    d = np.array([0.20, 0.25, 0.31], np.float32)
    q = quantise(d, 0.1)
    assert np.abs(dequantise(q, 0.1) - d).max() <= 6e-6


def test_visibility_occlusion_and_backfacing():
    sphere = trimesh.creation.icosphere(3, radius=0.005)
    sphere.apply_translation([0, 0, 0.005])
    wall = trimesh.creation.box(extents=[0.002, 0.05, 0.05])
    wall.apply_translation([0.02, 0, 0.02])
    scene_mesh = LabeledMesh.from_trimesh(trimesh.util.concatenate([sphere, wall]))
    cam = PinholeCamera(256, 256, 20.0)
    pts = np.array([[0.005, 0, 0.005], [-0.005, 0, 0.005]])     # +x side faces the wall
    nrm = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    T_front = look_at_opencv(np.array([0.25, 0, 0.005]), np.array([0, 0, 0.005]))
    T_back = look_at_opencv(np.array([-0.25, 0, 0.005]), np.array([0, 0, 0.005]))
    vis, cos, z = slot_visibility(scene_mesh, np.eye(4), pts, nrm, [T_front, T_back], cam, Occluder.belt_plane())
    assert not vis[0, 0]          # +x point hidden behind the wall
    assert not vis[0, 1]          # -x point back-facing for the +x camera
    assert vis[1, 1] and not vis[1, 0]
    assert z[1, 1] == pytest.approx(0.245, abs=1e-4)
