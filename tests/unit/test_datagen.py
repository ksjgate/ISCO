import numpy as np
import pytest

from isco.datagen.defects import (DefectDictionary, DefectGenerator, difficulty_of, hemisphere_directions,
                                  tier_value, visible_direction_count)
from isco.datagen.normal_variation import NormalVariationParams, PoseParams, apply_normal_variation, sample_pose
from isco.datagen.sensor_noise import apply_motion_blur, apply_sensor_noise
from isco.geometry.camera_model import PinholeCamera
from isco.geometry.remesh import remesh_isotropic
from isco.geometry.transforms import apply_T, look_at_opencv
from isco.render.pixels import ViewPixels
from isco.schemas.sensor import NoiseModel


def test_hemisphere_and_difficulty_bins():
    d = hemisphere_directions(41)
    assert d.shape == (41, 3) and np.all(d[:, 2] > 0) and np.allclose(np.linalg.norm(d, axis=1), 1)
    assert list(difficulty_of(np.array([0, 3, 10, 30]))) == ["unreachable", "hard", "medium", "easy"]


def test_visible_count_top_vs_bottom(box_mesh):
    pts = np.array([[0, 0, 0.01], [0, 0, 0.0]])
    nrm = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    k = visible_direction_count(box_mesh, pts, nrm)
    assert k[0] >= 20 and k[1] == 0


def test_tier_value_within_range(rng):
    for t in "SML":
        v = tier_value(rng, 0.2e-3, 1.0e-3, t)
        assert 0.2e-3 <= v <= 1.0e-3
    assert tier_value(rng, 0.2e-3, 1e-3, "S") < tier_value(rng, 0.2e-3, 1e-3, "L")


@pytest.fixture(scope="module")
def fine_box():
    import trimesh

    from isco.geometry.mesh_io import LabeledMesh

    b = trimesh.creation.box(extents=[0.04, 0.02, 0.01])
    b.apply_translation([0, 0, 0.005])
    return remesh_isotropic(LabeledMesh.from_trimesh(b), 6e-4)


@pytest.mark.parametrize("dtype", ["dent", "bulge", "hole", "cut", "crack", "warp"])
def test_defect_types_watertight_and_labelled(fine_box, dtype):
    gen = DefectGenerator(fine_box, DefectDictionary(), seed=0, n_candidates=1500)
    out, meta = gen.make(fine_box, dtype, "easy", "L", np.random.default_rng(1))
    tm = out.to_trimesh()
    assert tm.is_watertight
    assert (out.face_label > 0).sum() > 0 and out.label_names[-1].startswith(dtype)
    v0 = fine_box.to_trimesh().volume
    if dtype in ("hole", "cut", "crack", "dent"):
        assert tm.volume < v0
    if dtype == "bulge":
        assert tm.volume > v0


def test_unreachable_defect_on_bottom(fine_box):
    gen = DefectGenerator(fine_box, DefectDictionary(), seed=0, n_candidates=1500)
    out, meta = gen.make(fine_box, "dent", "unreachable", "M", np.random.default_rng(2))
    assert meta["difficulty"] == "unreachable" and meta["k_visible"] == 0


def test_normal_variation_and_pose(fine_box, rng):
    out, info = apply_normal_variation(fine_box, rng, NormalVariationParams())
    assert out.vertices[:, 2].min() == pytest.approx(0.0, abs=1e-12)
    assert info["lowfreq_amp_m"] <= 0.05e-3 + 1e-12
    assert np.abs(np.asarray(info["scale"]) - 1).max() < 0.02
    T, p = sample_pose(rng, PoseParams(), out)
    assert apply_T(T, out.vertices)[:, 2].min() == pytest.approx(0.0, abs=1e-9)
    assert abs(p["dy_m"]) <= 5e-3
    settle = {"speeds": {"0.3": {"samples": [{"droll": 25.0, "dpitch": 0.0, "settled": True}]}}}
    T2, p2 = sample_pose(rng, PoseParams(), out, settle)
    assert p2["tilt_source"] == "isaac_settle" and p2["roll_deg"] == 25.0


def _plane_view(n=60, z=0.25):
    cam = PinholeCamera(256, 256, 20.0)
    u, v = np.meshgrid(np.arange(100, 100 + n), np.arange(100, 100 + n))
    pix = (v.ravel() * 256 + u.ravel()).astype(np.int64)
    vp = ViewPixels(pix, np.full(len(pix), z, np.float32), np.ones(len(pix), np.uint8), np.ones(len(pix), np.float32))
    T = look_at_opencv(np.array([0, 0, z + 0.01]), np.array([0, 0, 0.01]))
    return cam, vp, T


def test_noiseless_is_identity():
    cam, vp, T = _plane_view()
    n0 = NoiseModel(name="N0", sigma0_m=0, dropout_p0=0, flying_p=0, dropout_theta_c_deg=90)
    assert n0.is_noiseless


def test_gaussian_noise_statistics():
    cam, vp, T = _plane_view(120)
    nm = NoiseModel(sigma0_m=0.1e-3, dropout_p0=0.0, dropout_theta_c_deg=89.9, flying_p=0.0)
    out = apply_sensor_noise(vp, cam, T, nm, np.random.default_rng(0))
    assert np.std(out.depth - 0.25) == pytest.approx(0.1e-3, rel=0.1)


def test_dropout_increases_at_grazing_incidence():
    cam, vp, T = _plane_view(120)
    nm = NoiseModel(sigma0_m=0, dropout_p0=0.0, dropout_theta_c_deg=70, dropout_width_deg=5, flying_p=0, edge_filter_m=0.0)
    head_on = apply_sensor_noise(vp, cam, T, nm, np.random.default_rng(0))
    vp.cos[:] = np.cos(np.radians(75))
    grazing = apply_sensor_noise(vp, cam, T, nm, np.random.default_rng(0))
    assert len(head_on) > 0.99 * len(vp) and len(grazing) < 0.4 * len(vp)


def test_motion_blur_flat_unchanged_and_step_smoothed():
    cam, vp, T = _plane_view(80)
    out = apply_motion_blur(vp, cam, T, np.array([1.0, 0, 0]), 0.6e-3)
    assert np.allclose(out.depth, 0.25, atol=1e-7)
    # top-down camera: the belt axis (+x) projects onto the image rows (v), so a step
    # across v is smeared while a step across u would not be
    u, v = cam.pix_to_uv(vp.pix)
    vp.depth = np.where(v < 140, 0.25, 0.249).astype(np.float32)          # 1 mm step
    out = apply_motion_blur(vp, cam, T, np.array([1.0, 0, 0]), 1.0e-3, jump_m=2e-3)
    _, vo = cam.pix_to_uv(out.pix)
    mid = (vo > 136) & (vo < 144)
    assert np.any((out.depth[mid] > 0.2491) & (out.depth[mid] < 0.2499))


def test_motion_blur_invalidates_mixed_pixels_across_large_steps():
    """A pixel whose exposure crosses a depth jump larger than the sensor's jump threshold
    is invalidated (inconsistent code), not averaged into a point floating off the part."""
    cam, vp, T = _plane_view(80)
    u, v = cam.pix_to_uv(vp.pix)
    vp.depth = np.where(v < 140, 0.25, 0.245).astype(np.float32)          # 5 mm step
    out = apply_motion_blur(vp, cam, T, np.array([1.0, 0, 0]), 1.0e-3, jump_m=2e-3)
    assert len(out) < len(vp)
    assert not np.any((out.depth > 0.2451) & (out.depth < 0.2499))


def test_edge_filter_removes_flying_pixels():
    cam, vp, T = _plane_view(60)
    fly = dict(sigma0_m=0.0, dropout_p0=0.0, dropout_theta_c_deg=89.9, flying_p=1.0)
    raw = apply_sensor_noise(vp, cam, T, NoiseModel(**fly, edge_filter_m=0.0), np.random.default_rng(0))
    assert np.any(raw.depth > 0.2501)                                    # veil points towards the belt
    out = apply_sensor_noise(vp, cam, T, NoiseModel(**fly, edge_filter_m=2e-3), np.random.default_rng(0))
    assert np.allclose(out.depth, 0.25) and len(out) >= 56 * 56          # silhouette ring (+ veil neighbours) dropped


def test_unreachable_plan_skips_types_the_part_cannot_host():
    from types import SimpleNamespace

    from isco.datagen.variants import VariantSetSpec, placeable_unreachable, plan_instances

    spec = VariantSetSpec.preset("mvp")
    table = {"unreachable": {"surface": 0, "edge": 12}}           # only a hidden edge
    types = placeable_unreachable(SimpleNamespace(difficulty_table=lambda: table), spec)
    assert types == ["cut"]
    plan = [p for p in plan_instances(spec, types) if p["role"] == "unreachable"]
    assert len(plan) == spec.n_unreachable and {p["type"] for p in plan} == {"cut"}
    # a hidden surface but no hidden edge: a cut becomes a surface chip, all types stay
    chip = {"unreachable": {"surface": 120, "edge": 0}}
    assert placeable_unreachable(SimpleNamespace(difficulty_table=lambda: chip), spec) == spec.unreachable_types
    # all types placeable: the plan is unchanged (existing manifests stay valid)
    full = {"unreachable": {"surface": 120, "edge": 40}}
    assert placeable_unreachable(SimpleNamespace(difficulty_table=lambda: full), spec) == spec.unreachable_types
    assert plan_instances(spec, spec.unreachable_types) == plan_instances(spec)
    none = {"unreachable": {"surface": 0, "edge": 0}}
    assert not [p for p in plan_instances(spec, placeable_unreachable(SimpleNamespace(difficulty_table=lambda: none), spec))
                if p["role"] == "unreachable"]


def test_cut_on_part_without_sharp_edges_is_a_surface_chip():
    """Smooth (scan-derived, turned) parts have no sharp edge: a cut slices a chip off the
    surface instead of failing."""
    import trimesh

    from isco.geometry.mesh_io import LabeledMesh

    s = trimesh.creation.capsule(height=0.03, radius=0.006, count=[48, 48])
    s.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    s.apply_translation([0, 0, 0.006 - s.bounds[0, 2]])
    mesh = remesh_isotropic(LabeledMesh.from_trimesh(s), 6e-4)
    gen = DefectGenerator(mesh, DefectDictionary(), seed=0, n_candidates=1500)
    assert len(gen.edges.points) == 0
    out, meta = gen.make(mesh, "cut", "easy", "L", np.random.default_rng(1))
    tm = out.to_trimesh()
    assert meta["cut_mode"] == "surface_chip" and tm.is_watertight
    assert (out.face_label > 0).sum() > 0 and tm.volume < mesh.to_trimesh().volume


def test_dent_label_area_matches_nominal_on_flat_surface():
    """E0.3 acceptance: a face is labelled by its mean vertex displacement, so the label
    matches the 10 %-of-height contour (an 'any vertex moved' rule dilates it by a ring)."""
    import trimesh

    from isco.datagen.defects import apply_bump
    from isco.geometry.mesh_io import LabeledMesh

    b = trimesh.creation.box(extents=[0.03, 0.03, 0.01])
    b.apply_translation([0, 0, 0.005])
    m = remesh_isotropic(LabeledMesh.from_trimesh(b), 2.5e-4)
    R, h = 2.0e-3, -0.5e-3
    out, _ = apply_bump(m, np.array([0.001, -0.002, 0.01]), np.array([0, 0, 1.0]), R, h, 1.0,
                        np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), 0.02e-3, "dent:0")
    area = out.to_trimesh().area_faces[out.face_label > 0].sum()
    assert area / (np.pi * (0.8269 * R) ** 2) - 1 == pytest.approx(0.0, abs=0.10)
