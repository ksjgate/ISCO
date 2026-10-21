"""MVTec 3D-AD anchor: BTF re-implementation and the E5b re-rendering of real scans,
on a synthetic organized scan (tilted support plane + box, off-centre crop)."""
import numpy as np
import pytest

from isco.eval import mvtec3d as M
from isco.geometry.camera_model import PinholeCamera, camera_pose_spherical
from isco.geometry.mesh_io import LabeledMesh
from isco.render.raycast_backend import RaycastRenderer
from isco.schemas.sensor import NoiseModel

IDEAL = NoiseModel(name="ideal", sigma0_m=0.0, dropout_p0=0.0, dropout_theta_c_deg=90.0, flying_p=0.0, edge_filter_m=0.0)


def _scene(bump: float = 0.0):
    import trimesh

    box = trimesh.creation.box([0.030, 0.012, 0.008])
    box.apply_translation([0, 0, 0.004])
    if bump > 0:
        cap = trimesh.creation.icosphere(subdivisions=3, radius=0.003)
        cap.apply_translation([0.006, 0.0, 0.008 - 0.003 + bump])
        box = trimesh.util.concatenate([box, cap])
    plane = trimesh.Trimesh([[-0.2, -0.2, 0], [0.2, -0.2, 0], [0.2, 0.2, 0], [-0.2, 0.2, 0]], [[0, 1, 2], [0, 2, 3]])
    m = trimesh.util.concatenate([box, plane])
    return LabeledMesh(np.asarray(m.vertices), np.asarray(m.faces))


def _scan(tmp_path, bump: float = 0.0, name: str = "000"):
    import tifffile

    cam = PinholeCamera(200, 200, 30.0)
    roi = (60, 40, 160, 140)                                  # principal point at (40, 60) of the crop
    T = camera_pose_spherical(np.zeros(3), 0.0, 70.0, 0.30)
    depth, _ = RaycastRenderer(cam).render_full_depth(_scene(bump), np.eye(4), T, roi)
    pix = cam.roi_pixels(roi)
    xyz = cam.unproject(pix, depth.ravel().astype(np.float64)).reshape(100, 100, 3).astype(np.float32)
    xyz[depth == 0] = 0
    p = tmp_path / f"{name}.tiff"
    tifffile.imwrite(str(p), xyz)
    return p, xyz, cam


def test_fitted_camera_and_embedding(tmp_path):
    p, xyz, cam = _scan(tmp_path)
    fc = M.fit_camera([p])
    assert abs(fc.fx - cam.fx) < 0.5 and abs(fc.cx - 40.0) < 0.05 and abs(fc.cy - 60.0) < 0.05 and fc.rms_px < 0.01
    cam2, roi2, err = fc.embed()
    assert err < 0.3 and roi2[2] - roi2[0] == 100 and roi2[3] - roi2[1] == 100


def test_support_frame_object_mask_and_ideal_rerender(tmp_path):
    p, xyz, _ = _scan(tmp_path)
    T = M.support_frame(xyz)
    h = M.height_above_support(xyz, T)
    assert np.nanmax(np.abs(h[:5])) < 1e-5                    # border rows lie on the support
    assert abs(np.nanmax(h) - 0.008) < 1e-4                   # box top 8 mm above it
    obj = M.object_mask(xyz, T)
    assert obj.sum() > 200 and np.all(h[obj] > 0.5e-3)
    cam2, roi2, _ = M.fit_camera([p]).embed()
    re = M.rerender(xyz, cam2, roi2, T, IDEAL, np.random.default_rng(0))
    both = M.valid_mask(re) & M.valid_mask(xyz)
    assert both.mean() > 0.97
    assert np.median(np.abs(re[..., 2] - xyz[..., 2])[both & obj]) < 2e-5
    ref, cos = M.ideal_render(xyz, cam2, roi2, T)
    st = M.scan_statistics(ref, ref, cos, obj, T)
    assert st["valid_object"] > 0.99 and st["sigma_part_mm"] < 1e-3 and st["outlier_frac_near_object"] < 0.01


def test_noise_levels_raise_measured_noise(tmp_path):
    p, xyz, _ = _scan(tmp_path)
    T = M.support_frame(xyz)
    obj = M.object_mask(xyz, T)
    cam2, roi2, _ = M.fit_camera([p]).embed()
    ref, cos = M.ideal_render(xyz, cam2, roi2, T)
    noisy = M.rerender(xyz, cam2, roi2, T, NoiseModel(name="t", sigma0_m=0.1e-3, dropout_p0=0.0, flying_p=0.0),
                       np.random.default_rng(1))
    st = M.scan_statistics(noisy, ref, cos, obj, T)
    # sigma0 0.1 mm at 0.25 m -> ~0.144 mm at 0.30 m (+ incidence gain); local fits see most of it
    assert 0.08 < st["sigma_part_mm"] < 0.2


def test_btf_pipeline(tmp_path):
    _, xyz, _ = _scan(tmp_path)
    pre = M.btf_preprocess(xyz)
    T = M.support_frame(xyz)
    h = M.height_above_support(np.where(M.valid_mask(pre)[..., None], pre, 0), T)
    assert M.valid_mask(pre).sum() > 100 and np.nanmin(h) > 0.0049     # support and lower 5 mm removed
    f = M.btf_features(xyz)
    assert f.shape == (784, 33)
    full = M.BTFPatchCore(f_coreset=1.0, device="cpu").fit([f])     # every patch kept: a copy scores ~0
    s_same, pmap = full.score(f)
    _, xyz_b, _ = _scan(tmp_path, bump=0.0015, name="001")
    f_b = M.btf_features(xyz_b)
    s_bump, _ = full.score(f_b)
    # BTF's torch.cdist (matrix-product path) leaves ~0.1 of round-off on FPFH histograms (values up to 100)
    assert pmap.shape == (28, 28) and abs(s_same) < 0.5 and s_bump > 10.0
    core = M.BTFPatchCore(device="cpu").fit([f])                       # 10 % greedy coreset from index 0
    assert core.lib.shape[0] == 78 and np.allclose(core.lib[0].numpy(), f[0])
    assert core.score(f_b)[0] > core.score(f)[0]


def test_resize_nearest_matches_torch():
    torch = pytest.importorskip("torch")
    a = np.random.default_rng(0).normal(size=(400, 400, 3)).astype(np.float32)
    t = torch.nn.functional.interpolate(torch.from_numpy(a).permute(2, 0, 1)[None], size=(224, 224), mode="nearest")
    assert np.array_equal(M.resize_nearest(a), t[0].permute(1, 2, 0).numpy())


def test_level_noise_native_and_embed_error(tmp_path):
    n2 = M.level_noise("N2", 0.55)
    nat = M.level_noise("N2@native", 0.55)
    assert n2.z0_m == 0.25 and nat.z0_m == 0.55 and nat.sigma0_m == n2.sigma0_m and nat.name == "N2@native"
    with pytest.raises(ValueError):
        M.level_noise("N2@far", 0.55)
    fc = M.FittedCamera(2745.0, 2749.0, 53.18, 251.75, 400, 400, 0.2)
    cam, roi, err = fc.embed()
    assert roi[2] - roi[0] == 400 and err < 0.6        # principal point within 0.25 px, focal 0.07 %
