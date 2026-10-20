import numpy as np
import pytest

from isco.detect import scorer
from isco.detect.coverage_scorer import CoverageScorer
from isco.detect.memory_bank import MemoryBank, greedy_coreset
from isco.detect.merge import voxel_average
from isco.geometry.camera_model import SlotGrid
from isco.optim.pareto import hypervolume, nondominated_mask
from isco.optim.space import SearchSpace
from isco.render.occluder import Occluder


def test_memory_bank_separates_clusters(rng):
    a = rng.normal(0, 1, (3000, 8)).astype(np.float32)
    b = rng.normal(6, 1, (200, 8)).astype(np.float32)
    mb = MemoryBank(coreset_ratio=0.2, coreset_max=600, device="cpu", presample=2000).fit([a])
    assert mb.size == 600
    assert mb.score(b).mean() > 3 * mb.score(a[:200]).mean()


def test_greedy_coreset_unique():
    import torch

    X = torch.randn(500, 16)
    idx = greedy_coreset(X, 50)
    assert len(set(idx.tolist())) == 50


def test_threshold_and_fnr_arithmetic(rng):
    val = rng.normal(0, 1, 1000)
    tau = scorer.threshold_at_fpr(val, 0.01)
    assert tau == pytest.approx(np.quantile(val, 0.99))
    assert scorer.fnr_at_fpr(val, [tau - 1, tau + 1, tau + 2], 0.01) == pytest.approx(1 / 3)
    f, lo, hi = scorer.bootstrap_fnr(val, rng.normal(3, 1, 200), B=200)
    assert lo <= f <= hi
    assert scorer.auroc([0, 0.1], [1, 2]) == 1.0
    assert scorer.threshold_at_fpr(val, 0.01, "gpd") == pytest.approx(tau, rel=0.25)


def test_aupro_perfect_detector():
    normal = np.zeros(1000)
    dps = [np.r_[np.ones(10), np.zeros(5)]]
    dms = [np.r_[np.ones(10, bool), np.zeros(5, bool)]]
    assert scorer.aupro(dps, dms, normal) == pytest.approx(1.0, abs=0.02)


def test_voxel_average_labels_max():
    xyz = np.array([[0, 0, 0], [1e-4, 0, 0], [5e-3, 0, 0]], float)
    lab = np.array([1, 2, 1], np.uint8)
    out, l, d = voxel_average(xyz, lab, np.tile([0, 0, 1.0], (3, 1)), 1e-3)
    assert len(out) == 2 and sorted(l.tolist()) == [1, 2]


def test_coverage_scorer_detects_missing_patch(rng):
    t = rng.uniform(0, 0.01, (2000, 3))
    t[:, 2] = 0
    cs = CoverageScorer(t, radius=0.3e-3, smooth_radius=1e-3)
    obs = [cs.observed(t + rng.normal(0, 1e-5, t.shape)) for _ in range(5)]
    cs.fit(obs)
    holey = t[np.linalg.norm(t[:, :2] - 0.005, axis=1) > 1.5e-3]
    assert cs.image_score(cs.observed(holey)) > 0.9 and cs.image_score(obs[0]) < 0.1


def test_pareto_and_hypervolume():
    F = np.array([[0.1, 0.5], [0.2, 0.2], [0.3, 0.3], [0.5, 0.1]])
    assert nondominated_mask(F).tolist() == [True, True, False, True]
    assert hypervolume(np.array([[0.5, 0.5]]), [1.0, 1.0]) == pytest.approx(0.25)


def test_search_space_constraints(rng):
    g = SlotGrid()
    sp = SearchSpace(g, np.array([0, 0, 0.005]), Occluder.belt_plane())
    assert len(sp.valid_slots) == 96
    a = g.slot_id(0, 1, 0)
    assert not sp.compatible(a, g.slot_id(0, 1, 1))           # same (az, el) cell
    assert not sp.compatible(a, g.slot_id(1, 1, 0))           # < 30 deg apart
    assert sp.compatible(a, g.slot_id(6, 1, 0))
    assert sp.repair([a, g.slot_id(1, 1, 0), g.slot_id(6, 1, 0)]) == [a, g.slot_id(6, 1, 0)]
    for _ in range(20):
        c = sp.random_config(rng)
        assert sp.feasible(c.slots)
    assert set(sp.baselines()) >= {"top_1cam", "ring3_el40"}


def test_shortlist_spreads_over_rotation_modes():
    from types import SimpleNamespace

    from isco.optim.search import SearchRunner
    from isco.schemas.config import InspectionConfig

    cands = []
    for n in (1, 2, 3):
        for rot, v in (("r90x3", 0.9), ("none", 0.5), ("r90x1", 0.7)):
            for j in range(3):
                c = InspectionConfig(slots=list(range(j * 10, j * 10 + n)), rotation=rot)
                cands.append({"config": c.model_dump(mode="json"), "config_id": c.config_id, "n_cam": n,
                              "surrogate": {"V_area": v - 0.01 * j}})
    sl = SearchRunner(SimpleNamespace(surrogate_metric="V_area")).shortlist(cands, k=9, n_random=0)
    assert {c.rotation for c in sl} == {"r90x3", "none", "r90x1"}
    assert {c.n_cam for c in sl} == {1, 2, 3}


def test_registrar_recovers_repositioning_error(box_mesh):
    import trimesh

    from isco.detect.merge import Registrar
    from isco.geometry.transforms import apply_T, make_T, rot_z

    tm = box_mesh.to_trimesh()
    pts, fid = trimesh.sample.sample_surface_even(tm, 20000, seed=0)
    nrm = np.asarray(tm.face_normals)[fid]
    reg = Registrar(pts, nrm)
    top = pts[pts[:, 2] > 0.002]                              # a partial (visible) cloud
    T_err = make_T(rot_z(1.0), [0.5e-3, -0.3e-3, 0.0])
    T, fit, rmse = reg.register(apply_T(T_err, top))
    resid = np.linalg.norm(apply_T(T @ T_err, top) - top, axis=1)
    assert np.median(resid) < 5e-5 and fit > 0.9
