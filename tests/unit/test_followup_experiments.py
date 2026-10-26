"""Follow-ups of 2026-10-26: statistics of the specialisation experiment and of the feature
experiment, the part descriptors, the fixed configuration grid and the held-out report
extension set."""
import numpy as np
import pytest

from isco.eval import hypotheses as H
from isco.eval import part_features as PF
from isco.geometry.camera_model import SlotGrid
from isco.optim.space import SearchSpace


def test_diagonal_advantage_and_exact_permutation():
    M = np.full((4, 4), 0.6)
    np.fill_diagonal(M, [0.2, 0.3, 0.25, 0.35])
    M[0, 1] = 0.5
    da = H._diag_advantage(M)
    want = np.mean([(M[:, j].sum() - M[j, j]) / 3 - M[j, j] for j in range(4)])
    assert np.isclose(da, want) and da > 0
    assert np.allclose(H._diag_advantage(np.stack([M, M.T])), [da, H._diag_advantage(M.T)])
    assert np.isclose(H._diag_perm_p(M), 1 / 24)            # only the identity labelling is as diagonal
    assert H._diag_perm_p(np.full((4, 4), 0.5)) == 1.0


def _rec(val, scores, ids):
    return {"val_scores": list(val), "report_defects": [{"inst_id": i, "score": s} for i, s in zip(ids, scores)]}


def test_fnr_table_is_paired_and_matches_point_values():
    rng = np.random.default_rng(0)
    val = rng.normal(size=200)
    ids = [f"d{i:03d}" for i in range(144)]
    good = rng.normal(3.0, 1.0, size=144)
    recs = {("a", "p"): _rec(val, good, ids), ("b", "p"): _rec(val, good, ids),
            ("c", "p"): _rec(val, good - 2.0, ids)}
    point, boot = H._fnr_table(recs, ["a", "b", "c"], ["p"], B=200, seed=0)
    assert point.shape == (3, 1) and boot.shape == (200, 3, 1)
    tau = H.scorer.threshold_at_fpr(val)
    assert np.isclose(point[0, 0], np.mean(good <= tau)) and point[2, 0] > point[0, 0]
    assert np.array_equal(boot[:, 0], boot[:, 1])               # identical configurations, identical replicates
    recs[("b", "p")] = _rec(val, good, ids[::-1])
    with pytest.raises(ValueError):                             # scores matched by instance id, not order
        H._fnr_table({**recs, ("b", "p"): _rec(val[:100], good, ids)}, ["a", "b"], ["p"], B=2, seed=0)


def test_permutation_spearman_and_mantel():
    rng = np.random.default_rng(1)
    x = np.arange(16.0)
    rho, p = H._perm_spearman(x, x + rng.normal(0, 0.5, 16), 2000, rng, +1)
    assert rho > 0.9 and p < 0.01
    rho, p = H._perm_spearman(x, -x, 2000, rng, -1)
    assert rho == -1.0 and p < 0.01
    pts = rng.normal(size=(12, 2))
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    r, p = H._mantel(dist, dist ** 2, 2000, rng)
    assert r > 0.99 and p < 0.01
    assert H._mantel(dist, np.ones_like(dist), 200, rng)[1] == 1.0          # nothing to explain
    assert H._perm_spearman(x, np.ones(16), 200, rng)[1] == 1.0


def test_loo_rule_picks_the_neighbours_configuration():
    labels = ["n1_el20_none", "n1_el20_r90x1", "n3_el60_none"]
    feats, fixed = {}, {}
    for i in range(8):                                           # two clusters with different winners
        a = i < 4
        feats[f"p{i}"] = np.array([0.0 if a else 5.0, 0.1 * i])
        fixed[f"p{i}"] = {"n1_el20_none": 0.5, "n1_el20_r90x1": 0.2 if a else 0.7, "n3_el60_none": 0.6 if a else 0.3}
    fixed["p0"]["n1_el20_r90x1"] = 0.21
    out = H._loo_rule(feats, fixed, labels, k=3)
    assert out["p1"]["rule"] == "n1_el20_r90x1" and out["p6"]["rule"] == "n3_el60_none"
    assert all(out[p]["universal"] in labels for p in out) and len(out["p0"]["neighbours"]) == 3


def test_nominal_descriptors_and_farthest_point_selection():
    grid = SlotGrid()
    V = np.zeros((grid.n_slots, 100), bool)
    V[0, :50] = True                                             # slot 0 sees one half ...
    V[grid.rotate(0, 90.0), 50:] = True                          # ... its 90-degree turn the other half
    d = PF.nominal_descriptors(V, [0.04, 0.02, 0.004], grid)
    assert np.isclose(d["height_ratio"], 0.1) and d["reachable"] == 1.0
    assert np.isclose(d["best_1view"], 0.5) and np.isclose(d["rotation_gain"], 0.5) and d["views_to_90pct"] == 2
    dv = np.zeros((4, grid.n_slots, 10), bool)
    dv[0, 3, :] = True
    dv[1, 5, :6] = True
    dv[2, 7, :4] = True
    assert np.isclose(PF.defect_hard_share(dv), 0.5)              # defects 2 (40 %) and 3 (unseen)
    z = {"a": np.array([0.0, 0.0]), "b": np.array([0.1, 0.0]), "c": np.array([3.0, 0.0]), "d": np.array([0.0, 2.0]),
         "e": np.array([1.5, 1.0])}
    assert PF.farthest_point_selection(["a"], ["b", "c", "d", "e"], z, 2) == ["c", "d"]


def test_fixed_configuration_grid():
    space = SearchSpace(SlotGrid(), np.array([0.0, 0.0, 0.016]))
    grid = H.fixed_configs(space)
    assert len(grid) == 20 and len(H.FIXED_GRID) == 20
    for lab, c in grid.items():
        n, el = int(lab.split("_")[0][1:]), float(lab.split("_")[1][2:])
        assert c.n_cam == n and all(space.grid.slot(k).elevation_deg == el for k in c.slots)
        assert c.speed_level == 2 and c.rotation == ("r90x1" if lab.endswith("r90x1") else "none")


@pytest.fixture(scope="module")
def tiny_root(tmp_path_factory):
    import os

    root = tmp_path_factory.mktemp("isco_ext")
    old = os.environ.get("ISCO_DATA_ROOT")
    os.environ["ISCO_DATA_ROOT"] = str(root)
    yield root
    if old is None:
        os.environ.pop("ISCO_DATA_ROOT", None)
    else:
        os.environ["ISCO_DATA_ROOT"] = old


@pytest.mark.slow
def test_report_extension_keeps_normals_and_scores_without_search_role(tiny_root):
    import trimesh

    from isco.datagen.parts import prepare_part
    from isco.datagen.variants import VariantSetSpec, build_report_extension, build_variant_set
    from isco.detect.pipeline import DetectorConfig
    from isco.optim.evaluator import ConfigEvaluator
    from isco.render.slot_cache import build_slot_cache
    from isco.schemas.config import InspectionConfig
    from isco.schemas.sensor import NoiseModel, SensorModel
    from isco.visibility.raycast import build_visibility

    a = trimesh.creation.box(extents=[0.04, 0.02, 0.004])
    b = trimesh.creation.box(extents=[0.004, 0.02, 0.015])
    b.apply_translation([0.018, 0, 0.0075])
    prepare_part("tiny", "procedural:tiny", "procedural", target_max_dim_m=0.04, target_edge_m=8e-4,
                 mesh=trimesh.boolean.union([a, b], engine="manifold"))
    base = build_variant_set("tiny", VariantSetSpec(set_id="t", n_bank=4, n_val=6, per_type_search=1, per_type_report=1,
                                                     n_unreachable=0, types=["dent", "hole"]), n_workers=2)
    ext = build_report_extension("tiny", base_set="t", set_id="t_ext", seed=1, n_val_extra=3, per_type_report=2, n_workers=2)
    assert ext["counts"] == {"bank": 4, "val": 9, "search": 0, "report": 4, "unreachable": 0}
    old = {r["inst_id"]: r for r in base["instances"] if r["role"] in ("bank", "val")}
    kept = [r for r in ext["instances"] if r["inst_id"] in old]
    assert len(kept) == 10 and all(r == old[r["inst_id"]] for r in kept)
    assert sorted(r["inst_id"] for r in ext["instances"] if r["role"] == "val")[-1] == "n_val_0008"
    assert build_report_extension("tiny", base_set="t", set_id="t_ext", seed=1, n_val_extra=3, per_type_report=2,
                                  n_workers=2)["spec_hash"] == ext["spec_hash"]    # rebuilt only on a spec change
    sensor = SensorModel(sensor_id="sl_default", width=256, height=256)
    grid = SlotGrid(azimuths_deg=(0.0, 90.0, 180.0, 270.0), elevations_deg=(40.0, 80.0), distances_m=(0.25,))
    build_slot_cache("tiny", "t_ext", sensor, grid, n_workers=2)
    build_visibility("tiny", "t_ext", sensor, grid, n_workers=2)
    det = DetectorConfig(voxel_m=0.6e-3, fpfh_radii_m=[1.5e-3, 3e-3], coreset_max=500, bank_presample=20000,
                         bootstrap_B=20, template_samples=4000)
    ev = ConfigEvaluator("tiny", "t_ext", det, noise=NoiseModel(), n_workers=2, use_cache=False)
    try:
        r = ev.evaluate(InspectionConfig(slots=[0, 6], rotation="none", speed_level=1), 0)
    finally:
        ev.close()
    assert r["search"] is None and len(r["report_defects"]) == 4 and len(r["val_scores"]) == 9
    assert r["objectives"]["fnr"] == r["report"]["fnr"]


class _FakeEv:
    """Deterministic stand-in for ConfigEvaluator: FNR depends on the configuration and the part."""

    def __init__(self, part, set_id, *a, **k):
        self.part, self.set_id = part, set_id
        self.space = SearchSpace(SlotGrid(), np.array([0.0, 0.0, 0.016]))

    def evaluate(self, cfg, seed=0):
        import zlib

        rng = np.random.default_rng(zlib.crc32(f"{self.part}|{self.set_id}|{cfg.config_id}".encode()))
        val = rng.normal(size=40)
        ids = [f"d{i:03d}" for i in range(30)]
        shift = 1.0 + 2.0 * rng.random()
        scores = rng.normal(shift, 1.0, size=30)
        tau = H.scorer.threshold_at_fpr(val)
        fnr = float(np.mean(scores <= tau))
        return {"val_scores": val.tolist(), "report_defects": [{"inst_id": i, "score": float(s)} for i, s in zip(ids, scores)],
                "report": {"fnr": fnr}, "search": {"fnr": fnr}, "objectives": {"fnr": fnr, "n_cam": cfg.n_cam, "cycle_time_s": 1.0},
                "n_cam": cfg.n_cam}

    def close(self):
        pass


def _cfg(slots, rot="none", v=1):
    from isco.schemas.config import InspectionConfig

    return InspectionConfig(slots=slots, rotation=rot, speed_level=v)


def test_specialization_runs_end_to_end_on_a_fake_evaluator(tmp_path, monkeypatch):
    import json

    parts = ["a", "b", "c"]
    opt40 = {"a": _cfg([1]), "b": _cfg([2, 8]), "c": _cfg([3])}
    optf = {"a": _cfg([1]), "b": _cfg([14]), "c": _cfg([5, 11])}
    for name, opt in (("e1", opt40), ("e1_full", optf)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text(json.dumps({"best_configs": {p: c.model_dump() for p, c in opt.items()}}))
    with open(tmp_path / "e1_full" / "search_a.jsonl", "w") as fh:          # only part a has extra candidates
        for slots in ([1], [4], [6, 30], [7], [9], [10]):
            c = _cfg(slots)
            fh.write(json.dumps({"stage": 2, "config": c.model_dump(), "config_id": c.config_id, "n_cam": c.n_cam,
                                 "objectives": {"fnr": 0.1 * len(slots), "cycle_time_s": 1.0}}) + "\n")
    for p in ("b", "c"):
        (tmp_path / "e1_full" / f"search_{p}.jsonl").write_text("")
    monkeypatch.setattr(H, "_ev", lambda p, s, n, **k: _FakeEv(p, s))
    d = H.specialization(parts, out=str(tmp_path / "out"), e1_dir=str(tmp_path / "e1"),
                         e1_full_dir=str(tmp_path / "e1_full"), top_k=3, B=50)
    s = json.loads((d / "summary.json").read_text())
    assert s["decision_specialisation"] in ("part-specific", "universal-equivalent", "inconclusive")
    assert s["decision_search_budget"] in ("stronger", "weaker", "inconclusive")
    assert len(s["candidates"]) == 3 + 3 and set(s["per_part"]) == set(parts)     # top-3 of a + three budget-40 optima
    assert np.array(s["matrix_full"]).shape == (3, 3) and 0 <= s["diag_perm_p_full"] <= 1
    assert s["universal"]["config_id"] in s["candidates"]


def test_feature_map_runs_end_to_end_on_a_fake_evaluator(tmp_path, monkeypatch):
    import json

    parts = [f"p{i}" for i in range(6)]
    (tmp_path / "e1").mkdir()
    (tmp_path / "e1" / "summary.json").write_text(json.dumps({"best_configs": {"p0": _cfg([1]).model_dump(),
                                                                               "p1": _cfg([2, 8]).model_dump()}}))
    rng = np.random.default_rng(0)
    feats = {p: {k: float(rng.random()) for k in PF.FEATURES} for p in parts}
    monkeypatch.setattr(PF, "part_descriptors", lambda p, s="mvp": feats[p])
    monkeypatch.setattr(H, "_ev", lambda p, s, n, **k: _FakeEv(p, s))
    searched = []

    def fake_search(ev, d, tag, method, budget, seed):
        searched.append(ev.part)
        return [ev.evaluate(_cfg([12 + len(searched)]), seed) | {"stage": 2, "config": _cfg([12 + len(searched)]).model_dump()}]

    monkeypatch.setattr(H, "_search", fake_search)
    monkeypatch.setattr(H, "_best", lambda rows, **k: rows[0])
    d = H.feature_map(parts, out=str(tmp_path / "out"), e1_dir=str(tmp_path / "e1"), n_perm=200)
    s = json.loads((d / "summary.json").read_text())
    assert searched == ["p2", "p3", "p4", "p5"]                   # budget-40 optima reused where they exist
    assert set(s["tests"]) == {"rotation", "transfer_loss", "sensitivity", "feature_distance"}
    assert all(0 < t["p"] <= 1 and t["p_holm"] >= t["p"] for t in s["tests"].values())
    assert len(s["fixed_grid"]["p0"]) == 20 and np.array(s["cross_matrix"]).shape == (6, 6)
    assert (d / "features.json").exists() and set(s["practical_rule"]["per_part"]) == set(parts)
