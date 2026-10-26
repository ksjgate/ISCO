"""End-to-end on a tiny problem (temporary data root, 256^2 sensor, 8 slots):
part -> variant set -> embree slot cache -> visibility -> detector -> evaluator
(cache hit) -> stage-1 + stage-2 search -> report."""
import numpy as np
import pytest
import trimesh


@pytest.fixture(scope="module")
def tiny_root(tmp_path_factory):
    import os

    root = tmp_path_factory.mktemp("isco_data")
    old = os.environ.get("ISCO_DATA_ROOT")
    os.environ["ISCO_DATA_ROOT"] = str(root)
    yield root
    if old is None:
        os.environ.pop("ISCO_DATA_ROOT", None)
    else:
        os.environ["ISCO_DATA_ROOT"] = old


@pytest.mark.slow
def test_tiny_end_to_end(tiny_root):
    from isco.datagen.parts import prepare_part
    from isco.datagen.variants import VariantSetSpec, build_variant_set
    from isco.detect.pipeline import DetectorConfig
    from isco.geometry.camera_model import SlotGrid
    from isco.optim.evaluator import ConfigEvaluator
    from isco.optim.search import SearchRunner
    from isco.render.slot_cache import build_slot_cache
    from isco.report.render_report import build_report
    from isco.report.store import ResultsStore
    from isco.schemas.config import InspectionConfig
    from isco.schemas.sensor import NoiseModel, SensorModel
    from isco.visibility.raycast import build_visibility

    # an L-bracket so that self-occlusion exists
    a = trimesh.creation.box(extents=[0.04, 0.02, 0.004])
    b = trimesh.creation.box(extents=[0.004, 0.02, 0.015])
    b.apply_translation([0.018, 0, 0.0075])
    part = trimesh.boolean.union([a, b], engine="manifold")
    spec = prepare_part("tiny", "procedural:tiny", "procedural", target_max_dim_m=0.04, target_edge_m=8e-4, mesh=part)
    assert spec.stats["fine"]["watertight"]
    vs = VariantSetSpec(set_id="t", n_bank=4, n_val=6, per_type_search=1, per_type_report=1, n_unreachable=1,
                        types=["dent", "hole"], unreachable_types=["dent"])
    man = build_variant_set("tiny", vs, n_workers=2)
    assert man["counts"] == {"bank": 4, "val": 6, "search": 2, "report": 2, "unreachable": 1}
    sensor = SensorModel(sensor_id="sl_default", width=256, height=256)
    grid = SlotGrid(azimuths_deg=(0.0, 90.0, 180.0, 270.0), elevations_deg=(40.0, 80.0), distances_m=(0.25,))
    cache = build_slot_cache("tiny", "t", sensor, grid, n_workers=2)
    assert cache.meta["truncated_views_total"] == 0 and cache.meta["pixel_counts"]["mean_per_view"] > 100
    build_visibility("tiny", "t", sensor, grid, n_workers=2)
    det = DetectorConfig(voxel_m=0.6e-3, fpfh_radii_m=[1.5e-3, 3e-3], coreset_max=500, bank_presample=20000,
                         bootstrap_B=50, template_samples=4000)
    ev = ConfigEvaluator("tiny", "t", det, noise=NoiseModel(), n_workers=2, use_cache=True)
    try:
        cfg = InspectionConfig(slots=[0, 6], rotation="none", speed_level=1)
        r1 = ev.evaluate(cfg, 0)
        assert 0.0 <= r1["search"]["fnr"] <= 1.0 and r1["n_cam"] == 2 and not r1["cache_hit"]
        assert r1["objectives"]["cycle_time_s"] == pytest.approx(1.0)
        assert len(r1["val_scores"]) == 6 and r1["diagnostics"]["points_mean"] > 100
        r2 = ev.evaluate(cfg, 0)
        assert r2["cache_hit"] and r2["search"]["fnr"] == r1["search"]["fnr"]
        s = ev.surrogate(InspectionConfig(slots=[0, 6], rotation="r180x1"))
        assert s["surrogate"]["n_views"] == 4
        st1 = ResultsStore(tiny_root / "run" / "stage1.jsonl")
        st2 = ResultsStore(tiny_root / "run" / "stage2.jsonl")
        r = SearchRunner(ev, st1)
        cands = r.stage1(n_trials=40, n_random=20)
        warm = r.shortlist(cands, k=3, n_random=1)
        rows = SearchRunner(ev, st2).run("bo_warm", budget=4, seed=0, warm=warm)
        assert len(rows) == 4
        rows += SearchRunner(ev, st2).run("greedy_vis", budget=2, seed=0)
        assert [x["greedy_n"] for x in rows[-2:]] == [1, 2]
    finally:
        ev.close()
    import json

    (tiny_root / "run" / "run.json").write_text(json.dumps({"part_id": "tiny", "set_id": "t"}))
    md = build_report(tiny_root / "run", max_cams=3, max_cycle_s=5.0)
    assert md.exists() and (md.parent / "pareto.png").exists() and (md.parent / "recommended.json").exists()
    assert np.isfinite(json.loads((md.parent / "recommended.json").read_text())["fnr_report"])


@pytest.mark.slow
def test_slot_cache_rerenders_only_changed_instances(tiny_root):
    """After the variant set is rebuilt with a new defect scale, a cache rebuild without
    ``overwrite`` re-renders the changed (defect) instances and keeps the identical
    (normal) ones - never silently reusing stale views."""
    import time

    from isco.datagen.defects import DefectDictionary
    from isco.datagen.parts import prepare_part
    from isco.datagen.variants import VariantSetSpec, build_variant_set
    from isco.geometry.camera_model import SlotGrid
    from isco.render.slot_cache import build_slot_cache, cache_dir
    from isco.schemas.sensor import SensorModel
    from isco.visibility.raycast import build_visibility

    part = trimesh.creation.box(extents=[0.04, 0.02, 0.01])
    prepare_part("tiny_box", "procedural:tiny_box", "procedural", target_max_dim_m=0.04, target_edge_m=1e-3, mesh=part)
    sensor = SensorModel(sensor_id="sl_default", width=128, height=128)
    grid = SlotGrid(azimuths_deg=(0.0, 180.0), elevations_deg=(60.0,), distances_m=(0.25,))

    def spec(k):
        return VariantSetSpec(set_id="c", n_bank=2, n_val=2, per_type_search=1, per_type_report=0, n_unreachable=0,
                              types=["dent"], dictionary=DefectDictionary(scale_k=k))

    build_variant_set("tiny_box", spec(1.0), n_workers=1)
    build_slot_cache("tiny_box", "c", sensor, grid, n_workers=1)
    vis = build_visibility("tiny_box", "c", sensor, grid, n_workers=1)
    views = cache_dir("tiny_box", "c", "sl_default") / "views"
    before = {p.name: p.stat().st_mtime_ns for p in views.glob("*.npz")}
    vis_before = vis.stat().st_mtime_ns
    time.sleep(0.05)
    assert build_visibility("tiny_box", "c", sensor, grid, n_workers=1).stat().st_mtime_ns == vis_before   # unchanged: reused
    build_variant_set("tiny_box", spec(1.4), n_workers=1, overwrite=True)
    build_slot_cache("tiny_box", "c", sensor, grid, n_workers=1)
    after = {p.name: p.stat().st_mtime_ns for p in views.glob("*.npz")}
    changed = {n for n in before if after[n] != before[n]}
    assert changed == {n for n in before if n.startswith("d_")}
    assert build_visibility("tiny_box", "c", sensor, grid, n_workers=1).stat().st_mtime_ns != vis_before    # defects changed
