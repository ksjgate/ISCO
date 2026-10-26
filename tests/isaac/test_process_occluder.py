"""Occluder export and physics settling (short run), then rendering still works."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytestmark = pytest.mark.isaac


def test_export_occluder(scene, tmp_path):
    from isco.render.occluder import Occluder
    from isco.sim.export_occluder import export_occluder

    out = export_occluder(scene, tmp_path / "occluder.npz", radius_m=0.6)
    occ = Occluder.load(tmp_path / "occluder.npz")
    assert len(occ.faces) > 100 and occ.source == "isaac_export"
    tri = occ.vertices[occ.faces]
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    assert np.all((hi >= -0.6 - 1e-6) & (lo <= 0.6 + 1e-6))   # every kept triangle touches the crop cube
    # the belt top surface near the focus is the plane z = 0 (ISCO world)
    assert abs(out["belt_surface_z_at_origin_m"]) < 1e-4
    assert out["min_abs_y_of_geometry_above_belt_m"] > 0.3     # only the side rails rise above the belt
    meta = json.loads((tmp_path / "conveyor_meta.json").read_text())
    assert meta["belt_top_z_stage"] == pytest.approx(scene.info["belt_top_z"])
    assert np.allclose(meta["isco_origin_stage"], scene.frame.origin_stage)


def test_settle_flat_box_short(scene, rig, part, defect_box, defect_box_index):
    from isco.sim.capture import capture_views
    from isco.sim.process import settle_part
    from isco.geometry.camera_model import SlotGrid

    res = settle_part(scene, defect_box, speeds_mps=[0.3], n_drops=6, settle_time_s=1.2, seed=0)
    s = res["speeds"]["0.3"]
    assert s["summary"]["n"] == 6
    assert s["summary"]["settled_fraction"] == 1.0
    assert s["summary"]["flipped_fraction"] == 0.0
    for smp in s["samples"]:
        assert abs(smp["dz"]) < 1e-3          # a box dropped flat stays flat
        assert abs(smp["droll"]) < 2 and abs(smp["dpitch"]) < 2
        assert abs(smp["dy"]) < 0.02
    # the stage is still usable for rendering after physics (worker interleaves job types)
    part.load(defect_box, key="test_defect_box")
    part.set_pose(np.eye(4))
    cams = rig.place([SlotGrid().T_world_cam(26, np.array([0.0, 0.0, 0.005]))])
    views, warns, _ = capture_views(cams, verify={"index": defect_box_index, "T_world_part": np.eye(4)})
    rig.park_all()
    assert not warns and len(views[0]) > 1000 and views[0].meta["verify"]["ok"]
