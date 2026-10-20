"""Isaac Sim test fixtures: one headless SimulationApp per pytest session.

Run with ``pytest -m isaac tests/isaac`` inside ``env_isaaclab``. Every test here is
marked ``isaac``; if ``isaacsim`` cannot be imported the tests are skipped.

At the end of the session the process exits via ``os._exit(status)`` instead of a full
Kit shutdown: ``SimulationApp.close()`` terminates the interpreter itself (losing the
pytest exit status) and leaving Kit running can hang the interpreter at exit.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

try:
    import isaacsim  # noqa: F401  (light-weight; does not start Kit)

    HAVE_ISAAC, ISAAC_ERR = True, ""
except Exception as e:  # pragma: no cover
    HAVE_ISAAC, ISAAC_ERR = False, repr(e)

_EXIT = {"status": 0, "launched": False}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tests/isaac" in str(item.fspath).replace(os.sep, "/"):
            item.add_marker(pytest.mark.isaac)
            if not HAVE_ISAAC:
                item.add_marker(pytest.mark.skip(reason=f"isaacsim not importable: {ISAAC_ERR}"))


def pytest_sessionfinish(session, exitstatus):
    _EXIT["status"] = int(exitstatus)


def pytest_unconfigure(config):
    if _EXIT["launched"]:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_EXIT["status"])


@pytest.fixture(scope="session")
def sim_app():
    if not HAVE_ISAAC:
        pytest.skip(f"isaacsim not importable: {ISAAC_ERR}")
    from isco.sim import app

    a = app.launch(headless=True)
    _EXIT["launched"] = True
    return a


@pytest.fixture(scope="session")
def scene(sim_app):
    from isco.sim.scene import IscoScene

    return IscoScene.build(speed=0.0)


@pytest.fixture(scope="session")
def rig(scene):
    from isco.schemas.sensor import SensorModel
    from isco.sim.rig import CameraRig

    return CameraRig(scene.frame, SensorModel(), n_cams=6, overview=False)


@pytest.fixture(scope="session")
def part(scene):
    from isco.sim.part_prim import PartPrim

    return PartPrim(scene.stage, scene.frame)


def make_defect_box():
    """30 x 20 x 10 mm box (rest frame) with a 4 mm-radius defect patch on its top face."""
    import trimesh

    from isco.geometry.mesh_io import LabeledMesh

    box = trimesh.creation.box(extents=[0.03, 0.02, 0.01])
    box.apply_translation([0.0, 0.0, 0.005])
    v, f = trimesh.remesh.subdivide_to_size(box.vertices, box.faces, max_edge=0.001)
    tm = trimesh.Trimesh(v, f, process=True)
    c = tm.triangles_center
    lab = ((c[:, 2] > 0.0099) & (np.hypot(c[:, 0] - 0.006, c[:, 1] - 0.002) < 0.004)).astype(np.int32)
    return LabeledMesh(tm.vertices, tm.faces, lab, ["part", "dent:0"], {"defect_center": [0.006, 0.002, 0.01]})


@pytest.fixture(scope="session")
def defect_box():
    return make_defect_box()


@pytest.fixture(scope="session")
def defect_box_index(defect_box):
    from isco.sim.capture import SurfaceIndex

    return SurfaceIndex.from_mesh(defect_box)


@pytest.fixture()
def placed_box(part, defect_box):
    part.load(defect_box, key="test_defect_box")
    part.set_pose(np.eye(4))
    part.set_visible(True)
    return part
