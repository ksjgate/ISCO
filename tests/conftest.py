"""Shared pytest fixtures (unit tests never need Isaac Sim)."""
import numpy as np
import pytest
import trimesh

from isco.geometry.mesh_io import LabeledMesh


@pytest.fixture
def box_mesh() -> LabeledMesh:
    """40 x 20 x 10 mm box resting on z = 0, centred in x/y (a rest-frame part)."""
    b = trimesh.creation.box(extents=[0.04, 0.02, 0.01])
    b.apply_translation([0, 0, 0.005])
    return LabeledMesh.from_trimesh(b)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
