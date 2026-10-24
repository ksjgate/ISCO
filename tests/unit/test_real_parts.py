"""MulSen-AD real-set registration and the E5 rank statistics."""
import numpy as np
import pytest
import trimesh

from isco.eval.hypotheses import _top_overlap
from isco.eval.real_parts import register_scan


def _scan(mesh: trimesh.Trimesh, n: int, seed: int) -> np.ndarray:
    return np.asarray(trimesh.sample.sample_surface(mesh, n, seed=seed)[0])


def _pose(seed: int) -> np.ndarray:
    T = trimesh.transformations.random_rotation_matrix(np.random.default_rng(seed).random(3))
    T[:3, 3] = [-230.0, 240.0, 2080.0]                  # scanner coordinates, far from the template
    return T


def test_register_scan_nearly_symmetric_part_far_away():
    pytest.importorskip("open3d")
    capsule = trimesh.creation.capsule(height=12.0, radius=4.0)        # ~20 x 8 x 8 mm pill
    tmpl = _scan(capsule, 20000, 0)
    voxel = float(max(capsule.extents) / 60.0)
    T_true = _pose(1)
    src = trimesh.transform_points(_scan(capsule, 18000, 1), T_true)
    T, fit4, _ = register_scan(src, tmpl, voxel, seed=0)
    assert fit4 > 0.95
    d = np.linalg.norm(trimesh.transform_points(src, T)[:, None] - tmpl[None, ::40], axis=2).min(axis=1)
    assert np.median(d) < 2 * voxel


def test_register_scan_rejects_a_different_part():
    pytest.importorskip("open3d")
    capsule = trimesh.creation.capsule(height=12.0, radius=4.0)
    plate = trimesh.creation.box([20.0, 14.0, 2.0])
    voxel = float(max(capsule.extents) / 60.0)
    src = trimesh.transform_points(_scan(plate, 18000, 2), _pose(3))
    _, fit4, _ = register_scan(src, _scan(capsule, 20000, 0), voxel, seed=0)
    assert fit4 < 0.95


def test_top_overlap_breaks_ties_at_random():
    rng = np.random.default_rng(0)
    a = np.arange(10.0)
    assert _top_overlap(a, a, 5, rng) == 5.0
    assert _top_overlap(a, a[::-1], 5, rng) == 0.0
    # all-tied b: index order would give 5 (a's best are listed first); random ties give k*k/n
    assert abs(_top_overlap(a, np.zeros(10), 5, rng) - 2.5) < 0.15
