"""Normal (defect-free) part-to-part variation and resting-pose jitter.

Protocol (plan "정상 변동 · 센서 모델"):

* tolerance: anisotropic scale ``s_i ~ N(1, sigma_s)``, sigma_s = 0.3 % (ISO 2768-m-like)
* low-frequency surface displacement: a few random harmonics with wavelength >= 10 mm,
  total amplitude ``A ~ U(0, 0.05 mm)`` along vertex normals (kept < 1/4 of the smallest
  defect depth so defects stay separable from normal variation)
* resting pose: guided yaw ``N(0, 3 deg)`` (ADR-0007) or uniform yaw (ablation), lateral
  offset ``U(-5, 5) mm``, small tilt from physics settling (measured in Isaac when
  ``data/settle/<part>.json`` exists, otherwise ``N(0, 0.5 deg)``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from ..geometry.mesh_io import LabeledMesh
from ..geometry.transforms import apply_T, make_T, pose_from_params


class NormalVariationParams(BaseModel):
    scale_sigma: float = 0.003
    lowfreq_amp_max_m: float = 0.05e-3
    lowfreq_n_harmonics: int = 3
    lowfreq_min_wavelength_m: float = 10e-3
    lowfreq_max_wavelength_m: float = 40e-3


class PoseParams(BaseModel):
    yaw_mode: Literal["guided", "random"] = "guided"
    yaw_sigma_deg: float = 3.0
    lateral_range_m: float = 5e-3
    tilt_sigma_deg: float = 0.5
    use_settle_data: bool = True


def vertex_normals(mesh: LabeledMesh) -> np.ndarray:
    return np.asarray(mesh.to_trimesh().vertex_normals, dtype=np.float64)


def lowfreq_field(points: np.ndarray, rng: np.random.Generator, p: NormalVariationParams) -> np.ndarray:
    """Smooth scalar field (metres) - a sum of random plane waves, rescaled so that its
    max |value| equals an amplitude drawn from U(0, A_max)."""
    if p.lowfreq_n_harmonics <= 0 or p.lowfreq_amp_max_m <= 0:
        return np.zeros(len(points))
    f = np.zeros(len(points))
    for _ in range(p.lowfreq_n_harmonics):
        lam = rng.uniform(p.lowfreq_min_wavelength_m, p.lowfreq_max_wavelength_m)
        k = rng.normal(size=3)
        k /= np.linalg.norm(k)
        f += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * (points @ k) / lam + rng.uniform(0, 2 * np.pi))
    amp = rng.uniform(0.0, p.lowfreq_amp_max_m)
    m = np.abs(f).max()
    return f * (amp / m) if m > 0 else f


def apply_normal_variation(mesh: LabeledMesh, rng: np.random.Generator, p: NormalVariationParams) -> tuple[LabeledMesh, dict]:
    out = mesh.copy()
    s = 1.0 + rng.normal(0.0, p.scale_sigma, size=3)
    out.vertices = out.vertices * s[None, :]          # rest frame: z_min=0 and xy-centre preserved
    disp = lowfreq_field(out.vertices, rng, p)
    out.vertices = out.vertices + vertex_normals(out) * disp[:, None]
    out.vertices[:, 2] -= out.vertices[:, 2].min()     # keep resting on z = 0
    info = {"scale": s.tolist(), "lowfreq_amp_m": float(np.abs(disp).max())}
    return out, info


def load_settle(part_id: str, data_root: Path) -> dict | None:
    p = data_root / "settle" / f"{part_id}.json"
    if not p.exists():
        return None
    import json

    return json.loads(p.read_text())


def _settle_samples(settle: dict) -> np.ndarray | None:
    """(droll, dpitch) of every drop that came to rest, pooled over speeds.

    Format written by the Isaac worker (``isco.sim.process.settle_part``):
    ``{"speeds": {"<v>": {"samples": [{"droll", "dpitch", "flipped", "settled", ...}]}}}``.
    *Flipped* rests (> 20 deg from the nominal pose) are kept: they are where the
    conveyor really leaves the part (e.g. the near-round bar fusion_22340 rolls onto a
    neighbouring facet in 62-100 % of drops), so bank, validation and test instances
    all see the physical pose distribution. The cache is rendered once, so one pooled
    distribution is used for all speeds (the per-speed statistics are reported, E0).
    """
    rows = []
    groups = settle.get("speeds") or settle.get("per_speed") or {}
    for sp in groups.values():
        for s in sp.get("samples", []):
            if not s.get("settled", True) or s.get("fell_off", False):
                continue
            rows.append([s.get("droll", s.get("droll_deg", 0.0)), s.get("dpitch", s.get("dpitch_deg", 0.0))])
    return np.asarray(rows, dtype=float) if rows else None


def sample_pose(rng: np.random.Generator, p: PoseParams, mesh: LabeledMesh, settle: dict | None = None) -> tuple[np.ndarray, dict]:
    """``T_world_part`` for one instance (relative to the nominal rest pose), lifted so
    the part's lowest point touches the belt (z = 0)."""
    yaw = rng.normal(0.0, p.yaw_sigma_deg) if p.yaw_mode == "guided" else rng.uniform(0.0, 360.0)
    dy = rng.uniform(-p.lateral_range_m, p.lateral_range_m)
    samples = _settle_samples(settle) if (settle and p.use_settle_data) else None
    if samples is not None and len(samples):
        roll, pitch = samples[rng.integers(len(samples))]
        tilt_src = "isaac_settle"
    else:
        roll, pitch = rng.normal(0.0, p.tilt_sigma_deg, size=2)
        tilt_src = "gaussian"
    T = pose_from_params(0.0, dy, 0.0, yaw, roll, pitch)
    zmin = apply_T(T, mesh.vertices)[:, 2].min()
    T = make_T(None, [0.0, 0.0, -zmin]) @ T
    return T, {"yaw_deg": float(yaw), "dy_m": float(dy), "roll_deg": float(roll), "pitch_deg": float(pitch),
               "tilt_source": tilt_src}
