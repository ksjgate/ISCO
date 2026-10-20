"""Depth-sensor model: geometry (pinhole) + parametric noise (ADR-0001: Isaac Sim 5.1
has no depth-noise model, so noise is applied as a documented post-process)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..geometry.camera_model import PinholeCamera


class NoiseModel(BaseModel):
    """Structured-light depth noise (applied per view, before unprojection).

    sigma_z(z, theta) = sigma0 * (z / z0)^2 * (1 + incidence_gain * (1 - cos theta))
    p_dropout(theta)  = p0 + (1 - p0) * sigmoid((theta - theta_c) / width)
    flying pixels     : pixels next to a depth jump > flying_jump_m are moved towards the
                        far side with probability flying_p (random interpolation)
    motion smear      : each point shifted along the belt by U(0, v * t_acq)
    """

    name: str = "N2"
    sigma0_m: float = 0.15e-3
    z0_m: float = 0.25
    incidence_gain: float = 0.5
    dropout_theta_c_deg: float = 70.0
    dropout_width_deg: float = 5.0
    dropout_p0: float = 0.02
    flying_p: float = 0.3
    flying_jump_m: float = 2.0e-3
    quant_m: float = 0.0
    t_acq_s: float = 0.002

    @property
    def is_noiseless(self) -> bool:
        return self.sigma0_m == 0 and self.dropout_p0 == 0 and self.flying_p == 0 and self.dropout_theta_c_deg >= 90


class SensorModel(BaseModel):
    sensor_id: str = "sl_default"
    width: int = 1024
    height: int = 1024
    hfov_deg: float = 20.0
    focal_length_mm: float = 24.0
    near_m: float = 0.05
    far_m: float = 2.0
    working_range_m: tuple[float, float] = (0.15, 0.55)

    def camera(self) -> PinholeCamera:
        return PinholeCamera(self.width, self.height, self.hfov_deg)


class VisibilityParams(BaseModel):
    """Stage-1 visibility test (plan: theta < 70 deg, footprint < 0.2 mm, in range)."""

    theta_max_deg: float = 70.0
    footprint_max_m: float = 0.2e-3
    occlusion_eps_m: float = 0.1e-3
    n_samples_per_defect: int = 200
    density_ref_footprint_m: float = 0.1e-3
    vis_threshold: float = Field(0.1, description="V_j >= this counts the defect as visible (FNR_vis)")
