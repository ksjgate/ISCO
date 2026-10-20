"""Inline process model (ADR-0006): how belt speed, trigger scheme and the rotation
stage enter the evaluation, and the cycle-time cost constants (plan, 실험 프로토콜)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

#: rotation stage -> part rotation angles (deg) at which the cameras capture
ROTATION_STAGES: dict[str, list[float]] = {
    "none": [0.0],
    "r90x1": [0.0, 90.0],
    "r180x1": [0.0, 180.0],
    "r90x3": [0.0, 90.0, 180.0, 270.0],
}


class ProcessParams(BaseModel):
    process_id: str = "inline_default"
    mode: Literal["inline", "static"] = "inline"
    speeds_mps: list[float] = [0.1, 0.3, 0.6]
    pitch_m: float = 0.3
    trigger_sigma_s: float = 0.005
    # rotation stage: total extra time per part (stop + turns + captures), protocol table
    rotation_time_s: dict[str, float] = {"none": 0.0, "r90x1": 0.9, "r180x1": 1.1, "r90x3": 1.7}
    rot_reposition_sigma_m: float = 0.5e-3
    rot_yaw_sigma_deg: float = 1.0
    register_icp: bool = True
    cycle_time_in_cost: bool = True

    def speed(self, level: int) -> float:
        if self.mode == "static":
            return 0.0
        return float(self.speeds_mps[level])

    def belt_time_s(self, level: int) -> float:
        v = float(self.speeds_mps[level])
        return self.pitch_m / v

    def cycle_time_s(self, speed_level: int, rotation: str) -> float:
        """T_cycle = L_pitch / v + T_rot (plan 지표; range ~[0.5, 4.7] s)."""
        return self.belt_time_s(speed_level) + self.rotation_time_s[rotation]

    @classmethod
    def static(cls) -> "ProcessParams":
        return cls(process_id="static_capture", mode="static", trigger_sigma_s=0.0, rot_reposition_sigma_m=0.0,
                   rot_yaw_sigma_deg=0.0, cycle_time_in_cost=False)
