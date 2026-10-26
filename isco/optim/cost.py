"""Cost model: camera count and cycle time (plan "지표" - T_cycle = L_pitch / v + T_rot)."""
from __future__ import annotations

from ..schemas.config import InspectionConfig
from ..schemas.process import ProcessParams


class CostModel:
    def __init__(self, process: ProcessParams, max_cams: int = 6, t_norm_s: float = 5.0):
        self.process = process
        self.max_cams = max_cams
        self.t_norm = t_norm_s

    def cycle_time(self, cfg: InspectionConfig) -> float:
        return self.process.cycle_time_s(cfg.speed_level, cfg.rotation)

    def objectives(self, cfg: InspectionConfig, fnr: float) -> dict:
        return {"fnr": float(fnr), "n_cam": cfg.n_cam, "cycle_time_s": self.cycle_time(cfg)}

    def normalised(self, obj: dict) -> list[float]:
        """Objectives in [0,1] for the hypervolume: (FNR, (n-1)/(N-1), T/T_norm)."""
        out = [obj["fnr"], (obj["n_cam"] - 1) / max(self.max_cams - 1, 1)]
        if self.process.cycle_time_in_cost:
            out.append(obj["cycle_time_s"] / self.t_norm)
        return out
