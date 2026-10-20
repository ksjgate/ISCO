"""Discrete configuration space over the slot grid (plan "최적화기").

Constraints
-----------
* slot validity: camera above the belt, outside the conveyor body, unoccluded line of
  sight to the focus point (occluder raycast), within the sensor working range;
* no two cameras in the same (azimuth, elevation) cell;
* minimum angle between viewing directions (default 30 deg, plan) and a minimum
  camera-centre distance (default 0.08 m, physical sensor size);
* 1-6 cameras; rotation stage, speed level and trigger mode are categorical.

Optuna encoding: ``n_cam``, ``cam0..cam5`` (categorical over valid slots), ``rotation``,
``speed``, ``trigger``. Decoding *repairs* infeasible proposals by dropping cameras
that violate a constraint against an earlier camera (recorded as ``repaired``).
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from ..geometry.camera_model import SlotGrid
from ..render.occluder import Occluder
from ..schemas.config import InspectionConfig


class SpaceParams(BaseModel):
    max_cams: int = 6
    rotations: list[str] = ["none", "r90x1", "r180x1", "r90x3"]
    speed_levels: list[int] = [0, 1, 2]
    trigger_modes: list[str] = ["simultaneous", "staggered"]
    min_angle_deg: float = 30.0
    min_cam_distance_m: float = 0.08


class SearchSpace:
    def __init__(self, grid: SlotGrid, focus: np.ndarray, occluder: Occluder | None = None,
                 params: SpaceParams | None = None, working_range=(0.15, 0.55)):
        self.grid = grid
        self.focus = np.asarray(focus, dtype=float)
        self.p = params or SpaceParams()
        self.occluder = occluder or Occluder.belt_plane()
        self.eyes = np.stack([grid.eye(k, self.focus) for k in range(grid.n_slots)])
        dirs = self.focus[None] - self.eyes
        self.dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        self.valid_slots, self.invalid_reason = self._validity(working_range)

    def _validity(self, working_range):
        import open3d as o3d

        reasons = {}
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(self.occluder.vertices.astype(np.float32)),
                            o3d.core.Tensor(self.occluder.faces.astype(np.uint32)))
        d = np.linalg.norm(self.focus[None] - self.eyes, axis=1)
        rays = np.concatenate([self.eyes, self.dirs], axis=1).astype(np.float32)
        t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        inside = self.occluder.point_inside_solid_bbox(self.eyes)
        valid = []
        for k in range(self.grid.n_slots):
            if self.eyes[k, 2] <= 0.0:
                reasons[k] = "below belt"
            elif inside[k]:
                reasons[k] = "inside conveyor body"
            elif np.isfinite(t[k]) and t[k] < d[k] - 0.03:
                reasons[k] = "line of sight blocked by conveyor"
            elif not (working_range[0] <= d[k] <= working_range[1]):
                reasons[k] = "outside working range"
            else:
                valid.append(k)
        return valid, reasons

    # -- constraints --------------------------------------------------------------
    def compatible(self, a: int, b: int) -> bool:
        sa, sb = self.grid.slot(a), self.grid.slot(b)
        if sa.cell == sb.cell:
            return False
        ang = np.degrees(np.arccos(np.clip(self.dirs[a] @ self.dirs[b], -1, 1)))
        if ang < self.p.min_angle_deg:
            return False
        return np.linalg.norm(self.eyes[a] - self.eyes[b]) >= self.p.min_cam_distance_m

    def feasible(self, slots: list[int]) -> bool:
        return all(s in self.valid_slots for s in slots) and all(
            self.compatible(a, b) for i, a in enumerate(slots) for b in slots[i + 1:])

    def repair(self, slots: list[int]) -> list[int]:
        kept: list[int] = []
        for s in slots:
            if s in self.valid_slots and s not in kept and all(self.compatible(s, k) for k in kept):
                kept.append(s)
        return kept

    # -- sampling ------------------------------------------------------------------
    def random_slots(self, n: int, rng: np.random.Generator, tries: int = 200) -> list[int]:
        for _ in range(tries):
            kept: list[int] = []
            order = rng.permutation(self.valid_slots)
            for s in order:
                if all(self.compatible(int(s), k) for k in kept):
                    kept.append(int(s))
                if len(kept) == n:
                    return kept
        return kept

    def random_config(self, rng: np.random.Generator, n: int | None = None) -> InspectionConfig:
        n = n or int(rng.integers(1, self.p.max_cams + 1))
        return InspectionConfig(slots=self.random_slots(n, rng), rotation=str(rng.choice(self.p.rotations)),
                                speed_level=int(rng.choice(self.p.speed_levels)),
                                trigger_mode=str(rng.choice(self.p.trigger_modes)))

    # -- optuna encoding -------------------------------------------------------------
    def suggest(self, trial) -> tuple[InspectionConfig, bool]:
        n = trial.suggest_int("n_cam", 1, self.p.max_cams)
        raw = [trial.suggest_categorical(f"cam{i}", self.valid_slots) for i in range(self.p.max_cams)]
        slots = self.repair(raw[:n])
        repaired = len(slots) != n
        cfg = InspectionConfig(slots=slots, rotation=trial.suggest_categorical("rotation", self.p.rotations),
                               speed_level=trial.suggest_categorical("speed", self.p.speed_levels),
                               trigger_mode=trial.suggest_categorical("trigger", self.p.trigger_modes))
        return cfg, repaired

    def to_params(self, cfg: InspectionConfig, rng: np.random.Generator | None = None) -> dict:
        """Optuna params reproducing ``cfg`` (unused camera slots filled arbitrarily)."""
        rng = rng or np.random.default_rng(0)
        cams = list(cfg.slots) + [int(rng.choice(self.valid_slots)) for _ in range(self.p.max_cams - cfg.n_cam)]
        p = {f"cam{i}": int(c) for i, c in enumerate(cams)}
        p.update(n_cam=cfg.n_cam, rotation=cfg.rotation, speed=cfg.speed_level, trigger=cfg.trigger_mode)
        return p

    # -- structured baselines -------------------------------------------------------
    def ring(self, n: int, el_deg: float, dist_m: float, az0_deg: float = 0.0) -> list[int]:
        out = []
        for i in range(n):
            az = (az0_deg + 360.0 * i / n) % 360.0
            out.append(self.grid.nearest_slot(az, el_deg, dist_m))
        return self.repair(out)

    def baselines(self, speed_level: int = 1) -> dict[str, InspectionConfig]:
        d0 = self.grid.distances_m[0]
        top = self.grid.elevations_deg[-1]
        base = {
            "top_1cam": InspectionConfig(slots=self.ring(1, top, d0), speed_level=speed_level),
            "ring2_el60": InspectionConfig(slots=self.ring(2, 60, d0), speed_level=speed_level),
            "ring3_el40": InspectionConfig(slots=self.ring(3, 40, d0), speed_level=speed_level),
            "ring4_el40": InspectionConfig(slots=self.ring(4, 40, d0), speed_level=speed_level),
            "ring6_el40": InspectionConfig(slots=self.ring(6, 40, d0), speed_level=speed_level),
            "top_1cam_r90x3": InspectionConfig(slots=self.ring(1, top, d0), rotation="r90x3", speed_level=speed_level),
            "ring2_el40_r90x1": InspectionConfig(slots=self.ring(2, 40, d0), rotation="r90x1", speed_level=speed_level),
        }
        return {k: v for k, v in base.items() if v.slots}
