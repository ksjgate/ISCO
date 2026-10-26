"""Inspection-cell configuration (the optimisation variable)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..utils.hashing import content_hash
from .process import ROTATION_STAGES

RotationMode = Literal["none", "r90x1", "r180x1", "r90x3"]
TriggerMode = Literal["simultaneous", "staggered"]


class CameraSpec(BaseModel):
    """A camera on the hemisphere around the focus point (free, not necessarily on-grid)."""

    azimuth_deg: float
    elevation_deg: float
    distance_m: float
    slot_id: int | None = None


class InspectionConfig(BaseModel):
    """Camera slots (1-6 distinct (azimuth, elevation) cells) + process variables.

    * ``rotation``: rotation stage programme (none / 90 deg once / 180 deg once / 90 deg x3)
    * ``speed_level``: index into ``ProcessParams.speeds_mps``
    * ``trigger_mode``: ``simultaneous`` (one trigger, common shift) or ``staggered``
      (each camera fires at its own time -> per-view along-belt misregistration)
    * ``trigger_offset_m``: deliberate along-belt offset of the capture position
      (kept from the proposal; fixed to 0 in the default search space)
    """

    slots: list[int] = Field(min_length=1, max_length=6)
    rotation: RotationMode = "none"
    speed_level: int = 1
    trigger_mode: TriggerMode = "simultaneous"
    trigger_offset_m: float = 0.0

    @field_validator("slots")
    @classmethod
    def _unique_sorted(cls, v: list[int]) -> list[int]:
        return sorted(set(int(x) for x in v))

    @property
    def n_cam(self) -> int:
        return len(self.slots)

    @property
    def stages_deg(self) -> list[float]:
        return ROTATION_STAGES[self.rotation]

    @property
    def config_id(self) -> str:
        return "c_" + content_hash(self.model_dump(mode="json"), length=10)

    def short(self) -> str:
        return f"{self.slots}|{self.rotation}|v{self.speed_level}|{self.trigger_mode[:4]}"
