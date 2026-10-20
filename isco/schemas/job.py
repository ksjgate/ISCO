"""Render-worker job contract (filesystem queue, ADR-0002).

A job file is JSON ``{"job_id", "job_type", "created_at", "priority", "payload"}``.
``payload`` is validated by the model registered for ``job_type``. All paths are
absolute or relative to the repository root. World frame = ISCO world (belt top z=0,
see ``isco.geometry.transforms``); the worker converts to the Isaac stage frame.
"""
from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from .config import CameraSpec, InspectionConfig
from .process import ProcessParams
from .sensor import SensorModel


class InstanceRef(BaseModel):
    inst_id: str
    mesh_path: str                       # LabeledMesh .npz in the part rest frame
    T_world_part: list[list[float]]      # 4x4, ISCO world frame


class EmptyPayload(BaseModel):
    """Payload of job types that carry no parameters (ping, shutdown)."""


class SettlePayload(BaseModel):
    """Drop the part on the running belt and record where it comes to rest (ADR-0006).

    ``tilt_sigma_deg`` / ``ang_vel_sigma_dps`` perturb the initial orientation / spin
    (sensitivity runs); ``physics_dt`` overrides the 240 Hz physics step. ``None`` keeps
    the worker defaults (0, 0, 1/240 s)."""

    part_id: str
    mesh_path: str
    speeds_mps: list[float] = [0.1, 0.3, 0.6]
    n_drops: int = 100
    drop_height_m: float = 0.02
    lateral_range_m: float = 0.005
    yaw_sigma_deg: float = 3.0
    settle_time_s: float = 1.5
    seed: int = 0
    output_path: str
    tilt_sigma_deg: float | None = None
    ang_vel_sigma_dps: float | None = None
    physics_dt: float | None = None


class RenderSlotsPayload(BaseModel):
    """Render instances from grid slots and write ViewPixels (same format as embree)."""

    part_id: str
    instances: list[InstanceRef]
    slot_ids: list[int]
    grid: dict
    sensor: SensorModel
    focus_m: list[float]
    output_dir: str
    save_rgb: bool = False


class ExportOccluderPayload(BaseModel):
    """Dump conveyor render meshes near the focus point into ISCO world coordinates."""

    output_path: str
    radius_m: float = 1.0
    conveyor_asset: str = "ConveyorBelt_A06"


class VerifyConfigPayload(BaseModel):
    """Direct (non-cached) inline render of one configuration for E0.2 / top-k re-checks.

    For every instance, rotation stage and camera the part is moved to the pose it has
    when that camera fires (speed x trigger time + sampled jitter); applied offsets are
    written next to the ViewPixels.
    """

    part_id: str
    config: InspectionConfig
    cameras: list[CameraSpec]
    process: ProcessParams
    instances: list[InstanceRef]
    sensor: SensorModel
    focus_m: list[float]
    output_dir: str
    seed: int = 0


class CaptureRgbPayload(BaseModel):
    """RGB snapshots for demos/reports: an overview of the rig + each camera's view."""

    part_id: str
    mesh_path: str
    cameras: list[CameraSpec]
    sensor: SensorModel
    focus_m: list[float]
    output_dir: str
    overview: bool = True


PAYLOADS: dict[str, type[BaseModel]] = {
    "settle": SettlePayload,
    "render_slots": RenderSlotsPayload,
    "export_occluder": ExportOccluderPayload,
    "verify_config": VerifyConfigPayload,
    "capture_rgb": CaptureRgbPayload,
    "ping": EmptyPayload,
    "shutdown": EmptyPayload,
}

JobType = Literal["settle", "render_slots", "export_occluder", "verify_config", "capture_rgb", "ping", "shutdown"]


class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: "j_" + uuid.uuid4().hex[:10])
    job_type: JobType
    created_at: float = Field(default_factory=time.time)
    priority: int = 50
    payload: dict = {}

    def parsed_payload(self) -> BaseModel:
        return PAYLOADS[self.job_type].model_validate(self.payload)

    @classmethod
    def make(cls, job_type: str, payload: BaseModel | dict | None = None, priority: int = 50) -> "Job":
        if isinstance(payload, BaseModel):
            payload = payload.model_dump(mode="json")
        return cls(job_type=job_type, payload=payload or {}, priority=priority)


class JobResult(BaseModel):
    job_id: str
    job_type: str
    status: Literal["ok", "failed"]
    outputs: dict = {}
    warnings: list[str] = []
    error: str | None = None
    wall_s: float = 0.0
    worker: dict = {}
