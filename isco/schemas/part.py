"""Part registry entry (written by ``isco parts prepare``)."""
from __future__ import annotations

from pydantic import BaseModel


class PartSpec(BaseModel):
    part_id: str
    source: str
    license: str
    redistributable: bool
    source_path: str
    target_max_dim_m: float
    mesh_coarse: str          # rest-frame coarse mesh (booleans are applied here)
    mesh_fine: str            # rest-frame remeshed nominal mesh
    extents_m: list[float]
    focus_m: list[float]      # world focus point (bbox centre of the nominal part)
    bounding_radius_m: float
    rest_info: dict = {}
    stats: dict = {}
