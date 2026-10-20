"""Configuration-level visibility metrics from the precomputed slot visibility.

Per defect j (S samples), a sample is *seen* by a view if it is visible, the
incidence is below ``theta_max``, its surface footprint ``z / (fx cos)`` is below
``footprint_max`` and the depth is inside the working range:

* ``V_j``        fraction of samples seen by >= 1 view   (V_area = mean_j V_j)
* ``V50``        fraction of defects with V_j >= 0.5
* ``V_density``  mean_j mean_s sum_views cos * (ref / footprint)^2 (expected samples per
                 reference pixel area; distance/incidence aware)
* ``V_multi``    mean_j fraction of samples seen by >= 2 views
* ``V_min_type`` min over defect types of the per-type mean V_j
* ``coverage_normal`` fraction of nominal-surface samples seen

Views of a configuration = its slots, plus for every rotation stage theta the slots
rotated by theta (``SlotGrid.rotate``), which is exact for the part (ADR-0003).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..geometry.camera_model import SlotGrid
from ..schemas.config import InspectionConfig
from ..schemas.sensor import SensorModel, VisibilityParams
from ..utils.io import read_json


def config_views(cfg: InspectionConfig, grid: SlotGrid) -> list[tuple[int, int, float]]:
    """(camera slot, cache slot, stage angle) for every capture of the configuration."""
    out = []
    for th in cfg.stages_deg:
        for s in cfg.slots:
            out.append((s, grid.rotate(s, th), th))
    return out


class VisibilityTable:
    def __init__(self, vis_npz: str | Path, roles: dict[str, str], types: dict[str, str],
                 params: VisibilityParams | None = None, sensor: SensorModel | None = None):
        z = np.load(vis_npz, allow_pickle=False)
        meta = read_json(Path(vis_npz).with_name("vis.json"))
        self.grid = SlotGrid.from_dict(meta["grid"])
        self.sensor = sensor or SensorModel.model_validate(meta["sensor"])
        self.params = params or VisibilityParams()
        self.ids = [str(x) for x in z["inst_ids"]]
        fx = self.sensor.camera().fx
        cos = z["cos"].astype(np.float32)
        zz = z["z"].astype(np.float32)
        p = self.params
        lo, hi = self.sensor.working_range_m
        with np.errstate(divide="ignore"):
            foot = zz / (fx * np.maximum(cos, 1e-6))
        self.ok = z["vis"] & (cos >= np.cos(np.radians(p.theta_max_deg))) & (foot <= p.footprint_max_m) & (zz >= lo) & (zz <= hi)
        self.dens = np.where(z["vis"], cos * (p.density_ref_footprint_m * fx / np.maximum(zz, 1e-6)) ** 2, 0.0).astype(np.float32)
        ncos = z["nominal_cos"].astype(np.float32)
        nz = z["nominal_z"].astype(np.float32)
        with np.errstate(divide="ignore"):
            nfoot = nz / (fx * np.maximum(ncos, 1e-6))
        self.nominal_ok = z["nominal_vis"] & (ncos >= np.cos(np.radians(p.theta_max_deg))) & (nfoot <= p.footprint_max_m)
        self.role = np.array([roles.get(i, "?") for i in self.ids])
        self.type = np.array([types.get(i, "?") for i in self.ids])

    @classmethod
    def for_set(cls, part_id: str, set_id: str, sensor_id: str = "sl_default", params: VisibilityParams | None = None):
        from ..datagen.variants import load_manifest
        from .raycast import vis_dir

        man = load_manifest(part_id, set_id)
        roles = {r["inst_id"]: r["role"] for r in man["instances"]}
        types = {r["inst_id"]: r["type"] for r in man["instances"]}
        return cls(vis_dir(part_id, set_id, sensor_id) / "vis.npz", roles, types, params)

    def per_defect(self, cfg: InspectionConfig, roles=("search",)) -> dict[str, np.ndarray]:
        views = sorted({v for _, v, _ in config_views(cfg, self.grid)})
        sel = np.isin(self.role, roles)
        ok = self.ok[sel][:, views, :]                     # (J, V, S)
        cnt = ok.sum(axis=1)
        Vj = (cnt >= 1).mean(axis=1)
        return {"ids": np.array(self.ids)[sel], "type": self.type[sel], "V": Vj,
                "V_multi": (cnt >= 2).mean(axis=1), "density": self.dens[sel][:, views, :].sum(axis=1).mean(axis=1)}

    def metrics(self, cfg: InspectionConfig, roles=("search",)) -> dict:
        d = self.per_defect(cfg, roles)
        views = sorted({v for _, v, _ in config_views(cfg, self.grid)})
        types = sorted(set(d["type"]))
        per_type = {t: float(d["V"][d["type"] == t].mean()) for t in types}
        cov = self.nominal_ok[views].any(axis=0).mean()
        return {"V_area": float(d["V"].mean()), "V50": float((d["V"] >= 0.5).mean()),
                "V_density": float(d["density"].mean()), "V_multi": float(d["V_multi"].mean()),
                "V_min_type": float(min(per_type.values())) if per_type else 0.0,
                "V_per_type": per_type, "coverage_normal": float(cov), "n_views": len(views)}
