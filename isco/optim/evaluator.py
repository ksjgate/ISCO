"""Configuration evaluator: stage 1 (visibility surrogate, ~ms) and stage 2 (detector in
the loop, cached). Entry point of the "one config in -> FNR + cost out" gate."""
from __future__ import annotations

import time

import numpy as np

from .. import __version__
from ..datagen.parts import load_part
from ..datagen.variants import load_manifest
from ..detect.pipeline import DetectionPipeline, DetectorConfig
from ..geometry.camera_model import SlotGrid
from ..render.occluder import Occluder
from ..render.slot_cache import SlotCache
from ..schemas.config import InspectionConfig
from ..schemas.process import ProcessParams
from ..schemas.sensor import NoiseModel
from ..visibility.surrogate import VisibilityTable
from .cache import EvalCache
from .cost import CostModel
from .space import SearchSpace, SpaceParams

#: bump when stage-2 semantics change so cached evaluations are not reused
EVAL_VERSION = "2026-10-20.1"


class ConfigEvaluator:
    def __init__(self, part_id: str, set_id: str = "mvp", detector: DetectorConfig | None = None,
                 process: ProcessParams | None = None, noise: NoiseModel | None = None, sensor_id: str = "sl_default",
                 n_workers: int = 20, use_cache: bool = True, space_params: SpaceParams | None = None,
                 surrogate_metric: str = "V_density"):
        self.part_id, self.set_id, self.sensor_id = part_id, set_id, sensor_id
        self.detector = detector or DetectorConfig()
        self.process = process or ProcessParams()
        self.noise = noise or NoiseModel()
        self.n_workers = n_workers
        self.use_cache = use_cache
        self.surrogate_metric = surrogate_metric
        self.part = load_part(part_id)
        self.manifest = load_manifest(part_id, set_id)
        cache = SlotCache.open(part_id, set_id, sensor_id)
        self.grid: SlotGrid = cache.grid
        self.vis = VisibilityTable.for_set(part_id, set_id, sensor_id)
        self.cost = CostModel(self.process)
        self.space = SearchSpace(self.grid, np.asarray(self.part.focus_m), Occluder.default(), space_params,
                                 cache.sensor.working_range_m)
        self.cache = EvalCache()
        self._pipe: DetectionPipeline | None = None
        self.n_stage2_computed = 0

    # ------------------------------------------------------------------------------
    def surrogate(self, cfg: InspectionConfig) -> dict:
        m = self.vis.metrics(cfg, roles=("search",))
        obj = self.cost.objectives(cfg, 1.0 - m["V_area"])
        return {"config": cfg.model_dump(mode="json"), "config_id": cfg.config_id, "stage": 1,
                "surrogate": m, "objectives": obj, "n_cam": cfg.n_cam}

    def _key(self, cfg: InspectionConfig, seed: int) -> str:
        return self.cache.key(v=EVAL_VERSION, cfg=cfg.model_dump(mode="json"), part=self.part_id, set=self.set_id,
                              man=self.manifest["spec_hash"], sensor=self.sensor_id,
                              process=self.process.model_dump(mode="json"), noise=self.noise.model_dump(mode="json"),
                              det=self.detector.model_dump(mode="json"), seed=seed)

    def pipeline(self) -> DetectionPipeline:
        if self._pipe is None:
            self._pipe = DetectionPipeline(self.part_id, self.set_id, self.detector, self.process, self.noise,
                                           self.sensor_id, self.n_workers)
        return self._pipe

    def evaluate(self, cfg: InspectionConfig, seed: int = 0) -> dict:
        key = self._key(cfg, seed)
        if self.use_cache:
            hit = self.cache.get(self.part_id, key)
            if hit is not None:
                hit["cache_hit"] = True
                return hit
        t0 = time.time()
        r = self.pipeline().evaluate(cfg, seed)
        s1 = self.surrogate(cfg)
        rec = {
            "config": cfg.model_dump(mode="json"), "config_id": cfg.config_id, "stage": 2, "seed": seed,
            "part_id": self.part_id, "set_id": self.set_id, "detector_id": self.detector.detector_id,
            "noise": self.noise.name, "process_id": self.process.process_id, "process_mode": self.process.mode,
            "n_cam": cfg.n_cam, "objectives": self.cost.objectives(cfg, r["search"]["fnr"]),
            "objectives_report": self.cost.objectives(cfg, r["report"]["fnr"]) if "report" in r else None,
            "surrogate": s1["surrogate"], "search": r.get("search"), "report": r.get("report"),
            "unreachable": r.get("unreachable"), "diagnostics": r.get("diagnostics"), "timing": r.get("timing"),
            "search_defects": r.get("search_defects"), "report_defects": r.get("report_defects"),
            "val_scores": r.get("val_scores"),
            "eval_s": round(time.time() - t0, 2), "isco_version": __version__, "eval_version": EVAL_VERSION,
            "cache_key": key,
        }
        if self.use_cache:
            self.cache.put(self.part_id, key, rec)
        self.n_stage2_computed += 1
        rec["cache_hit"] = False
        return rec

    def objective_vector(self, rec: dict, use_report: bool = False) -> list[float]:
        obj = rec["objectives_report"] if use_report else rec["objectives"]
        return self.cost.normalised(obj)

    def close(self) -> None:
        if self._pipe is not None:
            self._pipe.close()
            self._pipe = None
