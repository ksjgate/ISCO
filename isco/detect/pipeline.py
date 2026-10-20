"""Stage-2 evaluation: one inspection configuration -> FNR@1%FPR (+CI) and diagnostics.

Workers (spawned processes) compose each instance's merged cloud from the slot cache,
register it, and compute point features / CAD distances / coverage; the main process
builds the memory bank on the GPU, scores points, aggregates image scores, sets the
threshold on the validation normals and reports metrics on the *search* defects (what
the optimiser sees) and the held-out *report* defects (what we publish).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel

from ..geometry.mesh_io import LabeledMesh
from ..render.slot_cache import SlotCache, cache_dir
from ..schemas.config import InspectionConfig
from ..schemas.process import ProcessParams
from ..schemas.sensor import NoiseModel
from ..utils.hashing import content_hash
from ..utils.io import repo_root
from ..utils.logging import get_logger
from ..utils.seeds import core_order_env
from . import scorer

log = get_logger(__name__)

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class DetectorConfig(BaseModel):
    kind: Literal["fpfh_bank", "cad_distance", "dinov2_bank"] = "fpfh_bank"
    dino_model: str = "facebook/dinov2-small"
    dino_size: int = 224
    voxel_m: float = 0.4e-3
    normal_radius_m: float = 1.0e-3
    fpfh_radii_m: list[float] = [1.0e-3, 2.0e-3, 4.0e-3]
    fpfh_max_nn: int = 100
    xyz_weight: float = 0.2
    coreset_ratio: float = 0.03
    coreset_max: int = 10000
    bank_presample: int = 200_000
    smooth_k: int = 8
    image_agg: Literal["max", "topk"] = "max"
    topk_frac: float = 0.005
    use_coverage: bool = True
    coverage_radius_m: float = 0.6e-3
    coverage_smooth_m: float = 1.0e-3
    template_samples: int = 30000
    registration: Literal["icp", "oracle", "none"] = "icp"
    stage_registration: bool = True  # register each rotation stage separately before merging
    sor_neighbors: int = 20          # statistical outlier removal (0 disables)
    sor_std_ratio: float = 2.0
    fpr: float = 0.01
    bootstrap_B: int = 500
    keep_point_scores: bool = False
    min_defect_points: int = 30

    @property
    def detector_id(self) -> str:
        return f"{self.kind}_{content_hash(self.model_dump(mode='json'), 8)}"


# ------------------------------------------------------------------------------------
_W: dict = {}


def _init(part_id: str, set_id: str, sensor_id: str, det_json: str, proc_json: str, noise_json: str) -> None:
    from ..utils.seeds import pin_worker

    pin_worker(1)          # one core per worker: open3d's TBB/OpenMP pools follow the affinity mask
    from ..datagen.parts import load_part_mesh
    from .cad_distance import CadDistance
    from .coverage_scorer import CoverageScorer
    from .merge import Registrar

    det = DetectorConfig.model_validate_json(det_json)
    tmpl = load_part_mesh(part_id, "fine")
    tm = tmpl.to_trimesh()
    import trimesh

    pts, fid = trimesh.sample.sample_surface_even(tm, det.template_samples, seed=0)
    nrm = np.asarray(tm.face_normals)[fid]
    _W.update(cache=SlotCache.open(part_id, set_id, sensor_id), det=det,
              process=ProcessParams.model_validate_json(proc_json), noise=NoiseModel.model_validate_json(noise_json),
              registrar=Registrar(pts, nrm), template_pts=pts,
              coverage=CoverageScorer(pts, det.coverage_radius_m, det.coverage_smooth_m) if det.use_coverage else None,
              cad=CadDistance(tmpl) if det.kind == "cad_distance" else None)


def _work(task: tuple[dict, str, int]) -> dict:
    from .features import knn_indices, point_features
    from .merge import compose_instance

    inst, cfg_json, seed = task
    det: DetectorConfig = _W["det"]
    cfg = InspectionConfig.model_validate_json(cfg_json)
    t0 = time.time()
    mc = compose_instance(_W["cache"], inst, cfg, _W["process"], _W["noise"], seed, det.voxel_m,
                          _W["registrar"], det.registration,
                          (det.sor_neighbors, det.sor_std_ratio) if det.sor_neighbors > 0 else None,
                          keep_views=det.kind == "dinov2_bank", stage_registration=det.stage_registration)
    out = {"inst_id": inst["inst_id"], "n_points": int(len(mc.xyz)), "n_raw": mc.n_raw,
           "reg_fitness": mc.reg_fitness, "reg_rmse": mc.reg_rmse,
           "n_defect_points": int((mc.label >= 2).sum()), "label": mc.label}
    if len(mc.xyz) < 10:
        out.update(feats=None, cad=None, nbr=None, observed=None, xyz=mc.xyz, imgs=None)
        return out
    if det.kind == "fpfh_bank":
        f = point_features(mc.xyz, mc.cam_dir, det.normal_radius_m, det.fpfh_radii_m, det.xyz_weight, det.fpfh_max_nn)
        out["feats"] = f.astype(np.float16)
    elif det.kind == "dinov2_bank":
        from .features import pixel_to_image_coords, view_normal_image

        cam = _W["cache"].camera
        imgs, coords = [], np.zeros((len(mc.xyz), 2), np.float32)
        for vi, (vp, _slot) in enumerate(mc.views):
            img, crop = view_normal_image(vp, cam, det.dino_size)
            imgs.append(img)
            m = mc.rep_view == vi
            coords[m] = pixel_to_image_coords(mc.rep_pix[m], cam, crop, det.dino_size)
        out.update(imgs=np.stack(imgs), rep_view=mc.rep_view.astype(np.int16), coords=coords.astype(np.float16),
                   xyz=mc.xyz)
    else:
        out["cad"] = _W["cad"].point_scores(mc.xyz)
    out["nbr"] = knn_indices(mc.xyz, det.smooth_k) if det.smooth_k > 1 else None
    out["observed"] = _W["coverage"].observed(mc.xyz) if _W["coverage"] is not None else None
    if det.keep_point_scores:
        out["xyz"] = mc.xyz
    out["work_s"] = round(time.time() - t0, 3)
    return out


def _agg(point_scores: np.ndarray, nbr: np.ndarray | None, det: DetectorConfig) -> tuple[float, np.ndarray]:
    if len(point_scores) == 0:
        return 0.0, point_scores
    s = point_scores[nbr].mean(axis=1) if nbr is not None else point_scores
    if det.image_agg == "max":
        return float(s.max()), s
    k = max(1, int(round(det.topk_frac * len(s))))
    return float(np.sort(s)[-k:].mean()), s


class DetectionPipeline:
    def __init__(self, part_id: str, set_id: str, detector: DetectorConfig | None = None,
                 process: ProcessParams | None = None, noise: NoiseModel | None = None,
                 sensor_id: str = "sl_default", n_workers: int = 20, device: str = "cuda"):
        from ..datagen.variants import load_manifest

        self.part_id, self.set_id, self.sensor_id = part_id, set_id, sensor_id
        self.det = detector or DetectorConfig()
        self.process = process or ProcessParams()
        self.noise = noise or NoiseModel()
        self.n_workers = n_workers
        self.device = device
        self.man = load_manifest(part_id, set_id)
        self.insts = self.man["instances"]
        self._pool = None
        self._vis = None
        if not (cache_dir(part_id, set_id, sensor_id) / "cache.json").exists():
            raise FileNotFoundError(f"slot cache missing for {part_id}/{set_id}; run `isco cache build`")

    # -- resources -------------------------------------------------------------------
    def pool(self):
        if self._pool is None:
            ctx = mp.get_context("spawn")
            # one compute thread per worker process: open3d (OpenMP), BLAS and scipy would
            # otherwise each start one thread per core in every worker (oversubscription
            # made a 212-instance evaluation ~20x slower). Spawned children read these
            # variables at start-up, so they are set around the Pool creation only.
            saved = {k: os.environ.get(k) for k in _THREAD_VARS}
            for k in _THREAD_VARS:
                os.environ[k] = "1"
            try:
                with core_order_env():
                    self._pool = ctx.Pool(self.n_workers, initializer=_init, initargs=(
                        self.part_id, self.set_id, self.sensor_id, self.det.model_dump_json(),
                        self.process.model_dump_json(), self.noise.model_dump_json()))
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def visibility(self):
        if self._vis is None:
            from ..visibility.raycast import vis_dir
            from ..visibility.surrogate import VisibilityTable

            if (vis_dir(self.part_id, self.set_id, self.sensor_id) / "vis.npz").exists():
                self._vis = VisibilityTable.for_set(self.part_id, self.set_id, self.sensor_id)
        return self._vis

    # -- evaluation -------------------------------------------------------------------
    def evaluate(self, cfg: InspectionConfig, seed: int = 0, roles=("bank", "val", "search", "report", "unreachable")) -> dict:
        det = self.det
        t0 = time.time()
        # bank instances first: as soon as all of them are back, the memory bank / coverage
        # statistics are fitted while the workers keep processing the remaining
        # instances, which are then scored as they arrive (main-process work overlaps the
        # workers; scored features are dropped right away to bound memory)
        order = {"bank": 0, "val": 1, "search": 2, "report": 3, "unreachable": 4}
        todo = sorted([r for r in self.insts if r["role"] in roles], key=lambda r: order.get(r["role"], 9))
        tasks = [(r, cfg.model_dump_json(), seed) for r in todo]
        role_of = {r["inst_id"]: r["role"] for r in todo}
        bank_ids = {r["inst_id"] for r in todo if r["role"] == "bank"}
        res, pending = {}, []
        img, pts_scores, cov = {}, {}, {}
        state = {"score_fn": None, "cs": None, "bank_size": 0}
        t_score = 0.0

        def fit_bank():
            nonlocal t_score
            t1 = time.time()
            bank_out = [res[i] for i in bank_ids if i in res]
            if det.kind in ("fpfh_bank", "dinov2_bank"):
                from .memory_bank import MemoryBank

                if det.kind == "fpfh_bank":
                    feat = lambda o: o["feats"].astype(np.float32) if o.get("feats") is not None else None  # noqa: E731
                else:
                    dino = self._dino_model()

                    def feat(o):
                        if o.get("imgs") is None:
                            return None
                        f = dino(o["imgs"], o["rep_view"], o["coords"].astype(np.float32))
                        if det.xyz_weight > 0:
                            f = np.concatenate([f, o["xyz"] * 1e3 * det.xyz_weight], axis=1)
                        return f.astype(np.float32)
                feats = [f for f in (feat(o) for o in bank_out) if f is not None]
                bank = MemoryBank(det.coreset_ratio, det.coreset_max, self.device, seed, det.bank_presample).fit(feats)
                state["score_fn"] = lambda o: (bank.score(f) if (f := feat(o)) is not None else np.zeros(0))
                state["bank_size"] = bank.size
            else:
                state["score_fn"] = lambda o: o["cad"] if o.get("cad") is not None else np.zeros(0)
            if det.use_coverage:
                obs_bank = [o["observed"] for o in bank_out if o.get("observed") is not None]
                if obs_bank:
                    cs = self._coverage_scorer()
                    cs.fit(obs_bank)
                    state["cs"] = cs
            for o in bank_out:                      # bank features are no longer needed
                o.pop("feats", None)
                o.pop("imgs", None)
            t_score += time.time() - t1

        def score(o):
            nonlocal t_score
            t1 = time.time()
            i = o["inst_id"]
            s_img, s_pt = _agg(state["score_fn"](o), o.get("nbr"), det)
            img[i] = s_img
            if det.keep_point_scores:
                pts_scores[i] = s_pt
            if state["cs"] is not None:
                cov[i] = state["cs"].image_score(o["observed"]) if o.get("observed") is not None else 0.0
            for k in ("feats", "imgs", "coords", "cad", "nbr", "observed"):
                o.pop(k, None)
            t_score += time.time() - t1

        for out in self.pool().imap(_work, tasks, chunksize=2):
            res[out["inst_id"]] = out
            if role_of[out["inst_id"]] == "bank":
                if bank_ids.issubset(res):
                    fit_bank()
                    for o in pending:
                        score(o)
                    pending = []
                continue
            if state["score_fn"] is None:
                pending.append(out)
            else:
                score(out)
        if state["score_fn"] is None:            # no bank role requested
            fit_bank()
            for o in pending:
                score(o)
        t_work = time.time() - t0
        by_role = {role: [r for r in self.insts if r["role"] == role and r["inst_id"] in res] for role in roles}
        bank_size = state["bank_size"]
        # combine (z-score against validation normals)
        val_ids = [r["inst_id"] for r in by_role.get("val", [])]
        f_val = np.array([img[i] for i in val_ids])
        mu_f, sd_f = float(f_val.mean()), float(f_val.std() + 1e-12)
        final = {i: (s - mu_f) / sd_f for i, s in img.items()}
        if cov:
            c_val = np.array([cov[i] for i in val_ids])
            mu_c, sd_c = float(c_val.mean()), float(c_val.std() + 1e-9)
            final = {i: max(final[i], (cov[i] - mu_c) / sd_c) for i in final}
        # visibility per defect
        vis = self.visibility()
        Vmap = {}
        if vis is not None:
            pdft = vis.per_defect(cfg, roles=("search", "report", "unreachable"))
            Vmap = dict(zip(pdft["ids"], pdft["V"]))
        val_scores = np.array([final[i] for i in val_ids])

        def records(role):
            out = []
            for r in by_role.get(role, []):
                i = r["inst_id"]
                out.append({"inst_id": i, "score": final[i], "feat_score": img[i], "cov_score": cov.get(i),
                            "type": r["type"], "difficulty": r.get("difficulty"), "tier": r.get("tier"),
                            "V": float(Vmap[i]) if i in Vmap else np.nan,
                            "n_defect_points": res[i]["n_defect_points"], "n_points": res[i]["n_points"]})
            return out

        result = {"val_scores": val_scores.round(5).tolist(),
                  "config": cfg.model_dump(mode="json"), "config_id": cfg.config_id, "seed": seed,
                  "part_id": self.part_id, "set_id": self.set_id, "detector_id": det.detector_id,
                  "noise": self.noise.name, "process_id": self.process.process_id, "n_cam": cfg.n_cam}
        for role in ("search", "report"):
            recs = records(role)
            if recs:
                m = scorer.summarize(val_scores, recs, det.fpr, det.bootstrap_B, seed)
                m["fnr_detectable"] = scorer.fnr_at_fpr(val_scores, [d["score"] for d in recs if d["n_defect_points"] >= det.min_defect_points])
                result[role] = m
                result[f"{role}_defects"] = recs
        if by_role.get("unreachable"):
            recs = records("unreachable")
            tau = scorer.threshold_at_fpr(val_scores, det.fpr)
            result["unreachable"] = {"detected_frac": float(np.mean([d["score"] > tau for d in recs])), "n": len(recs)}
        if det.keep_point_scores and "report" in result:
            dps, dms = [], []
            for r in by_role["report"]:
                i = r["inst_id"]
                if i in pts_scores and len(pts_scores[i]):
                    dps.append(pts_scores[i])
                    dms.append(res[i]["label"] >= 2)
            nps = np.concatenate([pts_scores[i] for i in val_ids if i in pts_scores]) if val_ids else np.zeros(0)
            result["report"]["aupro_vis"] = scorer.aupro(dps, dms, nps)
        diag_ids = [r["inst_id"] for role in roles for r in by_role[role]]
        result["diagnostics"] = {
            "points_mean": float(np.mean([res[i]["n_points"] for i in diag_ids])),
            "raw_points_mean": float(np.mean([res[i]["n_raw"] for i in diag_ids])),
            "reg_fitness_mean": float(np.mean([res[i]["reg_fitness"] for i in diag_ids])),
            "reg_rmse_mean_m": float(np.mean([res[i]["reg_rmse"] for i in diag_ids])),
            "bank_size": bank_size, "val_mean_feat": mu_f, "val_sd_feat": sd_f,
        }
        result["timing"] = {"work_s": round(t_work, 2), "score_s": round(t_score, 2), "total_s": round(time.time() - t0, 2)}
        return result

    _cov_cache = None
    _dino = None

    def _dino_model(self):
        if self._dino is None:
            from .features import DinoFeatures

            self._dino = DinoFeatures(self.det.dino_model, self.device, self.det.dino_size)
        return self._dino

    def _coverage_scorer(self):
        if self._cov_cache is None:
            import trimesh

            from ..datagen.parts import load_part_mesh
            from .coverage_scorer import CoverageScorer

            tm = load_part_mesh(self.part_id, "fine").to_trimesh()
            pts, _ = trimesh.sample.sample_surface_even(tm, self.det.template_samples, seed=0)
            self._cov_cache = CoverageScorer(pts, self.det.coverage_radius_m, self.det.coverage_smooth_m)
        return self._cov_cache
