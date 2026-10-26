"""Two-stage configuration search (plan "최적화기", protocol E3/E8).

Stage 1 (surrogate, ~ms/config): Optuna NSGA-II on (1 - V_area, n_cam, T_cycle) plus
random and structured baselines -> a candidate pool; ``shortlist`` keeps the top
configurations by the surrogate per camera count (diversity) plus random ones.

Stage 2 (detector in the loop, budget B evaluations), methods:

``bo_warm``     multi-objective TPE (MOTPE) warm-started with the stage-1 shortlist
``bo_cold``     MOTPE from scratch
``nsga2``       Optuna NSGA-II
``random``      uniformly random feasible configurations
``greedy_vis``  coverage-greedy: nested sets n=1..6 adding the slot with the largest
                visibility gain (set-cover view planning), each evaluated once
``greedy_fnr``  forward selection with the detector (candidates pruned to the top-k
                visibility gains per step)

Objectives are minimised: (FNR_search, n_cam, cycle time) - cycle time is dropped when
the process does not count it (static baseline, E4).
"""
from __future__ import annotations

import time

import numpy as np

from ..schemas.config import InspectionConfig
from ..utils.logging import get_logger
from .evaluator import ConfigEvaluator
from .pareto import hypervolume, nondominated_mask

log = get_logger(__name__)


def _quiet_optuna():
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


class SearchRunner:
    def __init__(self, ev: ConfigEvaluator, store=None, fixed: dict | None = None):
        """``fixed`` pins process variables, e.g. ``{"rotation": "none", "speed": 1}``."""
        self.ev = ev
        self.store = store
        self.fixed = fixed or {}

    # -- helpers ------------------------------------------------------------------
    def _apply_fixed(self, cfg: InspectionConfig) -> InspectionConfig:
        d = cfg.model_dump()
        if "rotation" in self.fixed:
            d["rotation"] = self.fixed["rotation"]
        if "speed" in self.fixed:
            d["speed_level"] = self.fixed["speed"]
        if "trigger" in self.fixed:
            d["trigger_mode"] = self.fixed["trigger"]
        return InspectionConfig(**d)

    def _objs(self, rec: dict) -> tuple:
        o = rec["objectives"]
        if self.ev.process.cycle_time_in_cost:
            return (o["fnr"], o["n_cam"], o["cycle_time_s"])
        return (o["fnr"], o["n_cam"])

    def _log(self, rec: dict, method: str, seed: int, idx: int, extra: dict | None = None) -> dict:
        row = {**rec, "method": method, "search_seed": seed, "trial": idx, **(extra or {})}
        if self.store is not None:
            self.store.append(row)
        return row

    # -- stage 1 ----------------------------------------------------------------------
    def stage1(self, n_trials: int = 1500, n_random: int = 500, seed: int = 0) -> list[dict]:
        optuna = _quiet_optuna()
        ev, space = self.ev, self.ev.space
        seen: dict[str, dict] = {}

        def add(cfg: InspectionConfig, src: str):
            cfg = self._apply_fixed(cfg)
            if not cfg.slots or cfg.config_id in seen:
                return seen.get(cfg.config_id)
            r = ev.surrogate(cfg)
            r["source"] = src
            seen[cfg.config_id] = r
            return r

        for name, cfg in space.baselines().items():
            add(cfg, f"baseline:{name}")
        rng = np.random.default_rng(seed)
        for _ in range(n_random):
            add(space.random_config(rng), "random")

        def objective(trial):
            cfg, _ = space.suggest(trial)
            if not cfg.slots:
                return 1.0, 7.0, 10.0
            r = add(cfg, "nsga2")
            o = r["objectives"]
            return o["fnr"], o["n_cam"], o["cycle_time_s"]

        t0 = time.time()
        study = optuna.create_study(directions=["minimize"] * 3,
                                    sampler=optuna.samplers.NSGAIISampler(population_size=60, seed=seed))
        study.optimize(objective, n_trials=n_trials)
        cands = list(seen.values())
        log.info(f"stage 1: {len(cands)} unique candidates in {time.time() - t0:.1f}s")
        if self.store is not None:
            for r in cands:
                self.store.append({**r, "method": "stage1", "search_seed": seed})
        return cands

    def shortlist(self, cands: list[dict], k: int = 30, n_random: int = 10, metric: str | None = None,
                  seed: int = 0) -> list[InspectionConfig]:
        """Top surrogate configurations spread over (camera count x rotation mode) groups,
        round-robin, plus random ones. Grouping by rotation matters: the visibility
        surrogate always rewards extra (rotated) views, so a plain top-k per camera count
        warm-started the search with 90-deg x3 rigs only (demo run, 2026-10-20)."""
        metric = metric or self.ev.surrogate_metric
        rng = np.random.default_rng(seed)
        groups: dict[tuple, list[dict]] = {}
        for c in cands:
            groups.setdefault((c["n_cam"], c["config"]["rotation"]), []).append(c)
        for g in groups.values():
            g.sort(key=lambda c: -c["surrogate"][metric])
        keys = sorted(groups)
        chosen, ids, slot_sets = [], set(), set()
        n_top = max(1, k - n_random)
        depth = 0
        while len(chosen) < n_top and any(depth < len(groups[g]) for g in keys):
            for g in keys:
                if depth < len(groups[g]):
                    c = groups[g][depth]
                    key = (tuple(c["config"]["slots"]), c["config"]["rotation"])
                    if key not in slot_sets:
                        slot_sets.add(key)
                        chosen.append(c)
                        ids.add(c["config_id"])
                        if len(chosen) >= n_top:
                            break
            depth += 1
        rest = [c for c in cands if c["config_id"] not in ids]
        for i in rng.permutation(len(rest))[: max(0, k - len(chosen))]:
            chosen.append(rest[i])
        return [InspectionConfig(**c["config"]) for c in chosen[:k]]

    # -- stage 2 ----------------------------------------------------------------------
    def run(self, method: str, budget: int = 60, seed: int = 0, warm: list[InspectionConfig] | None = None) -> list[dict]:
        t0 = time.time()
        fn = {"bo_warm": self._optuna_tpe, "bo_cold": self._optuna_tpe, "nsga2": self._optuna_nsga2,
              "random": self._random, "greedy_vis": self._greedy_vis, "greedy_fnr": self._greedy_fnr}[method]
        if method == "bo_warm":
            rows = fn(budget, seed, warm or [], method)
        elif method == "bo_cold":
            rows = fn(budget, seed, [], method)
        else:
            rows = fn(budget, seed, method)
        log.info(f"stage 2 [{method}] {len(rows)} evals in {time.time() - t0:.0f}s "
                 f"({self.ev.n_stage2_computed} computed so far)")
        return rows

    def _objective_factory(self, method, seed, rows):
        space = self.ev.space

        def objective(trial):
            cfg, repaired = space.suggest(trial)
            cfg = self._apply_fixed(cfg)
            if not cfg.slots:
                return (1.0, 7.0, 10.0) if self.ev.process.cycle_time_in_cost else (1.0, 7.0)
            rec = self.ev.evaluate(cfg, seed)
            rows.append(self._log(rec, method, seed, len(rows), {"repaired": repaired}))
            return self._objs(rec)

        return objective

    def _optuna_tpe(self, budget, seed, warm, method):
        optuna = _quiet_optuna()
        rows: list[dict] = []
        n_obj = 3 if self.ev.process.cycle_time_in_cost else 2
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=0 if warm else 10)
        study = optuna.create_study(directions=["minimize"] * n_obj, sampler=sampler)
        rng = np.random.default_rng(seed)
        for cfg in warm[:budget]:
            study.enqueue_trial(self.ev.space.to_params(cfg, rng))
        study.optimize(self._objective_factory(method, seed, rows), n_trials=budget)
        return rows

    def _optuna_nsga2(self, budget, seed, method):
        optuna = _quiet_optuna()
        rows: list[dict] = []
        n_obj = 3 if self.ev.process.cycle_time_in_cost else 2
        study = optuna.create_study(directions=["minimize"] * n_obj,
                                    sampler=optuna.samplers.NSGAIISampler(population_size=10, seed=seed))
        study.optimize(self._objective_factory(method, seed, rows), n_trials=budget)
        return rows

    def _random(self, budget, seed, method):
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(budget):
            cfg = self._apply_fixed(self.ev.space.random_config(rng))
            rows.append(self._log(self.ev.evaluate(cfg, seed), method, seed, i))
        return rows

    def _base_process(self) -> dict:
        return {"rotation": self.fixed.get("rotation", "none"), "speed_level": self.fixed.get("speed", 1),
                "trigger_mode": self.fixed.get("trigger", "simultaneous")}

    def _greedy_vis(self, budget, seed, method):
        """Coverage-greedy nested camera sets (classic set-cover view planning)."""
        space, ev = self.ev.space, self.ev
        sel: list[int] = []
        rows = []
        for n in range(1, space.p.max_cams + 1):
            best, best_v = None, -1.0
            for s in space.valid_slots:
                if s in sel or not all(space.compatible(s, k) for k in sel):
                    continue
                cfg = InspectionConfig(slots=sel + [s], **self._base_process())
                m = ev.vis.metrics(cfg)
                v = m["V_area"] + 1e-3 * m["V_density"]
                if v > best_v:
                    best, best_v = s, v
            if best is None:
                break
            sel.append(best)
            cfg = InspectionConfig(slots=list(sel), **self._base_process())
            rows.append(self._log(ev.evaluate(cfg, seed), method, seed, len(rows), {"greedy_n": n}))
            if len(rows) >= budget:
                break
        return rows

    def _greedy_fnr(self, budget, seed, method, per_step: int = 15, max_n: int = 4):
        """Forward selection with the detector (candidates pruned by visibility gain)."""
        space, ev = self.ev.space, self.ev
        sel: list[int] = []
        rows = []
        for n in range(1, max_n + 1):
            cands = []
            for s in space.valid_slots:
                if s in sel or not all(space.compatible(s, k) for k in sel):
                    continue
                cfg = InspectionConfig(slots=sel + [s], **self._base_process())
                cands.append((ev.vis.metrics(cfg)["V_area"], s))
            cands.sort(reverse=True)
            best, best_f = None, 2.0
            for _, s in cands[:per_step]:
                cfg = InspectionConfig(slots=sel + [s], **self._base_process())
                rec = ev.evaluate(cfg, seed)
                rows.append(self._log(rec, method, seed, len(rows), {"greedy_n": n}))
                if rec["objectives"]["fnr"] < best_f:
                    best, best_f = s, rec["objectives"]["fnr"]
                if len(rows) >= budget:
                    break
            if best is None or len(rows) >= budget:
                break
            sel.append(best)
        return rows


def pareto_records(rows: list[dict], cost_normaliser, use_report: bool = False) -> list[dict]:
    """Unique-config non-dominated records (search objectives unless ``use_report``)."""
    uniq: dict[str, dict] = {}
    for r in rows:
        if r.get("stage") == 2:
            uniq[r["config_id"]] = r
    recs = list(uniq.values())
    if not recs:
        return []
    F = np.array([cost_normaliser(r["objectives_report"] if use_report else r["objectives"]) for r in recs])
    mask = nondominated_mask(F)
    return [r for r, m in zip(recs, mask) if m]


def recommend(rows: list[dict], max_cams: int | None = None, max_cycle_s: float | None = None) -> dict | None:
    """Lowest search-FNR evaluated configuration within the user constraints (ties -> fewer
    cameras, then shorter cycle)."""
    ok = []
    for r in rows:
        if r.get("stage") != 2:
            continue
        o = r["objectives"]
        if max_cams is not None and o["n_cam"] > max_cams:
            continue
        if max_cycle_s is not None and o["cycle_time_s"] > max_cycle_s + 1e-9:
            continue
        ok.append(r)
    if not ok:
        return None
    return min(ok, key=lambda r: (r["objectives"]["fnr"], r["objectives"]["n_cam"], r["objectives"]["cycle_time_s"]))


def hv_of(rows: list[dict], cost_normaliser, ref=None) -> float:
    F = np.array([cost_normaliser(r["objectives"]) for r in rows if r.get("stage") == 2])
    return hypervolume(F, ref) if len(F) else 0.0
