"""End-to-end search run: stage 1 -> shortlist -> stage 2 (one or more methods/seeds)
-> report. Everything a run needs to be reproduced is written to ``runs/<run_id>/``::

    run.json      resolved settings (part, set, detector, process, noise, search params)
    env.json      git sha, isco/isaac/torch/open3d versions, pip freeze excerpt
    stage1.jsonl  surrogate candidates
    stage2.jsonl  detector-in-the-loop evaluations (method, seed, trial, metrics)
    report/       index.md / index.html / recommended.json / figures
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from ..detect.pipeline import DetectorConfig
from ..report.store import ResultsStore
from ..schemas.process import ProcessParams
from ..schemas.sensor import NoiseModel
from ..utils.io import repo_root, write_json
from ..utils.logging import add_file_handler, get_logger
from .evaluator import ConfigEvaluator
from .search import SearchRunner
from .space import SpaceParams

log = get_logger(__name__)


def env_info() -> dict:
    info: dict = {"time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        info["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True).strip()
        info["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root(), text=True).strip())
    except Exception:
        pass
    import importlib.metadata as md

    for pkg in ("isco", "isaacsim", "torch", "open3d", "numpy", "optuna", "trimesh", "manifold3d", "pymeshlab"):
        try:
            info[pkg] = md.version(pkg)
        except Exception:
            info[pkg] = None
    return info


def run_search(part_id: str, set_id: str = "mvp", run_id: str | None = None, methods=("bo_warm",), budget: int = 60,
               seeds=(0,), detector: DetectorConfig | None = None, process: ProcessParams | None = None,
               noise: NoiseModel | None = None, stage1_trials: int = 1500, stage1_random: int = 500,
               shortlist_k: int = 30, shortlist_random: int = 10, fixed: dict | None = None, n_workers: int = 20,
               max_cams: int | None = 3, max_cycle_s: float | None = 1.5, space_params: SpaceParams | None = None,
               report: bool = True, runs_root: str | Path | None = None) -> Path:
    run_id = run_id or f"r_{time.strftime('%Y%m%d_%H%M%S')}_{part_id}"
    run_dir = Path(runs_root or repo_root() / "runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(run_dir / "log.txt")
    detector = detector or DetectorConfig()
    process = process or ProcessParams()
    noise = noise or NoiseModel()
    write_json(run_dir / "run.json", {
        "run_id": run_id, "part_id": part_id, "set_id": set_id, "methods": list(methods), "budget": budget,
        "seeds": list(seeds), "detector": detector.model_dump(mode="json"), "process": process.model_dump(mode="json"),
        "noise": noise.model_dump(mode="json"), "stage1_trials": stage1_trials, "stage1_random": stage1_random,
        "shortlist_k": shortlist_k, "shortlist_random": shortlist_random, "fixed": fixed or {},
        "constraints": {"max_cams": max_cams, "max_cycle_s": max_cycle_s},
        "space": (space_params or SpaceParams()).model_dump(mode="json")})
    write_json(run_dir / "env.json", env_info())
    ev = ConfigEvaluator(part_id, set_id, detector, process, noise, n_workers=n_workers, space_params=space_params)
    # rows are rewritten on every (re)run; finished evaluations come back from the
    # evaluation cache, so a resumed run reproduces the same rows without duplicates
    for f in ("stage1.jsonl", "stage2.jsonl"):
        if (run_dir / f).exists():
            (run_dir / f).unlink()
    from ..utils.tracking import Tracker

    tracker = Tracker(run_name=run_id, config={"part_id": part_id, "set_id": set_id, "methods": list(methods)})
    s1 = ResultsStore(run_dir / "stage1.jsonl")
    s2 = ResultsStore(run_dir / "stage2.jsonl", tracker)
    t0 = time.time()
    try:
        runner1 = SearchRunner(ev, s1, fixed)
        cands = runner1.stage1(stage1_trials, stage1_random, seed=seeds[0])
        warm = runner1.shortlist(cands, shortlist_k, shortlist_random, seed=seeds[0])
        runner2 = SearchRunner(ev, s2, fixed)
        for seed in seeds:
            for m in methods:
                runner2.run(m, budget, seed, warm=warm)
    finally:
        ev.close()
        tracker.finish()
    log.info(f"run {run_id}: {ev.n_stage2_computed} stage-2 evaluations computed in {time.time() - t0:.0f}s")
    if report:
        from ..report.render_report import build_report

        build_report(run_dir, max_cams, max_cycle_s, title=f"ISCO run {run_id}")
    return run_dir
