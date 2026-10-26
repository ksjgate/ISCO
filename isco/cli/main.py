"""``isco`` command-line interface (typer).

Typical flow::

    isco parts prepare-fusion                      # PoC parts (local only)
    isco parts prepare-tless --ids 1 5 20          # T-LESS CAD (CC BY 4.0)
    isco parts prepare-mulsen --categories nut flat_pad
    isco data build fusion_51876 --preset mvp      # variants + slot cache + visibility
    isco evaluate fusion_51876 --slots 12,16,20    # one config -> FNR + cost (JSON)
    isco search fusion_51876 --search smoke        # stage 1 + stage 2 + report
    isco run fusion_51876 --constraint "n<=3,T<=1.5"   # demo: build if needed + search + report
    isco report runs/<run_id>
    isco worker start|stop|status                  # Isaac Sim render worker
    isco exp e0|e1|e2|e3|e4|e5|e6|e7|e8 ...        # paper experiments
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import typer

from .. import config_loader as C
from ..utils.io import data_root, repo_root, resolve_data_path

app = typer.Typer(add_completion=False, help="ISCO - Inspection Setup Configuration Optimizer", no_args_is_help=True)
parts_app = typer.Typer(help="Prepare parts (rest-frame meshes)", no_args_is_help=True)
data_app = typer.Typer(help="Variant sets, slot cache, visibility tables", no_args_is_help=True)
worker_app = typer.Typer(help="Isaac Sim render worker (long-lived, filesystem queue)", no_args_is_help=True)
sim_app = typer.Typer(help="Isaac Sim jobs (occluder export, settling, verification, RGB)", no_args_is_help=True)
exp_app = typer.Typer(help="Paper experiments (E0-E8)", no_args_is_help=True)
app.add_typer(parts_app, name="parts")
app.add_typer(data_app, name="data")
app.add_typer(worker_app, name="worker")
app.add_typer(sim_app, name="sim")
app.add_typer(exp_app, name="exp")


def _echo(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, ensure_ascii=False, default=float))


def _parse_constraint(s: str | None) -> tuple[int | None, float | None]:
    if not s:
        return None, None
    n = re.search(r"n\s*<=\s*(\d+)", s)
    t = re.search(r"T\s*<=\s*([0-9.]+)", s)
    return (int(n.group(1)) if n else None), (float(t.group(1)) if t else None)


# --------------------------------------------------------------------------- parts
@parts_app.command("prepare-fusion")
def parts_fusion(overwrite: bool = False):
    """Fusion 360 Gallery PoC parts (non-redistributable; data/parts/raw/fusion360)."""
    from ..datagen.parts import prepare_fusion_parts

    for s in prepare_fusion_parts(overwrite):
        typer.echo(f"{s.part_id}: extents mm {[round(x * 1e3, 1) for x in s.extents_m]}")


@parts_app.command("prepare-tless")
def parts_tless(ids: List[int] = typer.Option([1, 5, 20, 27], help="T-LESS object ids"), overwrite: bool = False):
    from ..eval.real_parts import prepare_tless

    for s in prepare_tless(ids, overwrite):
        typer.echo(f"{s.part_id}: extents mm {[round(x * 1e3, 1) for x in s.extents_m]}")


@parts_app.command("prepare-mulsen")
def parts_mulsen(categories: List[str] = typer.Option(["nut", "flat_pad", "spring_pad"]), overwrite: bool = False):
    from ..eval.real_parts import prepare_mulsen

    for s in prepare_mulsen(categories, overwrite):
        typer.echo(f"{s.part_id}: extents mm {[round(x * 1e3, 1) for x in s.extents_m]}")


@parts_app.command("prepare-real3d")
def parts_real3d(categories: List[str] = typer.Option(["candybar", "starfish"]), overwrite: bool = False):
    from ..eval.real_parts import prepare_real3d

    for s in prepare_real3d(categories, overwrite):
        typer.echo(f"{s.part_id}: extents mm {[round(x * 1e3, 1) for x in s.extents_m]}")


@parts_app.command("prepare-dowel")
def parts_dowel(overwrite: bool = False):
    from ..datagen.parts import make_dowel_mesh, prepare_part

    s = prepare_part("dowel", "procedural:dowel", "procedural", target_max_dim_m=0.035, mesh=make_dowel_mesh(),
                     overwrite=overwrite)
    typer.echo(f"{s.part_id}: extents mm {[round(x * 1e3, 1) for x in s.extents_m]}")


@parts_app.command("list")
def parts_list():
    from ..datagen.parts import list_parts, load_part

    for pid in list_parts():
        s = load_part(pid)
        typer.echo(f"{pid:22s} {s.source:10s} redistributable={s.redistributable} faces={s.stats['fine']['n_faces']}")


# ---------------------------------------------------------------------------- data
@data_app.command("build")
def data_build(parts: List[str], preset: str = "mvp", workers: int = 6, overwrite: bool = False,
               yaw_mode: str = "guided", set_id: Optional[str] = None, dictionary: str = "default_dictionary",
               rebuild_sets: bool = typer.Option(False, help="regenerate the variant sets only; the slot cache and "
                                                          "visibility then re-render just the changed instances")):
    """Variant set + slot cache (embree) + visibility table for each part."""
    from ..datagen.variants import build_variant_set
    from ..render.slot_cache import build_slot_cache
    from ..visibility.raycast import build_visibility

    for pid in parts:
        spec = C.variant_spec(preset, dictionary)
        spec.pose.yaw_mode = yaw_mode
        if set_id:
            spec.set_id = set_id
        t0 = time.time()
        man = build_variant_set(pid, spec, n_workers=8, overwrite=overwrite or rebuild_sets)
        build_slot_cache(pid, spec.set_id, C.sensor(), C.grid(), n_workers=workers, overwrite=overwrite)
        build_visibility(pid, spec.set_id, C.sensor(), C.grid(), overwrite=overwrite)
        typer.echo(f"{pid}/{spec.set_id}: {man['counts']} ready in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------- evaluate
@app.command()
def evaluate(part: str, slots: str = typer.Option(..., help="comma-separated slot ids"), rotation: str = "none",
             speed: int = 1, trigger: str = "simultaneous", set_id: str = "mvp", noise: str = "N2",
             process: str = "inline_default", detector: str = "fpfh_bank", seed: int = 0, workers: int = 20,
             full: bool = False):
    """Evaluate ONE configuration (the "config in -> FNR + cost out" gate)."""
    from ..optim.evaluator import ConfigEvaluator
    from ..schemas.config import InspectionConfig

    cfg = InspectionConfig(slots=[int(x) for x in slots.split(",")], rotation=rotation, speed_level=speed,
                           trigger_mode=trigger)
    ev = ConfigEvaluator(part, set_id, C.detector(detector), C.process(process), C.noise(noise), n_workers=workers)
    try:
        rec = ev.evaluate(cfg, seed)
    finally:
        ev.close()
    if full:
        _echo(rec)
        return
    grid = ev.grid
    _echo({"config": rec["config"], "cameras": [{"slot": k, "azimuth_deg": grid.slot(k).azimuth_deg,
                                                 "elevation_deg": grid.slot(k).elevation_deg,
                                                 "distance_m": grid.slot(k).distance_m} for k in cfg.slots],
           "fnr_at_1fpr_search": rec["search"]["fnr"], "fnr_at_1fpr_report": rec["report"]["fnr"],
           "fnr_report_ci95": rec["report"]["fnr_ci"], "auroc_report": rec["report"]["auroc"],
           "fnr_by_type_report": rec["report"]["fnr_by_type"], "n_cam": rec["n_cam"],
           "cycle_time_s": rec["objectives"]["cycle_time_s"], "surrogate": rec["surrogate"],
           "timing": rec["timing"], "cache_hit": rec["cache_hit"]})


# -------------------------------------------------------------------------- search
@app.command()
def search(part: str, set_id: str = "mvp", search_cfg: str = typer.Option("smoke", "--search"), noise: str = "N2",
           process: str = "inline_default", detector: str = "fpfh_bank", run_id: Optional[str] = None,
           methods: Optional[str] = None, budget: Optional[int] = None, seeds: Optional[str] = None,
           workers: int = 20, constraint: Optional[str] = None):
    """Stage 1 + shortlist + stage 2 (+ report) for one part."""
    from ..optim.run import run_search

    sc = C.search(search_cfg)
    mc, mt = _parse_constraint(constraint)
    rd = run_search(part, set_id, run_id, methods=(methods.split(",") if methods else sc["methods"]),
                    budget=budget or sc["budget"], seeds=[int(s) for s in seeds.split(",")] if seeds else sc["seeds"],
                    detector=C.detector(detector), process=C.process(process), noise=C.noise(noise),
                    stage1_trials=sc["stage1_trials"], stage1_random=sc["stage1_random"],
                    shortlist_k=sc["shortlist_k"], shortlist_random=sc["shortlist_random"], n_workers=workers,
                    max_cams=mc if mc is not None else sc.get("max_cams"),
                    max_cycle_s=mt if mt is not None else sc.get("max_cycle_s"))
    typer.echo(f"run dir: {rd}\nreport: {rd / 'report' / 'index.html'}")


@app.command()
def run(part: str, constraint: str = "n<=3,T<=1.5", search_cfg: str = typer.Option("demo", "--search"),
        preset: str = "mvp", out: Optional[str] = None, workers: int = 20, noise: str = "N2"):
    """Demo flow: (build data if missing) -> stage-1 screening -> stage-2 search -> report."""
    from ..datagen.parts import list_parts
    from ..render.slot_cache import cache_dir

    if part not in list_parts():
        raise typer.BadParameter(f"unknown part {part}; run `isco parts ...` first")
    if not (cache_dir(part, preset, "sl_default") / "cache.json").exists():
        data_build([part], preset=preset)
    run_id = Path(out).name if out else None
    search(part, set_id=preset, search_cfg=search_cfg, noise=noise, process="inline_default", detector="fpfh_bank",
           run_id=run_id, methods=None, budget=None, seeds=None, workers=workers, constraint=constraint)


@app.command()
def report(run_dir: str, constraint: str = "n<=3,T<=1.5"):
    from ..report.render_report import build_report

    mc, mt = _parse_constraint(constraint)
    typer.echo(str(build_report(run_dir, mc, mt)))


# -------------------------------------------------------------------------- worker
@worker_app.command("start")
def worker_start(queue: str = "queue", headless: bool = True):
    """Start the long-lived Isaac Sim worker in the background (log: <queue>/worker.log)."""
    q = repo_root() / queue
    q.mkdir(parents=True, exist_ok=True)
    log = open(q / "worker.log", "a")
    args = [sys.executable, "-m", "isco.sim.worker", "--queue", str(q)] + (["--headless"] if headless else [])
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=repo_root(), start_new_session=True)
    typer.echo(f"worker pid {p.pid}; log {q / 'worker.log'}")


@worker_app.command("stop")
def worker_stop(queue: str = "queue"):
    from ..render.queue import FileJobQueue

    FileJobQueue(repo_root() / queue).request_shutdown()
    typer.echo("shutdown requested")


@worker_app.command("status")
def worker_status(queue: str = "queue"):
    from ..render.queue import FileJobQueue

    q = FileJobQueue(repo_root() / queue)
    _echo({"alive": q.worker_alive(), "counts": q.counts()})


def _submit(job_type: str, payload, timeout_s: float = 7200.0, queue: str = "queue") -> dict:
    from ..render.queue import FileJobQueue
    from ..schemas.job import Job

    q = FileJobQueue(repo_root() / queue)
    job = Job.make(job_type, payload)
    q.submit(job)
    res = q.wait([job.job_id], timeout_s=timeout_s)[job.job_id]
    return json.loads(res.model_dump_json())


@sim_app.command("export-occluder")
def sim_export_occluder(radius_m: float = 1.0):
    from ..schemas.job import ExportOccluderPayload

    _echo(_submit("export_occluder", ExportOccluderPayload(
        output_path=str(data_root() / "conveyor" / "occluder.npz"), radius_m=radius_m)))


@sim_app.command("settle")
def sim_settle(part: str, n_drops: int = 100, speeds: str = "0.1,0.3,0.6"):
    from ..datagen.parts import load_part
    from ..schemas.job import SettlePayload

    spec = load_part(part)
    _echo(_submit("settle", SettlePayload(part_id=part, mesh_path=str(resolve_data_path(spec.mesh_fine)), n_drops=n_drops,
                                          speeds_mps=[float(s) for s in speeds.split(",")],
                                          output_path=str(data_root() / "settle" / f"{part}.json"))))


@sim_app.command("check-rig")
def sim_check_rig(part: str, set_id: str = "mvp"):
    """Render the nominal part from all 96 slots in Isaac and cross-check slot validity
    (occluder ray test) against what the cameras actually see (plan: check-rig)."""
    import numpy as np

    from ..datagen.parts import load_part
    from ..optim.space import SearchSpace
    from ..render.occluder import Occluder
    from ..render.pixels import ViewPixels
    from ..render.slot_cache import SlotCache
    from ..schemas.job import InstanceRef, RenderSlotsPayload
    from ..utils.io import write_json

    spec = load_part(part)
    cache = SlotCache.open(part, set_id)
    grid = cache.grid
    out = repo_root() / "runs" / "check_rig" / part
    res = _submit("render_slots", RenderSlotsPayload(
        part_id=part, instances=[InstanceRef(inst_id="nominal", mesh_path=str(resolve_data_path(spec.mesh_fine)),
                                             T_world_part=np.eye(4).tolist())],
        slot_ids=list(range(grid.n_slots)), grid=grid.to_dict(), sensor=cache.sensor, focus_m=spec.focus_m,
        output_dir=str(out)))
    space = SearchSpace(grid, np.asarray(spec.focus_m), Occluder.default(), working_range=cache.sensor.working_range_m)
    px = {k: len(ViewPixels.load(out / "nominal" / f"slot_{k:03d}.npz")) for k in range(grid.n_slots)}
    med = float(np.median([v for v in px.values() if v > 0])) if any(px.values()) else 0.0
    rows = [{"slot": k, "valid": k in space.valid_slots, "isaac_part_px": px[k],
             "reason": space.invalid_reason.get(k)} for k in range(grid.n_slots)]
    flagged = [r for r in rows if r["valid"] and r["isaac_part_px"] < 0.05 * med]
    summary = {"job": res["status"], "n_valid": len(space.valid_slots), "n_slots": grid.n_slots,
               "median_part_px": med, "valid_but_part_not_seen": flagged,
               "invalid": [r for r in rows if not r["valid"]]}
    write_json(out / "check_rig.json", {"summary": summary, "rows": rows})
    _echo({k: v for k, v in summary.items() if k != "invalid"} | {"n_invalid": len(summary["invalid"])})


@sim_app.command("rgb")
def sim_rgb(part: str, slots: str, out: str = "runs/rgb"):
    """RGB snapshots of a rig (demo / report figures) through the Isaac worker."""
    from ..datagen.parts import load_part
    from ..schemas.config import CameraSpec
    from ..schemas.job import CaptureRgbPayload

    spec = load_part(part)
    grid = C.grid()
    cams = [CameraSpec(azimuth_deg=grid.slot(int(k)).azimuth_deg, elevation_deg=grid.slot(int(k)).elevation_deg,
                       distance_m=grid.slot(int(k)).distance_m, slot_id=int(k)) for k in slots.split(",")]
    _echo(_submit("capture_rgb", CaptureRgbPayload(part_id=part, mesh_path=str(resolve_data_path(spec.mesh_fine)), cameras=cams, sensor=C.sensor(),
                                                   focus_m=spec.focus_m, output_dir=str(repo_root() / out))))


# ---------------------------------------------------------------------- experiments
@exp_app.command("run")
def exp_run(name: str, parts: List[str] = typer.Option(["fusion_51876"]), set_id: str = "mvp",
            seeds: Optional[str] = None, budget: Optional[int] = None, workers: int = 20, out: Optional[str] = None):
    """Run a paper experiment: e0_5, e1, e2, e3, e4, e6, e7, e8 (see isco.eval.hypotheses).
    Without --budget / --seeds each experiment uses its pre-registered values."""
    from ..eval import hypotheses as H

    fn = getattr(H, name)
    kw = {}
    if seeds:
        kw["seeds"] = [int(s) for s in seeds.split(",")]
    if budget is not None:
        kw["budget"] = budget
    rd = fn(parts=list(parts), set_id=set_id, n_workers=workers, out=out, **kw)
    typer.echo(f"results: {rd}")


def main() -> None:  # console-script entry point
    app()


if __name__ == "__main__":
    main()
