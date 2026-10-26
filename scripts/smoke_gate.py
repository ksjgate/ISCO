"""Week-5 gate ("one configuration in -> FNR + cost out") in one command.

1. make sure the Isaac worker is alive (starts it if needed) and render 3 instances x 2
   slots through Isaac (``render_slots``), comparing them with the embree slot cache
   (mini E0.1);
2. evaluate one 2-camera configuration with the detector in the loop;
3. print one JSON object: Isaac/embree agreement + FNR@1%FPR + cost + timing.

    python scripts/smoke_gate.py --part fusion_51876 [--slots 12,20] [--stop-worker]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

import numpy as np

from isco import config_loader as C
from isco.datagen.parts import load_part
from isco.datagen.variants import load_manifest
from isco.optim.evaluator import ConfigEvaluator
from isco.render.pixels import ViewPixels
from isco.render.queue import FileJobQueue
from isco.render.slot_cache import SlotCache
from isco.schemas.config import InspectionConfig
from isco.schemas.job import InstanceRef, Job, RenderSlotsPayload
from isco.utils.io import repo_root, resolve_data_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="fusion_51876")
    ap.add_argument("--set-id", default="mvp")
    ap.add_argument("--slots", default="12,20")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--stop-worker", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    q = FileJobQueue(repo_root() / "queue")
    started = False
    if not q.worker_alive():
        subprocess.run(["bash", str(repo_root() / "scripts" / "start_worker.sh")], check=True)
        started = True
        for _ in range(240):
            if q.worker_alive():
                break
            time.sleep(1)
    slots = [int(s) for s in a.slots.split(",")]
    cache = SlotCache.open(a.part, a.set_id)
    man = load_manifest(a.part, a.set_id)
    insts = [r for r in man["instances"] if r["role"] in ("val", "report")][:3]
    spec = load_part(a.part)
    out = repo_root() / "runs" / "smoke_gate" / a.part
    job = Job.make("render_slots", RenderSlotsPayload(
        part_id=a.part, instances=[InstanceRef(inst_id=r["inst_id"], mesh_path=str(resolve_data_path(r["mesh"])),
                                               T_world_part=r["T_world_part"]) for r in insts],
        slot_ids=slots, grid=cache.grid.to_dict(), sensor=cache.sensor, focus_m=spec.focus_m, output_dir=str(out)))
    q.submit(job)
    res = q.wait([job.job_id], timeout_s=1800)[job.job_id]
    agree = []
    for r in insts:
        emb = cache.views(r["inst_id"], slots)
        for k in slots:
            isa = ViewPixels.load(out / r["inst_id"] / f"slot_{k:03d}.npz")
            common, ia, ib = np.intersect1d(isa.pix, emb[k].pix, return_indices=True)
            iou = len(common) / max(len(np.union1d(isa.pix, emb[k].pix)), 1)
            dd = np.abs(isa.depth[ia].astype(float) - emb[k].depth[ib].astype(float))
            agree.append({"inst": r["inst_id"], "slot": k, "iou": iou, "depth_med_um": float(np.median(dd) * 1e6),
                          "depth_p99_um": float(np.quantile(dd, 0.99) * 1e6)})
    ev = ConfigEvaluator(a.part, a.set_id, C.detector("fpfh_bank"), C.process("inline_default"), C.noise("N2"),
                         n_workers=a.workers)
    try:
        cfg = InspectionConfig(slots=slots, rotation="none", speed_level=1)
        rec = ev.evaluate(cfg)
    finally:
        ev.close()
    if a.stop_worker or started:
        q.request_shutdown()
    print(json.dumps({
        "part": a.part, "isaac_job": res.status, "isaac_vs_embree": {
            "views": len(agree), "iou_min": min(x["iou"] for x in agree),
            "depth_med_um_max": max(x["depth_med_um"] for x in agree),
            "depth_p99_um_max": max(x["depth_p99_um"] for x in agree)},
        "config": rec["config"], "fnr_at_1fpr": rec["search"]["fnr"], "fnr_at_1fpr_report": rec["report"]["fnr"],
        "auroc_report": rec["report"]["auroc"], "per_type_report": rec["report"]["fnr_by_type"],
        "cost": {"n_cam": rec["n_cam"], "cycle_time_s": rec["objectives"]["cycle_time_s"]},
        "eval_timing": rec["timing"], "cache_hit": rec["cache_hit"], "gate_wall_s": round(time.time() - t0, 1)},
        indent=2))


if __name__ == "__main__":
    main()
