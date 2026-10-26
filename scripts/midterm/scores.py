"""Per-point anomaly scores of the demo defect of each part under its recommended
configuration - the detector of the experiments (FPFH memory bank, inline process, main
noise level), run with the experiment's own worker functions so the numbers match the
evaluation record.

    python scripts/midterm/scores.py [--parts tless_13 ...] [--workers 4]
    -> runs/midterm_reports/scores/<part>.npz (+ <part>.json)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PARTS, WORK, best_record, save_json, showcase  # noqa: E402


def run_part(pid: str, show: dict, n_workers: int) -> None:
    from isco import config_loader as C
    from isco.datagen.variants import load_manifest
    from isco.detect import pipeline as P
    from isco.detect.memory_bank import MemoryBank
    from isco.schemas.config import InspectionConfig
    from isco.utils.seeds import core_order_env

    t0 = time.time()
    rec = best_record(pid)
    cfg = InspectionConfig.model_validate(rec["config"])
    det = C.detector("fpfh_bank")
    insts = {r["inst_id"]: r for r in load_manifest(pid, "mvp")["instances"]}
    bank = [r for r in insts.values() if r["role"] == "bank"]
    val = [r for r in insts.values() if r["role"] == "val"]
    tasks = [(r, cfg.model_dump_json(), 0) for r in bank + val + [insts[show["inst_id"]]]]
    saved = {k: os.environ.get(k) for k in P._THREAD_VARS}
    for k in P._THREAD_VARS:
        os.environ[k] = "1"
    try:
        with core_order_env():
            pool = mp.get_context("spawn").Pool(n_workers, initializer=P._init, initargs=(
                pid, "mvp", "sl_default", det.model_copy(update={"keep_point_scores": True}).model_dump_json(),
                C.process("inline_default").model_dump_json(), C.noise("N2").model_dump_json()))
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    outs = {o["inst_id"]: o for o in pool.imap(P._work, tasks, chunksize=2)}
    pool.close()
    pool.join()
    mb = MemoryBank(det.coreset_ratio, det.coreset_max, "cuda", 0, det.bank_presample).fit(
        [outs[r["inst_id"]]["feats"].astype(np.float32) for r in bank if outs[r["inst_id"]].get("feats") is not None])

    def pts(o):
        s = mb.score(o["feats"].astype(np.float32))
        return s[o["nbr"]].mean(axis=1) if o["nbr"] is not None else s

    val_img = np.array([pts(outs[r["inst_id"]]).max() if outs[r["inst_id"]].get("feats") is not None else 0.0
                        for r in val])
    o = outs[show["inst_id"]]
    s = pts(o)
    mu, sd = float(val_img.mean()), float(val_img.std() + 1e-12)
    feat_thr = mu + show["tau"] * sd               # feature-score level of the 1 %-false-alarm threshold
    normal_pts = np.concatenate([pts(outs[r["inst_id"]]) for r in val[:10]])
    meta = {"part": pid, **show, "feat_image_score": float(s.max()), "feat_z": (float(s.max()) - mu) / sd,
            "val_feat_mean": mu, "val_feat_sd": sd, "feat_threshold": feat_thr,
            "normal_point_median": float(np.median(normal_pts)), "normal_point_p99": float(np.quantile(normal_pts, 0.99)),
            "n_points": int(len(o["xyz"])), "n_defect_points": int((o["label"] >= 2).sum()),
            "reg_fitness": o["reg_fitness"], "seconds": round(time.time() - t0, 1)}
    np.savez_compressed(WORK / "scores" / f"{pid}.npz", xyz=o["xyz"].astype(np.float32), label=o["label"].astype(np.int16),
                        score=s.astype(np.float32))
    save_json(WORK / "scores" / f"{pid}.json", meta)
    print(f"[scores] {pid} {show['type']}: feature image score {s.max():.3f} (record {show['feat_score']:.3f}), "
          f"feature z {meta['feat_z']:.2f} vs tau {show['tau']:.2f}, {meta['seconds']}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*", default=PARTS)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    (WORK / "scores").mkdir(parents=True, exist_ok=True)
    shows = showcase()
    save_json(WORK / "showcase.json", shows)
    for pid in a.parts:
        run_part(pid, shows[pid], a.workers)


if __name__ == "__main__":
    main()
