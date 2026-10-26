"""Filesystem job queue for the long-lived Isaac render worker (ADR-0002).

Layout under ``root``::

    pending/<prio>_<ts>_<job_id>.json   submitted, waiting
    running/<job_id>.json               claimed by the worker
    done/<job_id>.json                  JobResult (status ok)
    failed/<job_id>.json                JobResult (status failed)
    control/shutdown                    ask the worker to exit
    worker.heartbeat                    JSON {pid, ts, ...}, refreshed every loop

All state changes are atomic ``os.replace`` renames, so a crashed worker leaves jobs
in ``running/`` which are re-queued on the next start (``recover()``).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..schemas.job import Job, JobResult
from ..utils.io import atomic_write_text, write_json


class FileJobQueue:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        for d in ("pending", "running", "done", "failed", "control"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    # -- client side ------------------------------------------------------------
    def submit(self, job: Job) -> str:
        name = f"{job.priority:03d}_{int(job.created_at * 1000):015d}_{job.job_id}.json"
        atomic_write_text(self.root / "pending" / name, job.model_dump_json())
        return job.job_id

    def result(self, job_id: str) -> JobResult | None:
        for d in ("done", "failed"):
            p = self.root / d / f"{job_id}.json"
            if p.exists():
                return JobResult.model_validate_json(p.read_text())
        return None

    def wait(self, job_ids: list[str], timeout_s: float = 3600.0, poll_s: float = 0.5,
             require_worker: bool = True) -> dict[str, JobResult]:
        t0 = time.time()
        out: dict[str, JobResult] = {}
        while True:
            for j in job_ids:
                if j not in out:
                    r = self.result(j)
                    if r is not None:
                        out[j] = r
            if len(out) == len(job_ids):
                return out
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"jobs not finished after {timeout_s}s: {sorted(set(job_ids) - set(out))}")
            if require_worker and time.time() - t0 > 180 and not self.worker_alive():
                raise RuntimeError("render worker is not alive (no heartbeat); start it with scripts/start_worker.sh")
            time.sleep(poll_s)

    def request_shutdown(self) -> None:
        atomic_write_text(self.root / "control" / "shutdown", str(time.time()))

    # -- worker side --------------------------------------------------------------
    def recover(self) -> int:
        n = 0
        for p in sorted((self.root / "running").glob("*.json")):
            job = Job.model_validate_json(p.read_text())
            self.submit(job)
            p.unlink()
            n += 1
        return n

    def claim_next(self) -> Job | None:
        for p in sorted((self.root / "pending").glob("*.json")):
            job_id = p.stem.split("_", 2)[-1]
            dst = self.root / "running" / f"{job_id}.json"
            try:
                os.replace(p, dst)
            except FileNotFoundError:
                continue
            return Job.model_validate_json(dst.read_text())
        return None

    def complete(self, result: JobResult) -> None:
        d = "done" if result.status == "ok" else "failed"
        write_json(self.root / d / f"{result.job_id}.json", json.loads(result.model_dump_json()))
        running = self.root / "running" / f"{result.job_id}.json"
        if running.exists():
            running.unlink()

    def shutdown_requested(self) -> bool:
        return (self.root / "control" / "shutdown").exists()

    def clear_shutdown(self) -> None:
        p = self.root / "control" / "shutdown"
        if p.exists():
            p.unlink()

    def heartbeat(self, **info) -> None:
        write_json(self.root / "worker.heartbeat", {"pid": os.getpid(), "ts": time.time(), **info})

    def worker_alive(self, max_age_s: float = 120.0) -> bool:
        p = self.root / "worker.heartbeat"
        if not p.exists():
            return False
        try:
            hb = json.loads(p.read_text())
        except Exception:
            return False
        return time.time() - float(hb.get("ts", 0)) < max_age_s

    def counts(self) -> dict[str, int]:
        return {d: len(list((self.root / d).glob("*.json"))) for d in ("pending", "running", "done", "failed")}
