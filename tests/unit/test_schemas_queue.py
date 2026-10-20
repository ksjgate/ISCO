import time

import pytest

from isco.render.queue import FileJobQueue
from isco.schemas.config import InspectionConfig
from isco.schemas.job import PAYLOADS, EmptyPayload, Job, JobResult, SettlePayload
from isco.schemas.process import ROTATION_STAGES, ProcessParams


def test_config_canonical_and_id_stable():
    a = InspectionConfig(slots=[20, 12, 16, 12], rotation="r90x1", speed_level=2)
    b = InspectionConfig(slots=[16, 20, 12], rotation="r90x1", speed_level=2)
    assert a.slots == [12, 16, 20] and a.n_cam == 3
    assert a.config_id == b.config_id and a.config_id.startswith("c_")
    assert InspectionConfig(slots=[12]).config_id != a.config_id
    assert a.stages_deg == [0.0, 90.0]


def test_config_rejects_too_many_cameras():
    with pytest.raises(Exception):
        InspectionConfig(slots=list(range(7)))


def test_process_cycle_time_and_static():
    p = ProcessParams()
    assert p.cycle_time_s(1, "none") == pytest.approx(1.0)
    assert p.cycle_time_s(2, "r90x3") == pytest.approx(0.5 + 1.7)
    assert p.cycle_time_s(0, "r90x1") == pytest.approx(3.0 + 0.9)
    s = ProcessParams.static()
    assert s.speed(2) == 0.0 and not s.cycle_time_in_cost
    assert set(ROTATION_STAGES) == {"none", "r90x1", "r180x1", "r90x3"}


def test_job_roundtrip_and_empty_payload():
    j = Job.make("settle", SettlePayload(part_id="p", mesh_path="/x.npz", output_path="/o.json"))
    j2 = Job.model_validate_json(j.model_dump_json())
    assert j2.parsed_payload().part_id == "p" and j2.parsed_payload().tilt_sigma_deg is None
    assert PAYLOADS["ping"] is EmptyPayload
    assert isinstance(Job.make("ping").parsed_payload(), EmptyPayload)


def test_file_queue_lifecycle(tmp_path):
    q = FileJobQueue(tmp_path)
    ids = [q.submit(Job.make("ping", priority=p)) for p in (50, 10)]
    first = q.claim_next()
    assert first.job_id == ids[1]                      # lower priority number first
    q.complete(JobResult(job_id=first.job_id, job_type="ping", status="ok"))
    assert q.result(first.job_id).status == "ok"
    second = q.claim_next()
    assert q.counts()["running"] == 1
    assert q.recover() == 1 and q.counts()["pending"] == 1   # crashed worker -> requeued
    assert not q.worker_alive()
    q.heartbeat(note="t")
    assert q.worker_alive()
    q.request_shutdown()
    assert q.shutdown_requested()
    q.clear_shutdown()
    assert not q.shutdown_requested()
    with pytest.raises(TimeoutError):
        q.wait([second.job_id], timeout_s=0.3, poll_s=0.05, require_worker=False)
