import os
import sys
import time

import pytest

from shad_helper import vision_budget
from shad_helper.jobs import read_json, write_json


def test_active_work_is_cumulative_and_explicit_increase_preserves_usage(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(vision_budget.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(vision_budget.time, "time", lambda: clock[0])
    request = {"start": 0, "end": 20}
    budget = vision_budget.Budget(tmp_path, request)
    with budget.active("model_load"):
        clock[0] += 3
    clock[0] += 100  # CPU selection and time between runs do not count.
    with budget.active("analysis"):
        clock[0] += 7
        with pytest.raises(vision_budget.BudgetExhausted):
            budget.check()
    assert budget.summary() == {"limit_seconds": 10, "used_seconds": 10, "remaining_seconds": 0}
    resumed = vision_budget.Budget(tmp_path, request)
    assert resumed.remaining == 0
    increased = vision_budget.Budget(tmp_path, {**request, "vision_budget_minutes": 1})
    assert increased.summary() == {"limit_seconds": 60, "used_seconds": 10, "remaining_seconds": 50}


@pytest.mark.parametrize("minutes", [True, 0, -1, float("inf"), float("nan"), "30"])
def test_invalid_budget_is_rejected(tmp_path, minutes):
    with pytest.raises(ValueError):
        vision_budget.Budget(tmp_path, {"vision_budget_minutes": minutes})


def test_interrupted_model_load_is_charged_on_resume(tmp_path, monkeypatch):
    write_json(tmp_path / vision_budget.BUDGET_FILE,
               {"version": 1, "limit_seconds": 100, "used_seconds": 10,
                "active_started": 50, "active_pid": 999999, "phase": "model_load"})
    write_json(tmp_path / vision_budget.HEARTBEAT_FILE,
               {"active_pid": 999999, "active_started": 50, "observed_time": 80})
    monkeypatch.setattr(vision_budget.time, "time", lambda: 2000)

    def dead(pid, signal):
        raise ProcessLookupError

    monkeypatch.setattr(vision_budget.os, "kill", dead)
    budget = vision_budget.Budget(tmp_path, {"start": 0, "end": 200})
    assert budget.state["used_seconds"] == 40
    assert budget.state["active_started"] is None


def test_concurrent_worker_is_not_silently_reset(tmp_path):
    write_json(tmp_path / vision_budget.BUDGET_FILE,
               {"version": 1, "limit_seconds": 100, "used_seconds": 0,
                "active_started": time.time(), "active_pid": os.getpid(), "phase": "analysis"})
    with pytest.raises(RuntimeError, match="still active"):
        vision_budget.Budget(tmp_path, {"start": 0, "end": 200})


def test_watchdog_interrupts_loading_and_recovers_completed_evidence(tmp_path):
    manifest = {"items": [{"id": "done", "image": "first.png", "timestamp": 1},
                          {"id": "later", "image": "second.png", "timestamp": 500,
                           "page_id": "page1", "page_start": 300, "page_end": 600}], "audit": {}}
    write_json(tmp_path / "sparse-selection.json", manifest)
    write_json(tmp_path / "sparse-vision-results.json",
               {"items": [{**manifest["items"][0], "status": "processed", "summary": "done",
                           "regions": [], "issues": []}], "audit": {}})
    state_path = tmp_path / vision_budget.BUDGET_FILE
    script = f"""import os,time,json
from pathlib import Path
path=Path({str(state_path)!r})
state={{'version':1,'limit_seconds':0.07,'used_seconds':0,'active_started':time.time(),
       'active_pid':os.getpid(),'phase':'model_load'}}
temporary=path.with_suffix('.tmp')
temporary.write_text(json.dumps(state));temporary.replace(path)
time.sleep(60)
"""
    response_path = tmp_path / "response.json"
    started = time.monotonic()
    result = vision_budget.run_budgeted_worker([sys.executable, "-c", script], tmp_path,
                                               {"start": 0, "end": 600, "items": []}, response_path)
    assert time.monotonic() - started < 3
    assert [item["status"] for item in result["items"]] == ["processed", "deferred"]
    assert result["items"][1]["regions"] == []
    assert result["items"][1]["page_id"] == "page1"
    assert result["audit"]["watchdog_terminated"] is True
    assert read_json(state_path)["active_started"] is None
    assert read_json(response_path) == result


def test_watchdog_does_not_hide_unrelated_worker_failure(tmp_path):
    with pytest.raises(vision_budget.subprocess.CalledProcessError):
        vision_budget.run_budgeted_worker([sys.executable, "-c", "raise SystemExit(7)"],
                                         tmp_path, {}, tmp_path / "response.json")


def test_deadline_kills_immediately_without_sigterm_grace(tmp_path, monkeypatch):
    signals = []
    killpg = vision_budget.os.killpg

    def recorded_killpg(pid, sig):
        signals.append(sig)
        killpg(pid, sig)

    monkeypatch.setattr(vision_budget.os, "killpg", recorded_killpg)
    state_path = tmp_path / vision_budget.BUDGET_FILE
    script = f"""import os,time,json,signal
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
path=Path({str(state_path)!r})
state={{'version':1,'limit_seconds':0.07,'used_seconds':0,'active_started':time.time(),
       'active_pid':os.getpid(),'phase':'model_load'}}
temporary=path.with_suffix('.tmp')
temporary.write_text(json.dumps(state));temporary.replace(path)
time.sleep(60)
"""
    result = vision_budget.run_budgeted_worker([sys.executable, "-c", script], tmp_path,
                                               {"start": 0, "end": 600}, tmp_path / "response.json")
    assert signals == [vision_budget.signal.SIGKILL]
    assert result["audit"]["watchdog_terminated"] is True
    assert read_json(state_path)["active_started"] is None


def test_recovery_rejects_stale_results_for_changed_occurrence_and_pixels(tmp_path):
    from PIL import Image

    image = tmp_path / "frame.png"
    Image.new("RGB", (30, 20), "red").save(image)
    item = {"id": "v1", "image": str(image), "timestamp": 10}
    write_json(tmp_path / "sparse-selection.json", {"items": [item], "audit": {}})
    saved = {**item, "status": "processed", "sha256": "old pixels", "regions": []}
    write_json(tmp_path / "sparse-vision-results.json", {"items": [saved], "audit": {}})
    result = vision_budget.recover_results(tmp_path, {})
    assert result["items"][0]["status"] == "deferred"
    saved.update(sha256=vision_budget.sha256_file(image), timestamp=9)
    write_json(tmp_path / "sparse-vision-results.json", {"items": [saved], "audit": {}})
    assert vision_budget.recover_results(tmp_path, {})["items"][0]["status"] == "deferred"
    saved.update(timestamp=10, selection_role="complementary", complement_of="page")
    write_json(tmp_path / "sparse-vision-results.json", {"items": [saved], "audit": {}})
    assert vision_budget.recover_results(tmp_path, {})["items"][0]["status"] == "deferred"


def test_watchdog_settles_crashed_load_before_raising(tmp_path):
    state_path = tmp_path / vision_budget.BUDGET_FILE
    script = f"""import os,time,json
from pathlib import Path
path=Path({str(state_path)!r})
path.write_text(json.dumps({{'version':1,'limit_seconds':60,'used_seconds':0,
                           'active_started':time.time(),'active_pid':os.getpid(),'phase':'model_load'}}))
time.sleep(0.07)
raise SystemExit(9)
"""
    with pytest.raises(vision_budget.subprocess.CalledProcessError):
        vision_budget.run_budgeted_worker([sys.executable, "-c", script], tmp_path, {},
                                         tmp_path / "response.json")
    state = read_json(state_path)
    assert state["active_started"] is None
    assert state["used_seconds"] >= 0.07


def test_job_lock_rejects_second_worker_before_overwriting_request(tmp_path, monkeypatch):
    request_path = tmp_path / "request.json"
    write_json(request_path, {"original": True})
    monkeypatch.setattr(vision_budget.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("worker started"))
    with (tmp_path / "sparse-qwen-worker.lock").open("a") as lock:
        vision_budget.fcntl.flock(lock, vision_budget.fcntl.LOCK_EX | vision_budget.fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            vision_budget.run_budgeted_worker([], tmp_path, {"replacement": True},
                                              tmp_path / "response.json", request_path=request_path)
    assert read_json(request_path) == {"original": True}


def test_worker_request_is_written_inside_lock_before_start(tmp_path):
    request_path, response_path = tmp_path / "request.json", tmp_path / "response.json"
    script = f"""import json
from pathlib import Path
request=json.loads(Path({str(request_path)!r}).read_text())
Path({str(response_path)!r}).write_text(json.dumps({{'items':[],'received':request}}))
"""
    result = vision_budget.run_budgeted_worker([sys.executable, "-c", script], tmp_path,
                                               {"input": "exact"}, response_path, request_path=request_path)
    assert result["received"] == {"input": "exact"}
