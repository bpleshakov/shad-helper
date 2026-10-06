"""Persist the cumulative Qwen allowance and enforce it outside model workers."""

import fcntl
import math
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from .jobs import read_json, sha256_file, write_json

BUDGET_FILE = "sparse-qwen-budget.json"
HEARTBEAT_FILE = "sparse-qwen-heartbeat.json"


class BudgetExhausted(RuntimeError):
    """No more active Qwen work is permitted for this job."""


def _heartbeat(job, state):
    write_json(Path(job) / HEARTBEAT_FILE, {"active_pid": state["active_pid"],
                                          "active_started": state["active_started"],
                                          "observed_time": time.time()})


def _settle(path, *, ended=None):
    state = read_json(path)
    started = state.get("active_started")
    if started is None:
        return state
    if ended is None:
        heartbeat_path = path.parent / HEARTBEAT_FILE
        heartbeat = read_json(heartbeat_path) if heartbeat_path.exists() else {}
        ended = (heartbeat.get("observed_time", started)
                 if heartbeat.get("active_pid") == state.get("active_pid")
                 and heartbeat.get("active_started") == started else started)
    state["used_seconds"] += max(0, ended - started)
    state.update(active_started=None, active_pid=None, phase=None)
    write_json(path, state)
    return state


def _limit(request):
    minutes = request.get("vision_budget_minutes")
    if minutes is None:
        return (request["end"] - request["start"]) * 0.5
    if isinstance(minutes, bool) or not isinstance(minutes, (int, float)):
        raise ValueError("The Qwen budget must be a positive number of minutes.")  # noqa: TRY004
    if not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("The Qwen budget must be a positive number of minutes.")
    return minutes * 60


class Budget:
    def __init__(self, job, request):
        self.path = Path(job) / BUDGET_FILE
        self.state = (read_json(self.path) if self.path.exists() else
                      {"version": 1, "limit_seconds": _limit(request), "used_seconds": 0.0,
                       "active_started": None, "active_pid": None, "phase": None})
        if self.state.get("version") != 1:
            raise ValueError("Unsupported Qwen budget version.")
        self._started = None
        self._last_heartbeat = -math.inf
        self.settle_interrupted()
        if request.get("vision_budget_minutes") is not None:
            self.state["limit_seconds"] = _limit(request)
        write_json(self.path, self.state)

    def settle_interrupted(self):
        started = self.state.get("active_started")
        if started is None:
            return
        pid = self.state.get("active_pid")
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, TypeError):
            pass
        else:
            raise RuntimeError("Another Qwen worker is still active for this job.")
        self.state = _settle(self.path)

    @property
    def remaining(self):
        active = 0 if self._started is None else time.monotonic() - self._started
        return max(0.0, self.state["limit_seconds"] - self.state["used_seconds"] - active)

    def check(self):
        if self._started is not None and time.monotonic() - self._last_heartbeat >= 0.25:
            _heartbeat(self.path.parent, self.state)
            self._last_heartbeat = time.monotonic()
        if self.remaining <= 0:
            raise BudgetExhausted("The cumulative Qwen time budget is exhausted.")

    @contextmanager
    def active(self, phase):
        self.check()
        self._started = time.monotonic()
        self.state.update(active_started=time.time(), active_pid=os.getpid(), phase=phase)
        write_json(self.path, self.state)
        _heartbeat(self.path.parent, self.state)
        try:
            yield
        finally:
            self.state["used_seconds"] += max(0, time.monotonic() - self._started)
            self._started = None
            self.state.update(active_started=None, active_pid=None, phase=None)
            write_json(self.path, self.state)

    def summary(self):
        return {"limit_seconds": self.state["limit_seconds"],
                "used_seconds": self.state["used_seconds"], "remaining_seconds": self.remaining}


def deferred(item, reason="Qwen time budget exhausted; visual evidence has not been interpreted."):
    return {**item, "status": "deferred", "kind": item.get("kind", "other"),
            "summary": "", "regions": [], "diagram": "", "issues": [reason],
            "selection_issues": list(item.get("selection_issues", []))}


def recover_results(job, request):
    """Only completed results survive a watchdog termination."""
    job = Path(job)
    manifest = job / "sparse-selection.json"
    selected = read_json(manifest) if manifest.exists() else {"items": [], "audit": {}}
    saved = job / "sparse-vision-results.json"
    result = read_json(saved) if saved.exists() else {"items": [], "audit": {}}
    completed = {item["id"]: item for item in result["items"] if item.get("status") in ("processed", "failed")}
    items = {item["id"]: item for item in [*selected["items"], *request.get("items", [])]}
    outputs = []
    for identifier, item in items.items():
        saved_item = completed.get(identifier)
        fields = ("image", "timestamp", "page_id", "page_start", "page_end", "selection_issues",
                  "selection_role", "complement_of")
        matching = saved_item is not None and all(saved_item.get(key) == item.get(key) for key in fields)
        image = Path(item["image"]) if item.get("image") else None
        if matching and image is not None and image.exists():
            matching = saved_item.get("sha256") == sha256_file(image)
        outputs.append(saved_item if matching else deferred(item))
    # A routing deadline can occur before the complete selection manifest exists.
    audit = {**selected.get("audit", {}), **result.get("audit", {}),
             "budget_exhausted": True, "watchdog_terminated": True}
    if not manifest.exists():
        audit["unprocessed_intervals"] = [{"start": request["start"], "end": request["end"],
                                           "reason": "routing_interrupted"}]
    state_path = job / BUDGET_FILE
    if state_path.exists():
        state = _settle(state_path, ended=time.time())
        audit["qwen_budget"] = {"limit_seconds": state["limit_seconds"],
                                "used_seconds": state["used_seconds"],
                                "remaining_seconds": max(0, state["limit_seconds"] - state["used_seconds"])}
    recovered = {"items": outputs, "audit": audit}
    write_json(saved, recovered)
    return recovered


def run_budgeted_worker(command, job, request, response_path, *, request_path=None):
    """Reject concurrent jobs before replacing their request or budget state."""
    with (Path(job) / "sparse-qwen-worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A Qwen worker is already running for this job.") from error
        if request_path is not None:
            write_json(request_path, request)
        return _run_budgeted_worker(command, job, request, response_path)


def _run_budgeted_worker(command, job, request, response_path):
    """Kill the worker process group when active Qwen work uses its allowance."""
    response_path = Path(response_path)
    response_path.unlink(missing_ok=True)
    process = subprocess.Popen(command, start_new_session=True)
    exhausted = False
    state_path = Path(job) / BUDGET_FILE
    last_heartbeat = -math.inf
    try:
        while process.poll() is None:
            if state_path.exists():
                state = read_json(state_path)
                if state.get("active_pid") == process.pid and state.get("active_started") is not None:
                    if time.monotonic() - last_heartbeat >= 0.25:
                        _heartbeat(job, state)
                        last_heartbeat = time.monotonic()
                    used = state["used_seconds"] + max(0, time.time() - state["active_started"])
                    if used >= state["limit_seconds"]:
                        exhausted = True
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                        break
            time.sleep(0.05)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if state_path.exists() and read_json(state_path).get("active_pid") == process.pid:
            _settle(state_path, ended=time.time())
    if exhausted:
        recovered = recover_results(job, request)
        write_json(response_path, recovered)
        return recovered
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return read_json(response_path)
