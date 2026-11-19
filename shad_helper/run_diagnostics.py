"""Current-invocation code revision and nested wall-clock step timings."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from inspect import signature
from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired
from subprocess import run as run_process
from threading import RLock
from time import monotonic

_current_run = ContextVar("run_diagnostics", default=None)
_request_kind = ContextVar("notes_request_kind", default=None)
_current_steps = ContextVar("run_diagnostic_steps", default=None)


@contextmanager
def record_requests(client, folder):
    """Archive the exact requests for one model operation, including credential fallback."""
    if not hasattr(client, "request_recorder"):
        yield
        return

    def record(method, url, payload):
        from .jobs import write_json

        number = len(list(folder.glob("http-*.request.json"))) + 1
        prefix = f"http-{number:03d}"
        body_file = f"{prefix}.body.json" if payload is not None else None
        if body_file:
            (folder / body_file).write_bytes(payload)
        write_json(folder / f"{prefix}.request.json", {"method": method, "url": url, "body": body_file})

    previous = client.request_recorder
    client.request_recorder = record
    try:
        yield
    finally:
        client.request_recorder = previous


@contextmanager
def request_operation(name):
    token = _request_kind.set(name)
    try:
        yield
    finally:
        _request_kind.reset(token)


def record_generation_request():
    run, kind = _current_run.get(), _request_kind.get()
    if run and kind:
        with run[3]:
            counts = run[1].setdefault("generation_requests", {})
            counts[kind] = counts.get(kind, 0) + 1


def chapter_overlap_seconds(report):
    """Wall time spent in chapter jobs while the vision subprocess was active."""
    stages = {step["name"]: step for step in report["steps"]}
    vision = next((s for s in stages["extract"]["steps"] if s["name"] == "visual_extraction"), None)
    if vision is None:
        return 0.0
    start = vision["start_offset_seconds"]
    end = start + vision["elapsed_seconds"]
    total = 0.0
    for step in stages["notes"]["steps"]:
        if step["name"].startswith("chapter/"):
            began = step["start_offset_seconds"]
            total += max(0.0, min(end, began + step["elapsed_seconds"]) - max(start, began))
    return total


def _revision():
    root = Path(__file__).resolve().parent.parent
    if not (root / ".git").exists():
        return {"commit_hash": None, "git_dirty": None}
    try:
        commit = run_process(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = run_process(["git", "status", "--porcelain"], cwd=root, check=True,
                            capture_output=True, text=True, timeout=5).stdout
        return {"commit_hash": commit, "git_dirty": bool(dirty)}
    except (OSError, CalledProcessError, TimeoutExpired):
        return {"commit_hash": None, "git_dirty": None}


def _save(run):
    from .jobs import write_json

    job, document, started, lock = run
    with lock:
        document["elapsed_seconds"] = monotonic() - started
        write_json(job / "run-diagnostics.json", document)


@contextmanager
def diagnostic_run(job, command):
    """Share one report across a job's pipeline stages; replace it on the next run."""
    job = Path(job).resolve()
    active = _current_run.get()
    if active is not None and active[0] == job:
        yield active[1]
        return
    started = monotonic()
    started_at = datetime.now(UTC).isoformat()
    document = {"schema_version": 1, "command": command, **_revision(),
                "started_at": started_at, "finished_at": None,
                "status": "running", "steps": []}
    run = (job, document, started, RLock())
    token = _current_run.set(run)
    steps_token = _current_steps.set(document["steps"])
    try:
        _save(run)
        yield document
        document["status"] = ("handoff" if any(step["status"] == "handoff"
                                               for step in document["steps"]) else "completed")
    except BaseException as error:
        document["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        document["error_type"] = type(error).__name__
        raise
    finally:
        _current_steps.reset(steps_token)
        _current_run.reset(token)
        document["finished_at"] = datetime.now(UTC).isoformat()
        _save(run)


@contextmanager
def timed_step(name):
    """Measure this invocation, including failed attempts and cache lookup time."""
    active = _current_run.get()
    record = {"name": name, "status": "running", "elapsed_seconds": None, "steps": []}
    if active is None:
        yield record
        return
    started = monotonic()
    record["start_offset_seconds"] = started - active[2]
    with active[3]:
        _current_steps.get().append(record)
    token = _current_steps.set(record["steps"])
    try:
        yield record
        if record["status"] == "running":
            record["status"] = "completed"
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        with active[3]:
            record["error_type"] = type(error).__name__
        raise
    finally:
        _current_steps.reset(token)
        record["elapsed_seconds"] = monotonic() - started
        _save(active)


def record_run(command, *, job_argument="job"):
    """Also record direct Python stage calls, without changing their API."""
    def decorate(function):
        parameters = signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            job = Path(parameters.bind(*args, **kwargs).arguments[job_argument]).resolve()
            with diagnostic_run(job, command), timed_step(command) as step:
                result = function(*args, **kwargs)
                if command == "notes" and result == job / "handoff":
                    step["status"] = "handoff"
                return result

        return wrapped
    return decorate
