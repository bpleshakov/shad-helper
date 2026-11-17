"""Current-invocation code revision and nested wall-clock step timings."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from inspect import signature
from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired
from subprocess import run as run_process
from time import monotonic

_current_run = ContextVar("run_diagnostics", default=None)
_current_steps = ContextVar("run_diagnostic_steps", default=None)


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

    job, document, started = run
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
    run = (job, document, started)
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
    _current_steps.get().append(record)
    token = _current_steps.set(record["steps"])
    try:
        yield record
        if record["status"] == "running":
            record["status"] = "completed"
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
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
