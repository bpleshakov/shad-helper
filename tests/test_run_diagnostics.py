import subprocess
from datetime import datetime
from types import SimpleNamespace

import pytest

from shad_helper import jobs, notes
from shad_helper import run_diagnostics as diagnostics
from shad_helper.jobs import read_json, write_json


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(diagnostics, "monotonic", lambda: clock.now)
    monkeypatch.setattr(diagnostics, "_revision", lambda: {
        "commit_hash": "a" * 40, "git_dirty": True})
    return clock


def test_nested_steps_record_current_wall_time_and_source_revision(tmp_path, clock):
    with diagnostics.diagnostic_run(tmp_path, "notes"):
        assert read_json(tmp_path / "run-diagnostics.json")["status"] == "running"
        with diagnostics.timed_step("chapter"):
            clock.now += 2
            with diagnostics.timed_step("compile"):
                clock.now += 3
            partial = read_json(tmp_path / "run-diagnostics.json")
            assert partial["steps"][0]["steps"][0]["elapsed_seconds"] == 3
            clock.now += 1
    result = read_json(tmp_path / "run-diagnostics.json")
    assert result["commit_hash"] == "a" * 40 and result["git_dirty"] is True
    assert result["command"] == "notes" and result["status"] == "completed"
    assert result["elapsed_seconds"] == result["steps"][0]["elapsed_seconds"] == 6
    assert result["steps"][0]["status"] == "completed"
    assert datetime.fromisoformat(result["started_at"]) <= datetime.fromisoformat(result["finished_at"])


@pytest.mark.parametrize("error,status", [
    (ValueError("bad source"), "failed"), (KeyboardInterrupt(), "interrupted"),
])
def test_failed_or_interrupted_step_is_persisted_without_swallowing_error(tmp_path, clock, error, status):
    with (
        pytest.raises(type(error)),
        diagnostics.diagnostic_run(tmp_path, "extract"),
        diagnostics.timed_step("speech_recognition"),
    ):
        clock.now += 4
        raise error
    result = read_json(tmp_path / "run-diagnostics.json")
    assert result["status"] == result["steps"][0]["status"] == status
    assert result["elapsed_seconds"] == result["steps"][0]["elapsed_seconds"] == 4
    assert result["steps"][0]["error_type"] == type(error).__name__
    # Failure must not leak the previous context into subsequent standalone work.
    with diagnostics.timed_step("outside_run"):
        clock.now += 1
    assert read_json(tmp_path / "run-diagnostics.json") == result


def test_cached_checkpoint_records_lookup_time_not_historic_runtime(tmp_path, clock, monkeypatch):
    def generate():
        clock.now += 12
        return {"value": 42}

    with diagnostics.diagnostic_run(tmp_path, "notes"):
        jobs.checkpoint(tmp_path, "notes/ch001", {}, generate)
    first = read_json(tmp_path / "run-diagnostics.json")
    assert first["steps"][0]["elapsed_seconds"] == 12
    checkpoint = tmp_path / "checkpoints/notes/ch001.json"
    cached = read_json(checkpoint)
    write_json(checkpoint, {**cached, "seconds": 123456})
    read = jobs.read_json

    def timed_read(path):
        clock.now += 0.25
        return read(path)

    monkeypatch.setattr(jobs, "read_json", timed_read)
    with diagnostics.diagnostic_run(tmp_path, "notes"):
        assert jobs.checkpoint(tmp_path, "notes/ch001", {}, generate) == {"value": 42}
    result = read_json(tmp_path / "run-diagnostics.json")
    assert len(result["steps"]) == 1
    assert result["steps"][0]["status"] == "cached"
    assert result["steps"][0]["elapsed_seconds"] == result["elapsed_seconds"] == 0.25
    assert read_json(checkpoint)["seconds"] == 123456


def test_new_run_replaces_old_steps_and_nested_jobs_are_isolated(tmp_path, clock):
    parent, child = tmp_path / "parent", tmp_path / "child"
    with diagnostics.diagnostic_run(parent, "run"), diagnostics.timed_step("parent_work"):
        clock.now += 1
        with diagnostics.diagnostic_run(child, "notes"), diagnostics.timed_step("child_work"):
            clock.now += 2
    assert read_json(parent / "run-diagnostics.json")["elapsed_seconds"] == 3
    assert read_json(child / "run-diagnostics.json")["elapsed_seconds"] == 2
    assert [step["name"] for step in read_json(parent / "run-diagnostics.json")["steps"]] == ["parent_work"]
    with diagnostics.diagnostic_run(parent, "render"), diagnostics.timed_step("compile"):
        clock.now += 4
    result = read_json(parent / "run-diagnostics.json")
    assert result["command"] == "render" and result["elapsed_seconds"] == 4
    assert [step["name"] for step in result["steps"]] == ["compile"]
    assert not (parent / "runs").exists()


def test_stage_decorator_supports_keyword_arguments_and_shares_pipeline_report(tmp_path, clock):
    @diagnostics.record_run("extract", job_argument="out")
    def extract(media, out):
        clock.now += 3
        return out / "evidence.json"

    @diagnostics.record_run("notes")
    def build_notes(job, state):
        clock.now += 2
        return job / "handoff"

    with diagnostics.diagnostic_run(tmp_path, "run"):
        assert extract(media="lecture.mp4", out=tmp_path) == tmp_path / "evidence.json"
        assert build_notes(tmp_path, state="auth") == tmp_path / "handoff"
    result = read_json(tmp_path / "run-diagnostics.json")
    assert result["command"] == "run" and result["status"] == "handoff"
    assert [(step["name"], step["elapsed_seconds"]) for step in result["steps"]] == [
        ("extract", 3), ("notes", 2)]
    assert result["steps"][1]["status"] == "handoff"


def test_failed_first_extraction_can_resume_from_diagnostics_only_directory(tmp_path, clock):
    with pytest.raises(ValueError), diagnostics.diagnostic_run(tmp_path, "extract"):
        raise ValueError("Missing dependency")
    jobs.check_job(tmp_path)
    (tmp_path / "unknown-file").touch()
    with pytest.raises(ValueError, match="new output directory"):
        jobs.check_job(tmp_path)


def test_revision_comes_from_code_checkout_not_current_directory(tmp_path, monkeypatch):
    repository = tmp_path / "source"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    (repository / "tracked.py").write_text("original\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=repository, check=True)
    subprocess.run(["git", "-c", "user.name=Diagnostics Test", "-c", "user.email=test@example.test",
                    "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Test revision"],
                   cwd=repository, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, check=True,
                            capture_output=True, text=True).stdout.strip()
    monkeypatch.setattr(diagnostics, "__file__", str(repository / "shad_helper/run_diagnostics.py"))
    monkeypatch.chdir(tmp_path)
    assert diagnostics._revision() == {"commit_hash": commit, "git_dirty": False}
    (repository / "tracked.py").write_text("changed\n")
    assert diagnostics._revision() == {"commit_hash": commit, "git_dirty": True}


@pytest.mark.parametrize("unavailable", ["no_checkout", "no_git"])
def test_unavailable_git_metadata_is_explicitly_null(tmp_path, monkeypatch, unavailable):
    monkeypatch.setattr(diagnostics, "__file__", str(tmp_path / "shad_helper/run_diagnostics.py"))
    if unavailable == "no_git":
        (tmp_path / ".git").mkdir()

        def missing(*args, **kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(diagnostics, "run_process", missing)
    assert diagnostics._revision() == {"commit_hash": None, "git_dirty": None}


def test_real_notes_build_records_cache_reconciliation_and_project_steps(tmp_path, clock):
    write_json(tmp_path / "evidence.json", {
        "schema_version": 3, "title": "Lecture", "media": {"duration": 30},
        "segments": [{"id": "s1", "start": 0, "end": 30, "text": "Statement"}]})

    class Client:
        model = "fake"

        def complete(self, instructions, text, schema=None):
            clock.now += 2
            return {"title": "Lecture", "chapters": [
                {"id": "ch001", "title": "Lecture", "start": 0, "end": 30, "blocks": [
                    {"id": "text", "kind": "text", "content": "Statement", "evidence_ids": ["s1"]}]}],
                "glossary": [], "issues": []}

    for expected_status in ("completed", "cached"):
        assert notes.generate_notes(tmp_path, tmp_path / "auth", client=Client()).exists()
        report = read_json(tmp_path / "run-diagnostics.json")
        assert report["command"] == "notes" and report["status"] == "completed"
        steps = report["steps"][0]["steps"]
        names = [step["name"] for step in steps]
        assert {"outline_preparation", "handoff_export", "checkpoint/notes/ch001",
                "chapter_merge", "figure_reconciliation", "latex_project"} <= set(names)
        generated = next(step for step in steps if step["name"] == "checkpoint/notes/ch001")
        assert generated["status"] == expected_status
        assert generated["elapsed_seconds"] == (2 if expected_status == "completed" else 0)
