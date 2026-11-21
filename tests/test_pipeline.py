"""Batch stage ordering, isolation, resumption, and Anki position allocation."""

import json
import sqlite3
import subprocess
import zipfile
from types import SimpleNamespace
from unittest.mock import ANY, Mock, call

import pymupdf
import pytest

from shad_helper import anki, pipeline
from shad_helper.jobs import read_json, rendered_pdf, write_json
from shad_helper.run_diagnostics import record_run


@pytest.fixture
def media(tmp_path):
    paths = [tmp_path / f"lecture-{number}.mp4" for number in range(3)]
    for path in paths:
        path.touch()
    return paths


@pytest.fixture
def stages(monkeypatch):
    calls = Mock()
    calls.extract.side_effect = lambda source, job, **kwargs: job / "evidence.json"
    calls.generate_notes.side_effect = lambda job, state, **kwargs: job / "notes" / "main.tex"
    calls.render.side_effect = lambda job, **kwargs: job / "notes" / "main.pdf"

    def export(job, state, *, minimum_position, **kwargs):
        write_json(job / "anki" / "report.json", {
            "end_position": max(100, minimum_position) + 2,
        })
        return job / "anki" / "deck.apkg"

    calls.generate_anki.side_effect = export
    for name in ("extract", "generate_notes", "render", "generate_anki"):
        monkeypatch.setattr(pipeline, name, getattr(calls, name))
    return calls


@pytest.mark.parametrize("mode", ["frames", "video"])
def test_runs_all_stages_for_each_file_and_forwards_options(media, stages, tmp_path, capsys, mode):
    state, collection = tmp_path / "auth", tmp_path / "collection.anki2"
    assert pipeline.run_pipeline(media[:2], tmp_path / "out", state, language="en", start=10,
                                 duration=30, vision_model="35b", vision_mode=mode, rebuild=True,
                                 proxy="http://localhost:8080", anki_collection=collection)
    jobs = [item.args[1] for item in stages.extract.call_args_list]
    expected = []
    for index, (source, job) in enumerate(zip(media, jobs)):
        expected.extend([
            call.extract(source, job, language="en", start=10, duration=30,
                         vision_model="35b", vision_mode=mode, chapters=ANY),
            call.generate_notes(job, state, rebuild=True, proxy="http://localhost:8080"),
            call.render(job, rebuild=True),
            call.generate_anki(job, state, pdf=job / "notes" / "main.pdf", rebuild=True,
                               proxy="http://localhost:8080", anki_collection=collection,
                               minimum_position=1 if index == 0 else 103),
        ])
    assert stages.mock_calls == expected
    assert len(set(jobs)) == 2
    output = capsys.readouterr()
    assert "2 completed, 0 failed" in output.out
    assert output.err == ""


def test_jobs_stable_across_reordered_runs_and_duplicate_stems(tmp_path, stages):
    sources = [tmp_path / "first" / "Лекция 1.mp4", tmp_path / "second" / "Лекция 1.mp4",
               tmp_path / "first" / "Лекция 1.mkv"]
    for source in sources:
        source.parent.mkdir(exist_ok=True)
        source.touch()
    assert pipeline.run_pipeline(sources, tmp_path / "out", tmp_path / "auth")
    jobs = {item.args[0]: item.args[1] for item in stages.extract.call_args_list}
    assert len(set(jobs.values())) == 3
    assert all(job.name.startswith("Лекция-1-") for job in jobs.values())
    stages.reset_mock()
    assert pipeline.run_pipeline(sources[::-1], tmp_path / "out", tmp_path / "auth")
    assert {item.args[0]: item.args[1] for item in stages.extract.call_args_list} == jobs


@pytest.mark.parametrize("failure", [None, "render", "interrupt"])
def test_each_job_keeps_a_combined_pipeline_diagnostic_report(
        media, stages, tmp_path, monkeypatch, failure):
    # The existing stage mocks stand in for the decorated public Python APIs.
    for attribute, command in (("extract", "extract"), ("generate_notes", "notes"),
                               ("render", "render"), ("generate_anki", "anki")):
        action = getattr(stages, attribute)
        if command == "extract":
            def invoke(media, out, *, _action=action, **kwargs):
                return _action(media, out, **kwargs)
        else:
            def invoke(job, *args, _action=action, **kwargs):
                return _action(job, *args, **kwargs)
        monkeypatch.setattr(pipeline, attribute, record_run(
            command, job_argument="out" if command == "extract" else "job")(invoke))
    if failure:
        stages.render.side_effect = KeyboardInterrupt() if failure == "interrupt" else ValueError("Bad TeX")
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            pipeline.run_pipeline(media[:2], tmp_path / "out", tmp_path / "auth", rebuild=True)
    else:
        assert pipeline.run_pipeline(media[:2], tmp_path / "out", tmp_path / "auth", rebuild=True) is (failure is None)
    reports = [read_json(call.args[1] / "run-diagnostics.json") for call in stages.extract.call_args_list]
    assert len(reports) == (1 if failure == "interrupt" else 2)
    for report in reports:
        assert report["command"] == "run"
        assert report["status"] == ("interrupted" if failure == "interrupt" else "failed" if failure else "completed")
        assert [step["name"] for step in report["steps"]] == (
            ["extract", "notes", "render"] if failure else ["extract", "notes", "render", "anki"])
        assert all(step["elapsed_seconds"] >= 0 for step in report["steps"])
        assert sum(step["elapsed_seconds"] for step in report["steps"]) <= report["elapsed_seconds"]


@pytest.mark.parametrize("invalid", ["empty", "missing", "directory", "repeated", "alias"])
def test_invalid_batch_is_rejected_before_any_stage(invalid, media, stages, tmp_path):
    inputs = media[:1]
    if invalid == "empty":
        inputs = []
    elif invalid == "missing":
        inputs.append(tmp_path / "missing.mp4")
    elif invalid == "directory":
        inputs.append(tmp_path)
    elif invalid == "repeated":
        inputs.append(inputs[0])
    else:
        alias = tmp_path / "alias.mp4"
        alias.symlink_to(inputs[0])
        inputs.append(alias)
    with pytest.raises(ValueError):
        pipeline.run_pipeline(inputs, tmp_path / "out", tmp_path / "auth")
    assert stages.mock_calls == []
    assert not (tmp_path / "out").exists()


def test_job_name_collision_is_rejected_before_any_stage(tmp_path, stages, monkeypatch):
    sources = [tmp_path / folder / "lecture.mp4" for folder in ("first", "second")]
    for source in sources:
        source.parent.mkdir()
        source.touch()
    monkeypatch.setattr(pipeline.hashlib, "sha256",
                        lambda value: SimpleNamespace(hexdigest=lambda: "0" * 64))
    with pytest.raises(ValueError, match="same job directory"):
        pipeline.run_pipeline(sources, tmp_path / "out", tmp_path / "auth")
    assert stages.mock_calls == []


@pytest.mark.parametrize(("stage", "error"), [
    ("extract", OSError("Cannot read video")),
    ("generate_notes", ValueError("Invalid response")),
    ("render", subprocess.CalledProcessError(1, "latexmk")),
    ("generate_anki", RuntimeError("Invalid PDF")),
    ("extract", ModuleNotFoundError("Missing local model")),
])
def test_stage_failure_stops_only_affected_job_and_batch_continues(
        media, stages, tmp_path, capsys, stage, error):
    failing = getattr(stages, stage)
    original = failing.side_effect
    attempted = False

    def fail_once(*args, **kwargs):
        nonlocal attempted
        if not attempted:
            attempted = True
            raise error
        return original(*args, **kwargs)

    failing.side_effect = fail_once
    assert not pipeline.run_pipeline(media, tmp_path / "out", tmp_path / "auth")
    names = [item[0] for item in stages.mock_calls]
    order = ["extract", "generate_notes", "render", "generate_anki"]
    assert names == order[:order.index(stage) + 1] + order * 2
    calls = stages.generate_anki.call_args_list
    assert calls[-2].kwargs["minimum_position"] == 1
    assert calls[-1].kwargs["minimum_position"] == 103
    output = capsys.readouterr()
    assert str(media[0]) in output.err
    assert f"{stage.removeprefix('generate_')} failed" in output.err
    assert "2 completed, 1 failed" in output.out


def test_handoff_stops_before_stale_rendered_artifacts(media, stages, tmp_path, capsys):
    first_job = pipeline._jobs(media[:1], tmp_path / "out")[0][1]
    (first_job / "notes").mkdir(parents=True)
    (first_job / "notes" / "main.tex").write_text("stale source")
    (first_job / "notes" / "main.pdf").write_bytes(b"stale PDF")
    stages.generate_notes.side_effect = lambda job, state, **kwargs: (
        job / "handoff" if job == first_job else job / "notes" / "main.tex")
    assert not pipeline.run_pipeline(media[:2], tmp_path / "out", tmp_path / "auth")
    assert stages.render.call_count == stages.generate_anki.call_count == 1
    assert stages.render.call_args.args[0] != first_job
    assert "manual completion" in capsys.readouterr().err


def test_failed_job_does_not_reset_position_cursor(media, stages, tmp_path):
    original = stages.render.side_effect
    attempted = 0

    def render(job):
        nonlocal attempted
        attempted += 1
        if attempted == 2:
            raise RuntimeError("PDF compilation failed")
        return original(job)

    stages.render.side_effect = render
    assert not pipeline.run_pipeline(media, tmp_path / "out", tmp_path / "auth")
    assert [item.kwargs["minimum_position"] for item in
            stages.generate_anki.call_args_list] == [1, 103]


def test_keyboard_interrupt_stops_batch_immediately(media, stages, tmp_path):
    stages.extract.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_pipeline(media, tmp_path / "out", tmp_path / "auth")
    assert stages.extract.call_count == 1
    stages.generate_notes.assert_not_called()


def test_batch_exports_real_independent_packages_with_disjoint_positions(
        media, tmp_path, monkeypatch):
    collection = tmp_path / "collection.anki2"
    with sqlite3.connect(collection) as connection:
        connection.execute("CREATE TABLE cards (type INT, due INT, odid INT, odue INT)")
        connection.execute("INSERT INTO cards VALUES (0, 700, 0, 0)")
        connection.execute("CREATE TABLE col (conf TEXT)")
        connection.execute("INSERT INTO col VALUES (?)", (json.dumps({"nextPos": 703}),))
    original_collection = collection.read_bytes()
    response = {"title": "Одинаковое название лекции", "notes": [
        {"note_type": "Basic", "front": "Определение", "back": "Ответ",
         "source": {"page": 1, "order": 1},
         "card_sources": [{"field": "Back", "page": 1, "location": "Определение"}]},
        {"note_type": "Theorem Full proof", "name": "Теорема", "statement": "Утверждение",
         "proof": "Доказательство", "source": {"page": 1, "order": 2},
         "card_sources": [{"field": field, "page": 1, "location": "Теорема"}
                          for field in ("Statement", "Full proof")]},
    ]}
    client = SimpleNamespace(model=anki.DEFAULT_MODEL, complete=Mock(return_value=response))
    monkeypatch.setattr(anki, "PlanClient", lambda state, **kwargs: client)
    monkeypatch.setattr(pipeline, "extract", lambda source, job, **kwargs:
                        write_json(job / "evidence.json", {"media": {"path": str(source)}}))
    monkeypatch.setattr(pipeline, "generate_notes",
                        lambda job, state, **kwargs: job / "notes" / "main.tex")

    def render(job):
        pdf = rendered_pdf(job)
        pdf.parent.mkdir(parents=True)
        with pymupdf.open() as document:
            document.new_page().insert_text((24, 24), "Lecture notes")
            document.save(pdf)
        return pdf

    monkeypatch.setattr(pipeline, "render", render)
    assert pipeline.run_pipeline(media[:2], tmp_path / "out", tmp_path / "auth",
                                 anki_collection=collection)
    packages = sorted((tmp_path / "out").glob("*/anki/*.apkg"))
    assert [package.name for package in packages] == ["lecture-0.apkg", "lecture-1.apkg"]
    decks, guids, ranges = [], [], []
    for package in packages:
        report = read_json(package.with_name("report.json"))
        decks.append(report["deck_id"])
        ranges.append(set(range(report["start_position"], report["end_position"] + 1)))
        with zipfile.ZipFile(package) as archive, sqlite3.connect(":memory:") as connection:
            connection.deserialize(archive.read("collection.anki2"))
            rows = connection.execute(
                "SELECT n.guid, c.due, c.did FROM cards c JOIN notes n ON n.id = c.nid"
            ).fetchall()
        by_note = {}
        for guid, due, deck_id in rows:
            assert deck_id == report["deck_id"]
            by_note.setdefault(guid, set()).add(due)
        assert len(rows) == 3
        assert len(by_note) == 2
        assert all(len(positions) == 1 for positions in by_note.values())
        guids.append(set(by_note))
    assert decks[0] != decks[1]
    assert guids[0].isdisjoint(guids[1])
    assert ranges == [{703, 704}, {705, 706}]
    assert collection.read_bytes() == original_collection
    assert client.complete.call_count == 2
