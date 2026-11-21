"""Ordered chapter overlap, evidence parity, cancellation and resumable publication."""

import json
import os
import pty
import sys
from contextlib import closing
from io import StringIO
from threading import Event, Thread, get_ident
from unittest.mock import Mock

import pytest
from PIL import Image

from shad_helper import chapter_pipeline, extraction, notes, pipeline, rendering, vision_worker
from shad_helper.chapter_progress import (
    ChapterCancelled,
    ChapterProgress,
    NotesCancellation,
    notes_display,
    notes_phase,
)
from shad_helper.jobs import read_json, sha256_file, write_json
from shad_helper.openai_client import ResponseError
from shad_helper.openai_usage import command_usage, track_request
from shad_helper.run_diagnostics import timed_step


class NotesClient:
    model = "test-model"

    def __init__(self):
        self.calls = []
        self.before = lambda context: None

    def complete(self, _, text, schema=None, *, images=()):
        payload = json.loads(text)
        self.calls.append((get_ident(), payload, [(name, sha256_file(path)) for name, path in images]))
        with track_request(api_key=False, model=self.model):
            self.before(payload)
        chapter = payload["chapter"]
        blocks = [{"id": chapter["id"] + "_text", "kind": "text", "topic": "Пример",
                   "content": "Объяснение.", "evidence_ids": [payload["segments"][0]["id"]]}]
        coverage = []
        for source in payload["required_illustration_ids"]:
            key = f"{chapter['id']}_{source}"
            blocks.append({"id": key, "kind": "screenshot", "illustration_id": source,
                           "evidence_ids": [source], "content": "Схема", "caption": "Схема"})
            blocks[0].setdefault("figure_refs", []).append(key)
            coverage.append({"illustration_id": source, "block_id": key})
        return {"title": "Лекция", "chapters": [{**chapter, "blocks": blocks,
                 "illustration_coverage": coverage}],
                "glossary": [{"term": chapter["id"], "definition": "Термин"}]}


@pytest.fixture
def lecture(tmp_path, monkeypatch):
    media = tmp_path / "lecture.mp4"
    media.write_bytes(b"lecture")
    out = tmp_path / "jobs"
    job = pipeline._jobs([media], out)[0][1]
    timestamps = [10, 898, 960, 962, 1900, 2700]
    monkeypatch.setattr(extraction, "probe", lambda _: {"duration": 2700})
    monkeypatch.setattr(extraction, "probe_video", lambda _: {"end": 2700})
    monkeypatch.setattr(extraction, "_audio_chunks", lambda *args: [{}])
    monkeypatch.setattr(extraction, "_worker", lambda *args: {"items": []})
    segments = [{"id": "s1", "start": 1, "end": 2650, "text": "Тема лекции",
                 "words": [], "diagnostics": {}}]
    monkeypatch.setattr(extraction, "stitch_segments", lambda _: segments)
    monkeypatch.setattr(extraction, "transcription_warnings", lambda _: [])
    def selection(media, job, start, end, media_hash, **kwargs):
        frames = []
        for index, timestamp in enumerate(timestamps):
            key = f"f{index}"
            path = job / "frames" / f"{key}_{media_hash[:12]}.png"
            path.parent.mkdir(exist_ok=True)
            Image.new("RGB", (96, 72), (index * 13, 20, 120)).save(path)
            frames.append({"id": key, "timestamp": timestamp, "image": str(path)})
        return {"items": frames, "diagnostics": {}}

    monkeypatch.setattr(extraction, "select_frames", selection)
    client = NotesClient()
    monkeypatch.setattr(chapter_pipeline, "PlanClient", lambda *args, **kwargs: client)
    monkeypatch.setattr("shad_helper.illustration_verification.refine_crop",
                        lambda *args: ("", False, "", False))
    monkeypatch.setattr(chapter_pipeline, "validate_chapter_latex", lambda *args, **kwargs: None)
    rendered = []
    def render(job, **kwargs):
        assert read_json(job / "extraction-status.json")["state"] == "complete"
        assert (job / "notes/main.tex").is_file()
        rendered.append(job)
        return job / "notes/main.pdf"

    def anki(job, *args, **kwargs):
        assert rendered[-1] == job
        write_json(job / "anki/report.json", {"end_position": 1})
        return job / "anki/deck.apkg"

    monkeypatch.setattr(pipeline, "render", render)
    monkeypatch.setattr(pipeline, "generate_anki", anki)
    return media, out, job, client, rendered


def simulate_vision(monkeypatch, after=lambda _: None, *, illustration=False):
    @timed_step("visual_extraction")
    def vision(job, request, *, on_batch, log, **kwargs):
        items = []
        sizes = request.get("batch_sizes", [1] * len(request["items"]))
        for size in sizes:
            batch = []
            for frame in request["items"][len(items):len(items) + size]:
                observations = ([{"source_kind": "printed", "description": "Схема", "box": [.1,.1,.7,.7]}]
                                if illustration and frame["id"] == "f1" else [])
                batch.append({**frame, "content": {"kind": "slides", "text": frame["id"],
                    "formulas": [], "diagrams": [], "illustrations": observations, "unreadable": []},
                    "diagnostics": {"status": "complete"}})
            on_batch(batch)
            items.extend(batch)
            after(batch[-1]["id"])
        return {"items": items, "diagnostics": {"fresh_items": len(items)}}
    monkeypatch.setattr(extraction, "_vision_worker", vision)


@pytest.mark.parametrize("mode", ["frames", "video"])
def test_vision_finishes_while_first_notes_job_is_blocked(lecture, monkeypatch, mode, capsys):
    media, out, job, client, rendered = lecture
    writing, vision_finished = Event(), Event()
    def before(context):
        if context["chapter"]["id"] == "ch001":
            writing.set()
            assert vision_finished.wait(5)
    client.before = before

    def after(key):
        if key == ("f3" if mode == "video" else "f2"):
            assert writing.wait(5)
            assert not (job / "illustrations/manifest.json").exists()
            assert not (job / "evidence.json").exists()
        if key == "f5":
            vision_finished.set()
    simulate_vision(monkeypatch, after, illustration=True)
    with command_usage("test") as usage:
        assert pipeline.run_pipeline([media], out, out, vision_mode=mode)
        assert usage.plan_requests == 3
    assert rendered == [job]
    assert len({thread for thread, _, _ in client.calls} | {get_ident()}) == 2
    full = read_json(job / "evidence.json")
    outline = read_json(job / "outline.json")
    assert [call[1]["chapter"] for call in client.calls] == outline["chapters"]
    for index, (_, context, attachments) in enumerate(client.calls):
        ordinary = notes.chapter_context(full, outline["chapters"][index])
        assert all(context[key] == value for key, value in ordinary.items())
        assert attachments == [(label, sha256_file(path))
                               for label, path in notes._chapter_images(job, full, context)]
        assert len(context["glossary"]) == index
    assert [v["id"] for v in client.calls[0][1]["visuals"]] == ["f0","f1","f2"]
    assert client.calls[0][1]["required_illustration_ids"] == ["f1_ill01"]
    assert (job / "notes/figures/illustration-f1_ill01.png").is_file()
    assert read_json(job / "checkpoints/chapter-pipeline/notes-state.json")["chapter"] == "ch003"
    report = read_json(job / "run-diagnostics.json")
    assert [s["name"] for s in report["steps"]] == ["extract", "notes"]
    assert report["generation_requests"] == {"drafting": 3}
    assert 0 < report["chapter_worker_overlap_seconds"] < report["elapsed_seconds"]
    output = capsys.readouterr().err
    summary = "Video frame filtering complete — 0 removed, 6 remaining for vision."
    assert output.index(summary) < min(output.index("Vision  ch001"), output.index("Notes  ch001"))
    assert "Vision  ch001  1/3" in output and "Notes  ch002  2/3" in output
    assert "\x1b" not in output
    # Compare the actual prompts, attachment hashes and final content with standalone notes.
    parallel = read_json(job / "notes.json")
    standalone = NotesClient()
    assert notes.generate_notes(job, out, client=standalone, rebuild=True).is_file()
    assert [call[1:] for call in standalone.calls] == [call[1:] for call in client.calls]
    assert read_json(job / "notes.json") == parallel


def test_failed_child_stops_only_notes_and_resumes_from_that_child(lecture, monkeypatch):
    media, out, job, client, rendered = lecture
    split = []
    def fail(context):
        key = context["chapter"]["id"]
        if key == "ch002" or "_b_" in key:
            split.append(key)
            if key == "ch002":
                raise ResponseError("too long", code="context_length_exceeded")
            raise OSError("offline")
    client.before = fail
    finished = []
    simulate_vision(monkeypatch, finished.append)
    assert not pipeline.run_pipeline([media], out, out)
    assert not rendered and finished[-1] == "f5"
    assert read_json(job / "extraction-status.json")["state"] == "complete"
    leaves = [ch["id"] for ch in read_json(job / "outline.json")["chapters"]]
    assert [p[1]["chapter"]["id"] for p in client.calls] == ["ch001", "ch002", *leaves[1:3]]
    assert read_json(job / "handoff/status.json")["chapter"] == leaves[2]
    assert (job / "handoff/inputs/ch003/context.json").exists()
    assert read_json(job / "handoff/outline.json")["chapters"][-1]["id"] == "ch003"
    client.before = lambda _: None
    client.calls.clear()
    assert pipeline.run_pipeline([media], out, out)
    assert [p[1]["chapter"]["id"] for p in client.calls] == leaves[2:]
    assert len(read_json(job / "checkpoints/chapter-pipeline/plan.json")["base"]["chapters"]) == 3
    client.calls.clear()
    monkeypatch.setattr(chapter_pipeline.ChapterNotes, "_verify", Mock(side_effect=AssertionError))
    assert pipeline.run_pipeline([media], out, out)
    assert not client.calls


def test_fatal_vision_wakes_consumer_and_does_not_render_stale_pdf(lecture, monkeypatch):
    media, out, job, client, rendered = lecture
    old = job / "notes/main.pdf"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    write_json(job / "extraction-config.json", {"schema_version": 5, "pipeline": "audio_video",
                                              "vision_model": "35b"})
    def fail(_):
        raise RuntimeError("vision unavailable")
    simulate_vision(monkeypatch, fail)
    assert not pipeline.run_pipeline([media], out, out)
    assert read_json(job / "extraction-status.json")["state"] == "failed"
    assert read_json(job / "handoff/run.json")["status"] == "interrupted"
    assert not client.calls and not rendered and old.read_bytes() == b"old"


def test_two_rows_and_child_logs_over_dedicated_pipe():
    master, slave = pty.openpty()
    data = bytearray()
    done = Event()
    def drain():
        while b"finished-test" not in data:
            data.extend(os.read(master, 4096))
        done.set()
    reader = Thread(target=drain, daemon=True)
    reader.start()
    try:
        with os.fdopen(slave, "w") as terminal:
            progress = ChapterProgress(terminal)
            progress.start()
            first_rows = list(progress.rows.values())
            progress.update("Vision", "ch002  2/3", "inspecting", total=3, unit="frames")
            progress.update("Notes", "ch001  1/3", "drafting")
            script = ("import json,os,sys; fd=int(sys.argv[-1]); "
                      "print('child warning', file=sys.stderr); "
                      "os.write(fd, (json.dumps([{'id':'f1'}])+'\\n').encode()); "
                      "print('child stdout')")
            batches = []
            extraction._stream_vision([sys.executable, "-c", script], batches.append, progress.log)
            print("retry message", file=sys.stderr, flush=True)
            print("library output")
            progress.update("Notes", "ch001  1/3", "retry")
            assert first_rows == list(progress.rows.values())
            progress.close()
            print("finished-test", file=terminal, flush=True)
            assert done.wait(5)
            reader.join()
            output = data.decode()
        assert batches == [[{"id":"f1"}]]
        assert output.count("library output") == 1
        assert output.count("child warning") == output.count("child stdout") == output.count("retry message") == 1
        assert "Vision  ch002" in output and "Notes  ch001" in output and "\x1b[A" in output
    finally:
        os.close(master)


@pytest.fixture
def timed_progress(monkeypatch):
    terminal = StringIO()
    monkeypatch.setattr(terminal, "isatty", lambda: True)
    clock = Mock(return_value=0.0)
    monkeypatch.setattr("shad_helper.chapter_progress.monotonic", clock)
    with closing(ChapterProgress(terminal)) as progress:
        progress.start()
        yield progress, clock


@pytest.mark.parametrize("outcome", ["complete", "failed", "interrupted"])
def test_vision_chapter_and_total_elapsed(timed_progress, outcome):
    progress, clock = timed_progress
    first, second = "ch001  1/2", "ch002  2/2"
    vision = progress.rows["Vision"]

    def show(seconds, chapter, phase, chapter_seconds, **counts):
        clock.return_value = seconds
        progress.update("Vision", chapter, phase, **counts)
        line = str(vision)
        assert f"[chapter 00:{chapter_seconds:02d} | total 00:{seconds:02d}]" in line
        return line

    show(0, first, "inspecting", 0, total=3, unit="frames")
    assert "1/3 frames" in show(5, first, "inspecting", 5, done=1, total=3, unit="frames")
    show(8, first, "cached f2", 8)
    show(12, first, "retrying f3", 12)
    assert "3/3 frames" in show(16, first, "inspecting", 16, done=3, total=3, unit="frames")
    assert "0/2 frames" in show(16, second, "inspecting", 0, total=2, unit="frames")
    finished = show(24, second, outcome, 8)
    clock.return_value += 60
    progress.update("Notes", second, "drafting")
    progress.log("Notes still running")
    assert str(vision) == finished


@pytest.mark.parametrize("outcome", ["complete", "manual handoff"])
def test_notes_elapsed_excludes_waits_and_keeps_split_parts_together(timed_progress, outcome):
    progress, clock = timed_progress
    first = "ch001  1/2"
    second = "a" * 80 + "  2/2"
    notes_row = progress.rows["Notes"]
    progress.update("Vision", first, "inspecting")
    progress.update("Notes", first, "waiting for vision")
    clock.return_value = 10
    assert "[chapter 00:00 | total 00:00]" in str(notes_row)
    progress.update("Notes", first, "drafting")
    clock.return_value = 16
    part_one = first + "  part 1/2 (ch001_a_12345678)"
    part_two = first + "  part 2/2 (ch001_b_12345678)"
    progress.update("Notes", part_one, "retry 2/3")
    assert "[chapter 00:06 | total 00:06]" in str(notes_row)
    clock.return_value = 18
    progress.update("Notes", part_one, "complete")
    progress.update("Notes", part_two, "Illustration coverage", done=4, total=8, unit="observation")
    clock.return_value = 25
    line = str(notes_row)
    assert "part 2/2" in line and "4/8 observation" in line and "coverage" in line
    assert "[chapter 00:15 | total 00:15]" in line
    assert "[chapter 00:25 | total 00:25]" in str(progress.rows["Vision"])

    progress.update("Notes", part_two, "complete")
    progress.update("Notes", second, "waiting for vision")
    clock.return_value = 45
    assert "[chapter 00:00 | total 00:15]" in str(notes_row)
    progress.update("Notes", second, "drafting")
    clock.return_value += 3602
    assert "2/2" in str(notes_row) and "[chapter 1:00:02 | total 1:00:17]" in str(notes_row)
    progress.update("Notes", second, outcome)
    finished = str(notes_row)
    clock.return_value += 60
    progress.update("Vision", second, "inspecting")
    progress.log("Vision still running")
    assert str(notes_row) == finished


def test_cancelled_notes_request_stops_its_clocks(timed_progress):
    progress, clock = timed_progress
    cancel = NotesCancellation()
    with pytest.raises(ChapterCancelled), notes_display(progress, "ch001  1/2", cancel):
        notes_phase("drafting")
        clock.return_value = 15
        cancel.set()
        notes_phase("checking LaTeX")
    clock.return_value = 90
    assert "[chapter 00:15 | total 00:15]" in str(progress.rows["Notes"])


def test_clip_plan_carries_last_slide_and_releases_empty_short_chapter():
    evidence = {"schema_version": 3, "title": "clip", "media": {"duration": 3000, "clip_start": 100,
                "clip_end": 1910}, "segments": [{"id": "bridge", "text": "speech", "start": 990,
                                                 "end": 1080}]}
    base = notes.default_outline(evidence)
    frames = [{"id": f"f{i}", "timestamp": t} for i,t in enumerate((130, 1060, 1900, 1910))]
    plan = chapter_pipeline.chapter_plan(evidence, base, frames, [1, 2, 4])
    assert [ch["stop"] for ch in plan] == [2, 4, 4]
    assert [ch["batch_end"] for ch in plan] == [2, 4, 4]
    assert plan[1]["frame_ids"] == ["f0", "f1", "f2", "f3"]
    assert plan[-1]["frame_ids"] == ["f1", "f2", "f3"]
    assert [ch["segment_ids"] for ch in plan] == [["bridge"], ["bridge"], []]
    empty = chapter_pipeline.chapter_plan(evidence, base, [], [])
    assert [ch["stop"] for ch in empty] == [0, 0, 0]


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_active_notes_stop_and_join_when_vision_exits(lecture, monkeypatch, exception):
    from shad_helper.chapter_progress import notes_connection

    media, out, job, client, rendered = lecture
    writing, closed = Event(), Event()
    def before(_):
        class Connection:
            def close(self):
                closed.set()
        with notes_connection(Connection()):
            writing.set()
            assert closed.wait(5)
            raise OSError("closed")
    client.before = before
    def stop(key):
        if key == "f2":
            assert writing.wait(5)
            raise exception("vision stopped")
    simulate_vision(monkeypatch, stop)
    if exception is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            pipeline.run_pipeline([media], out, out)
    else:
        assert not pipeline.run_pipeline([media], out, out)
    assert closed.is_set() and not rendered and len(client.calls) == 1
    assert read_json(job / "extraction-status.json")["state"] == "failed"
    assert read_json(job / "handoff/run.json")["status"] == "interrupted"


@pytest.mark.parametrize("mode", ["frames", "video"])
def test_persistent_vision_subprocess_streams_existing_checkpoints(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    content = {"kind": "slides", "text": "Теорема", "diagrams": [], "formulas": [],
               "illustrations": [], "unreadable": []}
    class Runtime:
        def infer(self, image, limit, retry):
            return json.dumps(content), {"finish_reason": "stop"}

        def infer_video(self, frames, limit, retry):
            return json.dumps({"observations": [
                {"frame_indices": list(range(len(frames))), **content}]}), {"finish_reason": "stop"}

    runtime = Mock(return_value=Runtime())
    monkeypatch.setattr(vision_worker, "_load_runtime", runtime)
    frames = []
    for i,timestamp in enumerate([0, 18, 20, 30]):
        path = tmp_path / f"f{i}.png"
        Image.new("RGB", (64, 32), (22, 55, i)).save(path)
        frames.append({"id": f"f{i}", "timestamp": timestamp, "image": str(path)})
    request = {"model": "offline-cache-fixture", "mode": mode, "items": frames}
    # Seed exactly the same checkpoints standalone extraction creates. The subprocess
    # has no fake runtime, so a mismatched key would try to load this nonexistent model.
    original = vision_worker.run(dict(request, job=str(tmp_path)))
    assert runtime.call_count == 1
    if mode == "video":
        request["batch_sizes"] = [3, 1]
    batches, status, logs = [], [], []
    result = extraction._vision_worker(tmp_path, request, on_batch=batches.append, log=logs.append,
                                       on_status=lambda phase, fid: status.append((phase, fid)))
    assert [item["id"] for batch in batches for item in batch] == [f["id"] for f in frames]
    assert [len(batch) for batch in batches] == ([3, 1] if mode == "video" else [1, 1, 1, 1])
    assert status == [("cached", batch[0]["id"]) for batch in batches]
    assert result["items"] == original["items"] and result["diagnostics"]["fresh_items"] == 0
    assert not any("%|" in line or "it/s]" in line for line in logs)


def test_completed_chapters_publish_renderable_pdf(lecture, monkeypatch):
    import shutil

    if not shutil.which("latexmk"):
        pytest.skip("latexmk is required for the PDF smoke test")
    media, out, job, _, _ = lecture
    simulate_vision(monkeypatch, illustration=True)
    monkeypatch.setattr(pipeline, "render", rendering.render)
    def anki(job, state, *, pdf, **kwargs):
        assert pdf == job / "notes" / f"{media.stem}.pdf"
        assert pdf.read_bytes().startswith(b"%PDF")
        write_json(job / "anki/report.json", {"end_position": 1})
        return job / "anki" / f"{media.stem}.apkg"
    monkeypatch.setattr(pipeline, "generate_anki", anki)
    assert pipeline.run_pipeline([media], out, out)
    assert (job / "notes" / f"{media.stem}.pdf").is_file()


def test_manual_failure_is_reported_while_qwen_continues_and_next_file_runs(lecture, monkeypatch):
    media, out, first, client, rendered = lecture
    next_media = media.with_name("next.mp4")
    next_media.write_bytes(b"next")
    failed = Event()
    update = ChapterProgress.update
    def report(self, worker, label, phase, **kwargs):
        update(self, worker, label, phase, **kwargs)
        if phase == "manual handoff":
            failed.set()
    monkeypatch.setattr(ChapterProgress, "update", report)
    visited = []
    def after(key):
        visited.append(key)
        if key == "f2":
            assert failed.wait(5)
        if key == "f3" and not rendered:
            assert failed.is_set()
    simulate_vision(monkeypatch, after)
    def before(context):
        if context["lecture_title"] == media.stem:
            raise OSError("notes offline")
    client.before = before
    assert not pipeline.run_pipeline([media, next_media], out, out)
    assert visited == ["f0","f1","f2","f3","f4","f5"] * 2
    assert read_json(first / "handoff/status.json")["chapter"] == "ch001"
    assert read_json(first / "extraction-status.json")["state"] == "complete"
    assert rendered == [pipeline._jobs([next_media], out)[0][1]]


def test_native_repair_catalog_extends_beyond_ready_qwen_frames(lecture, monkeypatch):
    from shad_helper.illustrations import native_frames

    media, out, _, _, _ = lecture
    repaired = Event()
    def refine(job, evidence, *args):
        assert [v["id"] for v in evidence["visuals"]] == ["f0", "f1", "f2"]
        assert [v["id"] for v in native_frames(job, evidence)] == [f"f{i}" for i in range(6)]
        repaired.set()
        return "", False, "", False
    monkeypatch.setattr("shad_helper.illustration_verification.refine_crop", refine)
    def after(key):
        if key == "f2":
            assert repaired.wait(5)
    simulate_vision(monkeypatch, after, illustration=True)
    assert pipeline.run_pipeline([media], out, out) and repaired.is_set()


def test_preflight_split_runs_in_order_and_rebuild_resets_base_once(lecture, monkeypatch):
    media, out, job, client, _ = lecture
    original = notes._chapter_budget_error
    def budget(context, *args, **kwargs):
        if context["chapter"]["id"] == "ch001":
            return "too many attached images"
        return original(context, *args, **kwargs)
    monkeypatch.setattr(notes, "_chapter_budget_error", budget)
    reset = Mock(wraps=notes.reset_notes)
    monkeypatch.setattr(notes, "reset_notes", reset)
    simulate_vision(monkeypatch)
    assert pipeline.run_pipeline([media], out, out, rebuild=True)
    reset.assert_called_once_with(job)
    ids = [chapter["id"] for chapter in read_json(job / "outline.json")["chapters"]]
    assert len(ids) == 4 and [p[1]["chapter"]["id"] for p in client.calls] == ids
    assert client.calls[1][1]["glossary"][0]["term"] == ids[0]
    assert not (job / "handoff/inputs/ch001").exists()


def test_changed_verdict_rechecks_the_chapter_without_redrafting(lecture, monkeypatch):
    media, out, job, client, _ = lecture
    simulate_vision(monkeypatch)
    verify = chapter_pipeline.ChapterNotes._verify
    calls = []
    verdict = job / "illustration-verification/prior/verdict.json"
    def verify_with_verdict(self, draft, evidence, previous):
        if not verdict.exists():
            write_json(verdict, {"decision": "first"})
        calls.append(draft["chapters"][0]["id"])
        read_json(verdict)
        return verify(self, draft, evidence, previous)
    monkeypatch.setattr(chapter_pipeline.ChapterNotes, "_verify", verify_with_verdict)
    assert pipeline.run_pipeline([media], out, out)
    assert len(calls) == len(client.calls) == 3
    calls.clear()
    assert pipeline.run_pipeline([media], out, out)
    assert calls == []
    write_json(verdict, {"decision": "corrected"})
    assert pipeline.run_pipeline([media], out, out)
    assert calls == ["ch001", "ch002", "ch003"] and len(client.calls) == 3
