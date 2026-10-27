import copy
import json
from pathlib import Path

import httpx
import pytest

from shad_helper import notes
from shad_helper.jobs import read_json, write_json
from shad_helper.openai_client import PlanClient, ResponseError


def prepare_job(job, duration=900):
    data = {
        "schema_version": 3, "title": "Лекция", "media": {"duration": duration},
        "segments": [{"id": f"s{number:03d}", "start": number * 450 + 5,
                      "end": number * 450 + 10, "text": "Определение и доказательство"}
                     for number in range(duration // 450)],
    }
    write_json(job / "evidence.json", data)
    write_json(job / "extraction-diagnostics.json", {
        "warnings": [{"description": "PRIVATE_EXTRACTION_DIAGNOSTIC"}],
    })
    return data


def chapter_reply(context):
    chapter = context["chapter"]
    return {
        "title": "Лекция", "language": "ru", "issues": [],
        "glossary": [{"term": chapter["id"], "definition": "Новое обозначение"}],
        "chapters": [{**chapter, "blocks": [{
            "id": f"{chapter['id']}_b1", "kind": "definition", "content": "Определение",
            "topic": "Метод ближайших соседей",
            "evidence_ids": [context["segments"][0]["id"]],
        }]}],
    }


class FakeClient:
    model = "fake"

    def __init__(self, behavior=None):
        self.behavior = behavior
        self.calls = []

    def complete(self, instructions, text, schema=None):
        context = json.loads(text)
        self.calls.append({"instructions": instructions, "text": text,
                           "schema": copy.deepcopy(schema), "model": self.model})
        if self.behavior:
            self.behavior(context)
        return chapter_reply(context)


@pytest.fixture(autouse=True)
def isolate_generation(monkeypatch):
    monkeypatch.setattr(notes.time, "sleep", lambda seconds: None)

    def write_project(job, payload):
        destination = job / "notes/main.tex"
        destination.parent.mkdir(exist_ok=True)
        destination.write_text("test project", encoding="utf-8")
        return destination

    monkeypatch.setattr("shad_helper.rendering.write_project", write_project)


def snapshot(folder):
    return {path.relative_to(folder): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


def test_automatic_handoff_matches_calls_including_glossary_and_topics(tmp_path):
    prepare_job(tmp_path, duration=1800)
    client = FakeClient()
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()

    run = tmp_path / "handoff"
    assert not (run / "runs").exists()
    assert read_json(run / "run.json") == {
        "mode": "automatic", "status": "completed", "rebuild": False,
    }
    assert read_json(run / "chapters.json") == {"ch001": "generated", "ch002": "generated"}
    assert read_json(run / "outline.json") == read_json(tmp_path / "outline.json")
    assert (run / "README.txt").exists()
    for call in client.calls:
        context = json.loads(call["text"])
        identifier = context["chapter"]["id"]
        assert read_json(run / "inputs" / identifier / "context.json") == context
        assert (run / "inputs" / identifier / "prompt.txt").read_text() == call["instructions"]
        assert read_json(run / "inputs/response-schema.json") == call["schema"]
        attempt = run / "requests" / identifier / "attempt-01"
        saved = read_json(attempt / "call.json")
        assert all(saved[key] == value for key, value in call.items())
        assert read_json(attempt / "status.json")["status"] == "success"
    assert json.loads(client.calls[1]["text"])["glossary"] == [
        {"term": "ch001", "definition": "Новое обозначение"}]
    assert json.loads(client.calls[0]["text"])["topics"] == []
    assert json.loads(client.calls[1]["text"])["topics"] == ["Метод ближайших соседей"]
    assert "PRIVATE_EXTRACTION_DIAGNOSTIC" not in repr(snapshot(run))


def test_cached_chapters_replay_normalized_topics_in_first_appearance_order(tmp_path):
    prepare_job(tmp_path, duration=2700)

    class TopicClient(FakeClient):
        def complete(self, instructions, text, schema=None):
            reply = super().complete(instructions, text, schema)
            chapter = reply["chapters"][0]
            if chapter["id"] == "ch002":
                chapter["blocks"][0]["topic"] = "  метод   ближайших Соседей  "
                chapter["blocks"].append({**chapter["blocks"][0], "id": "ch002_b2",
                                           "topic": "Линейная регрессия"})
            return reply

    first = TopicClient()
    notes.generate_notes(tmp_path, tmp_path / "auth", client=first)
    expected = [[], ["Метод ближайших соседей"],
                ["Метод ближайших соседей", "Линейная регрессия"]]
    assert [json.loads(call["text"])["topics"] for call in first.calls] == expected

    cached = TopicClient()
    notes.generate_notes(tmp_path, tmp_path / "auth", client=cached)
    assert cached.calls == []
    latest = tmp_path / "handoff"
    assert list(read_json(latest / "chapters.json").values()) == ["cached"] * 3
    assert [read_json(latest / "inputs" / f"ch{number:03d}" / "context.json")["topics"]
            for number in range(1, 4)] == expected


def test_cached_and_rebuilt_runs_replace_current_diagnostics_without_creating_archives(tmp_path):
    prepare_job(tmp_path)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=FakeClient())
    handoff = tmp_path / "handoff"
    assert len(list(handoff.glob("requests/**/call.json"))) == 1

    cached = FakeClient()
    notes.generate_notes(tmp_path, tmp_path / "auth", client=cached)
    assert cached.calls == []
    assert (handoff / "inputs/ch001/context.json").exists()
    assert list(handoff.glob("requests/**/call.json")) == []
    assert read_json(handoff / "chapters.json") == {"ch001": "cached"}

    checkpoint = tmp_path / "checkpoints/notes/ch001.json"
    assert read_json(checkpoint)["result"]["chapters"][0]["blocks"][0]["content"] == "Определение"

    class RebuiltClient(FakeClient):
        def complete(self, instructions, text, schema=None):
            assert not checkpoint.exists()
            reply = super().complete(instructions, text, schema)
            reply["chapters"][0]["blocks"][0]["content"] = "Новое определение"
            return reply

    rebuilt = RebuiltClient()
    notes.generate_notes(tmp_path, tmp_path / "auth", client=rebuilt, rebuild=True)
    assert len(rebuilt.calls) == 1
    assert read_json(checkpoint)["result"]["chapters"][0]["blocks"][0]["content"] == "Новое определение"
    assert len(list(handoff.glob("requests/**/call.json"))) == 1
    assert read_json(handoff / "run.json")["rebuild"] is True

    resumed = FakeClient()
    notes.generate_notes(tmp_path, tmp_path / "auth", client=resumed)
    assert resumed.calls == []
    assert read_json(handoff / "chapters.json") == {"ch001": "cached"}
    assert list(handoff.glob("requests/**/call.json")) == []
    assert read_json(handoff / "run.json")["rebuild"] is False
    assert not (handoff / "runs").exists()


def test_existing_run_archives_survive_normal_runs_but_are_removed_by_rebuild(tmp_path):
    prepare_job(tmp_path)
    legacy = tmp_path / "handoff/runs"
    write_json(legacy / "old-run/requests/ch001/response.json", {"legacy": True})
    before = snapshot(legacy)

    notes.generate_notes(tmp_path, tmp_path / "auth", client=FakeClient())
    assert snapshot(legacy) == before
    assert [path.name for path in legacy.iterdir()] == ["old-run"]
    notes.generate_notes(tmp_path, tmp_path / "auth", client=FakeClient(), rebuild=True)
    assert not legacy.exists()


@pytest.mark.parametrize("failure", ["validation", "generation"])
def test_rebuild_removes_all_stage_artifacts_before_validation_even_if_it_fails(
        tmp_path, monkeypatch, failure):
    prepare_job(tmp_path)
    directories = ("checkpoints/notes", "checkpoints/notes_candidates", "checkpoints/notes_final",
                   "handoff", "notes", "illustration-crops", "illustration-verification")
    files = ("notes.json", "review.json", "outline.json")
    stale = [tmp_path / name / "old.json" for name in directories]
    stale.extend(tmp_path / name for name in files)
    stale.extend(tmp_path / name for name in (
        "handoff/runs/legacy/response.json", "handoff/operator.json",
        "notes/main.tex", "notes/main.pdf", "notes/qa/page-001.png"))
    for path in stale:
        write_json(path, {"old": True})
    source_files = [tmp_path / name for name in (
        "illustrations/original.png", "native-frames/original.png",
        "checkpoints/extraction.json", "checkpoints/anki.json", "anki/deck.apkg",
        "review-resolutions.json")]
    for path in source_files:
        write_json(path, {})
    source_files.extend((tmp_path / "evidence.json", tmp_path / "extraction-diagnostics.json"))
    preserved = {path: path.read_bytes() for path in source_files}
    original_check = notes.check_job
    checked = []

    def check_after_cleanup(job):
        if not checked:
            assert all(not (job / name).exists() for name in (*directories, *files))
            assert all(path.read_bytes() == content for path, content in preserved.items())
            checked.append(job)
            if failure == "validation":
                raise ValueError("Validation failed")
        original_check(job)

    def fail_generation(context):
        assert all(not path.exists() for path in stale if path.name != "outline.json")
        raise RuntimeError("Generation failed")

    monkeypatch.setattr(notes, "check_job", check_after_cleanup)
    client = FakeClient(fail_generation)
    exception = ValueError if failure == "validation" else RuntimeError
    with pytest.raises(exception, match="failed"):
        notes.generate_notes(tmp_path, tmp_path / "auth", client=client, rebuild=True)

    assert checked == [tmp_path]
    assert len(client.calls) == (failure == "generation")
    assert all(not path.exists() for path in stale if path.name != "outline.json")
    assert all(path.read_bytes() == content for path, content in preserved.items())


def test_rebuild_stops_when_checkpoint_cannot_be_removed(tmp_path, monkeypatch):
    prepare_job(tmp_path)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=FakeClient())
    original = notes.shutil.rmtree

    def fail_notes_checkpoint(path, *args, **kwargs):
        if path == tmp_path / "checkpoints/notes":
            raise PermissionError("Cannot remove notes checkpoint")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(notes.shutil, "rmtree", fail_notes_checkpoint)
    client = FakeClient()
    with pytest.raises(PermissionError, match="Cannot remove notes checkpoint"):
        notes.generate_notes(tmp_path, tmp_path / "auth", client=client, rebuild=True)
    assert client.calls == []


def test_manual_and_import_replace_current_handoff_without_requests(tmp_path):
    prepare_job(tmp_path)
    handoff = notes.generate_notes(tmp_path, tmp_path / "auth", manual=True)
    assert handoff == tmp_path / "handoff"
    assert read_json(handoff / "run.json")["mode"] == "manual"
    assert read_json(handoff / "run.json")["status"] == "handoff"
    context = read_json(handoff / "inputs/ch001/context.json")
    reply = tmp_path / "reply.json"
    write_json(reply, chapter_reply(context))

    assert notes.generate_notes(tmp_path, tmp_path / "auth", imported=reply).exists()
    assert read_json(handoff / "run.json")["mode"] == "import"
    assert read_json(handoff / "run.json")["status"] == "completed"
    assert read_json(handoff / "outline.json") == read_json(tmp_path / "outline.json")
    assert (handoff / "inputs/ch001/context.json").exists()
    assert list(handoff.glob("requests/**/call.json")) == []
    assert not (handoff / "runs").exists()


def test_import_can_read_previous_current_response_before_diagnostics_are_replaced(tmp_path):
    prepare_job(tmp_path)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=FakeClient())
    handoff = tmp_path / "handoff"
    reply = handoff / "requests/ch001/attempt-01/response.json"
    assert reply.is_file()
    imported_notes = read_json(tmp_path / "notes.json")

    assert notes.generate_notes(tmp_path, tmp_path / "auth", imported=reply).exists()

    assert read_json(tmp_path / "notes.json") == imported_notes
    assert read_json(handoff / "run.json")["mode"] == "import"
    assert list(handoff.glob("requests/**/call.json")) == []
    assert not (handoff / "runs").exists()


def test_rebuild_deletes_import_reply_inside_previous_stage_outputs(tmp_path):
    prepare_job(tmp_path)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=FakeClient())
    reply = tmp_path / "handoff/requests/ch001/attempt-01/response.json"
    assert reply.is_file()

    with pytest.raises(FileNotFoundError):
        notes.generate_notes(tmp_path, tmp_path / "auth", imported=reply, rebuild=True)

    assert not reply.exists()
    assert not (tmp_path / "notes.json").exists()
    assert not (tmp_path / "notes").exists()


def test_each_failed_retry_survives_fallback_with_actual_glossary(tmp_path):
    prepare_job(tmp_path, duration=1800)

    def fail_second(context):
        if context["chapter"]["id"] == "ch002":
            raise ResponseError("Temporary failure", code="server_error")

    client = FakeClient(fail_second)
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client) == tmp_path / "handoff"
    run = tmp_path / "handoff"
    assert read_json(run / "run.json")["status"] == "handoff"
    attempts = sorted((run / "requests/ch002").iterdir())
    assert [path.name for path in attempts] == [f"attempt-{number:02d}" for number in range(1, 5)]
    for attempt in attempts:
        assert read_json(attempt / "status.json")["status"] == "failed"
        assert read_json(attempt / "call.json")["text"] == client.calls[-1]["text"]
    context = json.loads(client.calls[-1]["text"])
    assert context["glossary"] == [{"term": "ch001", "definition": "Новое обозначение"}]
    assert context["topics"] == ["Метод ближайших соседей"]
    assert read_json(run / "inputs/ch002/context.json") == context
    assert read_json(run / "status.json")["status"] == "manual_required"


def test_runtime_split_retains_rejected_parent_and_final_child_handoff(tmp_path):
    prepare_job(tmp_path)

    def reject_parent(context):
        if context["chapter"]["id"] == "ch001":
            raise ResponseError("Context limit", code="context_length_exceeded")

    client = FakeClient(reject_parent)
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    run = tmp_path / "handoff"
    identifiers = [json.loads(call["text"])["chapter"]["id"] for call in client.calls]
    assert len(identifiers) == 3 and identifiers[0] == "ch001"
    assert [json.loads(call["text"])["topics"] for call in client.calls] == [
        [], [], ["Метод ближайших соседей"]]
    assert {path.name for path in (run / "requests").iterdir()} == set(identifiers)
    assert read_json(run / "requests/ch001/attempt-01/status.json")["status"] == "failed"
    outline = read_json(run / "outline.json")
    assert [chapter["id"] for chapter in outline["chapters"]] == identifiers[1:]
    assert {path.name for path in (run / "inputs").iterdir() if path.is_dir()} == set(identifiers[1:])


def test_pre_request_configuration_failure_still_saves_run_and_handoff(tmp_path, monkeypatch):
    prepare_job(tmp_path)
    monkeypatch.setenv("SHAD_HELPER_OPENAI_PROXY", "")
    assert notes.generate_notes(tmp_path, tmp_path / "auth") == tmp_path / "handoff"
    run = tmp_path / "handoff"
    assert (run / "inputs/ch001/context.json").exists()
    assert read_json(run / "status.json")["status"] == "manual_required"
    assert list(run.glob("requests/**/call.json")) == []


def test_recorded_http_body_is_exact_network_payload_without_credentials(tmp_path):
    prepare_job(tmp_path)
    requests = []

    def handle(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"models": [{"slug": "fake", "visibility": "list"}]})
        body = json.loads(request.content)
        context = json.loads(body["input"][0]["content"][0]["text"])
        event = {"type": "response.completed", "response": {"status": "completed", "output": [
            {"content": [{"type": "output_text", "text": json.dumps(chapter_reply(context))}]}]}}
        return httpx.Response(200, text=f"data: {json.dumps(event)}\n\n")

    client = PlanClient(tmp_path / "auth", model="fake", proxy="http://private-user:private-pass@localhost:8080",
                        transport=httpx.MockTransport(handle), token_provider=lambda: "PRIVATE_ACCESS_TOKEN")
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    run = tmp_path / "handoff"
    attempt = run / "requests/ch001/attempt-01"
    records = [read_json(path) for path in sorted(attempt.glob("http-*.request.json"))]
    assert len(records) == len(requests) == 2
    for record, request in zip(records, requests, strict=True):
        assert record["method"] == request.method
        assert record["url"] == str(request.url)
        if request.method == "POST":
            assert (attempt / record["body"]).read_bytes() == request.content
        else:
            assert record["body"] is None
    archive = repr(snapshot(run))
    for secret in ("PRIVATE_ACCESS_TOKEN", "private-user", "private-pass", "PRIVATE_EXTRACTION_DIAGNOSTIC"):
        assert secret not in archive
    assert client.request_recorder is None


def test_request_body_write_failure_stops_before_post(tmp_path, monkeypatch):
    prepare_job(tmp_path)
    requests = []

    def handle(request):
        requests.append(request)
        assert request.method == "GET"
        return httpx.Response(200, json={"models": [{"slug": "fake", "visibility": "list"}]})

    original_write = Path.write_bytes

    def fail_archive(path, content):
        if path.name.endswith(".body.json"):
            raise OSError("Archive disk full")
        return original_write(path, content)

    monkeypatch.setattr(Path, "write_bytes", fail_archive)
    client = PlanClient(tmp_path / "auth", model="fake", proxy="http://localhost:8080",
                        transport=httpx.MockTransport(handle), token_provider=lambda: "private-token")
    with pytest.raises(OSError, match="Archive disk full"):
        notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert [request.method for request in requests] == ["GET"]
    run = tmp_path / "handoff"
    assert read_json(run / "run.json")["status"] == "failed"
    assert (run / "inputs/ch001/context.json").exists()
    assert client.request_recorder is None
