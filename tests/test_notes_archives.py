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


def runs(job):
    return sorted((job / "handoff/runs").iterdir())


def snapshot(folder):
    return {path.relative_to(folder): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


def test_automatic_handoff_matches_calls_including_glossary_and_topics(tmp_path):
    prepare_job(tmp_path, duration=1800)
    client = FakeClient()
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()

    [run] = runs(tmp_path)
    assert read_json(run / "run.json") == {
        "mode": "automatic", "status": "completed", "rebuild": False,
    }
    assert read_json(run / "chapters.json") == {"ch001": "generated", "ch002": "generated"}
    assert read_json(run / "outline.json") == read_json(tmp_path / "outline.json")
    assert (run / "README.txt").exists()
    for call in client.calls:
        context = json.loads(call["text"])
        identifier = context["chapter"]["id"]
        for root in (tmp_path / "handoff", run):
            assert read_json(root / "inputs" / identifier / "context.json") == context
            assert (root / "inputs" / identifier / "prompt.txt").read_text() == call["instructions"]
            assert read_json(root / "inputs/response-schema.json") == call["schema"]
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
    latest = runs(tmp_path)[-1]
    assert list(read_json(latest / "chapters.json").values()) == ["cached"] * 3
    assert [read_json(latest / "inputs" / f"ch{number:03d}" / "context.json")["topics"]
            for number in range(1, 4)] == expected


def test_cached_and_rebuilt_runs_keep_old_archives_and_distinguish_requests(tmp_path):
    prepare_job(tmp_path)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=FakeClient())
    [first] = runs(tmp_path)
    before = snapshot(first)

    cached = FakeClient()
    notes.generate_notes(tmp_path, tmp_path / "auth", client=cached)
    assert cached.calls == []
    first_again, second = runs(tmp_path)
    assert first_again == first and snapshot(first) == before
    assert (second / "inputs/ch001/context.json").exists()
    assert list(second.glob("requests/**/call.json")) == []
    assert read_json(second / "chapters.json") == {"ch001": "cached"}
    cached_snapshot = snapshot(second)

    rebuilt = FakeClient()
    notes.generate_notes(tmp_path, tmp_path / "auth", client=rebuilt, rebuild=True)
    assert len(rebuilt.calls) == 1
    assert len(runs(tmp_path)) == 3
    assert snapshot(first) == before and snapshot(second) == cached_snapshot
    assert len(list(runs(tmp_path)[-1].glob("requests/**/call.json"))) == 1


def test_manual_and_import_runs_save_handoffs_without_requests(tmp_path):
    prepare_job(tmp_path)
    handoff = notes.generate_notes(tmp_path, tmp_path / "auth", manual=True)
    assert handoff == tmp_path / "handoff"
    [manual_run] = runs(tmp_path)
    assert read_json(manual_run / "run.json")["mode"] == "manual"
    assert read_json(manual_run / "run.json")["status"] == "handoff"
    context = read_json(handoff / "inputs/ch001/context.json")
    assert read_json(manual_run / "inputs/ch001/context.json") == context
    before = snapshot(manual_run)
    reply = tmp_path / "reply.json"
    write_json(reply, chapter_reply(context))

    assert notes.generate_notes(tmp_path, tmp_path / "auth", imported=reply).exists()
    assert len(runs(tmp_path)) == 2
    imported_run = runs(tmp_path)[-1]
    assert read_json(imported_run / "run.json")["mode"] == "import"
    assert read_json(imported_run / "run.json")["status"] == "completed"
    assert snapshot(manual_run) == before
    assert read_json(imported_run / "outline.json") == read_json(tmp_path / "outline.json")
    assert (imported_run / "inputs/ch001/context.json").exists()
    assert list((handoff / "runs").glob("*/requests/**/call.json")) == []


def test_each_failed_retry_survives_fallback_with_actual_glossary(tmp_path):
    prepare_job(tmp_path, duration=1800)

    def fail_second(context):
        if context["chapter"]["id"] == "ch002":
            raise ResponseError("Temporary failure", code="server_error")

    client = FakeClient(fail_second)
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client) == tmp_path / "handoff"
    [run] = runs(tmp_path)
    assert read_json(run / "run.json")["status"] == "handoff"
    attempts = sorted((run / "requests/ch002").iterdir())
    assert [path.name for path in attempts] == [f"attempt-{number:02d}" for number in range(1, 5)]
    for attempt in attempts:
        assert read_json(attempt / "status.json")["status"] == "failed"
        assert read_json(attempt / "call.json")["text"] == client.calls[-1]["text"]
    context = json.loads(client.calls[-1]["text"])
    assert context["glossary"] == [{"term": "ch001", "definition": "Новое обозначение"}]
    assert context["topics"] == ["Метод ближайших соседей"]
    for root in (tmp_path / "handoff", run):
        assert read_json(root / "inputs/ch002/context.json") == context
        assert read_json(root / "status.json")["status"] == "manual_required"


def test_runtime_split_retains_rejected_parent_and_final_child_handoff(tmp_path):
    prepare_job(tmp_path)

    def reject_parent(context):
        if context["chapter"]["id"] == "ch001":
            raise ResponseError("Context limit", code="context_length_exceeded")

    client = FakeClient(reject_parent)
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    [run] = runs(tmp_path)
    identifiers = [json.loads(call["text"])["chapter"]["id"] for call in client.calls]
    assert len(identifiers) == 3 and identifiers[0] == "ch001"
    assert [json.loads(call["text"])["topics"] for call in client.calls] == [
        [], [], ["Метод ближайших соседей"]]
    assert {path.name for path in (run / "requests").iterdir()} == set(identifiers)
    assert read_json(run / "requests/ch001/attempt-01/status.json")["status"] == "failed"
    outline = read_json(run / "outline.json")
    assert [chapter["id"] for chapter in outline["chapters"]] == identifiers[1:]
    for root in (tmp_path / "handoff", run):
        assert {path.name for path in (root / "inputs").iterdir() if path.is_dir()} == set(identifiers[1:])


def test_pre_request_configuration_failure_still_saves_run_and_handoff(tmp_path, monkeypatch):
    prepare_job(tmp_path)
    monkeypatch.setenv("SHAD_HELPER_OPENAI_PROXY", "")
    assert notes.generate_notes(tmp_path, tmp_path / "auth") == tmp_path / "handoff"
    [run] = runs(tmp_path)
    assert (run / "inputs/ch001/context.json").exists()
    assert read_json(run / "status.json")["status"] == "manual_required"
    assert list(run.glob("requests/**/call.json")) == []


def test_archived_http_body_is_exact_network_payload_without_credentials(tmp_path):
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
    [run] = runs(tmp_path)
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
    [run] = runs(tmp_path)
    assert read_json(run / "run.json")["status"] == "failed"
    assert (run / "inputs/ch001/context.json").exists()
    assert client.request_recorder is None
