import copy
import json
from collections import Counter

import pytest

from shad_helper import notes
from shad_helper.auth import PlanAccessError
from shad_helper.jobs import read_json, write_json
from shad_helper.openai_client import ResponseError


def prepare_job(job, duration=900):
    write_json(job / "evidence.json", {
        "schema_version": 3, "title": "Лекция", "media": {"duration": duration},
        "segments": [{"id": f"s{number:03d}", "start": number * 450 + 5,
                      "end": number * 450 + 10, "text": "Определение и доказательство"}
                     for number in range(duration // 450)],
    })


def chapter_reply(payload):
    chapter = payload["chapter"]
    return {
        "title": "Лекция", "language": "ru", "glossary": [], "issues": [],
        "chapters": [{**chapter, "blocks": [{
            "id": f"{chapter['id']}_b1", "kind": "text", "content": "Определение",
            "evidence_ids": [payload["segments"][0]["id"]],
        }]}],
    }


class FakeClient:
    model = "fake"

    def __init__(self, behavior=None):
        self.behavior = behavior
        self.calls = []
        self.attempts = Counter()

    def complete(self, instructions, prompt, schema=None):
        payload = json.loads(prompt)
        self.calls.append((instructions, prompt, copy.deepcopy(schema)))
        identifier = payload["chapter"]["id"]
        self.attempts[identifier] += 1
        if self.behavior:
            result = self.behavior(payload, self.attempts[identifier])
            if result is not None:
                return result
        return chapter_reply(payload)


@pytest.fixture(autouse=True)
def isolate_generation(monkeypatch):
    delays = []
    monkeypatch.setattr(notes.time, "sleep", delays.append)

    def write_project(job, payload):
        destination = job / "notes" / "main.tex"
        destination.parent.mkdir(exist_ok=True)
        destination.write_text("test project", encoding="utf-8")
        return destination

    monkeypatch.setattr("shad_helper.rendering.write_project", write_project)
    return delays


@pytest.mark.parametrize("error_type", [ResponseError, PlanAccessError])
def test_chapter_succeeds_on_third_retry_with_identical_request(
        tmp_path, isolate_generation, error_type):
    prepare_job(tmp_path)

    def fail_first_three(payload, attempt):
        assert not (tmp_path / "checkpoints/notes/ch001.json").exists()
        if attempt <= 3:
            raise error_type("Temporary failure", code="server_error")

    client = FakeClient(fail_first_three)
    result = notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert result == tmp_path / "notes/main.tex"
    assert client.attempts == {"ch001": 4}
    assert isolate_generation == [5, 10, 20]
    assert all(call == client.calls[0] for call in client.calls)
    assert read_json(tmp_path / "checkpoints/notes/ch001.json")["result"]["chapters"][0][
        "id"] == "ch001"
    assert not (tmp_path / "handoff/status.json").exists()


@pytest.mark.parametrize("error_type", [ResponseError, PlanAccessError])
def test_exhausted_retries_export_handoff_without_failed_checkpoint(
        tmp_path, isolate_generation, error_type):
    prepare_job(tmp_path)

    def fail(payload, attempt):
        raise error_type(f"Temporary failure {attempt}", code="server_error")

    client = FakeClient(fail)
    result = notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert result == tmp_path / "handoff"
    assert client.attempts == {"ch001": 4}
    assert isolate_generation == [5, 10, 20]
    assert not (tmp_path / "checkpoints/notes/ch001.json").exists()
    assert not (tmp_path / "notes.json").exists()
    assert read_json(result / "status.json") == {
        "status": "manual_required", "reason": "Temporary failure 4",
    }


def test_successful_chapters_remain_cached_after_later_chapter_exhaustion(
        tmp_path, isolate_generation):
    prepare_job(tmp_path, duration=1800)

    def fail_second(payload, attempt):
        if payload["chapter"]["id"] == "ch002":
            raise ResponseError("Temporary failure", code="server_error")

    failed = FakeClient(fail_second)
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=failed) == tmp_path / "handoff"
    assert failed.attempts == {"ch001": 1, "ch002": 4}
    saved = (tmp_path / "checkpoints/notes/ch001.json").read_bytes()
    assert not (tmp_path / "checkpoints/notes/ch002.json").exists()
    assert isolate_generation == [5, 10, 20]

    resumed = FakeClient()
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=resumed).exists()
    assert resumed.attempts == {"ch002": 1}
    assert (tmp_path / "checkpoints/notes/ch001.json").read_bytes() == saved
    assert not (tmp_path / "handoff/status.json").exists()


def test_each_chapter_gets_its_own_three_retries(tmp_path, isolate_generation):
    prepare_job(tmp_path, duration=1800)

    def fail_first_three(payload, attempt):
        if attempt <= 3:
            raise ResponseError("Temporary failure", code="server_error")

    client = FakeClient(fail_first_three)
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    assert client.attempts == {"ch001": 4, "ch002": 4}
    assert isolate_generation == [5, 10, 20, 5, 10, 20]


def test_invalid_chapter_reply_is_retried_before_checkpointing(tmp_path, isolate_generation):
    prepare_job(tmp_path)

    def invalid_first(payload, attempt):
        assert not (tmp_path / "checkpoints/notes/ch001.json").exists()
        if attempt == 1:
            reply = chapter_reply(payload)
            reply["chapters"][0]["id"] = "unexpected"
            return reply

    client = FakeClient(invalid_first)
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    assert client.attempts == {"ch001": 2}
    assert isolate_generation == [5]
    assert read_json(tmp_path / "checkpoints/notes/ch001.json")["result"]["chapters"][0][
        "id"] == "ch001"


@pytest.mark.parametrize("error_type", [ResponseError, PlanAccessError])
def test_typed_context_limit_splits_immediately_without_retry_delay(
        tmp_path, isolate_generation, error_type):
    prepare_job(tmp_path)

    def fail_large(payload, attempt):
        chapter = payload["chapter"]
        if chapter["end"] - chapter["start"] > 450:
            raise error_type("Context limit", code="context_length_exceeded")

    client = FakeClient(fail_large)
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    assert len(client.calls) == 3
    assert all(count == 1 for count in client.attempts.values())
    assert isolate_generation == []
    assert not (tmp_path / "checkpoints/notes/ch001.json").exists()
    chapters = read_json(tmp_path / "outline.json")["chapters"]
    assert len(chapters) == 2
    assert [json.loads(call[1])["chapter"]["id"] for call in client.calls[1:]] == [
        chapter["id"] for chapter in chapters]
