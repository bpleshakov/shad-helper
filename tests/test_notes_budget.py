import copy
import json

import pytest

from shad_helper import notes
from shad_helper.auth import PlanAccessError
from shad_helper.jobs import read_json, write_json
from shad_helper.openai_client import ResponseError
from shad_helper.schemas import LectureNotes


def evidence(duration=900, *, dense=False):
    segments = [
        {"id": f"s{number:03d}", "start": number * 60 + 10,
         "end": number * 60 + 12, "text": f"Definition {number}. " + "x" * (9000 if dense else 10)}
        for number in range(int(duration // 60))
    ]
    data = {"schema_version": 3, "title": "Лекция", "media": {"duration": duration},
            "segments": segments}
    return data


def prepare_job(job, data, outline=None):
    write_json(job / "evidence.json", data)
    if outline is not None:
        write_json(job / "outline.json", outline)


def payload_from_prompt(prompt):
    payload = json.loads(prompt)
    assert set(payload) == {"lecture_title", "glossary", "topics", "chapter", "segments", "writing_budget"}
    return payload


def chapter_reply(payload):
    chapter = payload["chapter"]
    segments = payload["segments"]
    citations = [segments[0]["id"]] if segments else []
    return {"title": "Лекция с тематическими названиями", "language": "ru", "chapters": [
        {**chapter, "title": f"Определения и доказательства {chapter['id']}", "blocks": [
            {"id": f"{chapter['id']}_b1", "kind": "text" if citations else "clarification",
             "content": "Определение и объяснение", "evidence_ids": citations}]}],
        "glossary": [{"term": chapter["id"], "definition": "Обозначение из этой главы"}],
        "issues": []}


class FakeClient:
    model = "fake"

    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def complete(self, instructions, prompt, schema=None):
        payload = payload_from_prompt(prompt)
        self.calls.append({"payload": payload, "instructions": instructions, "prompt": prompt,
                           "schema": schema})
        if self.fail:
            self.fail(payload)
        return chapter_reply(payload)


@pytest.fixture(autouse=True)
def no_tex(monkeypatch):
    monkeypatch.setattr(notes.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(notes, "MAX_CHAPTER_TEXT_BYTES", 120_000)

    def write_project(job, payload):
        destination = job / "notes" / "main.tex"
        destination.parent.mkdir(exist_ok=True)
        destination.write_text("test project", encoding="utf-8")
        return destination

    monkeypatch.setattr("shad_helper.rendering.write_project", write_project)


def assert_coverage(outline, data):
    assert notes.validate_outline(outline, data) == outline
    assert all(chapter["end"] - chapter["start"] >= notes.MIN_CHAPTER_SECONDS
               for chapter in outline["chapters"])
    contexts = [notes.chapter_context(data, chapter) for chapter in outline["chapters"]]
    assert {segment["id"] for context in contexts for segment in context["segments"]} == {
        segment["id"] for segment in data["segments"]}
    return contexts


def test_preflight_splits_without_losing_text(tmp_path):
    data = evidence(dense=True)
    outline = notes.default_outline(data)
    original = copy.deepcopy(data)
    prepare_job(tmp_path, data)
    bounded = notes._bounded_outline(data, outline)
    assert len(bounded["chapters"]) > 1
    assert bounded == notes._bounded_outline(data, outline)
    assert data == original
    assert outline == notes.default_outline(data)
    contexts = assert_coverage(bounded, data)
    assert all(set(context) == {"chapter", "segments"} for context in contexts)


def test_automatic_requests_fit_budget_and_use_only_chapters(tmp_path):
    data = evidence(dense=True)
    prepare_job(tmp_path, data)
    canonical = (tmp_path / "evidence.json").read_bytes()
    client = FakeClient()
    result = notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert result.exists() and len(client.calls) > 1
    saved = read_json(tmp_path / "outline.json")
    assert_coverage(saved, data)
    assert [call["payload"]["chapter"]["id"] for call in client.calls] == [
        chapter["id"] for chapter in saved["chapters"]]
    assert not (tmp_path / "checkpoints/notes/outline.json").exists()
    assert (tmp_path / "evidence.json").read_bytes() == canonical
    for call in client.calls:
        assert call["instructions"] == notes.INSTRUCTIONS
        text_bytes = len((call["instructions"] + call["prompt"] + json.dumps(
            call["schema"], ensure_ascii=False, separators=(",", ":"))).encode("utf-8"))
        assert text_bytes <= notes.MAX_CHAPTER_TEXT_BYTES
        for segment in call["payload"]["segments"]:
            assert set(segment) == {"id", "start", "end", "text"}


def test_manual_split_ids_are_persisted_and_import_does_not_split_again(tmp_path, monkeypatch):
    data = evidence(dense=True)
    prepare_job(tmp_path, data)
    folder = notes.generate_notes(tmp_path, tmp_path / "auth", manual=True)
    outline = read_json(tmp_path / "outline.json")
    assert read_json(folder / "outline.json") == outline
    assert len(outline["chapters"]) > 1
    contexts = assert_coverage(outline, data)
    replies = tmp_path / "replies"
    replies.mkdir()
    for chapter, context in zip(outline["chapters"], contexts, strict=True):
        model_context = notes._model_context(context, outline, outline["glossary"])
        assert read_json(folder / "inputs" / chapter["id"] / "context.json") == model_context
        write_json(replies / f"{chapter['id']}.json", chapter_reply(model_context))
    monkeypatch.setattr(notes, "MAX_CHAPTER_TEXT_BYTES", 1)
    result = notes.generate_notes(tmp_path, tmp_path / "auth", imported=replies)
    assert result.exists()
    assert [chapter["id"] for chapter in read_json(tmp_path / "notes.json")["chapters"]] == [
        chapter["id"] for chapter in outline["chapters"]]


@pytest.mark.parametrize("error_type", [ResponseError, PlanAccessError])
def test_typed_context_error_splits_only_failed_chapter_and_retries(tmp_path, error_type):
    data = evidence()
    prepare_job(tmp_path, data)

    def fail_large(payload):
        chapter = payload["chapter"]
        if chapter["end"] - chapter["start"] > 450:
            raise error_type("Context limit", code="context_length_exceeded")

    client = FakeClient(fail_large)
    result = notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert result.exists()
    assert len(client.calls) == 3
    outline = read_json(tmp_path / "outline.json")
    assert len(outline["chapters"]) == 2
    assert_coverage(outline, data)
    assert [call["payload"]["chapter"]["id"] for call in client.calls[1:]] == [
        chapter["id"] for chapter in outline["chapters"]]


@pytest.mark.parametrize("error_type", [ResponseError, PlanAccessError])
def test_other_error_retries_then_falls_back_without_matching_message_text(tmp_path, error_type):
    data = evidence()
    prepare_job(tmp_path, data)

    def fail(payload):
        raise error_type("context_length_exceeded appears in unrelated text", code="server_error")

    client = FakeClient(fail)
    folder = notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert folder == tmp_path / "handoff" and len(client.calls) == 4
    assert read_json(tmp_path / "outline.json") == notes.default_outline(data)
    assert read_json(folder / "status.json")["status"] == "manual_required"
    assert read_json(folder / "inputs/ch001/context.json")["segments"] == notes.chapter_context(
        data, notes.default_outline(data)["chapters"][0])["segments"]


def test_irreducible_text_budget_preserves_all_evidence_in_manual_handoff(tmp_path):
    data = evidence(duration=120)
    data["segments"] = [{**data["segments"][0], "start": 0, "end": 120, "text": "x" * 150000}]
    prepare_job(tmp_path, data)
    client = FakeClient()
    folder = notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert folder == tmp_path / "handoff"
    assert client.calls == []
    outline = read_json(tmp_path / "outline.json")
    assert_coverage(outline, data)
    assert read_json(folder / "status.json")["status"] == "manual_required"
    for chapter in outline["chapters"]:
        assert read_json(folder / "inputs" / chapter["id"] / "context.json")["segments"][0]["text"] == "x" * 150000




def test_irreducible_context_error_stops_at_minimum_chapter(tmp_path):
    data = evidence(duration=notes.MIN_CHAPTER_SECONDS)
    data["segments"] = [{"id": "s000", "start": 1, "end": 2, "text": "Определение"}]
    prepare_job(tmp_path, data)

    def fail(payload):
        raise ResponseError("Context limit", code="context_length_exceeded")

    client = FakeClient(fail)
    folder = notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert folder == tmp_path / "handoff" and len(client.calls) == 1
    assert read_json(tmp_path / "outline.json") == notes.default_outline(data)
    assert read_json(folder / "inputs/ch001/context.json")["segments"][0]["id"] == "s000"


@pytest.mark.parametrize("error_type", [ResponseError, PlanAccessError])
def test_runtime_split_keeps_earlier_success_and_caches_both_children(tmp_path, error_type):
    data = evidence(duration=1800)
    prepare_job(tmp_path, data)

    def fail_second(payload):
        chapter = payload["chapter"]
        if chapter["id"] == "ch002":
            raise error_type("Context limit", code="context_length_exceeded")

    client = FakeClient(fail_second)
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    ids = [call["payload"]["chapter"]["id"] for call in client.calls]
    assert len(ids) == 4 and ids[:2] == ["ch001", "ch002"]
    assert ids.count("ch001") == 1
    outline = read_json(tmp_path / "outline.json")
    assert [chapter["id"] for chapter in outline["chapters"]] == ["ch001", *ids[2:]]
    assert_coverage(outline, data)
    cached = FakeClient()
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=cached).exists()
    assert cached.calls == []


def test_resume_reuses_successful_chapters_with_generated_titles_and_glossary(tmp_path):
    data = evidence(duration=1800)
    outline = notes.default_outline(data)
    outline["title"] = "Сохранённый план лекции"
    outline["glossary"] = [{"term": "A", "definition": "Исходное обозначение"}]
    prepare_job(tmp_path, data, outline)

    def fail_second(payload):
        if payload["chapter"]["id"] == "ch002":
            raise ResponseError("Temporary failure", code="server_error")

    first_client = FakeClient(fail_second)
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=first_client) == tmp_path / "handoff"
    assert len(first_client.calls) == 5
    assert read_json(tmp_path / "outline.json") == outline
    resumed = FakeClient()
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=resumed).exists()
    assert len(resumed.calls) == 1
    assert resumed.calls[0]["payload"]["chapter"]["id"] == "ch002"
    assert {entry["term"] for entry in resumed.calls[0]["payload"]["glossary"]} == {"A", "ch001"}
    assert read_json(tmp_path / "outline.json") == outline
    generated = read_json(tmp_path / "notes.json")
    assert generated["chapters"][0]["title"].startswith("Определения и доказательства")
    assert {entry["term"] for entry in generated["glossary"]} == {"A", "ch001", "ch002"}
    cached = FakeClient()
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=cached).exists()
    assert cached.calls == []


def test_deep_split_long_chapter_id_keeps_block_and_issue_ids_unique(tmp_path, monkeypatch):
    data = evidence(dense=True)
    for segment in data["segments"]:
        segment["text"] = "x" * 3000
    outline = notes.default_outline(data)
    outline["chapters"][0]["id"] = "a" * 80
    prepare_job(tmp_path, data, outline)
    # Keep a small evidence allowance independent of prompt/schema changes.
    overhead = notes.INSTRUCTIONS + json.dumps(LectureNotes.model_json_schema(), ensure_ascii=False)
    monkeypatch.setattr(notes, "MAX_CHAPTER_TEXT_BYTES", len(overhead.encode("utf-8")) + 14000)

    class UncertainClient(FakeClient):
        def complete(self, instructions, prompt, schema=None):
            reply = super().complete(instructions, prompt, schema=schema)
            reply["chapters"][0]["blocks"][0]["kind"] = "uncertain"
            return reply

    result = notes.generate_notes(tmp_path, tmp_path / "auth", client=UncertainClient())
    assert result.exists() and result != tmp_path / "handoff"
    generated = read_json(tmp_path / "notes.json")
    assert len(generated["chapters"]) >= 4
    block_ids = [block["id"] for chapter in generated["chapters"] for block in chapter["blocks"]]
    issue_ids = [issue["id"] for issue in generated["issues"]]
    assert len(set(block_ids)) == len(block_ids)
    assert len(set(issue_ids)) == len(issue_ids) == len(generated["chapters"])
    assert all(len(identifier) <= 80 for identifier in [*block_ids, *issue_ids])
    assert all(issue["status"] == "open" for issue in generated["issues"])


def test_tiny_chapter_reduces_default_neighbor_overlap_but_preserves_core_evidence():
    data = evidence(duration=180)
    data["segments"] = [{"id": "core", "start": 70, "end": 72, "text": "Текст главы"},
                        {"id": "neighbor", "start": 100, "end": 102, "text": "Соседняя глава"}]
    chapter = {"id": "tiny", "title": "Короткая глава", "start": 60, "end": 90}
    default = notes.chapter_context(data, chapter)
    explicit = notes.chapter_context(data, chapter, overlap=60)
    assert [segment["id"] for segment in default["segments"]] == ["core"]
    assert [segment["id"] for segment in explicit["segments"]] == ["core", "neighbor"]


def test_response_schema_change_invalidates_chapter_cache(tmp_path, monkeypatch):
    data = evidence()
    prepare_job(tmp_path, data)
    client = FakeClient()
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    assert len(client.calls) == 1
    changed_schema = LectureNotes.model_json_schema()
    changed_schema["description"] = "Revised lecture response contract"
    monkeypatch.setattr(LectureNotes, "model_json_schema", lambda: copy.deepcopy(changed_schema))
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    assert len(client.calls) == 2
    assert client.calls[-1]["schema"]["description"] == "Revised lecture response contract"
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    assert len(client.calls) == 2
