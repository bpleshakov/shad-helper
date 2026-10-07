import copy
import json

import pytest

from shad_helper import notes
from shad_helper.auth import PlanAccessError
from shad_helper.jobs import read_json, write_json
from shad_helper.openai_client import ResponseError
from shad_helper.schemas import LectureNotes


def evidence(duration=900, *, dense=False, images=False):
    segments = [
        {"id": f"s{number:03d}", "start": number * 60 + 10,
         "end": number * 60 + 12, "text": f"Definition {number}. " + "x" * (9000 if dense else 10),
         "words": [{"word": "Definition", "start": number * 60 + 10,
                    "end": number * 60 + 11, "probability": 0.91}],
         "diagnostics": {"avg_logprob": -0.1, "no_speech_prob": 0.01}}
        for number in range(int(duration // 60))
    ]
    data = {"title": "Лекция", "video": {"duration": duration}, "segments": segments,
            "visuals": [], "sources": [], "issues": []}
    if dense or images:
        board = {"id": "board", "timestamp": duration - 1, "page_id": "page1",
                 "page_start": 0, "page_end": duration, "image": "frames/board.png",
                 "original_image": "originals/board.png", "regions": [
                     {"id": "formula", "kind": "formula", "image": "crops/formula.png"},
                     {"id": "duplicate_crop", "kind": "text", "image": "crops/formula.png"}]}
        data["visuals"] = [board, {**board, "id": "board_alias", "regions": []}]
        data["sources"] = [{"id": "source1", "kind": "pdf", "source": "slides.pdf",
                            "location": "page 1", "matched_visual_ids": ["board"],
                            "images": ["assets/chart.png", "assets/chart.png"]}]
    if images:
        data["visuals"].extend(
            {"id": f"v{number:03d}", "timestamp": number * 28 + 14,
             "image": f"frames/{number:03d}.png", "regions": []}
            for number in range(32)
        )
    return data


def prepare_job(job, data, outline=None):
    write_json(job / "evidence.json", data)
    if outline is not None:
        write_json(job / "outline.json", outline)
    for reference in notes._context_image_references(
            {"visuals": data["visuals"], "sources": data["sources"]}):
        image = job / reference["image"]
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")


def payload_from_prompt(prompt):
    payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index("{"):])
    assert "context" in payload, "A whole-lecture outline request is forbidden"
    assert "chapter" in payload["context"]
    return payload


def chapter_reply(payload):
    chapter = payload["context"]["chapter"]
    segments = payload["context"]["segments"]
    citations = [segments[0]["id"]] if segments else []
    return {"title": "Лекция с тематическими названиями", "language": "ru", "chapters": [
        {**chapter, "title": f"Определения и доказательства {chapter['id']}", "blocks": [
            {"id": f"{chapter['id']}_b1", "kind": "text" if citations else "clarification",
             "content": "Определение и объяснение", "evidence_ids": citations}]}],
        "glossary": [{"term": chapter["id"], "definition": "Обозначение из этой главы"}],
        "issues": [], "illustrations": []}


class FakeClient:
    model = "fake"

    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def complete(self, instructions, prompt, images=None, schema=None):
        payload = payload_from_prompt(prompt)
        self.calls.append({"payload": payload, "instructions": instructions, "prompt": prompt,
                           "images": list(images or []), "schema": schema})
        if self.fail:
            self.fail(payload)
        return chapter_reply(payload)


@pytest.fixture(autouse=True)
def no_tex(monkeypatch):
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
    assert {visual["id"] for context in contexts for visual in context["visuals"]} == {
        visual["id"] for visual in data["visuals"]}
    return contexts


@pytest.mark.parametrize("dense,images", [(True, False), (False, True)])
def test_preflight_splits_without_losing_text_or_long_board_page(tmp_path, dense, images):
    data = evidence(dense=dense, images=images)
    outline = notes.default_outline(data)
    original = copy.deepcopy(data)
    prepare_job(tmp_path, data)
    bounded = notes._bounded_outline(tmp_path, data, outline)
    assert len(bounded["chapters"]) > 1
    assert bounded == notes._bounded_outline(tmp_path, data, outline)
    assert data == original
    assert outline == notes.default_outline(data)
    contexts = assert_coverage(bounded, data)
    for context in contexts:
        assert {"board", "board_alias"}.issubset({visual["id"] for visual in context["visuals"]})
        references = notes._context_image_references(context)
        paths = [reference["image"] for reference in references]
        assert len(paths) == len(set(paths)) <= notes.MAX_CHAPTER_IMAGES
        assert paths.count("crops/formula.png") == 1
        assert paths.count("assets/chart.png") == 1


@pytest.mark.parametrize("dense,images", [(True, False), (False, True)])
def test_automatic_requests_fit_budget_and_use_only_chapters(tmp_path, dense, images):
    data = evidence(dense=dense, images=images)
    prepare_job(tmp_path, data)
    canonical = (tmp_path / "evidence.json").read_bytes()
    client = FakeClient()
    result = notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert result.exists() and len(client.calls) > 1
    saved = read_json(tmp_path / "outline.json")
    assert_coverage(saved, data)
    assert [call["payload"]["context"]["chapter"]["id"] for call in client.calls] == [
        chapter["id"] for chapter in saved["chapters"]]
    assert not (tmp_path / "checkpoints/notes/outline.json").exists()
    assert (tmp_path / "evidence.json").read_bytes() == canonical
    for call in client.calls:
        text_bytes = len((call["instructions"] + call["prompt"] + json.dumps(
            call["schema"], ensure_ascii=False, separators=(",", ":"))).encode("utf-8"))
        assert text_bytes <= notes.MAX_CHAPTER_TEXT_BYTES
        assert len(call["images"]) <= notes.MAX_CHAPTER_IMAGES
        for segment in call["payload"]["context"]["segments"]:
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
        assert read_json(folder / chapter["id"] / "context.json") == notes._model_context(context)
        write_json(replies / f"{chapter['id']}.json", chapter_reply({"context": context}))
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
        chapter = payload["context"]["chapter"]
        if chapter["end"] - chapter["start"] > 450:
            raise error_type("Context limit", code="context_length_exceeded")

    client = FakeClient(fail_large)
    result = notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert result.exists()
    assert len(client.calls) == 3
    outline = read_json(tmp_path / "outline.json")
    assert len(outline["chapters"]) == 2
    assert_coverage(outline, data)
    assert [call["payload"]["context"]["chapter"]["id"] for call in client.calls[1:]] == [
        chapter["id"] for chapter in outline["chapters"]]


@pytest.mark.parametrize("error_type", [ResponseError, PlanAccessError])
def test_other_error_falls_back_once_without_matching_message_text(tmp_path, error_type):
    data = evidence()
    prepare_job(tmp_path, data)

    def fail(payload):
        raise error_type("context_length_exceeded appears in unrelated text", code="server_error")

    client = FakeClient(fail)
    folder = notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert folder == tmp_path / "handoff" and len(client.calls) == 1
    assert read_json(tmp_path / "outline.json") == notes.default_outline(data)
    assert read_json(folder / "status.json")["status"] == "manual_required"
    assert read_json(folder / "ch001/context.json")["segments"] == notes.chapter_context(
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
        assert read_json(folder / chapter["id"] / "context.json")["segments"][0]["text"] == "x" * 150000


def test_irreducible_image_budget_preserves_every_image_in_manual_handoff(tmp_path):
    data = evidence(duration=120)
    paths = [f"crops/{number:03d}.png" for number in range(notes.MAX_CHAPTER_IMAGES + 1)]
    data["visuals"] = [{"id": "board", "timestamp": 119, "page_id": "page1",
                        "page_start": 0, "page_end": 120, "regions": [
                            {"id": f"r{number:03d}", "kind": "formula", "image": path}
                            for number, path in enumerate(paths)]}]
    prepare_job(tmp_path, data)
    client = FakeClient()
    folder = notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert folder == tmp_path / "handoff" and client.calls == []
    outline = read_json(tmp_path / "outline.json")
    assert_coverage(outline, data)
    for chapter in outline["chapters"]:
        context = read_json(folder / chapter["id"] / "context.json")
        assert {reference["image"] for reference in notes._context_image_references(context)} == set(paths)
        assert all((folder / chapter["id"] / path).read_bytes() == b"image" for path in paths)


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
    assert read_json(folder / "ch001/context.json")["segments"][0]["id"] == "s000"


@pytest.mark.parametrize("error_type", [ResponseError, PlanAccessError])
def test_runtime_split_keeps_earlier_success_and_caches_both_children(tmp_path, error_type):
    data = evidence(duration=1800)
    prepare_job(tmp_path, data)

    def fail_second(payload):
        chapter = payload["context"]["chapter"]
        if chapter["id"] == "ch002":
            raise error_type("Context limit", code="context_length_exceeded")

    client = FakeClient(fail_second)
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    ids = [call["payload"]["context"]["chapter"]["id"] for call in client.calls]
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
        if payload["context"]["chapter"]["id"] == "ch002":
            raise ResponseError("Temporary failure", code="server_error")

    first_client = FakeClient(fail_second)
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=first_client) == tmp_path / "handoff"
    assert len(first_client.calls) == 2
    assert read_json(tmp_path / "outline.json") == outline
    resumed = FakeClient()
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=resumed).exists()
    assert len(resumed.calls) == 1
    assert resumed.calls[0]["payload"]["context"]["chapter"]["id"] == "ch002"
    assert {entry["term"] for entry in resumed.calls[0]["payload"]["glossary"]} == {"A", "ch001"}
    assert read_json(tmp_path / "outline.json") == outline
    generated = read_json(tmp_path / "notes.json")
    assert generated["chapters"][0]["title"].startswith("Определения и доказательства")
    assert {entry["term"] for entry in generated["glossary"]} == {"A", "ch001", "ch002"}
    cached = FakeClient()
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=cached).exists()
    assert cached.calls == []


def test_deep_split_long_chapter_id_keeps_deferred_gap_and_issue_ids_unique(tmp_path, monkeypatch):
    data = evidence(dense=True)
    for segment in data["segments"]:
        segment["text"] = "x" * 3000
    data["visuals"][0]["status"] = "deferred"
    outline = notes.default_outline(data)
    outline["chapters"][0]["id"] = "a" * 80
    prepare_job(tmp_path, data, outline)
    monkeypatch.setattr(notes, "MAX_CHAPTER_TEXT_BYTES", 22000)
    client = FakeClient()
    result = notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert result.exists() and result != tmp_path / "handoff"
    generated = read_json(tmp_path / "notes.json")
    assert len(generated["chapters"]) >= 4
    gap_blocks = [block for chapter in generated["chapters"] for block in chapter["blocks"]
                  if block["kind"] == "uncertain" and block["evidence_ids"] == ["board"]]
    assert len(gap_blocks) == len(generated["chapters"])
    block_ids = [block["id"] for chapter in generated["chapters"] for block in chapter["blocks"]]
    issue_ids = [issue["id"] for issue in generated["issues"]]
    assert len(set(block_ids)) == len(block_ids)
    assert len(set(issue_ids)) == len(issue_ids) == len(gap_blocks)
    assert all(len(identifier) <= 80 for identifier in [*block_ids, *issue_ids])
    assert all(issue["status"] == "open" and issue["evidence_ids"] == ["board"]
               for issue in generated["issues"])


def test_tiny_chapter_reduces_default_neighbor_overlap_but_preserves_core_evidence():
    data = evidence(duration=180)
    data["segments"] = [{"id": "core", "start": 70, "end": 72, "text": "Текст главы"},
                        {"id": "neighbor", "start": 100, "end": 102, "text": "Соседняя глава"}]
    data["visuals"] = [{"id": "core_visual", "timestamp": 75, "regions": []},
                       {"id": "neighbor_visual", "timestamp": 110, "regions": []}]
    chapter = {"id": "tiny", "title": "Короткая глава", "start": 60, "end": 90}
    default = notes.chapter_context(data, chapter)
    explicit = notes.chapter_context(data, chapter, overlap=60)
    assert [segment["id"] for segment in default["segments"]] == ["core"]
    assert [visual["id"] for visual in default["visuals"]] == ["core_visual"]
    assert [segment["id"] for segment in explicit["segments"]] == ["core", "neighbor"]
    assert [visual["id"] for visual in explicit["visuals"]] == ["core_visual", "neighbor_visual"]


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
