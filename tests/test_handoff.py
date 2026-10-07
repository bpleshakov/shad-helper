import json
from unittest.mock import Mock

import pytest

from shad_helper import notes
from shad_helper.jobs import read_json, write_json


def evidence():
    return {"schema_version": 3, "title": "Лекция", "media": {"duration": 1800}, "segments": [
        {"id": "speech", "start": 10, "end": 12, "text": "Определение"}]}


@pytest.mark.parametrize("mode", ["import", "automatic"])
def test_success_does_not_export_handoff_or_open_network(tmp_path, monkeypatch, mode):
    data = evidence()
    data["media"]["duration"] = 900
    write_json(tmp_path / "evidence.json", data)
    outline = notes.default_outline(data)
    reply = {"title": "Лекция", "language": "ru", "chapters": [
        {**outline["chapters"][0], "blocks": [
            {"id": "ch001_b1", "kind": "text", "content": "Определение",
             "evidence_ids": ["speech"]}]}], "glossary": [], "issues": []}
    export = Mock(side_effect=AssertionError("Unexpected handoff export"))
    network = Mock(side_effect=AssertionError("Unexpected network connection"))
    monkeypatch.setattr(notes, "export_handoff", export)
    monkeypatch.setattr("shad_helper.network.httpx.Client", network)
    if mode == "import":
        imported = tmp_path / "reply.json"
        write_json(imported, reply)
        result = notes.generate_notes(tmp_path, tmp_path / "auth", imported=imported)
    else:
        client = Mock(model="fake")
        client.complete.side_effect = [reply]
        result = notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
        assert client.complete.call_count == 1
    assert result.exists() and read_json(tmp_path / "review.json")["ready"]
    assert not (tmp_path / "handoff").exists()
    export.assert_not_called()
    network.assert_not_called()


def test_reasoning_effort_changes_invalidate_chapter_caches(tmp_path, monkeypatch):
    from shad_helper import openai_client

    data = evidence()
    data["media"]["duration"] = 900
    write_json(tmp_path / "evidence.json", data)
    outline = notes.default_outline(data)
    reply = {"title": "Лекция", "language": "ru", "chapters": [
        {**outline["chapters"][0], "blocks": [
            {"id": "ch001_b1", "kind": "text", "content": "Определение",
             "evidence_ids": ["speech"]}]}], "glossary": [], "issues": []}
    client = Mock(model="fake")
    client.complete.side_effect = [reply, reply]
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert client.complete.call_count == 1
    monkeypatch.setattr(openai_client, "DEFAULT_REASONING_EFFORT", "medium")
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert client.complete.call_count == 2

def test_handoff_exports_three_distinct_input_types(tmp_path):
    data = evidence()
    write_json(tmp_path / "evidence.json", data)
    outline = notes.default_outline(data)
    outline["title"] = "Unique lecture title"
    outline["glossary"] = [{"term": "Unique glossary term", "definition": "Unique definition"}]
    folder = notes.export_handoff(tmp_path, data, outline)
    inputs = folder / "inputs"
    assert {path.relative_to(inputs).as_posix() for path in inputs.rglob("*") if path.is_file()} == {
        "response-schema.json", "ch001/context.json", "ch001/prompt.txt",
        "ch002/context.json", "ch002/prompt.txt"}
    assert {path.name for path in folder.iterdir()} == {"inputs", "README.txt"}
    chapter = inputs / "ch001"
    context = read_json(chapter / "context.json")
    assert set(context) == {"lecture_title", "glossary", "chapter", "segments"}
    assert context["lecture_title"] == outline["title"]
    assert context["glossary"] == outline["glossary"]
    assert context["segments"] == data["segments"]
    prompt = (chapter / "prompt.txt").read_text()
    assert prompt == notes.INSTRUCTIONS
    schema_text = (inputs / "response-schema.json").read_text()
    for unique_data in ("Определение", "Unique lecture title", "Unique glossary term",
                        "Unique definition"):
        assert unique_data not in prompt and unique_data not in schema_text
    assert read_json(inputs / "ch002/context.json")["segments"] == []
    assert not list(folder.rglob("transcript.txt"))
    assert not (folder / "outline.json").exists()
    assert not list(folder.rglob("*.png")) and not list(folder.rglob("*.pdf"))
    assert "illustrations" not in read_json(inputs / "response-schema.json")["properties"]
    operator_instructions = (folder / "README.txt").read_text()
    assert "inputs" in operator_instructions and "replies" in operator_instructions


def test_manual_inputs_match_automatic_request_contents(tmp_path):
    data = evidence()
    data["media"]["duration"] = 900
    write_json(tmp_path / "evidence.json", data)
    outline = notes.default_outline(data)
    outline["glossary"] = [{"term": "A", "definition": "Исходное обозначение"}]
    write_json(tmp_path / "outline.json", outline)
    reply = {"title": "Лекция", "language": "ru", "chapters": [
        {**outline["chapters"][0], "blocks": [
            {"id": "ch001_b1", "kind": "text", "content": "Определение",
             "evidence_ids": ["speech"]}]}], "glossary": [], "issues": []}
    folder = notes.generate_notes(tmp_path, tmp_path / "auth", manual=True)
    client = Mock(model="fake")
    client.complete.return_value = reply
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    instructions, prompt = client.complete.call_args.args
    assert instructions == (folder / "inputs/ch001/prompt.txt").read_text()
    assert json.loads(prompt) == read_json(folder / "inputs/ch001/context.json")
    assert client.complete.call_args.kwargs["schema"] == read_json(
        folder / "inputs/response-schema.json")
    assert prompt == json.dumps(json.loads(prompt), ensure_ascii=False, separators=(",", ":"))


def test_reexport_replaces_obsolete_chapter_inputs_and_preserves_operator_files(tmp_path):
    data = evidence()
    write_json(tmp_path / "evidence.json", data)
    folder = notes.export_handoff(tmp_path, data, notes.default_outline(data))
    readme = (folder / "README.txt").read_bytes()
    status = {"status": "manual_required", "reason": "Operator review pending"}
    write_json(folder / "status.json", status)
    (folder / "operator-notes.txt").write_text("Keep these local notes", encoding="utf-8")
    write_json(folder / "replies/ch001.json", {"saved_reply": True})

    outline = notes.default_outline(data, chunk_seconds=600)
    outline["title"] = "Revised lecture title"
    for chapter in outline["chapters"]:
        chapter["id"] = f"revised_{chapter['id']}"
    write_json(tmp_path / "outline.json", outline)
    assert notes.export_handoff(tmp_path, data, outline) == folder

    inputs = folder / "inputs"
    assert {path.name for path in inputs.iterdir()} == {
        "response-schema.json", *(chapter["id"] for chapter in outline["chapters"])}
    assert not (inputs / "ch001").exists() and not (inputs / "ch002").exists()
    for chapter in outline["chapters"]:
        context = read_json(inputs / chapter["id"] / "context.json")
        assert context == notes._model_context(
            notes.chapter_context(data, chapter), outline, outline["glossary"])
        assert {path.name for path in (inputs / chapter["id"]).iterdir()} == {
            "context.json", "prompt.txt"}
    assert (folder / "README.txt").read_bytes() == readme
    assert read_json(folder / "status.json") == status
    assert (folder / "operator-notes.txt").read_text() == "Keep these local notes"
    assert read_json(folder / "replies/ch001.json") == {"saved_reply": True}
