from unittest.mock import Mock

import pytest

from shad_helper import notes
from shad_helper.jobs import read_json, write_json


def evidence():
    return {"schema_version": 2, "title": "Лекция", "media": {"duration": 1800}, "segments": [
        {"id": "speech", "start": 10, "end": 12, "text": "Определение"}], "issues": []}


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

def test_handoff_exports_scoped_text_without_attachments(tmp_path):
    data = evidence()
    write_json(tmp_path / "evidence.json", data)
    outline = notes.default_outline(data)
    folder = notes.export_handoff(tmp_path, data, outline)
    chapter = folder / "ch001"
    context = read_json(chapter / "context.json")
    assert set(context) == {"chapter", "segments"}
    assert context["segments"] == data["segments"]
    prompt = (chapter / "prompt.txt").read_text()
    assert "context.json" in prompt and "response-schema.json" in prompt
    assert "Определение" not in prompt and '"segments"' not in prompt
    assert (chapter / "transcript.txt").read_text() == "[speech] 00:00:10\nОпределение"
    assert read_json(folder / "ch002/context.json")["segments"] == []
    assert not list(folder.rglob("*.png")) and not list(folder.rglob("*.pdf"))
    assert "illustrations" not in read_json(folder / "response-schema.json")["properties"]
