from unittest.mock import Mock

import pytest

from shad_helper import notes
from shad_helper.jobs import read_json, write_json


def evidence():
    return {"title": "Лекция", "video": {"duration": 1800}, "segments": [
        {"id": "speech", "start": 10, "end": 12, "text": "Определение"}],
        "visuals": [], "sources": [], "issues": []}


def test_handoff_copies_all_scoped_image_paths_without_pdf(tmp_path, monkeypatch):
    data = evidence()
    visual = {"id": "first", "timestamp": 100, "page_id": "page1", "page_start": 0,
              "page_end": 900, "image": "frames/same.png",
              "original_image": "originals/same.png", "regions": [
                  {"id": "formula", "kind": "formula", "image": "crops/same.png"},
                  {"id": "text", "kind": "text", "image": "text/same.png"},
                  {"id": "duplicate", "kind": "diagram", "image": "crops/same.png"}]}
    future = {"id": "future", "timestamp": 1000, "page_id": "page2", "page_start": 900,
              "page_end": 1800, "image": "frames/future.png", "regions": []}
    data["visuals"] = [visual, {**visual, "id": "alias", "timestamp": 200}, future]
    data["sources"] = [{"id": "matched", "matched_visual_ids": ["first"],
                        "images": ["source_assets/chart.svg", "source_assets/chart.svg"]},
                       {"id": "unmatched", "matched_visual_ids": [],
                        "images": ["source_assets/unmatched.svg"]}]
    contents = {"frames/same.png": b"frame", "originals/same.png": b"original",
                "crops/same.png": b"formula", "text/same.png": b"text",
                "frames/future.png": b"future", "source_assets/chart.svg": b"<svg/>",
                "source_assets/unmatched.svg": b"<svg>unmatched</svg>"}
    for relative, content in contents.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    copied = []
    copy = notes.shutil.copy2

    def record_copy(source, destination):
        copied.append((source, destination))
        return copy(source, destination)

    monkeypatch.setattr(notes.shutil, "copy2", record_copy)
    outline = notes.default_outline(data)
    folder = notes.export_handoff(tmp_path, data, outline)
    chapter = folder / "ch001"
    assert read_json(chapter / "context.json") == notes._model_context(
        notes.chapter_context(data, outline["chapters"][0]))
    assert (chapter / "prompt.txt").read_text() == notes.INSTRUCTIONS + "\n" + notes._chapter_prompt(
        notes.chapter_context(data, outline["chapters"][0]), outline, outline["glossary"], manual=True)
    prompt = (chapter / "prompt.txt").read_text()
    assert "context.json" in prompt and "response-schema.json" in prompt
    assert "Определение" not in prompt and '"segments"' not in prompt
    assert (chapter / "transcript.txt").read_text() == "[speech] 00:00:10\nОпределение"
    context = notes.chapter_context(data, outline["chapters"][0])
    assert [path.relative_to(tmp_path).as_posix()
            for path in notes._evidence_image_paths(tmp_path, context)] == [
                "frames/same.png", "originals/same.png", "crops/same.png", "text/same.png",
                "source_assets/chart.svg"]
    for relative, content in contents.items():
        assert (tmp_path / relative).read_bytes() == content
        if relative not in ("frames/future.png", "source_assets/unmatched.svg"):
            assert (chapter / relative).read_bytes() == content
    assert not (chapter / "frames/future.png").exists()
    assert not list(folder.glob("*/source_assets/unmatched.svg"))
    assert sum(destination.is_relative_to(chapter) for _, destination in copied) == 5
    assert (folder / "ch002/frames/future.png").read_bytes() == b"future"
    assert not list(folder.rglob("*.pdf"))
    instructions = (folder / "README.txt").read_text()
    assert "attach context.json" in instructions and "evidence.pdf" not in instructions


def test_reexport_removes_only_legacy_handoff_pdfs(tmp_path):
    data = evidence()
    preserved = ["notes/main.pdf", "sources/evidence.pdf", "handoff/ch001/keep.pdf",
                 "handoff/ch001/reply.json", "handoff/evidence.pdf"]
    legacy = ["handoff/ch001/evidence.pdf", "handoff/ch001/evidence.tmp.pdf",
              "handoff/ch999/evidence.pdf"]
    for relative in preserved + legacy:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserve")
    folder = notes.export_handoff(tmp_path, data, notes.default_outline(data))
    for relative in preserved:
        assert (tmp_path / relative).read_bytes() == b"preserve"
    assert all(not (tmp_path / relative).exists() for relative in legacy)
    assert read_json(folder / "ch001/context.json")["visuals"] == []
    assert (folder / "response-schema.json").exists()


@pytest.mark.parametrize("image,exception", [("../outside.png", ValueError),
                                             ("missing.png", FileNotFoundError)])
def test_handoff_keeps_evidence_image_validation(tmp_path, image, exception):
    data = evidence()
    data["visuals"] = [{"id": "v1", "timestamp": 10, "image": "frame.png", "regions": [
        {"id": "text", "kind": "text", "image": image}]}]
    (tmp_path / "frame.png").write_bytes(b"frame")
    with pytest.raises(exception):
        notes.export_handoff(tmp_path, data, notes.default_outline(data))


@pytest.mark.parametrize("mode", ["import", "automatic"])
def test_success_does_not_export_handoff_or_open_network(tmp_path, monkeypatch, mode):
    data = evidence()
    data["video"]["duration"] = 900
    write_json(tmp_path / "evidence.json", data)
    outline = notes.default_outline(data)
    reply = {"title": "Лекция", "language": "ru", "chapters": [
        {**outline["chapters"][0], "blocks": [
            {"id": "ch001_b1", "kind": "text", "content": "Определение",
             "evidence_ids": ["speech"]}]}], "glossary": [], "issues": [], "illustrations": []}
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
    data["video"]["duration"] = 900
    write_json(tmp_path / "evidence.json", data)
    outline = notes.default_outline(data)
    reply = {"title": "Лекция", "language": "ru", "chapters": [
        {**outline["chapters"][0], "blocks": [
            {"id": "ch001_b1", "kind": "text", "content": "Определение",
             "evidence_ids": ["speech"]}]}], "glossary": [], "issues": [], "illustrations": []}
    client = Mock(model="fake")
    client.complete.side_effect = [reply, reply]
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert client.complete.call_count == 1
    monkeypatch.setattr(openai_client, "DEFAULT_REASONING_EFFORT", "medium")
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert client.complete.call_count == 2
