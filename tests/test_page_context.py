import hashlib
import json
from unittest.mock import Mock

import pytest
from PIL import Image

from shad_helper import notes
from shad_helper.jobs import evidence_index, read_json, write_json
from shad_helper.visual_grouping import group_visuals


def page(identifier, start, end, capture):
    return {"id": identifier, "timestamp": capture, "page_id": f"page_{identifier}",
            "page_start": start, "page_end": end, "image": f"{identifier}.png",
            "summary": "Итог страницы", "regions": [{"id": f"{identifier}_region"}]}


def evidence(visuals):
    return {"title": "Lecture", "video": {"duration": 1800}, "segments": [],
            "visuals": visuals, "sources": [], "issues": []}


def legacy_evidence():
    return evidence([{"id": "slide", "timestamp": 10, "summary": "legacy", "regions": []}])


def test_long_page_final_image_serves_earlier_chapter_and_only_its_matched_sources():
    first = page("first", 0, 1200, 1100)
    future = page("future", 1200, 1800, 1700)
    data = evidence([first, future,
                     {"id": "source_image", "timestamp": None, "source_id": "source_1"},
                     {"id": "future_source_image", "timestamp": None, "source_id": "source_2"}])
    data["sources"] = [
        {"id": "source_1", "source": "first.pdf", "location": "page 1",
         "matched_visual_ids": ["first"]},
        {"id": "source_2", "source": "future.pdf", "location": "page 2",
         "matched_visual_ids": ["future"]}]
    data["issues"] = [{"id": "review_first", "evidence_ids": ["first_region"]},
                      {"id": "review_future", "evidence_ids": ["future_region"]}]

    context = notes.chapter_context(data, {"start": 0, "end": 900})
    assert [v["id"] for v in context["visuals"]] == ["first", "source_image"]
    assert context["visuals"][0]["timestamp"] == 1100
    assert [s["id"] for s in context["sources"]] == ["source_1"]
    assert [issue["id"] for issue in context["issues"]] == ["review_first"]
    index = evidence_index(context)
    assert index["first"] == "Видео 00:00:00–00:20:00; итог страницы доски, кадр 00:18:20"
    assert index["first_region"] == index["first"]


def test_page_boundaries_are_half_open_without_neighboring_time_overlap():
    data = evidence([page("ended", 0, 900, 890), page("active", 900, 1200, 1190),
                     page("next", 1200, 1800, 1210)])
    context = notes.chapter_context(data, {"start": 900, "end": 1200})
    assert [v["id"] for v in context["visuals"]] == ["active"]


@pytest.mark.parametrize("start,end,expected", [(800, 950, ["board"]), (1100, 1200, [])])
def test_old_slide_is_not_resurrected_during_or_after_whiteboard_page(start, end, expected):
    data = evidence([{"id": "old_slide", "timestamp": 100}, page("board", 200, 1000, 900)])
    context = notes.chapter_context(data, {"start": start, "end": end})
    assert [v["id"] for v in context["visuals"]] == expected


def test_slide_after_page_can_carry_forward_and_ordinary_overlap_is_retained():
    data = evidence([{"id": "old_slide", "timestamp": 100}, page("board", 200, 1000, 900),
                     {"id": "new_slide", "timestamp": 1050}])
    context = notes.chapter_context(data, {"start": 1200, "end": 1350})
    assert [v["id"] for v in context["visuals"]] == ["new_slide"]

    data["visuals"].extend([{"id": "near_slide", "timestamp": 770},
                            {"id": "near_notebook", "timestamp": 1005}])
    context = notes.chapter_context(data, {"start": 800, "end": 950})
    assert [v["id"] for v in context["visuals"]] == ["board", "near_slide", "near_notebook"]


def test_legacy_context_citations_and_prompt_bytes_stay_unchanged():
    context = notes.chapter_context(legacy_evidence(), {"id": "ch001", "start": 0, "end": 900})
    assert [v["id"] for v in context["visuals"]] == ["slide"]
    assert evidence_index(context)["slide"] == "Видео 00:00:10"
    prompt = notes._chapter_prompt(context, {"title": "Lecture"}, [])
    # Captured before the page policy was implemented.
    assert hashlib.sha256(prompt.encode()).hexdigest() == (
        "dfd9479247a538f1bdfba4179e5b2ab61f385ce09dbf34a02681b9fdeaccba4f")
    assert hashlib.sha256(notes.INSTRUCTIONS.encode()).hexdigest() == (
        "b9625fa22dfbb4d2853e0ebb7cabf711afe29781cdce01f57d3f12705c9c54f1")


def test_page_guidance_is_shared_by_manual_handoff_and_chapter_prompt(tmp_path, monkeypatch):
    data = evidence([page("board", 0, 1800, 1700)])
    outline = notes.default_outline(data)
    chapter = outline["chapters"][0]
    context = notes.chapter_context(data, chapter)
    prompt = notes._chapter_prompt(context, outline, [])
    assert "Кадр может быть снят позже конца главы" in prompt
    assert "не переноси содержание следующей страницы" in prompt
    monkeypatch.setattr(notes, "_evidence_image_paths", lambda *args: [])
    monkeypatch.setattr(notes, "_context_pdf", lambda *args: None)
    folder = notes.export_handoff(tmp_path, data, outline)
    assert (folder / chapter["id"] / "prompt.txt").read_text() == notes.INSTRUCTIONS + "\n" + prompt
    saved = read_json(folder / chapter["id"] / "context.json")
    assert saved["visuals"][0]["timestamp"] == 1700
    assert saved["visuals"][0]["page_start"] == 0


@pytest.mark.parametrize("cited,accepted", [("first_region", True), ("future_region", False)])
def test_reply_validation_uses_page_intervals(cited, accepted):
    data = evidence([page("first", 0, 1200, 1100), page("future", 1200, 1800, 1700)])
    chapter = {"id": "ch001", "title": "First", "start": 0, "end": 900}
    reply = {"title": "Lecture", "language": "ru", "chapters": [
        {**chapter, "blocks": [{"id": "ch001_b1", "kind": "text", "content": "Statement",
                                "evidence_ids": [cited]}]}],
        "glossary": [], "issues": [], "illustrations": []}
    if accepted:
        assert notes._validate_chapter_reply(reply, data, chapter)["chapters"][0]["id"] == "ch001"
    else:
        with pytest.raises(ValueError, match="Unknown evidence references"):
            notes._validate_chapter_reply(reply, data, chapter)


class OutlineCaptured(RuntimeError):
    pass


def test_automatic_chapter_uses_the_common_consolidated_page_prompt(tmp_path, monkeypatch):
    data = evidence([page("board", 0, 1800, 1700)])
    write_json(tmp_path / "evidence.json", data)
    outline = notes.default_outline(data)
    client = Mock(model="fake")
    client.complete.side_effect = [outline, OutlineCaptured]
    monkeypatch.setattr(notes, "export_handoff", lambda *args: tmp_path / "handoff")
    monkeypatch.setattr(notes, "_evidence_image_paths", lambda *args: [])
    with pytest.raises(OutlineCaptured):
        notes.generate_notes(tmp_path, tmp_path / "state", client=client)
    expected = notes._chapter_prompt(notes.chapter_context(data, outline["chapters"][0]), outline, [])
    assert client.complete.call_args.args[1] == expected
    assert "Кадр может быть снят позже конца главы" in expected


@pytest.mark.parametrize("pages", [False, True])
def test_automatic_outline_includes_page_intervals_only_when_available(tmp_path, monkeypatch, pages):
    data = evidence([page("board", 0, 1800, 1700)]) if pages else legacy_evidence()
    write_json(tmp_path / "evidence.json", data)
    client = Mock(model="fake")
    client.complete.side_effect = OutlineCaptured
    monkeypatch.setattr(notes, "export_handoff", lambda *args: tmp_path / "handoff")
    with pytest.raises(OutlineCaptured):
        notes.generate_notes(tmp_path, tmp_path / "state", client=client)
    prompt = client.complete.call_args.args[1]
    visual = json.loads(prompt.split("\n", 1)[1])["visuals"][0]
    if pages:
        assert visual["page_id"] == "page_board"
        assert (visual["page_start"], visual["page_end"], visual["timestamp"]) == (0, 1800, 1700)
    else:
        assert visual == {"id": "slide", "timestamp": 10, "summary": "legacy"}
        assert hashlib.sha256(prompt.encode()).hexdigest() == (
            "6eff73de98567bb099b2160ad1c7f9d4f1b1ad7e7ad653b199f16528a08ad428")


def test_grouping_alias_keeps_own_page_interval_and_clears_untrusted_canonical_fields(tmp_path):
    items = []
    for identifier, capture, metadata in (
            ("first", 90, {"page_id": "page_1", "page_start": 0, "page_end": 100}),
            ("second", 190, {"page_id": "page_2", "page_start": 100, "page_end": 200}),
            ("ordinary", 210, {}), ("source", None, {"source_id": "source_1"})):
        path = tmp_path / f"{identifier}.png"
        Image.new("RGB", (40, 30), "white").save(path)
        items.append({"id": identifier, "timestamp": capture, "image": str(path), **metadata})

    def analyze(item):
        return {"kind": "slide", "summary": "Same content", "diagram": "", "regions": [],
                "issues": [], "page_id": "model_untrusted", "page_start": -1,
                "page_end": 9999, **item}

    outputs, _ = group_visuals(items, analyze, None)
    assert outputs[1]["analysis_id"] == "first"
    assert outputs[1]["image"] == items[0]["image"]
    assert outputs[1]["timestamp"] == 190
    assert (outputs[1]["page_id"], outputs[1]["page_start"], outputs[1]["page_end"]) == (
        "page_2", 100, 200)
    assert (outputs[0]["page_id"], outputs[0]["page_start"], outputs[0]["page_end"]) == (
        "page_1", 0, 100)
    for output in outputs[2:]:
        assert not {"page_id", "page_start", "page_end"}.intersection(output)


def test_exact_alias_keeps_own_fallback_warning_without_changing_clean_canonical(tmp_path):
    items = []
    warning = "Не удалось уверенно выбрать итоговую страницу; сохранены исходные состояния."
    for position in range(2):
        path = tmp_path / f"frame_{position}.png"
        Image.new("RGB", (40, 30), "white").save(path)
        items.append({"id": f"frame_{position}", "timestamp": position, "image": str(path)})
    items[1]["selection_issues"] = [warning, warning]
    analysis = {"kind": "slide", "summary": "Same content", "diagram": "", "regions": [],
                "issues": []}
    outputs, _ = group_visuals(items, lambda item: analysis, None)
    assert outputs[1]["analysis_id"] == "frame_0"
    assert outputs[1]["issues"] == [warning]
    assert outputs[0]["issues"] == analysis["issues"] == []
    outputs[1]["issues"].append("Alias-only edit")
    assert outputs[0]["issues"] == analysis["issues"] == []


@pytest.mark.parametrize("already_processed", [False, True])
def test_grouping_copies_baseline_issues_and_deduplicates_processed_warning(tmp_path, already_processed):
    path = tmp_path / "frame.png"
    Image.new("RGB", (40, 30), "white").save(path)
    warning = "Нужно проверить кадр."
    analysis = {"kind": "slide", "summary": "Visible", "diagram": "", "regions": [],
                "issues": [warning]}
    item = {"id": "frame", "timestamp": 0, "image": str(path)}
    if already_processed:
        item["selection_issues"] = [warning]
    outputs, _ = group_visuals([item], lambda item: analysis, None)
    assert outputs[0]["issues"] == [warning]
    outputs[0]["issues"].append("Occurrence-only edit")
    assert analysis["issues"] == [warning]
