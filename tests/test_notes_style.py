import json

import pytest

from shad_helper import notes
from shad_helper.jobs import read_json, write_json


def evidence(start=0, duration=900):
    return {"schema_version": 3, "title": "Лекция",
            "media": {"duration": start + duration, "clip_start": start,
                      "clip_end": start + duration},
            "segments": [{"id": "speech", "start": start, "end": start + duration,
                          "text": "Определение и доказательство"}]}


def test_three_hour_lecture_shares_one_writing_allowance():
    data = evidence(duration=10800)
    outline = notes.default_outline(data)
    budgets = [notes._model_context(notes.chapter_context(data, chapter), outline, [])[
        "writing_budget"] for chapter in outline["chapters"]]
    assert len(budgets) == 12
    assert sum(budget["target_words"] for budget in budgets) == 3000
    assert sum(budget["dense_words"] for budget in budgets) == 6000


@pytest.mark.parametrize("start,duration,expected", [
    (0, 900, {"target_words": 250, "dense_words": 500}),
    (3600, 900, {"target_words": 250, "dense_words": 500}),
    (3600, 30, {"target_words": 9, "dense_words": 17}),
])
def test_writing_allowance_uses_clip_duration_not_absolute_time(start, duration, expected):
    data = evidence(start, duration)
    outline = notes.default_outline(data)
    context = notes.chapter_context(data, outline["chapters"][0])
    assert notes._model_context(context, outline, [])["writing_budget"] == expected


def test_overlap_does_not_increase_writing_allowance():
    data = evidence(duration=2700)
    data["segments"] = [
        {"id": "previous", "start": 850, "end": 860, "text": "Предыдущая тема"},
        {"id": "core", "start": 950, "end": 960, "text": "Новая тема"},
        {"id": "next", "start": 1820, "end": 1830, "text": "Следующая тема"},
    ]
    outline = notes.default_outline(data)
    chapter = outline["chapters"][1]
    without_overlap = notes.chapter_context(data, chapter, overlap=0)
    with_overlap = notes.chapter_context(data, chapter)
    assert len(without_overlap["segments"]) == 1
    assert len(with_overlap["segments"]) == 3
    assert notes._model_context(without_overlap, outline, [])["writing_budget"] == {
        "target_words": 250, "dense_words": 500}
    assert notes._model_context(with_overlap, outline, [])["writing_budget"] == {
        "target_words": 250, "dense_words": 500}


def test_split_chapters_share_the_original_writing_allowance():
    data = evidence(start=7200)
    outline = notes.default_outline(data)
    split = notes._split_chapter(outline["chapters"][0])
    outline["chapters"] = split
    assert notes.validate_outline(outline, data) == outline
    budgets = [notes._model_context(notes.chapter_context(data, chapter), outline, [])[
        "writing_budget"] for chapter in split]
    assert budgets == [{"target_words": 125, "dense_words": 250}] * 2


def test_automatic_and_manual_allowances_match_without_truncating_notes(tmp_path):
    data = evidence(start=3600, duration=30)
    write_json(tmp_path / "evidence.json", data)
    handoff = notes.generate_notes(tmp_path, tmp_path / "auth", manual=True)
    manual_context = read_json(handoff / "inputs/ch001/context.json")
    content = " ".join(["Существенный шаг доказательства."] * 40)

    class Client:
        model = "fake"

        def complete(self, instructions, prompt, schema=None):
            context = json.loads(prompt)
            assert context == manual_context
            assert context["writing_budget"] == {"target_words": 9, "dense_words": 17}
            return {"title": data["title"], "chapters": [{**context["chapter"], "blocks": [
                {"id": "ch001_proof", "kind": "text", "content": content,
                 "evidence_ids": ["speech"]}]}], "glossary": [], "issues": []}

    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=Client()).exists()
    generated = read_json(tmp_path / "notes.json")
    assert generated["chapters"][0]["blocks"][0]["content"] == content


def test_admin_only_chapter_can_remain_empty_without_losing_outline_coverage():
    data = evidence(duration=1800)
    outline = notes.default_outline(data)
    parts = []
    for position, chapter in enumerate(outline["chapters"]):
        blocks = [] if position == 0 else [{
            "id": "ch002_definition", "kind": "text", "content": "Определение",
            "evidence_ids": ["speech"]}]
        reply = {"title": data["title"], "chapters": [{**chapter, "blocks": blocks}],
                 "glossary": [], "issues": []}
        parts.append(notes._validate_chapter_reply(reply, data, chapter))
    merged = notes._merge_chapters(parts, outline, data)
    assert len(merged["chapters"]) == 2
    assert merged["chapters"][0]["blocks"] == []
    assert merged["chapters"][1]["blocks"][0]["content"] == "Определение"
    assert [(chapter["start"], chapter["end"]) for chapter in merged["chapters"]] == [
        (0, 900), (900, 1800)]
