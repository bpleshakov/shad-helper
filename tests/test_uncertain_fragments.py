import pytest
from pydantic import ValidationError

from shad_helper import notes
from shad_helper.jobs import fingerprint, read_json, write_json
from shad_helper.schemas import Chapter, NoteBlock, UncertainFragment


def note_block(content, fragments, *, kind="text", evidence_ids=None):
    return NoteBlock.model_validate({
        "id": "b1", "kind": kind, "content": content,
        "evidence_ids": ["s1", "s2"] if evidence_ids is None else evidence_ids,
        "uncertain_fragments": fragments,
    })


def fragment(text, evidence_ids=None):
    return {"text": text, "evidence_ids": ["s1"] if evidence_ids is None else evidence_ids}


@pytest.mark.parametrize("kind", ["text", "clarification", "uncertain"])
def test_prose_fragments_preserve_cyrillic_and_complete_inline_math(kind):
    content = r"Оценка \(\hat{x}=x_1\) несмещённая, а дисперсия мала."
    selections = [fragment(r"Оценка \(\hat{x}=x_1\)"), fragment("дисперсия мала", ["s2"])]
    block = note_block(content, selections, kind=kind)
    assert block.content == content
    assert [item.model_dump() for item in block.uncertain_fragments] == selections


def test_existing_blocks_default_to_no_fragments():
    block = NoteBlock(id="b1", kind="math", content="x=1")
    assert block.uncertain_fragments == []


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
def test_fragment_text_cannot_be_empty_or_whitespace(text):
    with pytest.raises(ValidationError):
        UncertainFragment.model_validate(fragment(text))


def test_fragment_needs_evidence():
    with pytest.raises(ValidationError):
        UncertainFragment.model_validate(fragment("оценка", []))


@pytest.mark.parametrize("content,selection", [
    ("Оценка несмещённая.", "эффективная"),
    ("Оценка и оценка.", "ценка"),
    ("ааа", "аа"),
])
def test_fragment_must_match_content_once_even_for_overlapping_occurrences(content, selection):
    with pytest.raises(ValidationError, match="match exactly once"):
        note_block(content, [fragment(selection)])


def test_fragments_cannot_overlap():
    with pytest.raises(ValidationError, match="must not overlap"):
        note_block("Несмещённая оценка среднего.", [
            fragment("Несмещённая оценка"), fragment("оценка среднего"),
        ])


def test_adjacent_fragments_and_out_of_order_selections_are_supported():
    selections = [fragment("дисперсия"), fragment("Оценка и ")]
    block = note_block("Оценка и дисперсия", selections)
    assert [item.text for item in block.uncertain_fragments] == [item["text"] for item in selections]


@pytest.mark.parametrize("kind", ["math", "code", "diagram"])
def test_fragments_require_a_prose_block(kind):
    with pytest.raises(ValidationError, match="only in prose blocks"):
        note_block("x=1", [fragment("x=1")], kind=kind)


def test_fragment_cannot_add_evidence_outside_its_block():
    with pytest.raises(ValidationError, match="must belong to the block"):
        note_block("Оценка", [fragment("Оценка", ["s1", "other"])])


@pytest.mark.parametrize("selection", [
    "x_1", r"\(x_1", r"x_1\)", "(x_1", r"Оценка \(", r"\) несмещённая.",
])
def test_fragment_boundaries_cannot_cut_inline_math(selection):
    with pytest.raises(ValidationError, match="complete inline math expression"):
        note_block(r"Оценка \(x_1\) несмещённая.", [fragment(selection)])


def test_a_complete_inline_formula_can_be_selected_alone():
    formula = r"\(x_1\)"
    block = note_block(f"Оценка {formula} несмещённая.", [fragment(formula)])
    assert block.uncertain_fragments[0].text == formula


@pytest.mark.parametrize("content", [r"Оценка \(x", r"Оценка x\)", r"Оценка \(x \(y\)\)"])
def test_unbalanced_math_is_rejected_when_fragments_would_split_the_content(content):
    with pytest.raises(ValidationError, match="Unbalanced inline math"):
        note_block(content, [fragment("Оценка")])


def test_marked_block_cannot_split_a_paragraph_inside_unselected_inline_math():
    content = "Сомнительно. \\(x\n\n+y\\)"
    with pytest.raises(ValidationError, match="Inline math.*multiple paragraphs"):
        note_block(content, [fragment("Сомнительно")])
    assert note_block(content, []).content == content


def test_wrapped_lines_are_allowed_but_a_fragment_cannot_span_paragraphs():
    content = "Оценка\nнесмещённая."
    assert note_block(content, [fragment(content)]).content == content
    for separator in ("\n\n", "\n \n", "\r\n\r\n"):
        content = f"Оценка.{separator}Дисперсия."
        with pytest.raises(ValidationError, match="multiple paragraphs"):
            note_block(content, [fragment(content)])


def test_an_empty_chapter_retains_its_outline_interval():
    chapter = Chapter(id="c1", title="Организационные вопросы", start=60, end=120, blocks=[])
    assert chapter.model_dump() == {
        "id": "c1", "title": "Организационные вопросы", "start": 60, "end": 120, "blocks": [],
    }
    with pytest.raises(ValidationError, match="end must be after"):
        Chapter(id="c1", title="Вопросы", start=60, end=30, blocks=[])


@pytest.fixture(params=[3, 4])
def source_evidence(request):
    result = {
        "schema_version": request.param, "title": "Оценки", "media": {"duration": 1800},
        "segments": [
            {"id": identifier, "start": start, "end": start + 5, "text": "Оценка параметра"}
            for identifier, start in [("s1", 10), ("s2", 20), ("s3", 30), ("later", 1500)]
        ],
    }
    if request.param == 4:
        result["visuals"] = [
            {"id": identifier, "timestamp": start, "status": "complete", "kind": "whiteboard",
             "text": "Дисперсия", "formulas": [], "diagrams": [], "unreadable": []}
            for identifier, start in [("frame", 25), ("later_frame", 1500)]
        ]
    return result


@pytest.fixture
def notes_reply():
    return {
        "title": "Оценки", "chapters": [{
            "id": "c1", "title": "Свойства оценок", "start": 0, "end": 900,
            "blocks": [{
                "id": "b1", "kind": "text", "content": "Оценка несмещённая; дисперсия мала.",
                "evidence_ids": ["s1", "s2", "s3"],
                "uncertain_fragments": [
                    fragment("несмещённая"), fragment("дисперсия мала", ["s2", "s1"]),
                ],
            }],
        }],
    }


def test_validate_notes_retains_fragments_and_creates_one_issue_for_only_marked_sources(
        notes_reply, source_evidence):
    validated = notes.validate_notes(notes_reply, source_evidence)
    assert validated["chapters"][0]["blocks"][0]["uncertain_fragments"] == (
        notes_reply["chapters"][0]["blocks"][0]["uncertain_fragments"])
    assert validated["issues"] == [{
        "id": "review_b1", "description": "несмещённая; дисперсия мала",
        "evidence_ids": ["s1", "s2"], "status": "open", "resolution": "",
    }]
    assert not notes.review_ready(validated)


def test_validate_notes_checks_the_fragment_schema(notes_reply, source_evidence):
    notes_reply["chapters"][0]["blocks"][0]["uncertain_fragments"][0]["text"] = "Отсутствует"
    with pytest.raises(ValidationError, match="match exactly once"):
        notes.validate_notes(notes_reply, source_evidence)


@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("reference", ["later", "later_frame", "unknown"])
def test_fragment_cannot_cite_an_unknown_or_different_chapter_source(
        notes_reply, source_evidence, scoped, reference):
    chapter = notes_reply["chapters"][0]
    chapter["blocks"][0]["evidence_ids"] = [reference]
    for selection in chapter["blocks"][0]["uncertain_fragments"]:
        selection["evidence_ids"] = [reference]
    with pytest.raises(ValueError, match="Unknown evidence"):
        if scoped:
            notes._validate_chapter_reply(notes_reply, source_evidence, chapter)
        else:
            notes.validate_notes(notes_reply, source_evidence)


def test_resolution_requires_removing_fragment_markers_before_notes_are_ready(
        tmp_path, notes_reply, source_evidence):
    validated = notes.validate_notes(notes_reply, source_evidence)
    write_json(tmp_path / "evidence.json", source_evidence)
    write_json(tmp_path / "notes.json", validated)
    notes.resolve_issue(tmp_path, "review_b1", "Формулировка проверена по записи")
    resolved = read_json(tmp_path / "notes.json")
    records = read_json(tmp_path / "review-resolutions.json")
    assert resolved["issues"][0]["status"] == "resolved"
    assert not read_json(tmp_path / "review.json")["ready"]
    revalidated = notes.validate_notes(resolved, source_evidence, records)
    assert revalidated["issues"][0]["status"] == "resolved"
    assert not notes.review_ready(revalidated)
    revalidated["chapters"][0]["blocks"][0]["uncertain_fragments"] = []
    corrected = notes.validate_notes(revalidated, source_evidence, records)
    assert corrected["issues"][0]["status"] == "resolved"
    assert notes.review_ready(corrected)


@pytest.mark.parametrize("change", ["selection", "evidence"])
def test_changing_a_fragment_invalidates_its_recorded_resolution(
        notes_reply, source_evidence, change):
    validated = notes.validate_notes(notes_reply, source_evidence)
    recorded = {"review_b1": {
        **validated["issues"][0], "resolution": "Проверено по записи",
        "evidence_fingerprint": fingerprint(source_evidence),
    }}
    selection = validated["chapters"][0]["blocks"][0]["uncertain_fragments"][0]
    if change == "selection":
        selection["text"] = "Оценка"
    else:
        selection["evidence_ids"] = ["s3"]
    changed = notes.validate_notes(validated, source_evidence, recorded)
    assert changed["issues"][0]["status"] == "open"
    assert changed["issues"][0]["resolution"] == ""
