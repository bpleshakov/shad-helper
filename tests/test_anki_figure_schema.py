"""Original figure references and standalone PDF crops remain distinct inputs."""

import pytest

from shad_helper.anki_schema import Figure, SourceFigure, generation_schema, validate_response


def response_with_figure(figure):
    return {"title": "Lecture", "notes": [{
        "note_type": "Basic", "front": "Question", "back": "Answer",
        "source": {"page": 1, "order": 1}, "figures": [figure],
    }]}


@pytest.mark.parametrize(("source_figures", "expected", "absent"), [
    (True, "SourceFigure", "Figure"),
    (False, "Figure", "SourceFigure"),
])
def test_generation_schema_offers_only_the_available_figure_source(
        source_figures, expected, absent):
    definitions = generation_schema(source_figures=source_figures)["$defs"]
    assert absent not in definitions
    assert definitions[expected]["additionalProperties"] is False
    for kind in ("BasicNote", "DefinitionNote", "ProofNote", "StepsNote"):
        assert definitions[kind]["properties"]["figures"]["items"] == {
            "$ref": f"#/$defs/{expected}"}


def test_default_generation_schema_keeps_standalone_pdf_crops():
    assert generation_schema() == generation_schema(source_figures=False)


@pytest.mark.parametrize(("figure", "kind"), [
    ({"field": "Back", "block_id": "ch001-figure-1"}, SourceFigure),
    ({"field": "Back", "page": 1, "box": [0.1, 0.2, 0.8, 0.9]}, Figure),
])
def test_validate_response_accepts_original_references_and_legacy_crops(figure, kind):
    result = validate_response(response_with_figure(figure), page_count=1)
    assert isinstance(result.notes[0].figures[0], kind)
    assert result.notes[0].figures[0].model_dump() == figure


@pytest.mark.parametrize("figure", [
    {"field": "Back", "block_id": "ch001-figure-1", "page": 1, "box": [0, 0, 1, 1]},
    {"field": "Back", "block_id": "ch001-figure-1", "page": 1},
    {"field": "Back", "block_id": "   "},
    {"field": "Step 1", "block_id": "ch001-figure-1"},
    {"field": "Back", "page": 2, "box": [0, 0, 1, 1]},
    {"field": "Back", "page": 1, "box": [0.8, 0.2, 0.1, 0.9]},
    {"field": "Back", "page": 1, "box": [0, 0, 1.1, 1]},
])
def test_validate_response_rejects_ambiguous_or_invalid_figure_requests(figure):
    with pytest.raises(ValueError):
        validate_response(response_with_figure(figure), page_count=1)


def test_original_figure_cannot_target_an_empty_optional_field():
    response = {"title": "Lecture", "notes": [{
        "note_type": "Theorem Full proof", "name": "Theorem", "statement": "Claim",
        "proof": None, "review_flags": ["No proof supplied"],
        "source": {"page": 1, "order": 1},
        "figures": [{"field": "Full proof", "block_id": "ch001-figure-1"}],
    }]}
    with pytest.raises(ValueError, match="populated Anki field"):
        validate_response(response, page_count=1)
