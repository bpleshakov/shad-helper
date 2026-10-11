"""Structured PDF-to-Anki output, independent of Anki's template ordinals."""

from copy import deepcopy
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .anki_latex import normalize_latex

Text = Annotated[str, Field(min_length=1)]
Coordinate = Annotated[float, Field(ge=0, le=1)]


class Source(BaseModel):
    page: int = Field(ge=1)
    order: int = Field(ge=1, description="Unique note order on its first source page")


class AnswerSource(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    page: int = Field(ge=1, description="Physical PDF page containing this card's source")
    location: Text = Field(description="Heading, theorem/formula number or short exact "
                          "excerpt locating this card's source on the page")
    block_ids: list[Text] = Field(default_factory=list,
        description="Supporting block IDs from the supplied source catalog; empty only "
                    "when no catalog was provided")


class CardSource(AnswerSource):
    field: Text


class SourcedText(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: Text
    source: AnswerSource


class Figure(BaseModel):
    field: Text = Field(description="Exact destination Anki field name")
    page: int = Field(ge=1)
    box: list[Coordinate] = Field(min_length=4, max_length=4,
        description="[left, top, right, bottom], normalized to the displayed PDF page")

    @model_validator(mode="after")
    def positive_area(self):
        x0, y0, x1, y1 = self.box
        if x0 >= x1 or y0 >= y1:
            raise ValueError("Figure box must have positive width and height")
        return self


class NoteBase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    source: Source
    card_sources: list[CardSource] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    supplemented_fields: list[str] = Field(default_factory=list,
        description="Exact Anki fields containing reasoning added beyond the PDF")
    review_flags: list[str] = Field(default_factory=list)


class BasicNote(NoteBase):
    note_type: Literal["Basic"]
    front: Text
    back: Text


class DefinitionNote(NoteBase):
    note_type: Literal["Multistep definition"]
    name: Text
    overview: Text
    steps: list[Text] = Field(min_length=1, max_length=15)


class ProofNote(NoteBase):
    note_type: Literal["Theorem Full proof"]
    name: Text
    statement: Text
    proof: Text | None

    @model_validator(mode="after")
    def explain_missing_proof(self):
        if self.proof is None and not self.review_flags:
            raise ValueError("A missing proof requires a review flag")
        return self


class StepsNote(NoteBase):
    note_type: Literal["Theorem step by step"]
    name: Text
    statement: Text
    plan: Text
    steps: list[Text] = Field(min_length=1, max_length=15)


class AnkiResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: Text
    notes: list[BasicNote | DefinitionNote | ProofNote | StepsNote] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)


_ANSWER_FIELDS = {
    "Basic": {"back": "Back"},
    "Multistep definition": {"overview": "Definition steps"},
    "Theorem Full proof": {"statement": "Statement", "proof": "Full proof"},
    "Theorem step by step": {"statement": "Statement", "plan": "Steps to prove"},
}


def generation_schema() -> dict:
    """Require a source alongside each answer, without a parallel field-name list."""
    schema = AnkiResponse.model_json_schema()
    sourced = SourcedText.model_json_schema()
    schema["$defs"].update(sourced.pop("$defs"))
    schema["$defs"]["SourcedText"] = sourced
    schema["$defs"].pop("CardSource")
    for definition in schema["$defs"].values():
        properties = definition.get("properties", {})
        kind = properties.get("note_type", {}).get("const")
        if kind not in _ANSWER_FIELDS:
            continue
        properties.pop("card_sources")
        for field in _ANSWER_FIELDS[kind]:
            answer = {"$ref": "#/$defs/SourcedText"}
            properties[field] = {"anyOf": [answer, {"type": "null"}]} if field == "proof" else answer
        if "steps" in properties:
            properties["steps"]["items"] = {"$ref": "#/$defs/SourcedText"}
    return schema


def _unpack_sourced_answers(value: dict) -> dict:
    value = deepcopy(value)
    if not isinstance(value.get("notes"), list):
        return value
    for note in value["notes"]:
        if not isinstance(note, dict):
            continue
        answers = _ANSWER_FIELDS.get(note.get("note_type"), {})
        steps = note.get("steps", [])
        if not (any(isinstance(note.get(field), dict) for field in answers)
                or isinstance(steps, list) and any(isinstance(step, dict) for step in steps)):
            continue  # Existing caches and internal export data use plain text fields.
        if "card_sources" in note:
            raise ValueError("Sourced answers cannot also supply a separate card_sources list")
        references = []
        for field, target in answers.items():
            if note.get(field) is not None:
                answer = SourcedText.model_validate(note[field])
                note[field] = answer.text
                references.append({"field": target, **answer.source.model_dump()})
        if isinstance(steps, list) and "steps" in note:
            parsed = [SourcedText.model_validate(step) for step in steps]
            note["steps"] = [step.text for step in parsed]
            references.extend({"field": f"Step {i}", **step.source.model_dump()}
                              for i, step in enumerate(parsed, 1))
        note["card_sources"] = references
    return value


def note_fields(note: NoteBase) -> dict[str, str]:
    if isinstance(note, BasicNote):
        return {"Front": note.front, "Back": note.back}
    if isinstance(note, DefinitionNote):
        fields = {"Name": note.name, "Definition steps": note.overview}
    else:
        fields = {"Name": note.name, "Statement": note.statement}
        if isinstance(note, ProofNote):
            return {**fields, "Full proof": note.proof or ""}
        fields["Steps to prove"] = note.plan
    fields.update({f"Step {i + 1}": value for i, value in enumerate(note.steps)})
    return fields


def validate_response(value: dict, page_count: int) -> AnkiResponse:
    result = AnkiResponse.model_validate(_unpack_sourced_answers(value))
    sources = set()
    for note in result.notes:
        for name in ("front", "back", "name", "overview", "statement", "proof", "plan"):
            value = getattr(note, name, None)
            if value is not None:
                setattr(note, name, normalize_latex(value))
        if hasattr(note, "steps"):
            note.steps = [normalize_latex(value) for value in note.steps]
        source = (note.source.page, note.source.order)
        if source in sources:
            raise ValueError("Notes must have distinct source page/order pairs")
        sources.add(source)
        if note.source.page > page_count:
            raise ValueError("Note source is outside the PDF")
        fields = note_fields(note)
        for target in note.supplemented_fields:
            if not fields.get(target):
                raise ValueError("Supplement marker must refer to a populated Anki field")
        for figure in note.figures:
            if figure.page > page_count or not fields.get(figure.field):
                raise ValueError("Figure must refer to a PDF page and populated Anki field")
        for text in fields.values():
            if text and not text.strip():
                raise ValueError("Anki fields cannot contain only whitespace")
    result.notes.sort(key=lambda note: (note.source.page, note.source.order))
    return result
