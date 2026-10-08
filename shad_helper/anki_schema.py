"""Structured PDF-to-Anki output, independent of Anki's template ordinals."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .anki_latex import normalize_latex

Text = Annotated[str, Field(min_length=1)]
Coordinate = Annotated[float, Field(ge=0, le=1)]


class Source(BaseModel):
    page: int = Field(ge=1)
    order: int = Field(ge=1, description="Unique note order on its first source page")


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
    result = AnkiResponse.model_validate(value)
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
