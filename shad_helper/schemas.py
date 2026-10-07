"""The same validated note format is used by API and manual ChatGPT responses."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReviewIssue(StrictModel):
    id: Identifier
    description: str
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["open", "resolved"] = "open"
    resolution: str = ""

    @model_validator(mode="after")
    def resolved_explanation(self):
        if self.status == "resolved" and not self.resolution.strip():
            raise ValueError("A resolved review issue requires a resolution")
        return self


class NoteBlock(StrictModel):
    id: Identifier
    kind: Literal["text", "math", "code", "diagram", "clarification", "uncertain"]
    content: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    caption: str = ""
    language: str = ""


class Chapter(StrictModel):
    id: Identifier
    title: str = Field(min_length=1)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    blocks: list[NoteBlock] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_interval(self):
        if self.end <= self.start:
            raise ValueError("Chapter end must be after its start")
        return self


class GlossaryTerm(StrictModel):
    term: str
    definition: str


class LectureNotes(StrictModel):
    title: str = Field(min_length=1)
    language: Literal["ru"] = "ru"
    chapters: list[Chapter] = Field(min_length=1)
    glossary: list[GlossaryTerm] = Field(default_factory=list)
    issues: list[ReviewIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def stable_identifiers(self):
        identifiers = [chapter.id for chapter in self.chapters]
        identifiers += [block.id for chapter in self.chapters for block in chapter.blocks]
        identifiers += [issue.id for issue in self.issues]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Note, chapter, and issue IDs must be unique")
        return self


class OutlineChapter(StrictModel):
    id: Identifier
    title: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class LectureOutline(StrictModel):
    title: str
    chapters: list[OutlineChapter] = Field(min_length=1)
    glossary: list[GlossaryTerm] = Field(default_factory=list)
