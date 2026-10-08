"""The same validated note format is used by API and manual ChatGPT responses."""

import re
from itertools import pairwise
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


class UncertainFragment(StrictModel):
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def single_paragraph(self):
        if not self.text.strip():
            raise ValueError("An uncertain fragment must contain non-whitespace text")
        if re.search(r"\n\s*\n", self.text):
            raise ValueError("An uncertain fragment cannot span multiple paragraphs")
        return self


class NoteBlock(StrictModel):
    id: Identifier
    kind: Literal[
        "text", "definition", "theorem", "proof", "math", "code", "diagram", "clarification", "uncertain",
    ]
    content: str = Field(min_length=1)
    topic: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    caption: str = ""
    language: str = ""
    uncertain_fragments: list[UncertainFragment] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_uncertain_fragments(self):
        if not self.uncertain_fragments:
            return self
        if self.kind not in {"text", "definition", "theorem", "proof", "clarification", "uncertain"}:
            raise ValueError("Uncertain fragments are supported only in prose blocks")

        math_spans = []
        math_start = None
        for delimiter in re.finditer(r"\\[()]", self.content):
            if delimiter.group() == r"\(" and math_start is None:
                math_start = delimiter.start()
            elif delimiter.group() == r"\)" and math_start is not None:
                if re.search(r"\n\s*\n", self.content[math_start:delimiter.end()]):
                    raise ValueError("Inline math in a marked block cannot span multiple paragraphs")
                math_spans.append((math_start, delimiter.end()))
                math_start = None
            else:
                raise ValueError("Unbalanced inline math delimiters")
        if math_start is not None:
            raise ValueError("Unbalanced inline math delimiters")

        spans = []
        for fragment in self.uncertain_fragments:
            start = self.content.find(fragment.text)
            if start < 0 or self.content.find(fragment.text, start + 1) >= 0:
                raise ValueError("An uncertain fragment must match exactly once in block content")
            end = start + len(fragment.text)
            if not set(fragment.evidence_ids).issubset(self.evidence_ids):
                raise ValueError("Uncertain fragment evidence IDs must belong to the block")
            if any(left < start < right or left < end < right for left, right in math_spans):
                raise ValueError("An uncertain fragment must include the complete inline math expression")
            spans.append((start, end))
        spans.sort()
        if any(left[1] > right[0] for left, right in pairwise(spans)):
            raise ValueError("Uncertain fragments must not overlap")
        return self


class Chapter(StrictModel):
    id: Identifier
    title: str = Field(min_length=1)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    blocks: list[NoteBlock]

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
