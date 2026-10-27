import copy
import json
import shutil
from unittest.mock import Mock

import pytest

from shad_helper import notes, rendering
from shad_helper.jobs import read_json, write_json


def prepare_job(job):
    write_json(job / "evidence.json", {
        "schema_version": 3, "title": "Лекция", "media": {"duration": 900},
        "segments": [{"id": "s1", "start": 5, "end": 10, "text": "Субдифференциал"}],
    })


class Client:
    model = "fake"

    def __init__(self, content):
        self.content = content
        self.calls = []

    def complete(self, instructions, text, schema=None):
        context = json.loads(text)
        self.calls.append(copy.deepcopy(context))
        return {"title": "Лекция", "language": "ru", "glossary": [], "issues": [],
                "chapters": [{**context["chapter"], "blocks": [
                    {"id": "ch001_b1", "kind": "text", "content": "Определение",
                     "topic": "Субдифференциал", "evidence_ids": ["s1"]},
                    {"id": "ch001_b2", "kind": "math", "content": self.content,
                     "topic": "Субдифференциал", "evidence_ids": ["s1"]},
                ]}]}


def attempt(job):
    return job / "handoff/requests/ch001/attempt-01"


def test_compile_failure_retains_response_without_retry_or_checkpoint(tmp_path, monkeypatch):
    prepare_job(tmp_path)
    client = Client("x=0")
    checker = Mock(side_effect=rendering.RenderError("Chapter ch001: block ch001_b2 failed"))
    monkeypatch.setattr(rendering, "validate_chapter_latex", checker)
    sleep = Mock()
    monkeypatch.setattr(notes.time, "sleep", sleep)

    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client) == tmp_path / "handoff"
    assert len(client.calls) == 1
    sleep.assert_not_called()
    assert not (tmp_path / "checkpoints/notes/ch001.json").exists()
    assert not (tmp_path / "notes.json").exists()
    folder = attempt(tmp_path)
    assert read_json(folder / "response.json")["chapters"][0]["blocks"][1]["content"] == "x=0"
    assert read_json(folder / "status.json")["error_type"] == "RenderError"
    assert read_json(tmp_path / "handoff/status.json")["reason"] == checker.side_effect.args[0]
    assert "failed" not in json.dumps(client.calls)


def test_missing_compiler_stops_generation_locally(tmp_path, monkeypatch, chapter_latex_checker):
    prepare_job(tmp_path)
    monkeypatch.setattr(rendering, "validate_chapter_latex", chapter_latex_checker)
    monkeypatch.setattr(rendering.shutil, "which", lambda name: None)
    client = Client("x=0")
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client) == tmp_path / "handoff"
    assert len(client.calls) == 1
    assert not (tmp_path / "checkpoints/notes/ch001.json").exists()
    assert "requires latexmk and LuaLaTeX" in read_json(tmp_path / "handoff/status.json")["reason"]
    assert (attempt(tmp_path) / "latex-check/notes/main.tex").is_file()


@pytest.mark.skipif(not shutil.which("latexmk") or not shutil.which("lualatex"),
                    reason="requires TeX Live/MacTeX")
def test_real_cases_failure_then_correction_and_cache(tmp_path, monkeypatch, chapter_latex_checker):
    prepare_job(tmp_path)
    monkeypatch.setattr(rendering, "validate_chapter_latex", chapter_latex_checker)
    malformed = r"\partial f(x)=\begin{cases}\{1\},&x>0,\\\{-1\},&x<0,\\[-1,1],&x=0.\end{cases}"
    client = Client(malformed)
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client) == tmp_path / "handoff"
    assert len(client.calls) == 1
    assert not (tmp_path / "checkpoints/notes/ch001.json").exists()
    folder = attempt(tmp_path)
    error = read_json(folder / "status.json")["error"]
    assert "Chapter ch001" in error
    assert "block ch001_b2" in error
    assert "Illegal unit of measure" in error
    assert (folder / "latex-check/notes/compiler-output.log").is_file()

    client.content = malformed.replace(r"\\[-1,1]", r"\\{[-1,1]}")
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).is_file()
    assert len(client.calls) == 2
    assert read_json(tmp_path / "checkpoints/notes/ch001.json")["result"]["chapters"][0][
        "blocks"][1]["content"] == client.content
    assert not (tmp_path / "handoff/status.json").exists()
    pdf = attempt(tmp_path) / "latex-check/notes/main.pdf"
    assert pdf.read_bytes().startswith(b"%PDF")
    assert not (pdf.parent / "qa").exists()
    assert read_json(folder / "response.json")["chapters"][0]["blocks"][1][
        "content"] == client.content

    monkeypatch.setattr(rendering, "validate_chapter_latex", Mock(side_effect=AssertionError("Cached")))
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).is_file()
    assert len(client.calls) == 2
    assert not attempt(tmp_path).exists()
    assert not (tmp_path / "handoff/runs").exists()


@pytest.mark.skipif(not shutil.which("latexmk") or not shutil.which("lualatex"),
                    reason="requires TeX Live/MacTeX")
def test_valid_row_spacing_and_other_rendered_blocks(tmp_path, chapter_latex_checker):
    prepare_job(tmp_path)
    evidence = read_json(tmp_path / "evidence.json")
    client = Client(r"\begin{cases}1,&x>0,\\[2pt]{[-1,1]},&x=0.\end{cases}")
    result = client.complete("", json.dumps(notes.chapter_context(evidence, notes.default_outline(
        evidence)["chapters"][0])))
    result["chapters"][0]["blocks"].extend([
        {"id": "ch001_b3", "kind": "definition", "content": r"Вектор \(x\).",
         "evidence_ids": ["s1"]},
        {"id": "ch001_b4", "kind": "diagram", "content": r"\draw (0,0)--(1,1);",
         "caption": "Схема", "evidence_ids": ["s1"]},
        {"id": "ch001_b5", "kind": "code", "content": "print('Привет')",
         "evidence_ids": ["s1"]},
    ])
    result["glossary"] = [{"term": "Вектор", "definition": r"Элемент \(\mathbb{R}^d\)."}]
    chapter_latex_checker(result, evidence, tmp_path / "check")
    assert (tmp_path / "check/notes/main.pdf").is_file()
    assert (tmp_path / "check/notes/figures/diagram-ch001_b4.tex").is_file()
