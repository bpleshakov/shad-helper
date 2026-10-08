import copy
import json
from types import SimpleNamespace

import pytest

from shad_helper import notes, rendering
from shad_helper.jobs import evidence_index, fingerprint, read_json, write_json


def visual(identifier="frame_10", timestamp=10, **changes):
    return {"id": identifier, "timestamp": timestamp, "status": "complete",
            "kind": "whiteboard", "text": "Определение; поле issues относится к лекции.",
            "formulas": ["y=x^2"], "diagrams": [], "unreadable": [], **changes}


def evidence(*, duration=120, speech=True, visuals=None):
    return {"schema_version": 4, "title": "Лекция", "media": {"duration": duration},
            "segments": [{"id": "speech", "start": 0, "end": 20, "text": "Определение"}]
                        if speech else [],
            "visuals": [visual()] if visuals is None else visuals}


def prepare(job, data):
    write_json(job / "extraction-config.json",
               {"schema_version": 4, "pipeline": "audio_video", "vision_model": "9b"})
    write_json(job / "evidence.json", data)
    write_json(job / "extraction-diagnostics.json", {
        "raw": "PRIVATE_RAW_OUTPUT", "model": "PRIVATE_MODEL_NAME",
        "image": "/private/LOCAL_IMAGE.png", "selection_issues": ["PRIVATE_SELECTION_ISSUE"]})


def reply(chapter, identifier, *, diagram=False):
    content = (r"\begin{tikzpicture}\node[draw] (a) {Вход};"
               r"\node[draw,right=of a] (b) {Выход};\draw[->] (a)--(b);\end{tikzpicture}"
               if diagram else "Определение из кадра")
    return {"title": "Лекция", "chapters": [
        {**chapter, "blocks": [{"id": f"{chapter['id']}_b1",
                               "kind": "diagram" if diagram else "text", "content": content,
                               "caption": "Схема" if diagram else "", "evidence_ids": [identifier]}]}],
        "glossary": [], "issues": []}


class Client:
    model = "fake"

    def __init__(self, *, diagram=False):
        self.calls = []
        self.diagram = diagram

    def complete(self, instructions, prompt, schema=None):
        context = json.loads(prompt)
        self.calls.append({"instructions": instructions, "context": context, "schema": schema})
        return reply(context["chapter"], context["visuals"][0]["id"], diagram=self.diagram)


@pytest.mark.parametrize("speech", [True, False])
def test_video_automatic_and_manual_inputs_match_without_private_diagnostics(tmp_path, speech):
    data = evidence(speech=speech)
    prepare(tmp_path, data)
    client = Client()
    folder = notes.generate_notes(tmp_path, tmp_path / "auth", manual=True)
    manual_context = read_json(folder / "inputs/ch001/context.json")
    assert set(manual_context) == {
        "lecture_title", "glossary", "topics", "chapter", "segments", "visuals", "writing_budget"}
    assert manual_context["visuals"] == data["visuals"]
    assert "issues" in manual_context["visuals"][0]["text"]
    assert not list(folder.rglob("*.png"))
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    call = client.calls[0]
    assert call["context"] == manual_context
    assert call["instructions"] == (folder / "inputs/ch001/prompt.txt").read_text()
    assert call["schema"] == read_json(folder / "inputs/response-schema.json")
    assert not any(sentinel in json.dumps(call) for sentinel in
                   ("PRIVATE_RAW_OUTPUT", "PRIVATE_MODEL_NAME", "LOCAL_IMAGE", "PRIVATE_SELECTION"))
    assert read_json(tmp_path / "notes.json")["chapters"][0]["blocks"][0]["evidence_ids"] == ["frame_10"]
    assert evidence_index(data)["frame_10"] == "Видео 00:00:10"


def test_chapter_visual_context_retains_boundaries_and_one_preceding_state():
    frames = [visual(identifier, timestamp) for identifier, timestamp in
              [("ancient", 0), ("static", 100), ("start_edge", 840),
               ("inside", 1000), ("end_edge", 1260), ("future", 1260.1)]]
    data = evidence(duration=1800, visuals=list(reversed(frames)))
    chapter = {"id": "ch002", "title": "Продолжение", "start": 900, "end": 1200}
    original = copy.deepcopy(data)
    context = notes.chapter_context(data, chapter)
    assert [v["id"] for v in context["visuals"]] == ["static", "start_edge", "inside", "end_edge"]
    assert data == original
    static = evidence(duration=1800, visuals=[visual("long_slide", 50)])
    assert notes.chapter_context(static, chapter)["visuals"] == static["visuals"]


def test_visual_citations_are_scoped_and_static_context_remains_valid():
    data = evidence(duration=1800, visuals=[visual("older", 0), visual("carried", 100),
                                          visual("future", 1500)])
    chapter = {"id": "ch002", "title": "Продолжение", "start": 900, "end": 1200}
    accepted = notes._validate_chapter_reply(reply(chapter, "carried"), data, chapter)
    assert accepted["chapters"][0]["blocks"][0]["evidence_ids"] == ["carried"]
    for identifier in ("older", "future", "not_a_source"):
        with pytest.raises(ValueError, match="Unknown evidence"):
            notes._validate_chapter_reply(reply(chapter, identifier), data, chapter)


def test_visual_budget_splits_without_dropping_frame_evidence():
    frames = [visual(f"frame_{number}", number * 30, text="Теория " + "x" * 9000)
              for number in range(30)]
    data = evidence(duration=900, speech=False, visuals=frames)
    original = copy.deepcopy(data)
    bounded = notes._bounded_outline(data, notes.default_outline(data))
    assert len(bounded["chapters"]) > 1
    assert notes.validate_outline(bounded, data) == bounded
    contexts = [notes.chapter_context(data, chapter) for chapter in bounded["chapters"]]
    assert {v["id"] for context in contexts for v in context["visuals"]} == {v["id"] for v in frames}
    assert all(notes._chapter_budget_error(context, bounded, []) is None for context in contexts)
    assert data == original


def test_visual_content_invalidates_notes_cache_but_diagnostics_do_not(tmp_path):
    data = evidence()
    prepare(tmp_path, data)
    client = Client()
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    checkpoint = tmp_path / "checkpoints/notes/ch001.json"
    before = checkpoint.read_bytes()
    write_json(tmp_path / "extraction-diagnostics.json", {"raw": "changed private diagnostic"})
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert len(client.calls) == 1
    assert checkpoint.read_bytes() == before
    data["visuals"][0]["formulas"] = ["y=x^3"]
    write_json(tmp_path / "evidence.json", data)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert len(client.calls) == 2
    assert checkpoint.read_bytes() != before
    assert read_json(tmp_path / "notes/project.json")["evidence_fingerprint"] == fingerprint(data)


@pytest.mark.parametrize("frames", [[], [visual(text="", formulas=[], diagrams=[])],
                                   [visual(status="incomplete", text="", formulas=[],
                                           unreadable=["Нечитаемый кадр"])]])
def test_empty_visuals_do_not_bypass_missing_content_guard(tmp_path, frames):
    prepare(tmp_path, evidence(speech=False, visuals=frames))
    write_json(tmp_path / "checkpoints/notes/ch001.json", {"preserve": True})
    write_json(tmp_path / "notes.json", {"preserve": "previous notes"})
    with pytest.raises(ValueError, match="No speech or readable visual content"):
        notes.generate_notes(tmp_path, tmp_path / "auth", manual=True, rebuild=True)
    assert read_json(tmp_path / "checkpoints/notes/ch001.json") == {"preserve": True}
    assert read_json(tmp_path / "notes.json") == {"preserve": "previous notes"}
    runs = list((tmp_path / "handoff/runs").iterdir())
    assert len(runs) == 1
    assert read_json(runs[0] / "run.json") == {"mode": "manual", "status": "failed", "rebuild": True}
    assert {path.name for path in runs[0].iterdir()} == {"run.json"}
    assert not (tmp_path / "handoff/inputs").exists()


def test_model_context_does_not_forward_extra_top_level_fields():
    data = evidence()
    outline = notes.default_outline(data)
    context = notes.chapter_context(data, outline["chapters"][0])
    context.update(diagnostics={"secret": "PRIVATE"}, issues=["PRIVATE"])
    assert "PRIVATE" not in json.dumps(notes._model_context(context, outline, []))


def test_video_diagram_remains_editable_and_render_preserves_its_source(tmp_path, monkeypatch):
    data = evidence(speech=False, visuals=[visual(text="", formulas=[], diagrams=[{
        "description": "Вход слева, выход справа", "labels": ["Вход", "Выход"],
        "relationships": ["Стрелка от входа к выходу"]}])])
    prepare(tmp_path, data)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=Client(diagram=True))
    source = tmp_path / "notes/figures/diagram-ch001_b1.tex"
    assert source.exists()
    assert r"\draw[->] (a)--(b)" in source.read_text()
    assert "figures/diagram-ch001_b1.tex" in read_json(tmp_path / "notes/project.json")["tex_files"]
    corrected = source.read_text().replace("Выход", "Результат")
    source.write_text(corrected)

    def compile_pdf(*args, cwd, **kwargs):
        (cwd / "main.pdf").write_bytes(b"test PDF")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(rendering.shutil, "which", lambda name: name)
    monkeypatch.setattr(rendering.subprocess, "run", compile_pdf)
    monkeypatch.setattr(rendering, "_qa", lambda pdf: 1)
    assert rendering.render(tmp_path) == tmp_path / "notes/main.pdf"
    assert source.read_text() == corrected
    assert not list((tmp_path / "notes").rglob("*.png"))
