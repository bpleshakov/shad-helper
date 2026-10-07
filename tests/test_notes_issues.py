import json

import pytest

from shad_helper import notes
from shad_helper.jobs import read_json, write_json
from shad_helper.openai_client import ResponseError

DIAGNOSTIC_DESCRIPTIONS = ["UPSTREAM_DIAGNOSTIC", "VISUAL_DIAGNOSTIC", "VISUAL_SELECTION",
                          "REGION_DIAGNOSTIC", "REGION_SELECTION", "SOURCE_DIAGNOSTIC",
                          "SOURCE_SELECTION"]


def prepare_job(job):
    data = {"title": "Lecture", "video": {"duration": 300}, "segments": [
        {"id": "speech", "start": 10, "end": 12, "text": "Lecture statement"}],
        "visuals": [{"id": "board", "timestamp": 290, "page_id": "page1",
                     "page_start": 0, "page_end": 300, "image": "frames/board.png",
                     "status": "deferred", "summary": "A formula and a diagram",
                     "issues": ["VISUAL_DIAGNOSTIC"], "selection_issues": ["VISUAL_SELECTION"],
                     "regions": [{"id": "formula", "kind": "formula", "text": "x^2",
                                  "bbox": [0, 0, 1, 1], "image": "crops/formula.png",
                                  "issues": ["REGION_DIAGNOSTIC"],
                                  "selection_issues": ["REGION_SELECTION"]}]}],
        "sources": [{"id": "source", "kind": "ipynb", "source": "lecture.ipynb",
                     "location": "cell 1", "matched_visual_ids": ["board"],
                     "text": "Lecture source text", "code": "issues = {'exercise': 'proof'}",
                     "issues": ["SOURCE_DIAGNOSTIC"], "selection_issues": ["SOURCE_SELECTION"],
                     "outputs": [{"output_type": "display_data", "data": {
                         "application/json": {"issues": ["Real lecture output"]}}}]}],
        "issues": [{"id": "upstream", "description": "UPSTREAM_DIAGNOSTIC",
                    "evidence_ids": ["speech", "formula", "source"],
                    "status": "open", "resolution": ""}]}
    write_json(job / "evidence.json", data)
    for relative in ("frames/board.png", "crops/formula.png"):
        image = job / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
    return data


def assert_model_context(context, data):
    assert "issues" not in context
    serialized = json.dumps(context)
    assert not any(description in serialized for description in DIAGNOSTIC_DESCRIPTIONS)
    visual = context["visuals"][0]
    assert "issues" not in visual and "selection_issues" not in visual
    assert visual["status"] == "deferred"
    assert (visual["page_start"], visual["page_end"], visual["timestamp"]) == (0, 300, 290)
    region = visual["regions"][0]
    assert "issues" not in region and "selection_issues" not in region
    assert (region["text"], region["bbox"], region["image"]) == (
        "x^2", [0, 0, 1, 1], "crops/formula.png")
    source = context["sources"][0]
    assert "issues" not in source and "selection_issues" not in source
    assert (source["text"], source["code"], source["outputs"]) == (
        data["sources"][0]["text"], data["sources"][0]["code"], data["sources"][0]["outputs"])


class FakeClient:
    model = "fake"

    def __init__(self, fail=False):
        self.fail = fail
        self.prompts = []

    def complete(self, instructions, prompt, images, schema):
        self.prompts.append(prompt)
        payload = json.loads(prompt.rsplit("\n", 1)[1])
        chapter = payload["context"]["chapter"]
        if self.fail:
            raise ResponseError("Temporary synthesis failure", code="server_error")
        return {"title": "Lecture", "language": "ru", "chapters": [
            {**chapter, "blocks": [{"id": "ch001_text", "kind": "text", "content": "Statement",
                                    "evidence_ids": ["speech"]}]}], "glossary": [],
            "issues": [{"id": "model_issue", "description": "New model ambiguity",
                        "evidence_ids": ["formula"], "status": "open", "resolution": ""}],
            "illustrations": []}


def test_automatic_input_omits_diagnostics_but_local_review_preserves_all_issues(tmp_path, monkeypatch):
    data = prepare_job(tmp_path)
    original = (tmp_path / "evidence.json").read_bytes()
    destination = tmp_path / "project.tex"
    destination.write_text("test project")
    monkeypatch.setattr("shad_helper.rendering.write_project", lambda *args: destination)
    client = FakeClient()
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client) == destination
    assert len(client.prompts) == 1
    assert_model_context(json.loads(client.prompts[0].rsplit("\n", 1)[1])["context"], data)
    assert "не получили полную визуальную" in client.prompts[0]
    assert (tmp_path / "evidence.json").read_bytes() == original
    generated = read_json(tmp_path / "notes.json")
    issues = {issue["id"]: issue for issue in generated["issues"]}
    assert issues["upstream"] == data["issues"][0]
    assert issues["model_issue"]["description"] == "New model ambiguity"
    assert any(block["kind"] == "uncertain" and block["evidence_ids"] == ["board"]
               for chapter in generated["chapters"] for block in chapter["blocks"])
    assert read_json(tmp_path / "review.json") == {"issues": generated["issues"], "ready": False}
    assert notes.chapter_context(data, notes.default_outline(data)["chapters"][0])["issues"] == data["issues"]


@pytest.mark.parametrize("fallback", [False, True])
def test_manual_and_failed_fallback_handoffs_omit_diagnostics(tmp_path, fallback):
    data = prepare_job(tmp_path)
    original = (tmp_path / "evidence.json").read_bytes()
    write_json(tmp_path / "review.json", {"issues": data["issues"], "ready": False})
    client = FakeClient(fail=True)
    folder = notes.generate_notes(tmp_path, tmp_path / "auth", manual=not fallback, client=client)
    assert folder == tmp_path / "handoff"
    context = read_json(folder / "ch001/context.json")
    assert_model_context(context, data)
    prompt = (folder / "ch001/prompt.txt").read_text()
    assert not any(description in prompt for description in DIAGNOSTIC_DESCRIPTIONS)
    assert "не получили полную визуальную" in prompt
    assert (folder / "ch001/frames/board.png").read_bytes() == b"image"
    assert (folder / "ch001/crops/formula.png").read_bytes() == b"image"
    assert (tmp_path / "evidence.json").read_bytes() == original
    assert read_json(tmp_path / "review.json") == {"issues": data["issues"], "ready": False}
    if fallback:
        assert len(client.prompts) == 1
        assert_model_context(json.loads(client.prompts[0].rsplit("\n", 1)[1])["context"], data)
        assert read_json(folder / "status.json")["status"] == "manual_required"
    else:
        assert client.prompts == []
