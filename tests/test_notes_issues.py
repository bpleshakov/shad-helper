import json

import pytest

from shad_helper import notes
from shad_helper.jobs import read_json, write_json
from shad_helper.openai_client import ResponseError


def prepare_job(job):
    data = {"schema_version": 2, "title": "Lecture", "media": {"duration": 300},
            "segments": [{"id": "speech", "start": 0, "end": 300, "text": "Statement",
                          "words": [{"word": "Statement", "probability": 0.1}],
                          "diagnostics": {"avg_logprob": -1.5}}],
            "issues": [{"id": "upstream", "description": "UPSTREAM_DIAGNOSTIC",
                        "evidence_ids": ["speech"], "status": "open", "resolution": ""}]}
    write_json(job / "evidence.json", data)
    return data


class FakeClient:
    model = "fake"

    def __init__(self, fail=False):
        self.fail = fail
        self.prompts = []

    def complete(self, instructions, prompt, schema=None):
        self.prompts.append(prompt)
        payload = json.loads(prompt.rsplit("\n", 1)[1])
        chapter = payload["context"]["chapter"]
        if self.fail:
            raise ResponseError("Temporary synthesis failure", code="server_error")
        return {"title": "Lecture", "language": "ru", "chapters": [
            {**chapter, "blocks": [{"id": "ch001_text", "kind": "text", "content": "Statement",
                                   "evidence_ids": ["speech"]}]}], "glossary": [],
            "issues": [{"id": "model_issue", "description": "New model ambiguity",
                        "evidence_ids": ["speech"], "status": "open", "resolution": ""}]}


def assert_model_context(context, data):
    assert set(context) == {"chapter", "segments"}
    assert context["segments"] == [
        {key: segment[key] for key in ("id", "start", "end", "text")}
        for segment in data["segments"]]
    assert "UPSTREAM_DIAGNOSTIC" not in json.dumps(context)


def test_automatic_input_omits_diagnostics_but_local_review_preserves_issues(tmp_path):
    data = prepare_job(tmp_path)
    original = (tmp_path / "evidence.json").read_bytes()
    client = FakeClient()
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    assert len(client.prompts) == 1
    assert_model_context(json.loads(client.prompts[0].rsplit("\n", 1)[1])["context"], data)
    assert (tmp_path / "evidence.json").read_bytes() == original
    generated = read_json(tmp_path / "notes.json")
    issues = {issue["id"]: issue for issue in generated["issues"]}
    assert issues["upstream"] == data["issues"][0]
    assert issues["model_issue"]["description"] == "New model ambiguity"
    assert read_json(tmp_path / "review.json") == {"issues": generated["issues"], "ready": False}


@pytest.mark.parametrize("fallback", [False, True])
def test_manual_and_failed_fallback_handoffs_omit_diagnostics(tmp_path, fallback):
    data = prepare_job(tmp_path)
    original = (tmp_path / "evidence.json").read_bytes()
    write_json(tmp_path / "review.json", {"issues": data["issues"], "ready": False})
    client = FakeClient(fail=True)
    folder = notes.generate_notes(tmp_path, tmp_path / "auth", manual=not fallback, client=client)
    assert folder == tmp_path / "handoff"
    assert_model_context(read_json(folder / "ch001/context.json"), data)
    assert "UPSTREAM_DIAGNOSTIC" not in (folder / "ch001/prompt.txt").read_text()
    assert (tmp_path / "evidence.json").read_bytes() == original
    assert read_json(tmp_path / "review.json") == {"issues": data["issues"], "ready": False}
    assert not list(folder.rglob("*.png"))
    if fallback:
        assert len(client.prompts) == 1
        assert read_json(folder / "status.json")["status"] == "manual_required"
    else:
        assert client.prompts == []
