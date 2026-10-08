import copy
import json
from types import SimpleNamespace

import pytest

from shad_helper import cli, notes, rendering
from shad_helper.jobs import fingerprint, read_json, write_json
from shad_helper.openai_client import ResponseError


def prepare_job(job):
    data = {"schema_version": 3, "title": "Lecture", "media": {"duration": 300},
            "segments": [{"id": "speech", "start": 0, "end": 300, "text": "Statement"}]}
    write_json(job / "evidence.json", data)
    write_json(job / "extraction-diagnostics.json", {
        "schema_version": 1, "evidence_fingerprint": fingerprint(data), "models": {"asr": "fake"},
        "segments": [{"id": "speech", "words": [{"word": "Diagnostic word sentinel",
                      "start": 0, "end": 300, "probability": 0.1}],
                      "diagnostics": {"avg_logprob": -1.5}}],
        "warnings": [{"id": "upstream", "description": "UPSTREAM_DIAGNOSTIC",
                      "evidence_ids": ["speech"]}]})
    return data


class FakeClient:
    model = "fake"

    def __init__(self, *, fail=False, issues=True):
        self.fail = fail
        self.issues = issues
        self.prompts = []
        self.replies = []

    def complete(self, instructions, prompt, schema=None):
        assert instructions == notes.INSTRUCTIONS
        self.prompts.append(prompt)
        payload = json.loads(prompt)
        chapter = payload["chapter"]
        if self.fail:
            raise ResponseError("Temporary synthesis failure", code="server_error")
        reply = {"title": "Lecture", "language": "ru", "chapters": [
            {**chapter, "blocks": [{"id": "ch001_text", "kind": "text", "content": "Statement",
                                   "evidence_ids": ["speech"]}]}], "glossary": [],
            "issues": [{"id": "model_issue", "description": "New model ambiguity",
                        "evidence_ids": ["speech"], "status": "open", "resolution": ""}]
                      if self.issues else []}
        self.replies.append(copy.deepcopy(reply))
        return reply


def assert_model_context(context, data):
    assert set(context) == {"lecture_title", "glossary", "chapter", "segments", "writing_budget"}
    assert context["lecture_title"] == data["title"]
    assert context["segments"] == data["segments"]
    assert "UPSTREAM_DIAGNOSTIC" not in json.dumps(context)
    assert "Diagnostic word sentinel" not in json.dumps(context)


def test_automatic_input_and_local_review_exclude_extraction_diagnostics(tmp_path):
    data = prepare_job(tmp_path)
    original = (tmp_path / "evidence.json").read_bytes()
    diagnostic_bytes = (tmp_path / "extraction-diagnostics.json").read_bytes()
    client = FakeClient()
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    assert len(client.prompts) == 1
    assert_model_context(json.loads(client.prompts[0]), data)
    assert (tmp_path / "evidence.json").read_bytes() == original
    assert (tmp_path / "extraction-diagnostics.json").read_bytes() == diagnostic_bytes
    generated = read_json(tmp_path / "notes.json")
    assert [issue["id"] for issue in generated["issues"]] == ["model_issue"]
    assert generated["issues"][0]["description"] == "New model ambiguity"
    assert read_json(tmp_path / "review.json") == {"issues": generated["issues"], "ready": False}
    checkpoint = read_json(tmp_path / "checkpoints/notes/ch001.json")
    assert "UPSTREAM_DIAGNOSTIC" not in json.dumps(checkpoint)
    assert "Diagnostic word sentinel" not in json.dumps(checkpoint)


def test_extraction_warnings_do_not_block_generation_or_review_readiness(tmp_path, capsys):
    prepare_job(tmp_path)
    client = FakeClient(issues=False)
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    assert read_json(tmp_path / "notes.json")["issues"] == []
    assert read_json(tmp_path / "review.json") == {"issues": [], "ready": True}
    assert cli.main(["review", str(tmp_path)]) == 0
    assert "UPSTREAM_DIAGNOSTIC" not in capsys.readouterr().out
    before = {path.relative_to(tmp_path): path.read_bytes()
              for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="Unknown review issue"):
        notes.resolve_issue(tmp_path, "upstream", "Diagnostic warning ignored")
    assert {path.relative_to(tmp_path): path.read_bytes()
            for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("fallback", [False, True])
def test_manual_and_failed_fallback_handoffs_omit_diagnostics(tmp_path, monkeypatch, fallback):
    monkeypatch.setattr(notes.time, "sleep", lambda seconds: None)
    data = prepare_job(tmp_path)
    original = (tmp_path / "evidence.json").read_bytes()
    prior_review = {"issues": [{"id": "prior_note", "description": "Prior note ambiguity",
                               "evidence_ids": ["speech"], "status": "open", "resolution": ""}],
                    "ready": False}
    write_json(tmp_path / "review.json", prior_review)
    client = FakeClient(fail=True)
    folder = notes.generate_notes(tmp_path, tmp_path / "auth", manual=not fallback, client=client)
    assert folder == tmp_path / "handoff"
    assert_model_context(read_json(folder / "inputs/ch001/context.json"), data)
    assert "UPSTREAM_DIAGNOSTIC" not in (folder / "inputs/ch001/prompt.txt").read_text()
    assert (tmp_path / "evidence.json").read_bytes() == original
    assert read_json(tmp_path / "review.json") == prior_review
    assert not list(folder.rglob("*.png"))
    if fallback:
        assert len(client.prompts) == 4
        assert read_json(folder / "status.json")["status"] == "manual_required"
    else:
        assert client.prompts == []


@pytest.mark.parametrize("archive_state", ["changed", "missing", "malformed"])
def test_diagnostics_do_not_invalidate_cache_resolutions_review_or_render(
        tmp_path, monkeypatch, archive_state):
    data = prepare_job(tmp_path)
    client = FakeClient()
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    notes.resolve_issue(tmp_path, "model_issue", "The statement was checked")
    evidence_bytes = (tmp_path / "evidence.json").read_bytes()
    assert read_json(tmp_path / "review-resolutions.json")["model_issue"][
        "evidence_fingerprint"] == fingerprint(data)
    checkpoint = tmp_path / "checkpoints/notes/ch001.json"
    checkpoint_bytes = checkpoint.read_bytes()
    resolutions = (tmp_path / "review-resolutions.json").read_bytes()
    archive = tmp_path / "extraction-diagnostics.json"
    if archive_state == "changed":
        diagnostics = read_json(archive)
        diagnostics["models"]["asr"] = "changed-model"
        diagnostics["segments"][0]["words"][0]["probability"] = 0.99
        diagnostics["warnings"][0]["description"] = "Changed diagnostic warning"
        write_json(archive, diagnostics)
    elif archive_state == "missing":
        archive.unlink()
    else:
        archive.write_text("not valid JSON", encoding="utf-8")

    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    assert len(client.prompts) == 1
    assert checkpoint.read_bytes() == checkpoint_bytes
    reply = tmp_path / "reply.json"
    write_json(reply, client.replies[0])
    assert notes.generate_notes(tmp_path, tmp_path / "auth", imported=reply).exists()
    assert (tmp_path / "review-resolutions.json").read_bytes() == resolutions
    assert (tmp_path / "evidence.json").read_bytes() == evidence_bytes
    assert read_json(tmp_path / "notes.json")["issues"][0]["status"] == "resolved"
    assert read_json(tmp_path / "review.json")["ready"]
    manifest = read_json(tmp_path / "notes/project.json")
    assert manifest["evidence_fingerprint"] == fingerprint(data)
    assert not manifest["draft"]

    def compile_pdf(*args, cwd, **kwargs):
        (cwd / "main.pdf").write_bytes(b"test PDF")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(rendering.shutil, "which", lambda name: name)
    monkeypatch.setattr(rendering.subprocess, "run", compile_pdf)
    monkeypatch.setattr(rendering, "_qa", lambda pdf: 1)
    assert rendering.render(tmp_path) == tmp_path / "notes/main.pdf"
    assert read_json(tmp_path / "notes/render-report.json")["status"] == "rendered"
