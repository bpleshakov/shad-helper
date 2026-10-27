import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shad_helper.jobs import fingerprint, read_json, write_json
from shad_helper.notes import (
    _validate_evidence,
    chapter_context,
    default_outline,
    resolve_issue,
    review_ready,
)


def evidence():
    return {"schema_version": 3, "title": "Линейная алгебра", "media": {"duration": 1800},
            "segments": [{"id": "s1", "start": 10, "end": 12, "text": "Матрица"},
                         {"id": "s2", "start": 1200, "end": 1205, "text": "Следующий раздел"}]}


def issue():
    return {"id": "issue1", "description": "Неясный индекс",
            "evidence_ids": ["s1"], "status": "open", "resolution": ""}


def reply():
    return {"title": "Линейная алгебра", "language": "ru", "chapters": [
        {"id": "ch001", "title": "Матрицы", "start": 0, "end": 900,
         "blocks": [{"id": "ch001_b1", "kind": "math", "content": "A^T",
                     "evidence_ids": ["s1"]}]}], "glossary": [], "issues": []}


class NotesContextTests(unittest.TestCase):
    def test_context_includes_only_scoped_speech(self):
        context = chapter_context(evidence(), {"id": "ch001", "start": 0, "end": 900})
        self.assertEqual(set(context), {"chapter", "segments"})
        self.assertEqual([s["id"] for s in context["segments"]], ["s1"])

    def test_clip_outline_keeps_original_timestamps(self):
        data = evidence()
        data["media"].update(clip_start=500, clip_end=650)
        outline = default_outline(data)
        self.assertEqual(outline["chapters"][0]["start"], 500)
        self.assertEqual(outline["chapters"][0]["end"], 650)


    def test_resolved_issue_does_not_make_uncertain_content_ready(self):
        data = reply()
        data["chapters"][0]["blocks"][0]["kind"] = "uncertain"
        self.assertFalse(review_ready(data))

    def test_resolution_updates_readiness_only_with_explanation(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            write_json(job / "evidence.json", evidence())
            original = (job / "evidence.json").read_bytes()
            data = reply()
            data["issues"] = [issue()]
            write_json(job / "notes.json", data)
            with self.assertRaises(ValueError):
                resolve_issue(job, "issue1", "")
            resolve_issue(job, "issue1", "Проверен нижний индекс i; исправлена транскрипция")
            self.assertTrue(read_json(job / "review.json")["ready"])
            self.assertEqual(read_json(job / "notes.json")["issues"][0]["status"], "resolved")
            self.assertEqual((job / "evidence.json").read_bytes(), original)
            self.assertIn("evidence_fingerprint", read_json(job / "review-resolutions.json")["issue1"])

    def test_resolution_refreshes_review_without_overwriting_chapter_edits(self):
        from shad_helper.jobs import evidence_index
        from shad_helper.rendering import write_project
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            source = evidence()
            notes = reply()
            notes["issues"] = [issue()]
            write_json(job / "evidence.json", source)
            write_json(job / "notes.json", notes)
            write_project(job, dict(notes, evidence_index=evidence_index(source)))
            chapter = job / "notes/chapters/section-001.tex"
            edited = chapter.read_text() + "\nРучная правка.\n"
            chapter.write_text(edited)
            resolve_issue(job, "issue1", "Проверен индекс")
            self.assertEqual(chapter.read_text(), edited)
            self.assertFalse((job / "notes/review.tex").exists())
            self.assertEqual(read_json(job / "notes/project.json")["issues"][0]["status"], "resolved")
            self.assertNotIn("ЧЕРНОВИК", (job / "notes/main.tex").read_text())
            self.assertEqual(read_json(job / "notes/project.json")["evidence_fingerprint"],
                             fingerprint(read_json(job / "evidence.json")))


class NotesValidationTests(unittest.TestCase):
    def test_uncertain_block_always_opens_review_issue(self):
        data = reply()
        data["chapters"][0]["blocks"][0]["kind"] = "uncertain"
        notes = _validate_evidence(data, evidence())
        self.assertTrue(any(i["id"] == "review_ch001_b1" and i["status"] == "open"
                            for i in notes["issues"]))

    @unittest.skipUnless(importlib.util.find_spec("pydantic"), "requires installed pydantic")
    def test_chapter_cannot_cite_other_interval(self):
        from shad_helper.notes import _validate_chapter_reply
        data = reply()
        data["chapters"][0]["blocks"][0]["evidence_ids"] = ["s2"]
        with self.assertRaises(ValueError):
            _validate_chapter_reply(data, evidence(), {"id": "ch001", "start": 0, "end": 900})

    def test_whole_notes_cannot_cite_other_interval(self):
        data = reply()
        data["chapters"][0]["blocks"][0]["evidence_ids"] = ["s2"]
        with self.assertRaises(ValueError):
            _validate_evidence(data, evidence())

    def test_explicit_resolution_survives_import_but_changed_evidence_invalidates_it(self):
        data = reply()
        data["issues"] = [{"id": "note_issue", "description": "Неясный знак",
                           "evidence_ids": ["s1"], "status": "open", "resolution": ""}]
        source = evidence()
        signature = fingerprint(source)
        records = {"note_issue": {"description": "Неясный знак", "evidence_ids": ["s1"],
                                  "resolution": "Проверено: плюс", "evidence_fingerprint": signature}}
        self.assertEqual(_validate_evidence(data, source, records)["issues"][0]["status"], "resolved")
        source["segments"][0]["text"] = "Другой материал"
        self.assertEqual(_validate_evidence(data, source, records)["issues"][0]["status"], "open")

    def test_note_resolution_reopens_after_media_changes(self):
        source = evidence()
        data = reply()
        data["issues"] = [dict(issue(), status="resolved", resolution="Checked")]
        signature = fingerprint(source)
        records = {"issue1": dict(data["issues"][0], evidence_fingerprint=signature)}
        source["media"]["sha256"] = "changed-media"
        issues = _validate_evidence(data, source, records)["issues"]
        self.assertEqual(issues[0]["status"], "open")

    @unittest.skipUnless(importlib.util.find_spec("pydantic"), "requires installed pydantic")
    def test_file_identifier_cannot_escape_renderer_contract(self):
        from shad_helper.notes import validate_notes
        for identifier in ("_invalid", "x" * 81):
            data = reply()
            data["chapters"][0]["id"] = identifier
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                validate_notes(data, evidence())


    def test_hallucinated_source_id_is_rejected(self):
        data = reply()
        data["chapters"][0]["blocks"][0]["evidence_ids"] = ["invented"]
        with self.assertRaises(ValueError):
            _validate_evidence(data, evidence())

    def test_model_cannot_resolve_its_own_issue(self):
        data = reply()
        data["issues"] = [dict(issue(), status="resolved", resolution="guess")]
        notes = _validate_evidence(data, evidence())
        self.assertEqual(notes["issues"][0]["status"], "open")


    @unittest.skipUnless(importlib.util.find_spec("pydantic"), "requires installed pydantic")
    def test_manual_handoff_and_import_have_same_notes_contract(self):
        from shad_helper.notes import export_handoff, generate_notes
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary).resolve()
            data = evidence()
            data["media"]["duration"] = 900
            write_json(job / "evidence.json", data)
            write_json(job / "outline.json", default_outline(evidence()))
            response = reply()
            response["chapters"][0]["blocks"][0]["evidence_ids"] = ["s1"]
            response_file = job / "response.json"
            write_json(response_file, response)
            with patch.dict(os.environ, {"SHAD_HELPER_OPENAI_PROXY": ""}):
                folder = generate_notes(job, job / "auth", manual=True)
                self.assertEqual(folder, job / "handoff")
                self.assertTrue((folder / "inputs/ch001/prompt.txt").exists())
                self.assertFalse(list(folder.rglob("*.pdf")))
                self.assertEqual(read_json(folder / "run.json")["mode"], "manual")
                write_json(folder / "status.json", {"status": "manual_required"})
                with patch("shad_helper.notes.export_handoff", wraps=export_handoff) as export:
                    path = generate_notes(job, job / "auth", imported=response_file)
                export.assert_called_once_with(job, data, read_json(job / "outline.json"), contexts={})
                context = read_json(folder / "inputs/ch001/context.json")
                self.assertEqual(context["chapter"]["title"], "Матрицы")
                self.assertEqual(context["segments"], [data["segments"][0]])
                self.assertFalse((folder / "runs").exists())
                self.assertEqual(read_json(folder / "run.json")["mode"], "import")
                self.assertTrue(path.exists())
                self.assertTrue((job / "notes.json").exists())
                self.assertTrue(read_json(job / "review.json")["ready"])
                self.assertFalse((folder / "status.json").exists())

    @unittest.skipUnless(importlib.util.find_spec("pydantic") and importlib.util.find_spec("httpx"),
                         "requires installed pydantic and httpx")
    def test_missing_proxy_exports_handoff_without_openai_requests(self):
        from shad_helper.notes import generate_notes
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary).resolve()
            data = evidence()
            data["media"]["duration"] = 900
            write_json(job / "evidence.json", data)
            with patch.dict(os.environ, {"SHAD_HELPER_OPENAI_PROXY": ""}), \
                    patch("shad_helper.network.httpx.Client") as client:
                folder = generate_notes(job, job / "auth")
            client.assert_not_called()
            status = read_json(folder / "status.json")
            self.assertEqual(status["status"], "manual_required")
            self.assertIn("--openai-proxy", status["reason"])
            self.assertTrue((folder / "inputs/ch001/context.json").exists())
            self.assertFalse(list(folder.rglob("*.pdf")))


if __name__ == "__main__":
    unittest.main()
