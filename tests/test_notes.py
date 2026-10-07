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
    return {"title": "Линейная алгебра", "video": {"duration": 1800},
            "segments": [{"id": "s1", "start": 10, "end": 12, "text": "Матрица"},
                         {"id": "s2", "start": 1200, "end": 1205, "text": "Следующий раздел"}],
            "visuals": [{"id": "v1", "timestamp": 11, "image": "evidence/frame.png",
                         "regions": [{"id": "r1", "kind": "formula", "text": "A^T"}]},
                        {"id": "pv1", "timestamp": None, "source_id": "p1",
                         "image": "evidence/source.png", "regions": []}],
            "sources": [{"id": "p1", "kind": "pdf", "source": "slides.pdf",
                         "location": "page 1", "matched_visual_ids": ["v1"], "text": "A^T"},
                        {"id": "p2", "kind": "pdf", "source": "other.pdf",
                         "location": "page 2", "matched_visual_ids": [], "text": "Unrelated"}],
            "issues": [{"id": "issue1", "description": "Неясный индекс",
                        "evidence_ids": ["r1"], "status": "open", "resolution": ""}]}


def reply():
    return {"title": "Линейная алгебра", "language": "ru", "chapters": [
        {"id": "ch001", "title": "Матрицы", "start": 0, "end": 900,
         "blocks": [{"id": "ch001_b1", "kind": "math", "content": "A^T",
                     "evidence_ids": ["r1"]}]}], "glossary": [], "issues": [], "illustrations": []}


class NotesContextTests(unittest.TestCase):
    def test_context_includes_only_matched_sources(self):
        context = chapter_context(evidence(), {"id": "ch001", "start": 0, "end": 900})
        self.assertEqual([s["id"] for s in context["segments"]], ["s1"])
        self.assertEqual([s["id"] for s in context["sources"]], ["p1"])
        self.assertEqual([v["id"] for v in context["visuals"]], ["v1", "pv1"])
        self.assertEqual([i["id"] for i in context["issues"]], ["issue1"])

    def test_clip_outline_keeps_original_timestamps(self):
        data = evidence()
        data["video"].update(clip_start=500, clip_end=650)
        outline = default_outline(data)
        self.assertEqual(outline["chapters"][0]["start"], 500)
        self.assertEqual(outline["chapters"][0]["end"], 650)

    def test_persistent_slide_and_matched_source_carry_into_next_chapter(self):
        context = chapter_context(evidence(), {"id": "ch002", "start": 900, "end": 1800})
        self.assertEqual([v["id"] for v in context["visuals"]], ["v1", "pv1"])
        self.assertEqual([s["id"] for s in context["sources"]], ["p1"])

    def test_newer_slide_replaces_persistent_context(self):
        data = evidence()
        data["visuals"].append({"id": "v2", "timestamp": 800, "regions": []})
        context = chapter_context(data, {"id": "ch002", "start": 900, "end": 1800})
        self.assertEqual([v["id"] for v in context["visuals"]], ["v2"])
        self.assertEqual(context["sources"], [])

    def test_resolved_issue_does_not_make_uncertain_content_ready(self):
        data = reply()
        data["chapters"][0]["blocks"][0]["kind"] = "uncertain"
        self.assertFalse(review_ready(data))

    def test_resolution_updates_readiness_only_with_explanation(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            write_json(job / "evidence.json", evidence())
            with self.assertRaises(ValueError):
                resolve_issue(job, "issue1", "")
            resolve_issue(job, "issue1", "Проверен нижний индекс i; исправлен текст OCR")
            self.assertTrue(read_json(job / "review.json")["ready"])
            self.assertEqual(read_json(job / "evidence.json")["issues"][0]["status"], "resolved")
            self.assertIn("evidence_fingerprint", read_json(job / "review-resolutions.json")["issue1"])

    def test_resolution_refreshes_review_without_overwriting_chapter_edits(self):
        from shad_helper.jobs import evidence_index
        from shad_helper.rendering import write_project
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            source = evidence()
            notes = reply()
            notes["issues"] = source["issues"]
            write_json(job / "evidence.json", source)
            write_json(job / "notes.json", notes)
            write_project(job, dict(notes, evidence_index=evidence_index(source)))
            chapter = job / "notes/chapters/ch001.tex"
            edited = chapter.read_text() + "\nРучная правка.\n"
            chapter.write_text(edited)
            resolve_issue(job, "issue1", "Проверен индекс")
            self.assertEqual(chapter.read_text(), edited)
            self.assertIn("Решено", (job / "notes/review.tex").read_text())
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
                           "evidence_ids": ["r1"], "status": "open", "resolution": ""}]
        source = evidence()
        signature = fingerprint({key: value for key, value in source.items() if key != "issues"})
        records = {"note_issue": {"description": "Неясный знак", "evidence_ids": ["r1"],
                                  "resolution": "Проверено: плюс", "evidence_fingerprint": signature}}
        self.assertEqual(_validate_evidence(data, source, records)["issues"][0]["status"], "resolved")
        source["segments"][0]["text"] = "Другой материал"
        self.assertEqual(_validate_evidence(data, source, records)["issues"][0]["status"], "open")

    def test_upstream_resolution_reopens_after_material_changes(self):
        source = evidence()
        signature = fingerprint({key: value for key, value in source.items() if key != "issues"})
        source["issues"][0].update(status="resolved", resolution="Checked")
        records = {"issue1": dict(source["issues"][0], evidence_fingerprint=signature)}
        source["video"]["sha256"] = "changed-video"
        issues = _validate_evidence(reply(), source, records)["issues"]
        self.assertEqual(issues[0]["status"], "open")

    @unittest.skipUnless(importlib.util.find_spec("pydantic"), "requires installed pydantic")
    def test_file_identifier_cannot_escape_renderer_contract(self):
        from shad_helper.notes import validate_notes
        for identifier in ("_invalid", "x" * 81):
            data = reply()
            data["chapters"][0]["id"] = identifier
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                validate_notes(data, evidence())

    def test_unmatched_source_cannot_become_lecture_content(self):
        data = reply()
        data["chapters"][0]["blocks"][0]["evidence_ids"] = ["p2"]
        with self.assertRaises(ValueError):
            _validate_evidence(data, evidence())

    def test_hallucinated_source_id_is_rejected(self):
        data = reply()
        data["chapters"][0]["blocks"][0]["evidence_ids"] = ["invented"]
        with self.assertRaises(ValueError):
            _validate_evidence(data, evidence())

    def test_model_cannot_resolve_upstream_issue(self):
        data = reply()
        data["issues"] = [dict(evidence()["issues"][0], status="resolved", resolution="guess")]
        notes = _validate_evidence(data, evidence())
        self.assertEqual(notes["issues"][0]["status"], "open")

    def test_illustration_cannot_reference_evidence_image(self):
        data = reply()
        data["illustrations"] = [{"id": "fig1", "chapter_id": "ch001", "prompt": "Draw",
                                  "caption": "Example", "path": "evidence/frame.png"}]
        with self.assertRaises(ValueError):
            _validate_evidence(data, evidence())

    @unittest.skipUnless(importlib.util.find_spec("pydantic"), "requires installed pydantic")
    def test_manual_handoff_and_import_have_same_notes_contract(self):
        from shad_helper.notes import generate_notes
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary).resolve()
            data = evidence()
            data["video"]["duration"] = 900
            data["visuals"], data["sources"] = [], []
            data["issues"] = []
            write_json(job / "evidence.json", data)
            write_json(job / "outline.json", default_outline(evidence()))
            response = reply()
            response["chapters"][0]["blocks"][0]["evidence_ids"] = ["s1"]
            response_file = job / "response.json"
            write_json(response_file, response)
            with patch.dict(os.environ, {"SHAD_HELPER_OPENAI_PROXY": ""}):
                folder = generate_notes(job, job / "auth", manual=True)
                self.assertEqual(folder, job / "handoff")
                self.assertTrue((folder / "ch001/prompt.txt").exists())
                self.assertFalse(list(folder.rglob("*.pdf")))
                write_json(folder / "status.json", {"status": "manual_required"})
                with patch("shad_helper.notes.export_handoff") as export:
                    path = generate_notes(job, job / "auth", imported=response_file)
                export.assert_not_called()
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
            data["video"]["duration"] = 900
            data["visuals"], data["sources"], data["issues"] = [], [], []
            write_json(job / "evidence.json", data)
            with patch.dict(os.environ, {"SHAD_HELPER_OPENAI_PROXY": ""}), \
                    patch("shad_helper.network.httpx.Client") as client:
                folder = generate_notes(job, job / "auth")
            client.assert_not_called()
            status = read_json(folder / "status.json")
            self.assertEqual(status["status"], "manual_required")
            self.assertIn("--openai-proxy", status["reason"])
            self.assertTrue((folder / "ch001/context.json").exists())
            self.assertFalse(list(folder.rglob("*.pdf")))


if __name__ == "__main__":
    unittest.main()
