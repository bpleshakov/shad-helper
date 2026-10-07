import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shad_helper import rendering
from shad_helper.jobs import AUDIO_CONFIG, fingerprint, write_json


def sample_notes():
    return {
        "title": "Лекция: вероятность & статистика 100%",
        "language": "ru",
        "chapters": [{
            "id": "chapter-1", "title": "Случайные величины", "start": 61, "end": 125,
            "blocks": [
                {"id": "text-1", "kind": "text", "content": r"Для \(X \in \mathbb{R}\) оценим p_value & 50%.", "evidence_ids": ["ev-1"]},
                {"id": "math-1", "kind": "math", "content": r"\mathbb{E}[X] = \sum_{i=1}^{n} x_i p_i", "evidence_ids": []},
                {"id": "code-1", "kind": "code", "content": 'print("Привет")\n# \\input{secret}\n', "language": "python", "evidence_ids": []},
                {"id": "diagram-1", "kind": "diagram", "content": r"\begin{tikzpicture}\node[draw] (a) {Начало};\node[draw,right=of a] (b) {Конец};\draw[->] (a)--(b);\end{tikzpicture}", "caption": "Схема", "evidence_ids": []},
                {"id": "clarification-1", "kind": "clarification", "content": r"Условие: \(p > 0\).", "evidence_ids": []},
                {"id": "uncertain-1", "kind": "uncertain", "content": "Проверьте индекс.", "evidence_ids": ["ev-1"]},
            ],
        }],
        "glossary": [{"term": "Ожидание", "definition": r"Среднее значение \(X\)."}],
        "issues": [{"id": "issue-1", "description": "Неясный индекс", "status": "open", "resolution": "", "evidence_ids": ["ev-1"]}],
        "evidence_index": {"ev-1": "Аудио 00:01:05 - 00:01:12"},
    }


class RenderingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.job = Path(self.directory.name)
        self.notes = sample_notes()
        write_json(self.job / "extraction-config.json", AUDIO_CONFIG)

    def tearDown(self):
        self.directory.cleanup()

    def project(self):
        path = rendering.write_project(self.job, self.notes)
        root = path.parent
        return root, json.loads((root / "project.json").read_text())

    def test_escape_all_tex_special_characters(self):
        self.assertEqual(rendering.escape_text(r"\{}$&#%_~^"), r"\textbackslash{}\{\}\$\&\#\%\_\textasciitilde{}\textasciicircum{}")

    def test_cyrillic_math_code_and_draft_source(self):
        root, manifest = self.project()
        rendering._validate_project(root, manifest)
        main = (root / "main.tex").read_text()
        chapter = (root / "chapters/chapter-1.tex").read_text()
        self.assertIn("CMU Serif", main)
        self.assertIn("CMU Typewriter Text", main)
        self.assertIn("Latin Modern Math", main)
        self.assertIn("ЧЕРНОВИК", main)
        self.assertIn(r"\(X \in \mathbb{R}\)", chapter)
        self.assertIn(r"p\_value \& 50\%", chapter)
        self.assertIn("00:01:01 - 00:02:05", chapter)
        self.assertIn("Требует проверки", chapter)
        self.assertIn("Пояснение", chapter)
        self.assertEqual((root / "code/code-1.txt").read_text(), self.notes["chapters"][0]["blocks"][2]["content"])
        self.assertEqual(len(manifest["review"]), 2)
        self.assertIn("Аудио 00:01:05", chapter)

    def test_invalid_inline_math_is_blocked(self):
        for content in (r"Не закрыто \(x", r"Опасно \(\input{secret}\)"):
            with self.subTest(content=content):
                self.notes["chapters"][0]["blocks"][0]["content"] = content
                with self.assertRaises(rendering.RenderError):
                    self.project()

    def test_inner_math_environments_are_wrapped_but_display_environments_are_not(self):
        inner = {
            "aligned": r"a &= b \\ c &= d",
            "bmatrix": r"1 & 2 \\ 3 & 4",
            "pmatrix": r"1 & 2 \\ 3 & 4",
            "cases": r"x & x > 0 \\ 0 & x \leq 0",
            "array": r"{cc}1 & 2 \\ 3 & 4",
        }
        contents = [(r"\begin{" + environment + "}" + body + r"\end{" + environment + "}", True)
                    for environment, body in inner.items()]
        contents.extend((r"\begin{" + environment + "}a = b" + r"\end{" + environment + "}", False)
                        for environment in ("equation", "equation*", "align", "align*", "gather", "gather*", "multline", "multline*"))
        contents.append((r"\[a = b\]", False))
        for content, wrapped in contents:
            with self.subTest(content=content):
                self.notes["chapters"][0]["blocks"][1]["content"] = content
                root, manifest = self.project()
                rendering._validate_project(root, manifest)
                chapter = (root / "chapters/chapter-1.tex").read_text()
                display = "\\[\n" + content + "\n\\]"
                if wrapped:
                    self.assertIn(display, chapter)
                else:
                    self.assertIn(content, chapter)
                    self.assertNotIn(display, chapter)

    def test_unsafe_diagram_commands_are_blocked(self):
        for content in (r"\directlua{os.execute('id')}", r"\input{../evidence/frame.png}", r"\write18{ls}", r"^^5cinput{file}", r"\addplot table {data.csv};"):
            with self.subTest(content=content):
                self.notes["chapters"][0]["blocks"][3]["content"] = content
                with self.assertRaises(rendering.RenderError):
                    self.project()


    def test_manual_tex_edits_are_preserved_and_report_is_written(self):
        root, _ = self.project()
        source = root / "chapters/chapter-1.tex"
        corrected = source.read_text() + "\nИсправленная формулировка.\n"
        source.write_text(corrected)

        def compiler(arguments, *, cwd, env, stdout, **kwargs):
            self.assertIn("-norc", arguments)
            self.assertIn("-latexoption=-no-shell-escape", arguments)
            self.assertEqual(env["openin_any"], "p")
            (cwd / "main.pdf").write_bytes(b"mocked compiler output")
            stdout.write("Overfull \\hbox (2pt too wide)\nMissing character: U+1234\n")
            return subprocess.CompletedProcess(arguments, 0)

        with patch.object(rendering.shutil, "which", return_value="/bin/test"), patch.object(rendering.subprocess, "run", side_effect=compiler), patch.object(rendering, "_qa", return_value=3):
            self.assertEqual(rendering.render(self.job), root / "main.pdf")
        self.assertEqual(source.read_text(), corrected)
        report = json.loads((root / "render-report.json").read_text())
        self.assertEqual(report["pages"], 3)
        self.assertTrue(report["draft"])
        self.assertEqual(len(report["overfull"]), 1)
        self.assertEqual(len(report["missing_glyphs"]), 1)

    def test_manual_evidence_reference_is_rejected(self):
        root, manifest = self.project()
        source = root / "chapters/chapter-1.tex"
        for malicious in (r"\includegraphics{../evidence/frame.png}", r"\input{../sources/slides.tex}", r"\usepackage{shellesc}", r"\renewcommand\input{bad}", r"\setmainfont[Path=../]{CMU Serif}"):
            original = source.read_text()
            source.write_text(original + malicious)
            with self.subTest(malicious=malicious), self.assertRaises(rendering.RenderError):
                rendering._validate_project(root, manifest)
            source.write_text(original)

    def test_symlink_asset_escape_is_rejected(self):
        root, manifest = self.project()
        source = root / "chapters/chapter-1.tex"
        source.unlink()
        outside = self.job / "outside.tex"
        outside.write_text("external")
        source.symlink_to(outside)
        with self.assertRaises(rendering.RenderError):
            rendering._validate_project(root, manifest)


    def test_resolved_notes_have_no_draft_footer(self):
        self.notes["issues"][0]["status"] = "resolved"
        self.notes["issues"][0]["resolution"] = "Индекс исправлен."
        self.notes["chapters"][0]["blocks"][5]["kind"] = "text"
        root, manifest = self.project()
        rendering._validate_project(root, manifest)
        self.assertFalse(manifest["draft"])
        self.assertEqual(manifest["review"], [])
        self.assertNotIn("ЧЕРНОВИК", (root / "main.tex").read_text())

    def test_evidence_fingerprint_ignores_formatting_and_rejects_changed_content(self):
        evidence_path = self.job / "evidence.json"
        evidence = {"schema_version": 3, "title": "Лекция", "media": {"duration": 1800},
                    "segments": [{"id": "ev-1", "start": 61, "end": 125,
                                  "text": "Исходная формулировка"}]}
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False))
        root, manifest = self.project()
        self.assertEqual(manifest["evidence_fingerprint"], fingerprint(evidence))
        evidence_path.write_text(json.dumps(dict(reversed(list(evidence.items()))), indent=4))
        with patch.object(rendering.shutil, "which", return_value=None):
            with self.assertRaisesRegex(rendering.RenderError, "requires latexmk"):
                rendering.render(self.job)
        evidence["segments"][0]["text"] = "Изменённая формулировка"
        evidence_path.write_text(json.dumps(evidence))
        with patch.object(rendering.subprocess, "run") as compiler:
            with self.assertRaisesRegex(rendering.RenderError, "regenerate notes or import corrected notes"):
                rendering.render(self.job)
            compiler.assert_not_called()
        report = json.loads((root / "render-report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("Evidence changed", report["error"])

    def test_evidence_introduction_and_deletion_invalidate_project(self):
        _, manifest = self.project()
        self.assertIsNone(manifest["evidence_fingerprint"])
        evidence_path = self.job / "evidence.json"
        evidence_path.write_text('{"schema_version": 3, "title": "Лекция", "media": {"duration": 1800}, "segments": []}')
        with self.assertRaisesRegex(rendering.RenderError, "Evidence changed"):
            rendering.render(self.job)
        self.project()
        evidence_path.unlink()
        with self.assertRaisesRegex(rendering.RenderError, "Evidence changed"):
            rendering.render(self.job)

    def test_review_refresh_preserves_edits_and_uncertain_blocks_keep_draft(self):
        evidence_path = self.job / "evidence.json"
        evidence = {"schema_version": 3, "title": "Лекция", "media": {"duration": 1800},
                    "segments": []}
        evidence_path.write_text(json.dumps(evidence))
        root, _ = self.project()
        edited_paths = [root / "chapters/chapter-1.tex", root / "figures/diagram-diagram-1.tex", root / "code/code-1.txt"]
        for path in edited_paths:
            path.write_text(path.read_text() + "\nРучная правка.\n")
        edited = {path: path.read_bytes() for path in edited_paths}
        main_path = root / "main.tex"
        main_path.write_text(main_path.read_text().replace("Среднее значение", "Авторское определение"))
        original_main = main_path.read_text()
        self.notes["title"] = "Название не должно перезаписать TEX"
        self.notes["glossary"][0]["definition"] = "Не заменять авторское определение"
        self.notes["issues"][0].update(status="resolved", resolution=r"Уточнено: \(i = 1\).")
        self.assertEqual(rendering.refresh_review(self.job, self.notes), main_path)
        manifest = json.loads((root / "project.json").read_text())
        self.assertEqual(manifest["evidence_fingerprint"], fingerprint(evidence))
        self.assertEqual(manifest["issues"], self.notes["issues"])
        self.assertTrue(manifest["draft"])
        self.assertEqual([item["id"] for item in manifest["review"]], ["uncertain-1"])
        appendix = (root / "review.tex").read_text()
        self.assertIn("Решено", appendix)
        self.assertIn(r"Уточнено: \(i = 1\).", appendix)
        self.assertEqual(main_path.read_text(), original_main)
        for path, content in edited.items():
            self.assertEqual(path.read_bytes(), content)
        self.notes["chapters"][0]["blocks"][5]["kind"] = "text"
        rendering.refresh_review(self.job, self.notes)
        manifest = json.loads((root / "project.json").read_text())
        self.assertFalse(manifest["draft"])
        self.assertEqual(manifest["review"], [])
        self.assertNotIn("ЧЕРНОВИК", main_path.read_text())
        self.assertEqual(main_path.read_text(), original_main.replace(rendering._draft_footer([{"id": "uncertain-1"}]), ""))
        for path, content in edited.items():
            self.assertEqual(path.read_bytes(), content)
        rendering._validate_project(root, manifest)
        with patch.object(rendering.shutil, "which", return_value=None):
            with self.assertRaisesRegex(rendering.RenderError, "requires latexmk"):
                rendering.render(self.job)

    def test_review_refresh_requires_markers_before_mutating_any_file(self):
        root, _ = self.project()
        main = root / "main.tex"
        main.write_text(main.read_text().replace(rendering._DRAFT_START, "% custom footer"))
        before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(rendering.RenderError, "regenerate or import notes"):
            rendering.refresh_review(self.job, self.notes)
        self.assertEqual({path: path.read_bytes() for path in root.rglob("*") if path.is_file()}, before)

    def test_compiler_failure_retains_logs_and_sources(self):
        root, _ = self.project()

        def compiler(arguments, *, stdout, **kwargs):
            stdout.write("fontspec error: font cannot be found\n")
            return subprocess.CompletedProcess(arguments, 1)

        with patch.object(rendering.shutil, "which", return_value="/bin/test"), patch.object(rendering.subprocess, "run", side_effect=compiler):
            with self.assertRaises(rendering.RenderError):
                rendering.render(self.job)
        self.assertTrue((root / "main.tex").exists())
        self.assertIn("fontspec error", (root / "compiler-output.log").read_text())
        report = json.loads((root / "render-report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["missing_fonts"])

    def test_font_database_initialization_is_not_a_missing_font(self):
        root, _ = self.project()
        notice = "luaotfload | db : Font names database not found, generating new one."
        error = "Font missing: fontspec error - Requested Font cannot be found"

        def compiler(arguments, *, stdout, **kwargs):
            stdout.write(notice + "\n" + error + "\n")
            (root / "main.pdf").write_bytes(b"%PDF-1.7\n")
            return subprocess.CompletedProcess(arguments, 0)

        with patch.object(rendering.shutil, "which", return_value="/bin/test"), \
                patch.object(rendering.subprocess, "run", side_effect=compiler), \
                patch.object(rendering, "_qa", return_value=1):
            rendering.render(self.job)
        report = json.loads((root / "render-report.json").read_text())
        self.assertEqual(report["missing_fonts"], [error])

    @unittest.skipUnless(shutil.which("latexmk") and shutil.which("lualatex") and importlib.util.find_spec("pymupdf"), "LuaLaTeX, latexmk, and PyMuPDF are required")
    def test_actual_cyrillic_math_diagram_code_pdf_and_qa(self):
        self.project()
        pdf = rendering.render(self.job)
        self.assertGreater(pdf.stat().st_size, 1000)
        self.assertTrue((pdf.parent / "qa/page-001.png").is_file())
        report = json.loads((pdf.parent / "render-report.json").read_text())
        self.assertEqual(report["missing_fonts"], [])
        self.assertEqual(report["missing_glyphs"], [])


if __name__ == "__main__":
    unittest.main()
