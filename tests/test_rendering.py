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

    def evidence(self, segments, visuals=()):
        data = {"schema_version": 4 if visuals else 3, "title": "Лекция",
                "media": {"duration": 10800},
                "segments": [{"id": identifier, "start": start, "end": start + 1,
                              "text": "Формулировка"} for identifier, start in segments]}
        if visuals:
            data["visuals"] = [{"id": identifier, "timestamp": timestamp, "status": "complete",
                                "kind": "whiteboard", "text": "Формула", "formulas": [],
                                "diagrams": [], "unreadable": []}
                               for identifier, timestamp in visuals]
        write_json(self.job / "evidence.json", data)

    def test_escape_all_tex_special_characters(self):
        self.assertEqual(rendering.escape_text(r"\{}$&#%_~^"), r"\textbackslash{}\{\}\$\&\#\%\_\textasciitilde{}\textasciicircum{}")

    def test_cyrillic_math_code_and_draft_source(self):
        root, manifest = self.project()
        rendering._validate_project(root, manifest)
        main = (root / "main.tex").read_text()
        chapter = (root / "chapters/section-001.tex").read_text()
        self.assertIn("CMU Serif", main)
        self.assertIn("CMU Typewriter Text", main)
        self.assertIn("Latin Modern Math", main)
        self.assertIn("ЧЕРНОВИК", main)
        self.assertIn(r"\(X \in \mathbb{R}\)", chapter)
        self.assertIn(r"p\_value \& 50\%", chapter)
        self.assertNotIn("Лекция:", chapter)
        self.assertNotIn("Требует проверки", chapter)
        self.assertIn(r"Условие: \(p > 0\).", chapter)
        self.assertNotIn("Пояснение.", chapter)
        self.assertNotIn(r"\begin{quote}", chapter)
        self.assertIn(r"\textcolor{blue!60!black}{Проверьте индекс.}\footnote{00:01:01}", chapter)
        self.assertEqual((root / "code/code-1.txt").read_text(), self.notes["chapters"][0]["blocks"][2]["content"])
        self.assertEqual(len(manifest["review"]), 2)

    def test_evidence_and_review_metadata_do_not_add_pdf_footnotes_or_appendix(self):
        self.notes["chapters"][0]["blocks"][5]["kind"] = "text"
        for block in self.notes["chapters"][0]["blocks"]:
            block["evidence_ids"] = ["ev-1"]
        root, manifest = self.project()
        rendering._validate_project(root, manifest)
        for relative in manifest["tex_files"]:
            source = (root / relative).read_text()
            self.assertNotIn(r"\footnote", source)
            self.assertNotIn("Источники:", source)
            self.assertNotIn("Аудио 00:01:05", source)
            self.assertNotIn("Вопросы и замечания", source)
        self.assertFalse((root / "review.tex").exists())
        self.assertNotIn("review.tex", manifest["tex_files"])
        self.assertNotIn(r"\input{review.tex}", (root / "main.tex").read_text())
        self.assertEqual(manifest["issues"], self.notes["issues"])
        self.assertTrue(manifest["draft"])
        self.assertIn("Ожидание", (root / "main.tex").read_text())
        self.assertEqual([block["evidence_ids"] for block in self.notes["chapters"][0]["blocks"]],
                         [["ev-1"]] * 6)

    def test_only_uncertain_fragments_are_colored_with_one_deduplicated_footnote(self):
        self.notes["issues"] = []
        self.notes["evidence_index"] = {"ev-1": "Аудио 99:99:99"}
        content = r"Матрица симметрична, \(X_1 \in \mathbb{R}\), индекс \(j\)."
        self.notes["chapters"][0]["blocks"] = [{
            "id": "selected", "kind": "text", "content": content,
            "evidence_ids": ["ev-1", "ev-2", "frame-1", "ordinary"],
            "uncertain_fragments": [
                {"text": r"индекс \(j\)", "evidence_ids": ["ev-2", "frame-1"]},
                {"text": r"\(X_1 \in \mathbb{R}\)", "evidence_ids": ["frame-1", "ev-1"]},
            ],
        }]
        self.evidence([("ev-1", 3661.2), ("ev-2", 3661.9), ("ordinary", 7200)],
                      [("frame-1", 3723.7)])
        root, manifest = self.project()
        rendering._validate_project(root, manifest)
        chapter = (root / "chapters/section-001.tex").read_text()
        self.assertIn(r"Матрица симметрична, \textcolor{blue!60!black}{\(X_1 \in \mathbb{R}\)}, "
                      r"\textcolor{blue!60!black}{индекс \(j\)}.\footnote{01:01:01, 01:02:03}", chapter)
        self.assertEqual(chapter.count(r"\footnote"), 1)
        self.assertNotIn("02:00:00", chapter)
        self.assertNotIn("99:99:99", chapter)
        self.assertEqual([flag["id"] for flag in manifest["review"]], ["selected"])
        self.assertTrue(manifest["draft"])

    def test_many_uncertain_references_use_a_short_range_per_paragraph(self):
        self.notes["chapters"][0]["blocks"] = [{
            "id": "selected", "kind": "clarification",
            "content": "Первое утверждение.\n\nВторое утверждение.\n\nНадёжный вывод.",
            "evidence_ids": [f"ev-{index}" for index in range(5)],
            "uncertain_fragments": [
                {"text": "Первое", "evidence_ids": ["ev-3", "ev-1", "ev-2", "ev-0", "ev-0"]},
                {"text": "Второе", "evidence_ids": ["ev-4"]},
            ],
        }]
        self.evidence([(f"ev-{index}", 60 * index) for index in range(5)])
        root, manifest = self.project()
        rendering._validate_project(root, manifest)
        chapter = (root / "chapters/section-001.tex").read_text()
        self.assertIn(r"\textcolor{blue!60!black}{Первое} утверждение.\footnote{00:00:00–00:03:00}", chapter)
        self.assertIn(r"\textcolor{blue!60!black}{Второе} утверждение.\footnote{00:04:00}", chapter)
        self.assertEqual(chapter.count(r"\footnote"), 2)
        self.assertIn("\n\nНадёжный вывод.", chapter)

    def test_uncertain_fragments_require_real_evidence_timestamps(self):
        self.notes["chapters"][0]["blocks"] = [{
            "id": "selected", "kind": "text", "content": "Индекс неясен.",
            "evidence_ids": ["missing"],
            "uncertain_fragments": [{"text": "Индекс", "evidence_ids": ["missing"]}],
        }]
        with self.assertRaisesRegex(rendering.RenderError, "without timestamps: missing"):
            self.project()

    def test_invalid_or_unsafe_math_in_highlighted_fragments_is_rejected(self):
        for content, fragment in ((r"Индекс \(x_i\).", "x_i"),
                                  (r"Индекс \(\input{secret}\).", r"\(\input{secret}\)"),
                                  (r"Индекс \(x_i.", r"\(x_i")):
            self.notes["chapters"][0]["blocks"] = [{
                "id": "selected", "kind": "text", "content": content,
                "evidence_ids": ["ev-1"],
                "uncertain_fragments": [{"text": fragment, "evidence_ids": ["ev-1"]}],
            }]
            self.evidence([("ev-1", 61)])
            with self.subTest(content=content), self.assertRaises(rendering.RenderError):
                self.project()

    def test_empty_chapters_and_standalone_front_matter_are_omitted(self):
        self.notes["chapters"].insert(0, {
            "id": "empty", "title": "Организационные вопросы", "start": 0, "end": 61, "blocks": [],
        })
        root, manifest = self.project()
        rendering._validate_project(root, manifest)
        main = (root / "main.tex").read_text()
        self.assertNotIn("chapters/empty.tex", manifest["tex_files"])
        self.assertNotIn("Организационные вопросы", main)
        self.assertNotIn(r"\tableofcontents", main)
        self.assertNotIn(r"\newpage", main)
        self.assertNotIn(r"\maketitle", main)
        self.assertIn(r"\input{chapters/section-001.tex}", main)

    def test_sections_follow_topics_across_processing_chapters(self):
        self.notes["glossary"] = []
        self.notes["chapters"][0]["blocks"] = [
            {"id": "knn-1", "kind": "text", "topic": "  Метод   ближайших соседей  ",
             "content": "Первое определение."},
            {"id": "regression-1", "kind": "text", "topic": "Линейная регрессия",
             "content": "Модель регрессии."},
        ]
        self.notes["chapters"].append({
            "id": "chapter-2", "title": "Второй интервал", "start": 125, "end": 190,
            "blocks": [
                {"id": "knn-2", "kind": "text", "topic": "метод ближайших\nсоседей",
                 "content": "Продолжение первой темы."},
                {"id": "regression-2", "kind": "text", "topic": "Линейная регрессия",
                 "content": "Продолжение второй темы."},
            ],
        })
        root, manifest = self.project()
        rendering._validate_project(root, manifest)
        self.assertEqual(manifest["tex_files"], ["main.tex", "chapters/section-001.tex",
                                               "chapters/section-002.tex"])
        first = (root / "chapters/section-001.tex").read_text()
        second = (root / "chapters/section-002.tex").read_text()
        self.assertEqual(first.count(r"\section"), 1)
        self.assertIn(r"\section{Метод ближайших соседей}", first)
        self.assertLess(first.index("Первое определение."), first.index("Продолжение первой темы."))
        self.assertNotIn("Модель регрессии.", first)
        self.assertIn(r"\section{Линейная регрессия}", second)
        self.assertLess(second.index("Модель регрессии."), second.index("Продолжение второй темы."))
        self.assertNotIn("Второй интервал", first + second)
        self.assertNotIn("Лекция:", first + second)

    def test_legacy_blocks_use_chapter_title_and_topics_cannot_set_asset_paths(self):
        self.notes["chapters"][0]["blocks"] = [
            {"id": "legacy", "kind": "text", "topic": "", "content": "Старый блок."},
            {"id": "blank-topic", "kind": "text", "topic": " \n ", "content": "Продолжение."},
            {"id": "new", "kind": "text", "topic": r"../../Матрицы & \input{secret}",
             "content": "Новая тема."},
        ]
        root, manifest = self.project()
        rendering._validate_project(root, manifest)
        self.assertIn(r"\section{Случайные величины}", (root / "chapters/section-001.tex").read_text())
        self.assertIn("Продолжение.", (root / "chapters/section-001.tex").read_text())
        self.assertIn(r"\section{../../Матрицы \& \textbackslash{}input\{secret\}}",
                      (root / "chapters/section-002.tex").read_text())

    def test_formal_blocks_keep_inline_math_and_uncertainty_inside_environments(self):
        self.notes["chapters"][0]["blocks"] = [
            {"id": kind, "kind": kind, "topic": "Линейная алгебра",
             "content": r"Матрица \(A\) симметрична.", "evidence_ids": ["ev-1"],
             "uncertain_fragments": [{"text": "симметрична", "evidence_ids": ["ev-1"]}]}
            for kind in ("definition", "theorem", "proof")
        ]
        self.evidence([("ev-1", 3661)])
        root, manifest = self.project()
        rendering._validate_project(root, manifest)
        main = (root / "main.tex").read_text()
        chapter = (root / "chapters/section-001.tex").read_text()
        self.assertIn(r"\usepackage{amsthm}", main)
        self.assertIn(r"\newtheorem{theorem}{Теорема}[section]", main)
        self.assertIn(r"\newtheorem{definition}[theorem]{Определение}", main)
        for kind in ("definition", "theorem", "proof"):
            heading = "[Доказательство]" if kind == "proof" else ""
            self.assertIn("\\begin{" + kind + "}" + heading + "\n"
                          + r"Матрица \(A\) \textcolor{blue!60!black}{симметрична}.\footnote{01:01:01}"
                          + "\n\\end{" + kind + "}", chapter)
        self.assertEqual(chapter.count(r"\footnote"), 3)
        self.assertTrue(manifest["draft"])

    def test_formal_block_prose_cannot_inject_tex_environments(self):
        self.notes["chapters"][0]["blocks"] = [
            {"id": "definition", "kind": "definition",
             "content": r"\end{definition}\input{secret}"},
        ]
        root, manifest = self.project()
        rendering._validate_project(root, manifest)
        chapter = (root / "chapters/section-001.tex").read_text()
        self.assertIn(r"\textbackslash{}end\{definition\}\textbackslash{}input\{secret\}", chapter)
        for command in (r"\newtheorem{other}{Другое}", r"\theoremstyle{plain}",
                        r"\begin{proof}x\end{proof}"):
            self.notes["chapters"][0]["blocks"][0]["content"] = r"\(" + command + r"\)"
            with self.subTest(command=command), self.assertRaises(rendering.RenderError):
                self.project()

    def test_only_fixed_formal_environment_declarations_are_allowed(self):
        root, manifest = self.project()
        main = root / "main.tex"
        original = main.read_text()
        declarations = (r"\newtheorem{other}{Другое}", r"\newtheorem*{theorem}{Теорема}",
                        r"\newtheorem{theorem}{Другое}[section]", r"\theoremstyle{plain}",
                        r"\newtheorem{input}{Теорема}")
        for declaration in declarations:
            main.write_text(original + declaration)
            with self.subTest(declaration=declaration), self.assertRaises(rendering.RenderError):
                rendering._validate_project(root, manifest)
        main.write_text(original)
        chapter = root / "chapters/section-001.tex"
        original = chapter.read_text()
        for declaration in (r"\newtheorem{theorem}{Теорема}[section]", r"\theoremstyle{definition}"):
            chapter.write_text(original + declaration)
            with self.subTest(declaration=declaration), self.assertRaisesRegex(rendering.RenderError, "only in main.tex"):
                rendering._validate_project(root, manifest)

    def test_regeneration_removes_obsolete_review_appendix(self):
        root, _ = self.project()
        (root / "review.tex").write_text(r"\section{Вопросы и замечания}")
        root, manifest = self.project()
        self.assertFalse((root / "review.tex").exists())
        self.assertNotIn("review.tex", manifest["tex_files"])
        self.assertNotIn(r"\input{review.tex}", (root / "main.tex").read_text())
        rendering._validate_project(root, manifest)

    def test_invalid_inline_math_is_blocked(self):
        for content in (r"Не закрыто \(x", r"Опасно \(\input{secret}\)"):
            with self.subTest(content=content):
                self.notes["chapters"][0]["blocks"][0]["content"] = content
                with self.assertRaises(rendering.RenderError):
                    self.project()

    def test_unmarked_multiline_inline_math_is_preserved(self):
        expression = "\\(a\n\nb\\)"
        self.notes["chapters"][0]["blocks"][0]["content"] = "Формула " + expression
        root, _ = self.project()
        self.assertIn(expression, (root / "chapters/section-001.tex").read_text())

    def test_function_colon_is_supported_in_inline_and_display_math(self):
        expression = r"f\colon \mathbb{R} \to \mathbb{R}"
        self.notes["chapters"][0]["blocks"][0]["content"] = "Функция \\(" + expression + r"\)."
        self.notes["chapters"][0]["blocks"][1]["content"] = expression
        root, manifest = self.project()
        rendering._validate_project(root, manifest)
        chapter = (root / "chapters/section-001.tex").read_text()
        self.assertIn(r"\(" + expression + r"\)", chapter)
        self.assertIn("\\[\n" + expression + "\n\\]", chapter)

    def test_norm_conditional_and_tilde_commands_are_supported(self):
        expressions = (r"w_j = \|x-x_{(j)}\|^{-1}",
                       r"P(Y=1\mid X=x)", r"\tilde{x}=x-\bar{x}")
        for expression in expressions:
            with self.subTest(expression=expression):
                self.notes["chapters"][0]["blocks"][0]["content"] = r"Формула \(" + expression + r"\)."
                self.notes["chapters"][0]["blocks"][1]["content"] = expression
                root, manifest = self.project()
                rendering._validate_project(root, manifest)
                chapter = (root / "chapters/section-001.tex").read_text()
                self.assertIn(r"\(" + expression + r"\)", chapter)
                self.assertIn("\\[\n" + expression + "\n\\]", chapter)

    def test_safe_math_commands_do_not_allow_unsafe_commands(self):
        for name, command in (("input", r"\input{secret}"), ("write", r"\write18{ls}"),
                              ("directlua", r"\directlua{os.execute('id')}")):
            with self.subTest(command=command), self.assertRaisesRegex(rendering.RenderError, name):
                rendering._validate_commands(r"\|\tilde{x}\|\mid y " + command)

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
                chapter = (root / "chapters/section-001.tex").read_text()
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
        source = root / "chapters/section-001.tex"
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
        source = root / "chapters/section-001.tex"
        for malicious in (r"\includegraphics{../evidence/frame.png}", r"\input{../sources/slides.tex}", r"\usepackage{shellesc}", r"\renewcommand\input{bad}", r"\setmainfont[Path=../]{CMU Serif}"):
            original = source.read_text()
            source.write_text(original + malicious)
            with self.subTest(malicious=malicious), self.assertRaises(rendering.RenderError):
                rendering._validate_project(root, manifest)
            source.write_text(original)

    def test_symlink_asset_escape_is_rejected(self):
        root, manifest = self.project()
        source = root / "chapters/section-001.tex"
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
        edited_paths = [root / "chapters/section-001.tex", root / "figures/diagram-diagram-1.tex", root / "code/code-1.txt"]
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
        self.assertFalse((root / "review.tex").exists())
        self.assertNotIn("review.tex", manifest["tex_files"])
        self.assertNotIn(r"\input{review.tex}", main_path.read_text())
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
        self.notes["chapters"][0]["blocks"][0]["uncertain_fragments"] = [
            {"text": r"\(X \in \mathbb{R}\)", "evidence_ids": ["ev-1"]}]
        self.evidence([("ev-1", 3661)])
        self.notes["chapters"][0]["blocks"][1]["content"] = (
            r"f\colon \mathbb{R} \to \mathbb{R},\quad x \mapsto x^2,\quad "
            r"\|\tilde{x}\|,\quad P(Y=1\mid X=x)")
        self.notes["chapters"][0]["blocks"].extend([
            {"id": "definition-1", "kind": "definition",
             "content": r"Матрица называется симметричной, если \(A = A^{\mathsf{T}}\).",
             "evidence_ids": ["ev-1"],
             "uncertain_fragments": [{"text": "симметричной", "evidence_ids": ["ev-1"]}]},
            {"id": "theorem-1", "kind": "theorem",
             "content": r"Если \(X\) постоянно, то его дисперсия равна нулю."},
            {"id": "proof-1", "kind": "proof",
             "content": r"По определению \(\operatorname{Var}(X) = \mathbb{E}[(X-\mathbb{E}[X])^2] = 0\)."},
            {"id": "theorem-2", "kind": "theorem", "topic": "Следующая тема",
             "content": r"Для \(a > 0\) выполнено \(a^2 > 0\)."},
        ])
        self.project()
        pdf = rendering.render(self.job)
        self.assertGreater(pdf.stat().st_size, 1000)
        self.assertTrue((pdf.parent / "qa/page-001.png").is_file())
        report = json.loads((pdf.parent / "render-report.json").read_text())
        self.assertEqual(report["missing_fonts"], [])
        self.assertEqual(report["missing_glyphs"], [])
        import pymupdf
        with pymupdf.open(pdf) as document:
            text = "\n".join(page.get_text() for page in document)
        self.assertIn("Случайные величины", text)
        self.assertRegex(text, r"Определение\s+1\.1")
        self.assertRegex(text, r"Теорема\s+1\.2")
        self.assertRegex(text, r"Теорема\s+2\.1")
        self.assertIn("Доказательство", text)
        self.assertIn("Проверьте индекс.", text)
        self.assertIn("01:01:01", text)
        self.assertNotIn("Требует проверки", text)
        self.assertNotIn("Источники:", text)
        self.assertNotIn("Аудио 00:01:05", text)
        self.assertNotIn("Вопросы и замечания", text)
        self.assertNotIn("Неясный индекс", text)


if __name__ == "__main__":
    unittest.main()
