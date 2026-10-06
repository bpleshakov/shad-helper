import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from shad_helper.sources import import_sources, match_sources


class SourceTests(unittest.TestCase):
    def test_notebook_preserves_code_latex_and_recorded_outputs_without_execution(self):
        code = "raise RuntimeError('must never execute')\n\nx = 1  # точный код\n"
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><text>μ</text></svg>'
        outputs = [{"output_type": "stream", "name": "stdout", "text": ["hello\n", "мир\n"]},
                   {"output_type": "display_data", "data": {"text/latex": "$x^2$", "image/svg+xml": svg}, "metadata": {}}]
        notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
            {"id": "intro", "cell_type": "markdown", "source": ["# Теорема\n", "$\\alpha=1$"], "metadata": {}},
            {"id": "code", "cell_type": "code", "source": code, "execution_count": 7,
             "outputs": outputs, "metadata": {}}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "lecture.ipynb"
            path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
            records = import_sources([path], root / "job")
            self.assertEqual(records[0]["text"], "# Теорема\n$\\alpha=1$")
            self.assertEqual(records[1]["code"], code)
            self.assertEqual(records[1]["outputs"], outputs)
            self.assertEqual(records[1]["execution_count"], 7)
            self.assertIn("hello\nмир\n", records[1]["text"])
            self.assertEqual((root / "job" / records[1]["images"][0]).read_text(), svg)
            self.assertEqual((root / "job" / records[0]["original"]).read_bytes(), path.read_bytes())
            self.assertEqual(records[0]["source"], "lecture.ipynb")
            original_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertTrue(all(record["sha256"] == original_digest for record in records))
            notebook["cells"][0]["source"] = "Changed source, identical filename and cell ID"
            path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
            changed = import_sources([path], root / "job")
            self.assertNotEqual(changed[0]["sha256"], original_digest)
            self.assertEqual(changed[0]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_matching_russian_source_to_ocr_does_not_mutate_input(self):
        records = [{"id": "a", "text": "градиент функции вычисляется методом обратного распространения ошибки", "matched_visual_ids": []},
                   {"id": "b", "text": "определитель матрицы линейная алгебра собственные значения", "matched_visual_ids": []}]
        visuals = [{"id": "v", "timestamp": 42, "summary": "градиент функции методом обратного распространения",
                    "regions": [{"text": "ошибки"}]},
                   {"id": "s", "timestamp": None, "source_id": "a", "summary": records[0]["text"]}]
        result = match_sources(records, visuals)
        self.assertEqual(result[0]["matched_visual_ids"], ["v"])
        self.assertEqual(result[1]["matched_visual_ids"], [])
        self.assertEqual(records[0]["matched_visual_ids"], [])

    def test_ambiguous_and_short_matches_remain_unmatched(self):
        records = [{"id": "a", "text": "один второй третий четвертый"},
                   {"id": "b", "text": "один второй третий четвертый"}]
        result = match_sources(records, [{"id": "v", "timestamp": 1, "summary": records[0]["text"]}])
        self.assertTrue(all(not record["matched_visual_ids"] for record in result))
        result = match_sources([{"id": "a", "text": "один второй третий"}],
                               [{"id": "v", "timestamp": 1, "summary": "один второй третий"}])
        self.assertEqual(result[0]["matched_visual_ids"], [])

    def test_math_only_slide_uses_exact_source_formula_and_rejects_ties(self):
        records = [{"id": "a", "text": ""}, {"id": "b", "text": ""}]
        visuals = [{"id": "source_view", "timestamp": None, "source_id": "a", "regions": [
            {"kind": "formula", "text": r"$$L = \sum_i x_i^2$$"}]},
            {"id": "video_view", "timestamp": 10, "regions": [
                {"kind": "formula", "text": r"\(L=\sum_i x_i^2\)"}]}]
        result = match_sources(records, visuals)
        self.assertEqual(result[0]["matched_visual_ids"], ["video_view"])
        visuals.append({**visuals[0], "id": "source_second", "source_id": "b"})
        self.assertTrue(all(not record["matched_visual_ids"] for record in match_sources(records, visuals)))

    def test_notebook_latex_and_code_can_match_without_long_prose(self):
        code = "kernel_matrix = features @ features.T"
        records = [{"id": "a", "text": r"$a^2+b^2=c^2$"}, {"id": "b", "text": "", "code": code}]
        visuals = [{"id": "formula", "timestamp": 1, "regions": [{"kind": "formula", "text": "a^2+b^2=c^2"}]},
                   {"id": "code", "timestamp": 2, "regions": [{"kind": "code", "text": code}]}]
        result = match_sources(records, visuals)
        self.assertEqual(result[0]["matched_visual_ids"], ["formula"])
        self.assertEqual(result[1]["matched_visual_ids"], ["code"])


if __name__ == "__main__":
    unittest.main()
