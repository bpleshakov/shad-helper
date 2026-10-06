"""Reject incomplete model output without mistaking an inner object for evidence."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shad_helper.local_worker import parse_model_json, run, validate_visual_result


def visual_result():
    return {
        "kind": "whiteboard",
        "summary": "На доске записана формула и показан переход между двумя узлами.",
        "regions": [
            {"kind": "formula", "bbox": [0, 0.1, 0.4, 0.5]},
            {"kind": "diagram", "bbox": [0.4, 0.2, 1, 1]},
        ],
        "diagram": {
            "entities": [{"id": "A", "label": "Начало"}, {"id": "B", "label": "Конец"}],
            "arrows": [{"from": "A", "to": "B", "label": "x > 0"}],
        },
        "issues": ["Подпись второго узла частично перекрыта рукой."],
    }


class ModelJsonTests(unittest.TestCase):
    def test_complete_object_with_whitespace_fences_and_prior_thinking(self):
        expected = visual_result()
        payload = json.dumps(expected, ensure_ascii=False)
        cases = [
            f" \n{payload}\t ",
            f"\n```json\n{payload}\n```\n",
            f"\n```\n{payload}\n```\t",
            f'<think>{{"wrong": "reasoning"}}</think>\n{payload}',
            f'<think>{{"wrong": "reasoning"}}</think>\n```json\n{payload}\n```',
        ]
        for text in cases:
            with self.subTest(text=text[:40]):
                self.assertEqual(parse_model_json(text), expected)

    def test_literal_thinking_marker_inside_summary_or_code_is_preserved(self):
        expected = visual_result()
        expected["summary"] = "На доске показан буквальный HTML-тег </think>."
        expected["regions"] = [{"kind": "code", "bbox": [0, 0, 1, 1],
                                "code": 'print("</think>")'}]
        payload = json.dumps(expected, ensure_ascii=False)
        for text in (payload, f"```json\n{payload}\n```",
                     f"<think>Prior reasoning.</think>\n{payload}"):
            with self.subTest(text=text[:40]):
                self.assertEqual(parse_model_json(text), expected)

    def test_unfinished_leading_thinking_is_rejected(self):
        for text in ("<think>", '<think>Reasoning with {"kind":"whiteboard"}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_model_json(text)

    def test_truncated_outer_object_does_not_accept_a_complete_inner_region(self):
        text = ('```json\n{"kind":"whiteboard","summary":"Доска",'
                '"regions":[{"kind":"text","bbox":[0.03,0.05,0.12,0.11]}],'
                '"issues":["' + "'konde'/" * 100 + "'k")
        with self.assertRaises(ValueError):
            parse_model_json(text)

    def test_output_must_be_exactly_one_complete_json_object(self):
        cases = [
            "", "[]", "null", "true", "42", '"lecture"',
            'Here is the result: {"kind":"whiteboard"}',
            '{"kind":"whiteboard"} additional explanation',
            '{"kind":"whiteboard"}\n{"kind":"slide"}',
            '```json\n{"kind":"whiteboard"}',
            '```json\n{"kind":"whiteboard"}\n``` trailing prose',
            '```python\n{"kind":"whiteboard"}\n```',
            '{"kind":"whiteboard"}\n```',
        ]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_model_json(text)

    def test_duplicate_keys_and_nonfinite_constants_are_rejected(self):
        cases = [
            '{"kind":"whiteboard","kind":"slide"}',
            '{"diagram":{"label":"A","label":"B"}}',
            '{"value":NaN}', '{"value":Infinity}', '{"value":-Infinity}',
        ]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_model_json(text)

    def test_json_number_overflow_is_rejected(self):
        for text in ('{"value":1e400}', '{"diagram":{"value":-1e400}}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_model_json(text)


class VisualResultTests(unittest.TestCase):
    def test_valid_nested_diagram_and_extra_evidence_fields_are_preserved(self):
        value = visual_result()
        value["regions"][0]["caption"] = "Формула слева"
        value["extra_context"] = "Дополнительная видимая подпись"
        original = copy.deepcopy(value)
        self.assertEqual(validate_visual_result(value), original)
        self.assertEqual(value, original)

    def test_supported_kinds_and_empty_or_literal_diagram_are_valid(self):
        for kind in ("slide", "notebook", "whiteboard", "other"):
            for diagram in (None, "Стрелка от A к B", {}):
                with self.subTest(kind=kind, diagram=diagram):
                    value = visual_result()
                    value.update(kind=kind, regions=[], diagram=diagram, issues=[])
                    self.assertEqual(validate_visual_result(value), value)
        for kind in ("text", "code", "formula", "table", "chart", "diagram"):
            with self.subTest(region_kind=kind):
                value = visual_result()
                value["regions"] = [{"kind": kind, "bbox": [0, 0, 1, 1]}]
                self.assertEqual(validate_visual_result(value), value)

    def test_all_top_level_fields_are_required(self):
        for field in ("kind", "summary", "regions", "diagram", "issues"):
            value = visual_result()
            del value[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_visual_result(value)

    def test_invalid_top_level_types_and_values_are_rejected(self):
        for value in (None, [], "whiteboard"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_visual_result(value)
        cases = [
            ("kind", "text"), ("kind", ["whiteboard"]),
            ("summary", None), ("summary", 42),
            ("regions", {}), ("regions", None),
            ("diagram", 42), ("diagram", []),
            ("issues", "Unreadable"), ("issues", [None]), ("issues", [42]),
        ]
        for field, replacement in cases:
            value = visual_result()
            value[field] = replacement
            with self.subTest(field=field, replacement=replacement), self.assertRaises(ValueError):
                validate_visual_result(value)

    def test_region_kind_and_coordinates_are_required(self):
        cases = [None, "text", {}, {"kind": "text"}, {"bbox": [0, 0, 1, 1]},
                 {"kind": "whiteboard", "bbox": [0, 0, 1, 1]},
                 {"kind": ["text"], "bbox": [0, 0, 1, 1]}]
        for region in cases:
            value = visual_result()
            value["regions"] = [region]
            with self.subTest(region=region), self.assertRaises(ValueError):
                validate_visual_result(value)

    def test_coordinates_must_be_finite_normalized_numbers_with_positive_area(self):
        cases = [
            None, (0, 0, 1, 1), [0, 0, 1], [0, 0, 1, 1, 1],
            [False, 0, 1, 1], [0, 0, True, 1], ["0", 0, 1, 1],
            [math.nan, 0, 1, 1], [0, 0, math.inf, 1], [0, -math.inf, 1, 1],
            [10 ** 400, 0, 1, 1], [0, 0, 10 ** 400, 1],
            [-0.1, 0, 1, 1], [0, 0, 1.1, 1],
            [0.5, 0, 0.5, 1], [0.8, 0, 0.2, 1],
            [0, 0.5, 1, 0.5], [0, 0.8, 1, 0.2],
        ]
        for bbox in cases:
            value = visual_result()
            value["regions"] = [{"kind": "text", "bbox": bbox}]
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                validate_visual_result(value)

    def test_worker_preserves_raw_truncation_and_records_explicit_review_issue(self):
        raw = ('{"kind":"whiteboard","regions":[{"kind":"text",'
               '"bbox":[0,0,1,1]}],"summary":"' + "'konde'/" * 100)
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            image = job / "frame.png"
            image.write_bytes(b"pixels are not used by mocked generation")
            with patch("shad_helper.local_worker._vlm_model", return_value=(None, None)), \
                    patch("shad_helper.local_worker._generate", return_value=raw):
                result = run({"job": str(job), "model": "test", "items": [
                    {"id": "visual_99", "image": str(image), "timestamp": 431.68}]}, "vision")
            visual = result["items"][0]
            self.assertEqual(visual["kind"], "other")
            self.assertEqual(visual["summary"], "")
            self.assertEqual(visual["regions"], [])
            self.assertTrue(visual["issues"])
            self.assertEqual(visual["image"], str(image))
            artifact = json.loads((job / "raw/vision/visual_99.json").read_text())
            self.assertEqual(artifact["raw"], raw)


if __name__ == "__main__":
    unittest.main()
