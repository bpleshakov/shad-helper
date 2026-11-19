import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from shad_helper import vision_worker

CONTENT = {"kind": "whiteboard", "text": "Определение", "formulas": [r"\frac{x_i}{\theta} = 1"],
           "diagrams": [{"description": "A слева, B справа", "labels": ["A", "B"],
                         "relationships": ["стрелка A → B"]}],
           "illustrations": [{"source_kind": "drawn", "box": [0.1, 0.2, 0.8, 0.9],
                              "description": "Стрелка от A к B"}], "unreadable": []}


def response(content=CONTENT, **metrics):
    return json.dumps(content, ensure_ascii=False), {
        "seconds": 1.0, "generation_tokens": 100, "finish_reason": "stop", **metrics,
    }


class ParseOutputTests(unittest.TestCase):
    def test_missing_unreadable_keeps_whiteboard_content_and_unlocalized_figure(self):
        content = {**CONTENT, "kind": "whiteboard", "illustrations": []}
        del content["unreadable"]
        parsed = vision_worker.parse_output(json.dumps(content))
        self.assertEqual(parsed["unreadable"], [])
        self.assertEqual(parsed["diagrams"], content["diagrams"])
        self.assertEqual(parsed["formulas"], content["formulas"])
        self.assertEqual(parsed["illustrations"], [{"source_kind": "unknown", "box": [0, 0, 1, 1],
                         "description": content["diagrams"][0]["description"]}])
        localized = {**content, "illustrations": CONTENT["illustrations"]}
        self.assertEqual(vision_worker.parse_output(json.dumps(localized)), {**localized, "unreadable": []})
        complete = {**content, "unreadable": []}
        self.assertEqual(vision_worker.parse_output(json.dumps(complete)), complete)
        del content["formulas"]
        with self.assertRaises(ValueError):
            vision_worker.parse_output(json.dumps(content))

    def test_json_and_single_fence_preserve_formulas(self):
        raw, _ = response()
        self.assertEqual(vision_worker.parse_output(raw), CONTENT)
        self.assertEqual(vision_worker.parse_output(f"```json\n{raw}\n```"), CONTENT)

    def test_reject_extra_text_invalid_types_and_unknown_fields(self):
        raw, _ = response()
        values = [f"Here is JSON: {raw}", raw[:-1], "null", "[]", "{}"]
        for key, value in (("kind", []), ("kind", "presentation"), ("text", None),
                           ("formulas", "x=1"), ("formulas", [1]), ("unreadable", False),
                           ("diagrams", "diagram"), ("unknown", "field")):
            values.append(json.dumps({**CONTENT, key: value}))
        for diagram in (None, {}, {"description": "a", "labels": [], "relationships": [1]},
                        {**CONTENT["diagrams"][0], "code": "invented"}):
            values.append(json.dumps({**CONTENT, "diagrams": [diagram]}))
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                vision_worker.parse_output(value)

    def test_reject_single_json_backslashes_that_corrupt_formulas(self):
        for formula in ("\theta", "\frac{a}{b}", "\beta", "\rho", "\nu"):
            with self.subTest(formula=formula), self.assertRaisesRegex(ValueError, "escape LaTeX"):
                vision_worker.parse_output(json.dumps({**CONTENT, "formulas": [formula]}))

    def test_reject_invalid_illustration_boxes_and_classification(self):
        valid = CONTENT["illustrations"][0]
        invalid = [None, {}, {**valid, "source_kind": "slide"}, {**valid, "description": ""}]
        invalid += [{**valid, "box": box} for box in
                    ([0, 0, 1], [0, 0, 1, float("nan")], [0, 0, 2, 1],
                     [0.5, 0, 0.4, 1], [0, 0.5, 1, 0.5], [False, 0, 1, 1])]
        for observation in invalid:
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                vision_worker.parse_output(json.dumps({**CONTENT, "illustrations": [observation]}))


class VisionWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / "frame.jpg"
        self.image.write_bytes(b"image bytes")
        self.request = {"job": str(self.root / "job"), "model": "test/model", "revision": "abc",
                        "items": [{"id": "v0001", "image": str(self.image), "timestamp": 5.0}]}
        self.runtime = Mock()
        self.runtime.infer.return_value = response()
        self.loader = self.enterContext(patch.object(
            vision_worker, "_load_runtime", return_value=self.runtime,
        ))
        self.enterContext(patch.object(vision_worker, "_versions", return_value={"mlx-vlm": "test"}))

    def test_load_once_and_resume_without_loading_model(self):
        self.request["items"].append({"id": "v0002", "image": str(self.image), "timestamp": 10})
        first = vision_worker.run(self.request)
        self.assertEqual(first["items"][0]["content"], CONTENT)
        self.assertEqual(first["diagnostics"]["fresh_items"], 2)
        diagnostics = vision_worker.read_json(self.root / "job/diagnostics/vision/v0001.json")
        cached = vision_worker.read_json(self.root / "job/checkpoints/vision/v0001.json")
        self.assertEqual(diagnostics["fingerprint"], cached["fingerprint"])
        self.assertEqual(diagnostics["status"], "complete")
        self.loader.assert_called_once_with("test/model", "abc")
        self.assertEqual(self.runtime.infer.call_count, 2)
        self.loader.reset_mock()
        self.runtime.reset_mock()
        resumed = vision_worker.run(self.request)
        self.assertEqual(resumed["items"], first["items"])
        self.assertEqual(resumed["diagnostics"]["cached_items"], 2)
        self.assertEqual(resumed["diagnostics"]["model_load_seconds"], 0.0)
        self.loader.assert_not_called()
        self.runtime.infer.assert_not_called()

    def test_retry_invalid_json_with_larger_token_budget(self):
        self.runtime.infer.side_effect = [("{broken", {"generation_tokens": 20}), response()]
        item = vision_worker.run(self.request)["items"][0]
        self.assertEqual(item["content"], CONTENT)
        self.assertEqual([call.args[1] for call in self.runtime.infer.call_args_list], [2048, 4096])
        self.assertEqual([call.kwargs["retry"] for call in self.runtime.infer.call_args_list],
                         [False, True])
        self.assertIn("error", item["diagnostics"]["attempts"][0])
        self.assertEqual(item["diagnostics"]["attempts"][0]["raw"], "{broken")

    def test_invalid_box_uses_existing_retry(self):
        invalid = {**CONTENT, "illustrations": [{**CONTENT["illustrations"][0],
                                                 "box": [-1, 0, 1, 1]}]}
        self.runtime.infer.side_effect = [response(invalid), response()]
        item = vision_worker.run(self.request)["items"][0]
        self.assertEqual(item["content"], CONTENT)
        self.assertEqual(len(item["diagnostics"]["attempts"]), 2)

    def test_matrix_description_without_illustration_preserves_text_and_is_cached(self):
        content = {**CONTENT, "text": "Матрица признаков X и вектор ответов y.",
                   "formulas": [r"X = \begin{pmatrix} x_{11} & x_{12} \\ x_{21} & x_{22} \end{pmatrix}"],
                   "diagrams": [{"description": "Матричные обозначения на правой доске.",
                                 "labels": ["X", "y"], "relationships": []}],
                   "illustrations": []}
        self.runtime.infer.return_value = response(content)
        item = vision_worker.run(self.request)["items"][0]
        self.assertEqual(item["diagnostics"]["status"], "complete")
        self.assertEqual(item["content"], content)
        self.assertEqual(len(item["diagnostics"]["attempts"]), 1)
        self.runtime.infer.reset_mock()
        resumed = vision_worker.run(self.request)
        self.assertEqual(resumed["diagnostics"]["cached_items"], 1)
        self.assertEqual(resumed["items"][0]["content"], content)
        self.runtime.infer.assert_not_called()

    def test_token_truncation_retries_even_when_json_happens_to_be_valid(self):
        for metrics in ({"finish_reason": "length"},
                        {"finish_reason": None, "generation_tokens": 2048}):
            with self.subTest(metrics=metrics):
                self.runtime.infer.side_effect = [response(**metrics), response()]
                self.request["revision"] += "x"
                item = vision_worker.run(self.request)["items"][0]
                self.assertEqual(len(item["diagnostics"]["attempts"]), 2)
                self.assertIn("token limit", item["diagnostics"]["attempts"][0]["error"])

    def test_double_failure_is_explicit_and_not_cached(self):
        self.runtime.infer.return_value = ("not json", {"generation_tokens": 5})
        item = vision_worker.run(self.request)["items"][0]
        self.assertEqual(item["diagnostics"]["status"], "incomplete")
        self.assertTrue(item["content"]["unreadable"])
        self.assertEqual(item["content"]["illustrations"][0]["source_kind"], "unknown")
        self.assertEqual(item["content"]["illustrations"][0]["box"], [0, 0, 1, 1])
        self.assertFalse((self.root / "job/checkpoints/vision/v0001.json").exists())
        diagnostics = vision_worker.read_json(self.root / "job/diagnostics/vision/v0001.json")
        self.assertEqual(diagnostics["status"], "incomplete")
        self.assertEqual(diagnostics["attempts"], item["diagnostics"]["attempts"])
        self.runtime.infer.return_value = response()
        repaired = vision_worker.run(self.request)["items"][0]
        self.assertEqual(repaired["content"], CONTENT)

    def test_interrupted_run_retains_successful_frames(self):
        self.request["items"].append({"id": "v0002", "image": str(self.image), "timestamp": 10})
        self.runtime.infer.side_effect = [response(), RuntimeError("interrupted")]
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            vision_worker.run(self.request)
        self.assertTrue((self.root / "job/checkpoints/vision/v0001.json").exists())
        self.assertFalse((self.root / "job/checkpoints/vision/v0002.json").exists())
        self.runtime.infer.side_effect = None
        self.runtime.infer.reset_mock()
        result = vision_worker.run(self.request)
        self.assertEqual(result["diagnostics"]["cached_items"], 1)
        self.runtime.infer.assert_called_once()

    def test_failed_attempt_survives_interruption_during_retry(self):
        self.runtime.infer.side_effect = [("{malformed", {"generation_tokens": 20}),
                                          KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt):
            vision_worker.run(self.request)
        path = self.root / "job/diagnostics/vision/v0001.json"
        diagnostics = vision_worker.read_json(path)
        self.assertEqual(diagnostics["status"], "retry_pending")
        self.assertEqual(diagnostics["item"], self.request["items"][0])
        self.assertEqual(diagnostics["inputs"]["model"], self.request["model"])
        self.assertEqual(diagnostics["inputs"]["revision"], self.request["revision"])
        self.assertEqual(diagnostics["fingerprint"],
                         vision_worker.fingerprint(diagnostics["inputs"]))
        self.assertEqual(diagnostics["attempts"][0]["raw"], "{malformed")
        self.assertEqual(diagnostics["attempts"][0]["generation_tokens"], 20)
        self.assertIn("error", diagnostics["attempts"][0])
        self.assertFalse((self.root / "job/checkpoints/vision/v0001.json").exists())
        self.runtime.infer.side_effect = None
        self.runtime.infer.reset_mock()
        result = vision_worker.run(self.request)
        self.runtime.infer.assert_called_once()
        self.assertEqual(result["items"][0]["content"], CONTENT)
        self.assertEqual(vision_worker.read_json(path)["status"], "complete")
        self.assertEqual(vision_worker.read_json(path)["fingerprint"], diagnostics["fingerprint"])

    def test_cache_tracks_image_model_revision_prompt_settings_and_dependencies(self):
        vision_worker.run(self.request)
        mutations = [lambda: self.image.write_bytes(b"changed image"),
                     lambda: self.request.update(model="test/other-model"),
                     lambda: self.request.update(revision="def"),
                     lambda: self.request.update(max_tokens=1024)]
        for change in mutations:
            change()
            self.assertEqual(vision_worker.run(self.request)["diagnostics"]["fresh_items"], 1)
        with patch.object(vision_worker, "PROMPT", "new prompt"):
            self.assertEqual(vision_worker.run(self.request)["diagnostics"]["fresh_items"], 1)
        with patch.object(vision_worker, "_versions", return_value={"mlx-vlm": "changed"}):
            self.assertEqual(vision_worker.run(self.request)["diagnostics"]["fresh_items"], 1)

    def test_cached_content_keeps_current_timestamp_and_image_path(self):
        vision_worker.run(self.request)
        other = self.root / "copied.jpg"
        other.write_bytes(self.image.read_bytes())
        self.request["items"][0].update(timestamp=500, image=str(other))
        resumed = vision_worker.run(self.request)
        self.assertEqual(resumed["items"][0]["timestamp"], 500)
        self.assertEqual(resumed["items"][0]["image"], str(other))
        self.assertEqual(resumed["diagnostics"]["cached_items"], 1)

    def test_unreadable_source_is_successfully_extracted_uncertainty(self):
        content = {**CONTENT, "unreadable": ["Знаменатель закрыт лектором"]}
        self.runtime.infer.return_value = response(content)
        first = vision_worker.run(self.request)
        self.assertEqual(first["items"][0]["content"]["unreadable"], content["unreadable"])
        self.assertEqual(vision_worker.run(self.request)["diagnostics"]["cached_items"], 1)

    def test_reject_invalid_settings_and_checkpoint_ids_before_model_load(self):
        for value in (0, -1, 4097, True, "2048"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                vision_worker.run({**self.request, "max_tokens": value})
        for item_id in ("../escape", "/tmp/escape", "a/b", ".", "..", ""):
            request = copy.deepcopy(self.request)
            request["items"][0]["id"] = item_id
            with self.subTest(item_id=item_id), self.assertRaises(ValueError):
                vision_worker.run(request)
        self.loader.assert_not_called()


class RuntimeAdapterTests(unittest.TestCase):
    def test_measures_actual_tensor_size_and_disables_thinking(self):
        import sys

        runtime = vision_worker._MLXRuntime.__new__(vision_worker._MLXRuntime)
        runtime.model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5"))
        runtime.processor = SimpleNamespace(image_processor=SimpleNamespace(patch_size=16))
        grid = Mock()
        grid.tolist.return_value = [[1, 68, 120]]
        image = Mock()
        image.__enter__ = Mock(return_value=SimpleNamespace(size=(1920, 1080)))
        image.__exit__ = Mock(return_value=False)
        prepare = Mock(return_value={"input_ids": [1], "pixel_values": [2],
                                     "image_grid_thw": grid, "attention_mask": [3]})
        template = Mock(return_value="templated prompt")
        generated = SimpleNamespace(text=json.dumps(CONTENT), prompt_tokens=2200,
                                    generation_tokens=50, prompt_tps=100, generation_tps=20,
                                    peak_memory=5.0, finish_reason="stop")
        generate = Mock(return_value=generated)
        modules = {
            "mlx_vlm": SimpleNamespace(generate=generate),
            "mlx_vlm.prompt_utils": SimpleNamespace(apply_chat_template=template),
            "mlx_vlm.utils": SimpleNamespace(prepare_inputs=prepare,
                                              should_add_special_tokens=lambda *_: False),
            "PIL": SimpleNamespace(Image=SimpleNamespace(open=Mock(return_value=image))),
        }
        with patch.dict(sys.modules, modules):
            raw, metrics = runtime.infer("frame.jpg", 2048)
        self.assertEqual(json.loads(raw), CONTENT)
        self.assertFalse(template.call_args.kwargs["enable_thinking"])
        self.assertEqual(prepare.call_args.kwargs["max_pixels"], 2_200_000)
        self.assertEqual(metrics["source_size"], [1920, 1080])
        self.assertEqual(metrics["processed_size"], [1920, 1088])
        self.assertEqual(generate.call_args.kwargs["input_ids"], [1])
        self.assertEqual(generate.call_args.kwargs["mask"], [3])
        self.assertEqual(generate.call_args.kwargs["temperature"], 0.0)
        self.assertNotIn("image", generate.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
