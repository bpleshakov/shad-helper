import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import ocr_pilot


class OCRPilotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = self.root / "frame.png"
        self.image.write_bytes(b"native frame")
        self.request = {
            "job": str(self.root / "pilot"), "model": ocr_pilot.MODEL, "revision": "pinned",
            "items": [{"id": "page-1", "image": str(self.image), "task": "text",
                       "timestamp": 5015, "crop_box": [0, 0, 100, 100]}],
        }
        self.raw = "encoder = OneHotEncoder()\n# Нельзя обучать на тестовых данных!\n"
        self.runtime = Mock()
        self.runtime.infer.return_value = (
            self.raw, {"generation_tokens": 20, "finish_reason": "stop", "seconds": 0.5},
        )
        self.loader = self.enterContext(patch.object(
            ocr_pilot, "_GLMRuntime", return_value=self.runtime,
        ))
        self.enterContext(patch.object(ocr_pilot, "_versions", return_value={"mlx-vlm": "test"}))

    def test_raw_comments_retained_and_only_task_prompt_sent(self):
        self.request["items"][0]["annotation"] = "Reference must never enter the prompt"
        result = ocr_pilot.run(self.request)
        self.assertEqual(result["items"][0]["raw"], self.raw)
        self.assertEqual(result["items"][0]["crop_box"], [0, 0, 100, 100])
        self.runtime.infer.assert_called_once_with(str(self.image), "Text Recognition:", 4096)
        self.assertEqual(result["items"][0]["status"], "complete")
        diagnostic = ocr_pilot.read_json(self.root / "pilot/diagnostics/ocr/page-1.json")
        cached = ocr_pilot.read_json(self.root / "pilot/checkpoints/ocr/page-1.json")
        self.assertEqual(diagnostic["raw"], self.raw)
        self.assertEqual(diagnostic["fingerprint"], cached["fingerprint"])

    def test_resume_does_not_load_model_and_keeps_current_metadata(self):
        self.request["items"].append({"id": "page-2", "image": str(self.image), "task": "formula"})
        ocr_pilot.run(self.request)
        self.loader.assert_called_once_with(ocr_pilot.MODEL, "pinned")
        self.assertEqual(self.runtime.infer.call_count, 2)
        self.assertEqual(self.runtime.infer.call_args.args[1], "Formula Recognition:")
        self.loader.reset_mock()
        self.runtime.infer.reset_mock()
        self.request["items"][0]["timestamp"] = 5016
        result = ocr_pilot.run(self.request)
        self.assertEqual(result["diagnostics"]["cached_items"], 2)
        self.assertEqual(result["items"][0]["timestamp"], 5016)
        self.loader.assert_not_called()
        self.runtime.infer.assert_not_called()

    def test_input_task_model_settings_and_image_invalidate_cache(self):
        ocr_pilot.run(self.request)
        for field, value in (("revision", "new-pinned"), ("model", "local-model"),
                             ("max_tokens", 2048)):
            self.request[field] = value
            self.assertEqual(ocr_pilot.run(self.request)["diagnostics"]["fresh_items"], 1)
        self.request["items"][0]["task"] = "formula"
        self.assertEqual(ocr_pilot.run(self.request)["diagnostics"]["fresh_items"], 1)
        self.image.write_bytes(b"new source pixels")
        self.assertEqual(ocr_pilot.run(self.request)["diagnostics"]["fresh_items"], 1)

    def test_empty_and_truncated_responses_are_durable_but_not_cached_or_retried(self):
        for raw, metrics, status in (
            (" \n", {"finish_reason": "stop"}, "empty"),
            ("unfinished", {"finish_reason": "length", "generation_tokens": 5}, "truncated"),
            ("unfinished", {"finish_reason": None, "generation_tokens": 4096}, "truncated"),
        ):
            with self.subTest(status=status, metrics=metrics):
                self.runtime.infer.reset_mock()
                self.runtime.infer.return_value = (raw, metrics)
                result = ocr_pilot.run(self.request)
                self.assertEqual(result["items"][0]["status"], status)
                self.runtime.infer.assert_called_once()
                self.assertFalse((self.root / "pilot/checkpoints/ocr/page-1.json").exists())
                diagnostic = ocr_pilot.read_json(self.root / "pilot/diagnostics/ocr/page-1.json")
                self.assertEqual(diagnostic["raw"], raw)
                self.assertEqual(diagnostic["status"], status)

    def test_interruption_preserves_prior_raw_response_for_resume(self):
        self.request["items"].append({"id": "page-2", "image": str(self.image), "task": "formula"})
        self.runtime.infer.side_effect = [(self.raw, {"finish_reason": "stop"}), KeyboardInterrupt]
        with self.assertRaises(KeyboardInterrupt):
            ocr_pilot.run(self.request)
        diagnostic = ocr_pilot.read_json(self.root / "pilot/diagnostics/ocr/page-1.json")
        self.assertEqual(diagnostic["raw"], self.raw)
        self.assertFalse((self.root / "pilot/checkpoints/ocr/page-2.json").exists())
        self.runtime.infer.side_effect = None
        self.runtime.infer.reset_mock()
        result = ocr_pilot.run(self.request)
        self.assertEqual(result["diagnostics"]["cached_items"], 1)
        self.runtime.infer.assert_called_once()

    def test_invalid_request_rejected_before_model_load(self):
        for field, value in (("id", "../escape"), ("id", "a/b"), ("task", "summarize")):
            request = copy.deepcopy(self.request)
            request["items"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                ocr_pilot.run(request)
        duplicate = copy.deepcopy(self.request)
        duplicate["items"].append(duplicate["items"][0].copy())
        with self.assertRaises(ValueError):
            ocr_pilot.run(duplicate)
        for field, value in (("revision", ""), ("max_tokens", 0), ("max_tokens", True)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                ocr_pilot.run({**self.request, field: value})
        self.loader.assert_not_called()


class GLMRuntimeTests(unittest.TestCase):
    def test_temporal_pixel_budget_and_actual_prepared_geometry(self):
        processor = SimpleNamespace(image_processor=SimpleNamespace(
            patch_size=14, merge_size=2, temporal_patch_size=2, min_pixels=12544,
            max_pixels=100000, use_transformers_backend=True,
        ))
        model = SimpleNamespace(config=SimpleNamespace(model_type="glm_ocr"))
        generated = SimpleNamespace(text=r"\hat{x}", prompt_tokens=2800, generation_tokens=5,
                                    prompt_tps=100, generation_tps=100, peak_memory=2.2,
                                    finish_reason="stop")
        generate = Mock(return_value=generated)
        load = Mock(return_value=(model, processor))
        grid = Mock()
        grid.tolist.return_value = [[1, 78, 138]]
        prepare = Mock(return_value={"input_ids": [1], "pixel_values": [2],
                                     "attention_mask": [3], "image_grid_thw": grid})
        template = Mock(return_value="formatted")
        source = Mock()
        source.__enter__ = Mock(return_value=SimpleNamespace(size=(1920, 1080)))
        source.__exit__ = Mock(return_value=False)
        modules = {
            "mlx_vlm": SimpleNamespace(load=load, generate=generate),
            "mlx_vlm.prompt_utils": SimpleNamespace(apply_chat_template=template),
            "mlx_vlm.utils": SimpleNamespace(prepare_inputs=prepare,
                                              should_add_special_tokens=lambda *_: False),
            "PIL": SimpleNamespace(Image=SimpleNamespace(open=Mock(return_value=source))),
        }
        with patch.dict(sys.modules, modules):
            runtime = ocr_pilot._GLMRuntime(ocr_pilot.MODEL, "pinned")
            raw, metrics = runtime.infer("native.png", "Formula Recognition:", 4096)
        load.assert_called_once_with(ocr_pilot.MODEL, revision="pinned")
        self.assertEqual(processor.image_processor.max_pixels, 4_400_000)
        self.assertFalse(processor.image_processor.use_transformers_backend)
        self.assertEqual(metrics["source_size"], [1920, 1080])
        self.assertEqual(metrics["processed_size"], [1932, 1092])
        self.assertEqual(raw, r"\hat{x}")
        self.assertEqual(template.call_args.args[2], "Formula Recognition:")
        self.assertNotIn("max_pixels", prepare.call_args.kwargs)
        self.assertEqual(generate.call_args.kwargs["input_ids"], [1])
        self.assertEqual(generate.call_args.kwargs["mask"], [3])
        json.dumps(metrics)


if __name__ == "__main__":
    unittest.main()
