import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import ocr_theory_pilot as pilot


class TheoryPilotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.response = self.root / "ocr.json"
        self.job = self.root / "theory"
        self.source = {"items": [
            {"id": "notebook-1", "task": "text", "image": "/never/open.png",
             "raw": "# Пояснение понятия\nx = [1, 2]", "status": "truncated",
             "timestamp": 12, "annotation": "Reference must not enter the model"},
            {"id": "notebook-2", "task": "text", "raw": "Другое понятие", "status": "complete"},
        ], "diagnostics": {"model": "ocr-model", "revision": "ocr-pinned"}}
        self.save_source()
        self.runtime = Mock()
        self.runtime.infer.return_value = ("Пояснение понятия", {"finish_reason": "stop"})
        self.loader = self.enterContext(patch.object(pilot, "_Runtime", return_value=self.runtime))
        self.enterContext(patch.object(pilot, "version", return_value="test"))

    def save_source(self):
        pilot.write_json(self.response, self.source)

    def run_pilot(self, ids=None):
        return pilot.run(self.response, self.job, ids or ["notebook-1"])

    def test_only_selected_raw_text_enters_model_and_provenance_is_retained(self):
        result = self.run_pilot()
        self.runtime.infer.assert_called_once_with(self.source["items"][0]["raw"])
        item = result["items"][0]
        self.assertEqual(item["text"], "Пояснение понятия")
        self.assertEqual(item["provenance"]["ocr_item"]["status"], "truncated")
        self.assertEqual(item["provenance"]["ocr_run"]["revision"], "ocr-pinned")
        diagnostic = pilot.read_json(self.job / "diagnostics/theory/notebook-1.json")
        cached = pilot.read_json(self.job / "checkpoints/theory/notebook-1.json")
        self.assertEqual(diagnostic["fingerprint"], cached["fingerprint"])
        self.assertEqual(diagnostic["text"], item["text"])

    def test_resume_avoids_model_and_uses_current_provenance(self):
        self.run_pilot(["notebook-1", "notebook-2"])
        self.loader.assert_called_once()
        self.source["items"][0]["timestamp"] = 13
        self.save_source()
        self.loader.reset_mock()
        self.runtime.infer.reset_mock()
        result = self.run_pilot(["notebook-1", "notebook-2"])
        self.assertEqual(result["diagnostics"]["cached_items"], 2)
        self.assertEqual(result["items"][0]["provenance"]["ocr_item"]["timestamp"], 13)
        self.loader.assert_not_called()
        self.runtime.infer.assert_not_called()
        self.source["items"][0]["raw"] = "Изменённый OCR"
        self.save_source()
        result = self.run_pilot(["notebook-1", "notebook-2"])
        self.assertEqual(result["diagnostics"]["fresh_items"], 1)
        self.runtime.infer.assert_called_once_with("Изменённый OCR")

    def test_model_prompt_and_settings_invalidate_cache(self):
        for field, value in (("REVISION", "new-pinned"), ("PROMPT", "new prompt"),
                             ("SETTINGS", {**pilot.SETTINGS, "max_tokens": 2048})):
            self.run_pilot()
            with self.subTest(field=field), patch.object(pilot, field, value):
                self.assertEqual(self.run_pilot()["diagnostics"]["fresh_items"], 1)

    def test_incomplete_outputs_are_durable_without_retry_or_success_cache(self):
        for raw, metrics, status in (
            (" ", {"finish_reason": "stop"}, "empty"),
            ("часть", {"finish_reason": "length"}, "truncated"),
            ("часть", {"generation_tokens": 4096}, "truncated"),
        ):
            with self.subTest(status=status):
                self.runtime.infer.reset_mock()
                self.runtime.infer.return_value = (raw, metrics)
                result = self.run_pilot()
                self.runtime.infer.assert_called_once()
                self.assertEqual(result["items"][0]["status"], status)
                self.assertFalse((self.job / "checkpoints/theory/notebook-1.json").exists())
                diagnostic = pilot.read_json(self.job / "diagnostics/theory/notebook-1.json")
                self.assertEqual(diagnostic["text"], raw)

    def test_interruption_keeps_completed_item(self):
        self.runtime.infer.side_effect = [("понятие", {"finish_reason": "stop"}), KeyboardInterrupt]
        with self.assertRaises(KeyboardInterrupt):
            self.run_pilot(["notebook-1", "notebook-2"])
        self.assertTrue((self.job / "diagnostics/theory/notebook-1.json").exists())
        self.runtime.infer.side_effect = None
        self.runtime.infer.reset_mock()
        result = self.run_pilot(["notebook-1", "notebook-2"])
        self.assertEqual(result["diagnostics"]["cached_items"], 1)
        self.runtime.infer.assert_called_once()

    def test_invalid_selection_or_formula_input_rejected_before_load(self):
        for ids in (["../escape"], ["notebook-1", "notebook-1"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.run_pilot(ids)
        self.source["items"][0]["task"] = "formula"
        self.save_source()
        with self.assertRaises(ValueError):
            self.run_pilot()
        self.loader.assert_not_called()


class TextOnlyRuntimeTests(unittest.TestCase):
    def test_chat_has_only_system_instruction_and_raw_ocr_no_media(self):
        model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5"))
        processor = object()
        load = Mock(return_value=(model, processor))
        template = Mock(return_value="formatted")
        generated = SimpleNamespace(text="Понятие", prompt_tokens=80, generation_tokens=10,
                                    prompt_tps=100, generation_tps=100, peak_memory=2.2,
                                    finish_reason="stop")
        generate = Mock(return_value=generated)
        with patch.dict(sys.modules, {
            "mlx_vlm": SimpleNamespace(load=load, generate=generate),
            "mlx_vlm.prompt_utils": SimpleNamespace(apply_chat_template=template),
        }):
            runtime = pilot._Runtime()
            text, metrics = runtime.infer("# OCR comment\nvalue = 2")
        load.assert_called_once_with(pilot.MODEL, revision=pilot.REVISION)
        messages = template.call_args.args[2]
        self.assertEqual(messages, [{"role": "system", "content": pilot.PROMPT},
                                    {"role": "user", "content": "# OCR comment\nvalue = 2"}])
        self.assertEqual(template.call_args.kwargs["num_images"], 0)
        self.assertFalse(template.call_args.kwargs["enable_thinking"])
        generate.assert_called_once_with(model, processor, "formatted", verbose=False,
                                         **pilot.SETTINGS)
        self.assertEqual(text, "Понятие")
        self.assertEqual(metrics["generation_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
