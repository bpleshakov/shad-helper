"""Benchmark qualification requires actual complete results and human review."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from shad_helper.jobs import read_json, write_json
from shad_helper.vision_benchmark import MODELS, main, qualify, run

VALID = json.dumps({"kind": "whiteboard", "summary": "Graph", "regions": [
    {"kind": "diagram", "bbox": [0.1, 0.1, 0.9, 0.9]}], "diagram": "A → B", "issues": []})
ROUTE = json.dumps({"kind": "whiteboard", "lecture_surface_bbox": [0, 0, 1, 1],
                    "lecturer_bbox": None, "issues": []})


class VisionBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        cases = []
        for i in range(40):
            image = self.root / f"case-{i}.png"
            image.write_bytes(f"mock image {i}".encode())
            cases.append({"id": f"case-{i}", "image": image.name,
                          "split": "tuning" if i < 20 else "holdout"})
        self.manifest = self.root / "cases.json"
        write_json(self.manifest, {"cases": cases})
        self.reviews = self.root / "reviews.json"
        write_json(self.reviews, {"reviews": [
            {"model": model, "case_id": case["id"], "reviewed": True,
             "routing_correct": True,
             "additional_critical_errors": 0, "missed_regions": 0, "missing_final_content": 0}
            for model in MODELS for case in cases if case["split"] == "holdout"]})
        self.profile = self.root / "profile.json"

    @staticmethod
    def response(raw=VALID, finish_reason="stop"):
        return iter([SimpleNamespace(text=raw, finish_reason=finish_reason, prompt_tokens=100,
                                     generation_tokens=80, prompt_tps=123, generation_tps=45,
                                     peak_memory=16)])

    def valid_stream(self, _model, _processor, _prompt, _image, settings):
        return self.response(ROUTE if settings["max_tokens"] == 128 else VALID)

    def report(self, stream=None, loader=None):
        with patch("shad_helper.vision_benchmark._load", loader or
                   (lambda model: (model, "processor"))) as load, \
                patch("shad_helper.vision_benchmark._stream", stream or
                      self.valid_stream), \
                patch("shad_helper.vision_benchmark._release") as release:
            path = run(self.manifest, self.root / "benchmark")
        self.assertEqual(release.call_count, 3)
        return path, load

    def test_run_loads_models_sequentially_and_preserves_raw_results(self):
        events = []

        def loader(model):
            events.append(("load", model))
            return model, "processor"

        def stream(model, _processor, prompt, image, settings):
            events.append(("stream", model))
            self.assertTrue(Path(image).is_absolute())
            self.assertIn(settings["max_tokens"], [128, 1024])
            if settings["max_tokens"] == 1024:
                self.assertIn("Do not transcribe ordinary text", prompt)
            else:
                self.assertEqual(settings["max_pixels"], 262144)
            return self.valid_stream(model, _processor, prompt, image, settings)

        path, _ = self.report(stream=stream, loader=loader)
        report = read_json(path)
        self.assertEqual(len(report["results"]), 120)
        self.assertEqual([x for x in events if x[0] == "load"], [("load", model) for model in MODELS])
        self.assertEqual([events[i * 81][0] for i in range(3)], ["load"] * 3)
        first = report["results"][0]
        self.assertEqual(first["raw"], VALID)
        self.assertEqual(first["parsed"]["diagram"], "A → B")
        self.assertEqual(first["telemetry"]["peak_memory"], 16)
        self.assertTrue(first["valid"])
        self.assertTrue(first["routing"]["valid"])

    def test_invalid_json_and_truncation_do_not_qualify(self):
        def stream(model, processor, prompt, image, settings):
            valid = ROUTE if settings["max_tokens"] == 128 else VALID
            return self.response("{bad" if model == MODELS[1] else valid,
                                 "length" if model == MODELS[2] else "stop")

        path, _ = self.report(stream=stream)
        profile = qualify(path, self.reviews, self.profile)
        self.assertFalse(profile["smaller_model_qualified"])
        self.assertFalse(profile["sparse_default"])
        self.assertEqual(profile["primary_model"], MODELS[0])
        self.assertTrue(all(not result["valid"] for result in read_json(path)["results"]
                            if result["model"] != MODELS[0]))

    def test_json_validity_without_human_reviews_never_qualifies(self):
        path, _ = self.report()
        write_json(self.reviews, {"reviews": []})
        profile = qualify(path, self.reviews, self.profile)
        self.assertFalse(profile["smaller_model_qualified"])

    def test_fastest_reviewed_holdout_model_wins_without_default_promotion(self):
        path, _ = self.report()
        report = read_json(path)
        for result in report["results"]:
            result["seconds"] = {MODELS[0]: 20, MODELS[1]: 10, MODELS[2]: 5}[result["model"]]
        write_json(path, report)
        profile = qualify(path, self.reviews, self.profile)
        self.assertEqual(profile["primary_model"], MODELS[2])
        self.assertTrue(profile["qualified"])
        self.assertFalse(profile["sparse_default"])
        self.assertEqual(profile["holdout_mean_seconds"][MODELS[2]], 5)

    def test_critical_error_or_missing_region_disqualifies_fast_model(self):
        path, _ = self.report()
        reviews = read_json(self.reviews)
        for review in reviews["reviews"]:
            if review["model"] == MODELS[2]:
                review["additional_critical_errors"] = 1
            if review["model"] == MODELS[1]:
                review["missed_regions"] = 1
        write_json(self.reviews, reviews)
        profile = qualify(path, self.reviews, self.profile)
        self.assertFalse(profile["smaller_model_qualified"])
        self.assertEqual(profile["primary_model"], MODELS[0])

    def test_duplicate_or_incomplete_reviews_do_not_qualify(self):
        path, _ = self.report()
        human = read_json(self.reviews)
        human["reviews"] = [review for review in human["reviews"] if review["model"] == MODELS[0]]
        human["reviews"].extend([dict(human["reviews"][0], model=MODELS[1])] * 20)
        write_json(self.reviews, human)
        self.assertFalse(qualify(path, self.reviews, self.profile)["smaller_model_qualified"])

    def test_review_numbers_must_be_explicit_integers(self):
        path, _ = self.report()
        human = read_json(self.reviews)
        for review in human["reviews"]:
            if review["model"] != MODELS[0]:
                review["missing_final_content"] = False
        write_json(self.reviews, human)
        self.assertFalse(qualify(path, self.reviews, self.profile)["smaller_model_qualified"])

    def test_loader_failure_preserves_report_and_runs_remaining_models(self):
        def load(model):
            if model == MODELS[1]:
                raise RuntimeError("model unavailable")
            return model, "processor"

        path, _ = self.report(loader=load)
        report = read_json(path)
        self.assertEqual(len(report["results"]), 120)
        self.assertTrue(all(not result["valid"] for result in report["results"]
                            if result["model"] == MODELS[1]))
        self.assertTrue(all(result["valid"] for result in report["results"]
                            if result["model"] == MODELS[2]))

    def test_case_sha_mismatch_blocks_qualification(self):
        path, _ = self.report()
        report = read_json(path)
        for result in report["results"]:
            if result["model"] != MODELS[0]:
                result["case_sha256"] = "different-case"
        write_json(path, report)
        self.assertFalse(qualify(path, self.reviews, self.profile)["smaller_model_qualified"])

    def test_report_fingerprint_prevents_swapped_case_definitions(self):
        path, _ = self.report()
        report = read_json(path)
        report["cases"][20]["sha256"] = "different-case"
        write_json(path, report)
        with self.assertRaises(ValueError):
            qualify(path, self.reviews, self.profile)

    def test_actual_reviewed_hour_pilot_enables_sparse_default(self):
        path, _ = self.report()
        report = read_json(path)
        for result in report["results"]:
            result["seconds"] = 3 if result["model"] == MODELS[2] else 10
        write_json(path, report)
        pilot = self.root / "pilot.json"
        write_json(pilot, {"video_seconds": 3600, "baseline_frame_count": 928,
                           "exported_frame_count": 40, "qwen_seconds": 1700,
                           "deferred_count": 0, "failed_count": 0, "final_content_omissions": 0,
                           "reviewed": True, "primary_model": MODELS[2],
                           "benchmark_fingerprint": report["benchmark_fingerprint"]})
        profile = qualify(path, self.reviews, self.profile, pilot)
        self.assertTrue(profile["qualified"])
        self.assertTrue(profile["sparse_default"])

    def test_pilot_with_deferral_or_boolean_timing_does_not_promote_default(self):
        path, _ = self.report()
        pilot = self.root / "pilot.json"
        for change in ({"deferred_count": 1}, {"qwen_seconds": False},
                       {"exported_frame_count": 120}, {"final_content_omissions": 1},
                       {"video_seconds": 600}, {"reviewed": False}, {"failed_count": 1}):
            write_json(pilot, {"video_seconds": 3600, "baseline_frame_count": 928,
                               "exported_frame_count": 40, "qwen_seconds": 1700,
                               "deferred_count": 0, "final_content_omissions": 0,
                               "failed_count": 0, "reviewed": True,
                               "primary_model": MODELS[2],
                               "benchmark_fingerprint": read_json(path)["benchmark_fingerprint"],
                               **change})
            self.assertFalse(qualify(path, self.reviews, self.profile, pilot)["sparse_default"])

    def test_wrong_model_or_unrelated_pilot_cannot_promote_default(self):
        path, _ = self.report()
        pilot = self.root / "pilot.json"
        profile = qualify(path, self.reviews, self.profile)
        valid = {"video_seconds": 3600, "baseline_frame_count": 928,
                 "exported_frame_count": 40, "qwen_seconds": 1700, "deferred_count": 0,
                 "failed_count": 0, "final_content_omissions": 0, "reviewed": True,
                 "primary_model": profile["primary_model"],
                 "benchmark_fingerprint": profile["benchmark_fingerprint"]}
        for change in ({"primary_model": "different-model"}, {"benchmark_fingerprint": "other"}):
            write_json(pilot, {**valid, **change})
            self.assertFalse(qualify(path, self.reviews, self.profile, pilot)["sparse_default"])

    def test_baseline_can_promote_sparse_when_both_smaller_models_fail(self):
        path, _ = self.report()
        human = read_json(self.reviews)
        for review in human["reviews"]:
            if review["model"] != MODELS[0]:
                review["routing_correct"] = False
        write_json(self.reviews, human)
        pilot = self.root / "pilot.json"
        write_json(pilot, {"video_seconds": 3600, "baseline_frame_count": 928,
                           "exported_frame_count": 40, "qwen_seconds": 1700, "deferred_count": 0,
                           "failed_count": 0, "final_content_omissions": 0, "reviewed": True,
                           "primary_model": MODELS[0],
                           "benchmark_fingerprint": read_json(path)["benchmark_fingerprint"]})
        profile = qualify(path, self.reviews, self.profile, pilot)
        self.assertTrue(profile["qualified"])
        self.assertFalse(profile["smaller_model_qualified"])
        self.assertEqual(profile["primary_model"], MODELS[0])
        self.assertTrue(profile["sparse_default"])

    def test_failed_first_generation_does_not_break_subsequent_cases(self):
        seen = set()

        def stream(model, processor, prompt, image, settings):
            if model not in seen:
                seen.add(model)
                return self.response("{bad")
            return self.valid_stream(model, processor, prompt, image, settings)

        path, _ = self.report(stream=stream)
        results = read_json(path)["results"]
        for model in MODELS:
            model_results = [x for x in results if x["model"] == model]
            self.assertFalse(model_results[0]["routing"]["valid"])
            self.assertTrue(model_results[1]["routing"]["valid"])
            self.assertTrue(model_results[1]["valid"])

    def test_parse_validation_is_rechecked_instead_of_trusting_valid_flag(self):
        path, _ = self.report()
        report = read_json(path)
        for result in report["results"]:
            if result["model"] != MODELS[0]:
                result["routing"]["raw"] = "{bad"
        write_json(path, report)
        self.assertFalse(qualify(path, self.reviews, self.profile)["smaller_model_qualified"])

    def test_manifest_requires_exact_split_counts_and_unique_ids(self):
        cases = read_json(self.manifest)
        cases["cases"][20]["id"] = cases["cases"][0]["id"]
        write_json(self.manifest, cases)
        with self.assertRaises(ValueError):
            run(self.manifest, self.root / "benchmark")

    def test_module_cli_uses_workspace_huggingface_cache(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch("shad_helper.vision_benchmark.run", return_value="report.json"):
            main(["run", str(self.manifest), "--out", str(self.root / "benchmark")])
            self.assertEqual(os.environ["HF_HOME"], str(Path(".cache/huggingface").resolve()))
            self.assertEqual(os.environ["HF_HUB_DISABLE_TELEMETRY"], "1")


if __name__ == "__main__":
    unittest.main()
