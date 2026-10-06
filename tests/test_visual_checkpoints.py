import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from shad_helper import local_worker
from shad_helper.extraction import _prepare_visuals
from shad_helper.jobs import fingerprint, read_json, sha256_file, write_json


class VisualCheckpointTests(unittest.TestCase):
    def _request(self, job, identifier="visual_00099"):
        image = job / f"{identifier}.png"
        image.write_bytes(b"pixels read only by mocked inference")
        item = {"id": identifier, "image": str(image), "timestamp": 99}
        return {"job": str(job), "model": "mock-vision", "items": [item]}

    def _visual(self):
        return {"kind": "whiteboard", "summary": "На доске записана формула x + y.",
                "diagram": "", "regions": [{"kind": "formula", "bbox": [0.1, 0.2, 0.8, 0.7]}],
                "issues": []}

    def _cache(self, request, result):
        item = request["items"][0]
        signature = {"model": request["model"], "task": "vision", "item": item,
                     "language": request.get("language"),
                     "sha256": sha256_file(Path(item["image"])),
                     "prompt": local_worker.VISION_PROMPT}
        path = Path(request["job"]) / "checkpoints" / "vision" / f"{item['id']}.json"
        write_json(path, {"fingerprint": fingerprint(signature), "seconds": 2.0, "result": result})
        return path

    def _assert_full_image_ocr_fallback(self, visual, job):
        class Image:
            width, height = 100, 100

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def crop(self, bbox):
                self.bbox = bbox
                return self

            def save(self, path):
                Path(path).write_bytes(b"full image OCR crop")

        image = Image()
        pillow = SimpleNamespace(Image=SimpleNamespace(open=lambda path: image))
        with patch.dict(sys.modules, {"PIL": pillow}):
            visuals, crops = _prepare_visuals([visual], job)
        self.assertTrue(visuals[0]["issues"])
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0]["kind"], "text")
        self.assertEqual(visuals[0]["regions"][0]["bbox"], [0, 0, 1, 1])
        self.assertEqual(image.bbox, (0, 0, 100, 100))

    def test_invalid_nested_region_cache_reruns_only_that_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            request = self._request(job)
            valid_request = self._request(job, "visual_00098")
            valid_visual = {**self._visual(), **valid_request["items"][0]}
            valid_path = self._cache(valid_request, valid_visual)
            valid_bytes = valid_path.read_bytes()
            invalid = {**self._visual(), **request["items"][0],
                       "regions": [{"kind": ["formula"], "bbox": [0.1, 0.2, 0.8, 0.7]}]}
            invalid_path = self._cache(request, invalid)
            request["items"].insert(0, valid_request["items"][0])
            raw = json.dumps(self._visual(), ensure_ascii=False)
            with patch.object(local_worker, "_vlm_model", return_value=(None, None)) as load, \
                    patch.object(local_worker, "_generate", return_value=raw) as generate:
                results = local_worker.run(request, "vision")["items"]
            self.assertEqual(results[0], valid_visual)
            self.assertEqual(valid_path.read_bytes(), valid_bytes)
            self.assertEqual(results[1]["regions"], self._visual()["regions"])
            self.assertEqual(read_json(invalid_path)["result"], results[1])
            load.assert_called_once_with("mock-vision")
            generate.assert_called_once_with(None, None, request["items"][1]["image"],
                                             local_worker.VISION_PROMPT, 4096, detect_repetition=True)

    def test_valid_legacy_cache_does_not_load_or_generate(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            request = self._request(job)
            visual = {**self._visual(), **request["items"][0]}
            path = self._cache(request, visual)
            previous = path.read_bytes()
            with patch.object(local_worker, "_vlm_model", side_effect=AssertionError("Loaded model")), \
                    patch.object(local_worker, "_generate", side_effect=AssertionError("Generated output")):
                result = local_worker.run(request, "vision")
            self.assertEqual(result, {"items": [visual]})
            self.assertEqual(path.read_bytes(), previous)

    def test_repetition_preserves_partial_raw_and_full_image_ocr_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            request = self._request(job)
            raw = "partial repeated output repeated output repeated output"
            with patch.object(local_worker, "_vlm_model", return_value=(None, None)), \
                    patch.object(local_worker, "_generate", side_effect=local_worker.RepetitionError(raw)):
                visual = local_worker.run(request, "vision")["items"][0]
            artifact = job / "raw" / "vision" / "visual_00099.json"
            self.assertEqual(read_json(artifact), {"raw": raw, "error": "repetition"})
            self._assert_full_image_ocr_fallback(visual, job)

    def test_invalid_json_preserves_raw_and_full_image_ocr_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            request = self._request(job)
            raw = "The visible content is unreadable; there is no JSON object."
            with patch.object(local_worker, "_vlm_model", return_value=(None, None)), \
                    patch.object(local_worker, "_generate", return_value=raw):
                visual = local_worker.run(request, "vision")["items"][0]
            artifact = job / "raw" / "vision" / "visual_00099.json"
            self.assertEqual(read_json(artifact), {"raw": raw, "error": "invalid_json"})
            self._assert_full_image_ocr_fallback(visual, job)


if __name__ == "__main__":
    unittest.main()
