import tempfile
import unittest
from pathlib import Path

from shad_helper.jobs import checkpoint, evidence_index, read_json, write_json


class CheckpointTests(unittest.TestCase):
    def test_resume_and_input_invalidation(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            calls = []
            action = lambda: calls.append(True) or {"value": len(calls)}
            self.assertEqual(checkpoint(job, "asr/000", {"audio": "a"}, action), {"value": 1})
            self.assertEqual(checkpoint(job, "asr/000", {"audio": "a"}, action), {"value": 1})
            self.assertEqual(checkpoint(job, "asr/000", {"audio": "b"}, action), {"value": 2})
            self.assertEqual(len(calls), 2)

    def test_failed_action_is_not_cached(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            with self.assertRaises(RuntimeError):
                checkpoint(job, "broken", {}, lambda: (_ for _ in ()).throw(RuntimeError("stop")))
            self.assertFalse((job / "checkpoints/broken.json").exists())

    def test_reject_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            for key in ("../escape", "/tmp/escape", "a//b"):
                with self.assertRaises(ValueError):
                    checkpoint(Path(temporary), key, {}, lambda: None)

    def test_unicode_json_and_source_visual_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.json"
            evidence = {"segments": [{"id": "s1", "start": 65}],
                        "visuals": [{"id": "v1", "timestamp": None, "source_id": "p1",
                                     "regions": [{"id": "r1"}]}],
                        "sources": [{"id": "p1", "source": "лекция.pdf", "location": "page 2"}]}
            write_json(path, evidence)
            self.assertEqual(read_json(path), evidence)
            index = evidence_index(evidence)
            self.assertEqual(index["s1"], "Видео 00:01:05")
            self.assertEqual(index["r1"], "лекция.pdf: page 2")


if __name__ == "__main__":
    unittest.main()
