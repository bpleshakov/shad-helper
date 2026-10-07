import tempfile
import unittest
from pathlib import Path

from shad_helper.jobs import (
    checkpoint,
    evidence_index,
    read_json,
    validate_audio_evidence,
    write_json,
)


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

    def test_unicode_json_and_audio_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.json"
            evidence = {"segments": [{"id": "s1", "start": 65, "text": "Теорема"}]}
            write_json(path, evidence)
            self.assertEqual(read_json(path), evidence)
            self.assertEqual(evidence_index(evidence), {"s1": "Аудио 00:01:05"})

    def test_evidence_rejects_diagnostics_and_incomplete_segments(self):
        evidence = {"schema_version": 3, "title": "Лекция", "media": {"duration": 120},
                    "segments": [{"id": "s1", "start": 0, "end": 10, "text": "Теорема"}]}
        self.assertEqual(validate_audio_evidence(evidence), evidence)
        for key in ("issues", "models", "diagnostics", "visuals"):
            with self.subTest(root_key=key), self.assertRaises(ValueError):
                validate_audio_evidence(dict(evidence, **{key: []}))
        for key in ("words", "diagnostics", "probability"):
            segment = dict(evidence["segments"][0], **{key: []})
            with self.subTest(segment_key=key), self.assertRaises(ValueError):
                validate_audio_evidence(dict(evidence, segments=[segment]))
        for segment in ({"id": "s1", "text": "Теорема"}, None):
            with self.subTest(segment=segment), self.assertRaises(ValueError):
                validate_audio_evidence(dict(evidence, segments=[segment]))

    def test_evidence_rejects_unsupported_containers(self):
        for value in (None, [], {"schema_version": 3},
                      {"schema_version": 3, "title": "Лекция", "media": {}, "segments": {}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_audio_evidence(value)


if __name__ == "__main__":
    unittest.main()
