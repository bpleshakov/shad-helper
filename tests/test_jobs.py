import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from shad_helper.jobs import (
    AUDIO_CONFIG,
    check_job,
    checkpoint,
    evidence_index,
    read_json,
    remove_artifacts,
    validate_audio_evidence,
    validate_evidence,
    write_json,
)


def video_evidence():
    return {"schema_version": 4, "title": "Лекция", "media": {"duration": 120},
            "segments": [{"id": "s1", "start": 0, "end": 10, "text": "Теорема"}],
            "visuals": [{"id": "frame_1", "timestamp": 65, "status": "complete",
                         "kind": "whiteboard", "text": "Определение", "formulas": ["y=x"],
                         "diagrams": [{"description": "Стрелка", "labels": ["A", "B"],
                                       "relationships": ["A → B"]}], "unreadable": []}]}


class CheckpointTests(unittest.TestCase):
    def test_artifact_cleanup_removes_outputs_and_links_without_touching_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = root / "job"
            write_json(job / "checkpoints/notes/ch001.json", {})
            write_json(job / "checkpoints/asr/ch001.json", {"source": True})
            (job / ".notes.json.123.tmp").write_text("interrupted output")
            write_json(root / "outside/keep.json", {"source": True})
            (job / "notes").symlink_to(root / "outside", target_is_directory=True)
            (job / "handoff").symlink_to(root / "missing", target_is_directory=True)

            remove_artifacts(job, ("checkpoints/notes", "notes", "handoff", "notes.json"))

            self.assertFalse((job / "checkpoints/notes").exists())
            self.assertFalse((job / "notes").is_symlink())
            self.assertFalse((job / "handoff").is_symlink())
            self.assertFalse((job / ".notes.json.123.tmp").exists())
            self.assertEqual(read_json(job / "checkpoints/asr/ch001.json"), {"source": True})
            self.assertEqual(read_json(root / "outside/keep.json"), {"source": True})

    def test_artifact_cleanup_rejects_escaping_parent_before_any_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = root / "job"
            write_json(job / "notes.json", {"keep": True})
            write_json(root / "outside/notes/ch001.json", {"keep": True})
            (job / "checkpoints").symlink_to(root / "outside", target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "escapes the job"):
                remove_artifacts(job, ("notes.json", "checkpoints/notes"))

            self.assertEqual(read_json(job / "notes.json"), {"keep": True})
            self.assertEqual(read_json(root / "outside/notes/ch001.json"), {"keep": True})

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

    def test_visual_evidence_and_timestamps_are_indexed(self):
        evidence = video_evidence()
        self.assertEqual(validate_evidence(evidence), evidence)
        self.assertEqual(evidence_index(evidence),
                         {"s1": "Аудио 00:00:00", "frame_1": "Видео 00:01:05"})
        evidence["visuals"][0]["status"] = "incomplete"
        self.assertEqual(validate_evidence(evidence), evidence)

    def test_visual_evidence_rejects_private_fields_and_bad_content(self):
        for key, value in (("image", "/local/frame.png"), ("diagnostics", {}),
                           ("timestamp", float("nan")), ("timestamp", -1),
                           ("status", "unknown"), ("kind", "unknown"),
                           ("formulas", "y=x"), ("unreadable", [None]),
                           ("diagrams", [{"description": "diagram"}])):
            evidence = video_evidence()
            evidence["visuals"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_evidence(evidence)
        for key in ("diagnostics", "models", "issues"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_evidence({**video_evidence(), key: {}})

    def test_visual_and_speech_ids_cannot_collide(self):
        evidence = video_evidence()
        for identifier in ("s1", "frame_1"):
            changed = deepcopy(evidence)
            changed["visuals"].append({**changed["visuals"][0], "id": identifier})
            with self.subTest(identifier=identifier), self.assertRaisesRegex(ValueError, "unique"):
                validate_evidence(changed)

    def test_supported_job_versions_allow_resume_and_model_switching(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            video = video_evidence()
            audio = {key: value for key, value in video.items() if key != "visuals"}
            audio["schema_version"] = 3
            for config in (AUDIO_CONFIG, *({"schema_version": 4, "pipeline": "audio_video",
                                           "vision_model": model}
                                          for model in ("4b", "9b", "35b"))):
                for evidence in (audio, video):
                    with self.subTest(config=config, version=evidence["schema_version"]):
                        write_json(job / "extraction-config.json", config)
                        write_json(job / "evidence.json", evidence)
                        check_job(job)

    def test_unsupported_video_preset_in_job_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            write_json(job / "extraction-config.json",
                       {"schema_version": 4, "pipeline": "audio_video", "vision_model": "unknown"})
            with self.assertRaisesRegex(ValueError, "Unsupported job format"):
                check_job(job)


if __name__ == "__main__":
    unittest.main()
