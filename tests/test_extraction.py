import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from shad_helper.extraction import (_artifact_checkpoint, _audio_chunks, _frames, _prepare_visuals, select_keyframes,
                                   source_glossary, stitch_segments, transcription_warnings,
                                   write_subtitles)
from shad_helper.local_worker import parse_model_json, run


class ExtractionTests(unittest.TestCase):
    def test_overlapping_chunks_have_one_owner_and_original_timestamps(self):
        chunks = [
            {"offset": 95.0, "core_start": 100.0, "core_end": 110.0, "segments": [
                {"start": 3, "end": 16, "text": " before first second after", "words": [
                    {"word": " before", "start": 3, "end": 4},
                    {"word": " first", "start": 5.1, "end": 6},
                    {"word": " second", "start": 14.7, "end": 15.3},
                    {"word": " after", "start": 15.4, "end": 16}], "avg_logprob": -0.2}]},
            {"offset": 105.0, "core_start": 110.0, "core_end": 120.0, "segments": [
                {"start": 4.7, "end": 7, "text": " second after", "words": [
                    {"word": " second", "start": 4.7, "end": 5.3},
                    {"word": " after", "start": 5.4, "end": 6}]}]},
        ]
        result = stitch_segments(list(reversed(chunks)))
        self.assertEqual([segment["text"] for segment in result], ["first", "second after"])
        self.assertAlmostEqual(result[0]["start"], 100.1)
        self.assertEqual(result[1]["start"], 110.0)
        self.assertEqual(result[1]["words"][0]["start"], 109.7)
        self.assertEqual(result[0]["diagnostics"], {"avg_logprob": -0.2})

    def test_segment_fallback_and_silence(self):
        self.assertEqual(stitch_segments([]), [])
        result = stitch_segments([{"offset": 3600, "core_start": 3600, "core_end": 4200,
                                   "segments": [{"start": 1, "end": 3, "text": " Теорема "}]}])
        self.assertEqual((result[0]["start"], result[0]["end"], result[0]["text"]), (3601, 3603, "Теорема"))

    def test_incremental_board_states_and_pre_erasure_are_retained(self):
        samples = [{"timestamp": round(position / 10, 1),
                    "change": 0.8 if position in (0, 20, 50) else 0,
                    "sharpness": 10} for position in range(80)]
        times = [sample["timestamp"] for sample in select_keyframes(samples, max_gap=30)]
        self.assertTrue({0.0, 1.9, 4.9, 7.9}.issubset(times))
        self.assertEqual(times, sorted(set(times)))

    def test_continuous_motion_has_periodic_evidence(self):
        samples = [{"timestamp": position / 10, "change": 0.8, "sharpness": position % 7}
                   for position in range(61)]
        times = [sample["timestamp"] for sample in select_keyframes(samples, max_gap=2)]
        self.assertTrue({0.0, 2.0, 4.0, 6.0}.issubset(times))
        self.assertEqual(select_keyframes([]), [])

    def test_pre_erasure_content_survives_when_earlier_frame_is_sharper(self):
        samples = [{"timestamp": position / 10, "change": 1 if position == 20 else 0,
                    "sharpness": 100 - position} for position in range(30)]
        times = {sample["timestamp"] for sample in select_keyframes(samples)}
        self.assertIn(1.9, times)

    def test_subtitles_keep_pilot_offset_and_russian_text(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            write_subtitles([{"start": 3601.125, "end": 3603.75, "text": "Теорема"}], out)
            self.assertIn("01:00:01,125 --> 01:00:03,750\nТеорема", (out / "transcript.srt").read_text())
            self.assertTrue((out / "transcript.vtt").read_text().startswith("WEBVTT\n\n"))

    def test_model_json_handles_fences_and_thinking(self):
        self.assertEqual(parse_model_json('<think>{"wrong": 1}</think>\n```json\n{"kind":"whiteboard"}\n```'),
                         {"kind": "whiteboard"})
        with self.assertRaises(ValueError):
            parse_model_json("Unreadable image")

    def test_pilot_audio_cache_cannot_hide_changed_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            commands = []

            def ffmpeg(command, **kwargs):
                commands.append(command)
                Path(command[-1]).write_bytes(b"audio")

            with patch("shad_helper.extraction.subprocess.run", side_effect=ffmpeg):
                _audio_chunks(Path("lecture.mp4"), out, 0, 600, "hash")
                _audio_chunks(Path("lecture.mp4"), out, 0, 1200, "hash")
            self.assertEqual([command[command.index("-t") + 1] for command in commands],
                             ["600", "605", "605"])

    def test_deleted_cached_artifact_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            image = out / "frame.png"
            calls = []

            def action():
                calls.append(1)
                image.write_bytes(b"frame")
                return {"image": str(image)}

            _artifact_checkpoint(out, "frames", {}, action, lambda result: [result["image"]])
            image.unlink()
            _artifact_checkpoint(out, "frames", {}, action, lambda result: [result["image"]])
            self.assertEqual(len(calls), 2)

    def test_exported_frame_hash_binds_saved_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch("shad_helper.extraction.subprocess.run",
                       side_effect=lambda command, **kwargs: Path(command[-1]).write_bytes(b"frame pixels")):
                frames = _frames(Path("lecture.mp4"), out, [{"timestamp": 42}])
            self.assertEqual(frames[0]["sha256"], hashlib.sha256(b"frame pixels").hexdigest())

    def test_visual_and_normal_or_fallback_crop_hashes_bind_saved_artifacts(self):
        class Image:
            width, height = 100, 50

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def crop(self, bbox):
                return self

            def save(self, path):
                Path(path).write_bytes(b"crop pixels")

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            image = out / "frame.png"
            image.write_bytes(b"source pixels")
            for regions in ([{"kind": "formula", "bbox": [0, 0, 0.5, 1]}], [{"bbox": []}]):
                with patch.dict(sys.modules, {"PIL": SimpleNamespace(Image=SimpleNamespace(open=lambda path: Image()))}):
                    visuals, crops = _prepare_visuals([{"id": "visual_0", "image": str(image),
                                                        "sha256": "untrusted", "regions": regions}], out)
                self.assertEqual(visuals[0]["sha256"], hashlib.sha256(b"source pixels").hexdigest())
                digest = hashlib.sha256(b"crop pixels").hexdigest()
                self.assertEqual(visuals[0]["regions"][0]["sha256"], digest)
                self.assertEqual(crops[0]["sha256"], digest)

    def test_whisper_risk_flags_and_bounded_source_glossary(self):
        warnings = transcription_warnings({"text": "метод опорных векторов", "words": [
            {"word": " векторов", "probability": 0.1}], "diagnostics":
            {"avg_logprob": -1.5, "no_speech_prob": 0.9, "compression_ratio": 3}})
        self.assertEqual(len(warnings), 4)
        self.assertEqual(transcription_warnings({"text": "Лемма", "diagnostics": {}}), [])
        prompt = source_glossary([{"text": "# Опорные векторы\n" + "дальше " * 500,
                                  "code": "kernel_matrix = numpy.dot(features, features.T)"}] * 100)
        self.assertLessEqual(len(prompt), 1000)
        self.assertIn("kernel_matrix", prompt)
        self.assertIsNone(source_glossary([]))

    def test_malformed_vision_keeps_evidence_and_explicit_uncertainty(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            image = job / "frame.png"
            image.write_bytes(b"not needed by mocked model")
            with patch("shad_helper.local_worker._vlm_model", return_value=(None, None)), \
                    patch("shad_helper.local_worker._generate", return_value="I cannot see this image"):
                result = run({"job": str(job), "model": "test", "items": [
                    {"id": "visual_0", "image": str(image), "timestamp": 42}]}, "vision")
            visual = result["items"][0]
            self.assertEqual(visual["image"], str(image))
            self.assertEqual(visual["summary"], "")
            self.assertTrue(visual["issues"])
            self.assertTrue((job / "raw/vision/visual_0.json").exists())


if __name__ == "__main__":
    unittest.main()
