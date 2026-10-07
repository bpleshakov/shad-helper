import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shad_helper.extraction import (
    _artifact_checkpoint,
    _audio_chunks,
    stitch_segments,
    transcription_warnings,
    write_subtitles,
)


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


    def test_subtitles_keep_pilot_offset_and_russian_text(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            write_subtitles([{"start": 3601.125, "end": 3603.75, "text": "Теорема"}], out)
            self.assertIn("01:00:01,125 --> 01:00:03,750\nТеорема", (out / "transcript.srt").read_text())
            self.assertTrue((out / "transcript.vtt").read_text().startswith("WEBVTT\n\n"))


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
            audio = out / "chunk.wav"
            calls = []

            def action():
                calls.append(1)
                audio.write_bytes(b"audio")
                return {"audio": str(audio)}

            _artifact_checkpoint(out, "audio", {}, action, lambda result: [result["audio"]])
            audio.unlink()
            _artifact_checkpoint(out, "audio", {}, action, lambda result: [result["audio"]])
            self.assertEqual(len(calls), 2)


    def test_whisper_risk_flags(self):
        warnings = transcription_warnings({"text": "метод опорных векторов", "words": [
            {"word": " векторов", "probability": 0.1}], "diagnostics":
            {"avg_logprob": -1.5, "no_speech_prob": 0.9, "compression_ratio": 3}})
        self.assertEqual(len(warnings), 4)
        self.assertEqual(transcription_warnings({"text": "Лемма", "diagnostics": {}}), [])


if __name__ == "__main__":
    unittest.main()
