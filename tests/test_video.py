import io
import json
import shutil
import subprocess
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import numpy as np

from shad_helper.video import (
    SELECTION_SETTINGS,
    _proxy_frames,
    _select_proxy_frames,
    packet_bounds,
    probe_video,
    select_frames,
)


class VideoTests(unittest.TestCase):
    def test_probe_ignores_cover_and_preserves_stream_offset(self):
        data = {"streams": [
            {"index": 0, "codec_type": "video", "disposition": {"attached_pic": 1}},
            {"index": 2, "codec_type": "video", "width": 1920, "height": 1080,
             "start_time": "12", "duration": "20"},
        ], "format": {"start_time": "10", "duration": "22"}}
        with patch("shad_helper.video.subprocess.run") as run:
            run.return_value.stdout = json.dumps(data)
            result = probe_video(Path("lecture.mp4"))
        self.assertEqual((result["stream_index"], result["start"], result["end"]), (2, 2, 22))

    def test_probe_has_no_video_for_audio_with_cover(self):
        with patch("shad_helper.video.subprocess.run") as run:
            run.return_value.stdout = json.dumps({"streams": [
                {"codec_type": "audio"},
                {"codec_type": "video", "disposition": {"attached_pic": 1}},
            ]})
            self.assertIsNone(probe_video(Path("audio.mp3")))

    def test_packet_bounds_use_pts_extrema_not_packet_order(self):
        output = io.StringIO("pts_time=15|duration_time=0.5\n"
                             "pts_time=N/A|duration_time=8\n"
                             "pts_time=13|duration_time=1\n"
                             "pts_time=16|duration_time=N/A\n")
        with patch("shad_helper.video.subprocess.Popen") as popen:
            process = popen.return_value
            process.stdout = output
            process.wait.return_value = process.poll.return_value = 0
            self.assertEqual(packet_bounds(Path("lecture.mkv"), "a:0", 10), (3, 6))
        self.assertTrue(output.closed)
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("-select_streams") + 1], "a:0")

    def test_matroska_video_uses_packet_bounds_instead_of_inaccurate_metadata(self):
        data = {"streams": [{"index": 2, "codec_type": "video", "width": 320,
                             "height": 180, "start_time": "10", "duration": "18"}],
                "format": {"start_time": "10", "duration": "18", "format_name": "matroska,webm"}}
        with (patch("shad_helper.video.subprocess.run") as run,
              patch("shad_helper.video.packet_bounds", return_value=(0, 8)) as bounds):
            run.return_value.stdout = json.dumps(data)
            result = probe_video(Path("lecture.mkv"))
        bounds.assert_called_once_with(Path("lecture.mkv"), 2, 10)
        self.assertEqual((result["start"], result["end"], result["duration"]), (0, 8, 8))

    def test_incremental_writing_accumulates_against_retained_state(self):
        frames = []
        for step in range(25):
            gray = np.full((180, 320), 255, np.uint8)
            for line in range(min(step, 12)):
                gray[20 + line * 8:23 + line * 8, 20:44] = 0
            frames.append((step / 2, gray))
        selected, count = _select_proxy_frames(iter(frames))
        self.assertEqual(count, 25)
        self.assertLess(len(selected), 6)
        self.assertGreaterEqual(selected[-1]["timestamp"], 6)
        self.assertTrue(any(frame["reason"] == "settled_change" for frame in selected))

    def test_pre_erasure_retains_short_lived_writing(self):
        blank = np.full((180, 320), 255, np.uint8)
        formula = blank.copy()
        formula[20:23, 20:44] = 0
        frames = [(0, blank), (0.5, formula), (1, formula), (1.5, blank), (2, blank)]
        selected, _ = _select_proxy_frames(frames)
        self.assertIn((1, "before_erasure"),
                      [(frame["timestamp"], frame["reason"]) for frame in selected])
        self.assertEqual(selected[-1]["timestamp"], 2)

    def test_brief_equal_area_formula_replacement_preserves_both_states(self):
        blank = np.full((180, 320), 255, np.uint8)
        first = blank.copy()
        first[20:30, 20:40] = 0
        second = blank.copy()
        second[20:30, 80:100] = 0
        frames = [(0, blank), (0.5, first), (1, first), (1.5, first), (2, first),
                  (2.5, second), (3, second), (3.5, second), (4, second)]
        selected, _ = _select_proxy_frames(frames)
        self.assertTrue(any(0.5 <= frame["timestamp"] < 2.5 for frame in selected))
        self.assertTrue(any(frame["timestamp"] >= 2.5 for frame in selected))

    def test_pure_small_cursor_motion_does_not_expand_selection(self):
        frames = []
        for step in range(80):
            gray = np.full((180, 320), 255, np.uint8)
            gray[80:84, step * 3:step * 3 + 4] = 0
            frames.append((step / 2, gray))
        selected, _ = _select_proxy_frames(frames)
        self.assertEqual(len(selected), 1)

    def test_brief_slide_survives_next_transition_with_sidebar_motion(self):
        frames = []
        for step, shade in enumerate([255, 210, 210, 210, 100, 100, 100]):
            gray = np.full((180, 320), 255, np.uint8)
            gray[:, 80:] = shade
            gray[20:30, step * 10:step * 10 + 10] = 0
            frames.append((step / 2, gray))
        selected, _ = _select_proxy_frames(frames)
        self.assertTrue(any(0.5 <= frame["timestamp"] < 2 for frame in selected))
        self.assertTrue(any(frame["timestamp"] >= 2 for frame in selected))

    def test_continuously_changing_board_has_fifteen_second_guard(self):
        gray = np.full((240, 480), 255, np.uint8)
        frames = []
        for step in range(81):
            gray = gray.copy()
            row, column = divmod(step, 20)
            gray[row * 30:row * 30 + 8, column * 22:column * 22 + 14] = 0
            frames.append((step / 2, gray))
        selected, _ = _select_proxy_frames(frames)
        timestamps = [frame["timestamp"] for frame in selected]
        self.assertTrue(all(end - start <= 15 for start, end in pairwise(timestamps)))
        self.assertTrue(any(frame["reason"] == "change_time_guard" for frame in selected))

    def test_two_boards_changing_independently_keep_both_states(self):
        blank = np.full((180, 320), 255, np.uint8)
        left = blank.copy()
        left[20:50, 20:60] = 0
        both = left.copy()
        both[20:50, 220:260] = 0
        frames = [(0, blank), (0.5, left), (1, left), (1.5, left), (2, left),
                  (2.5, both), (3, both), (3.5, both), (4, both)]
        selected, _ = _select_proxy_frames(frames)
        self.assertTrue(any(0.5 <= frame["timestamp"] < 2.5 for frame in selected))
        self.assertTrue(any(frame["timestamp"] >= 2.5 for frame in selected))

    def test_scroll_overlap_and_revisit_are_not_globally_deduplicated(self):
        page = np.random.default_rng(7).integers(0, 256, (500, 240), dtype=np.uint8)
        offsets = [0, 20, 40, 60, 80, 60, 40, 20, 0]
        frames = [(step / 2, page[offset:offset + 160].copy())
                  for step, offset in enumerate(offsets)]
        selected, _ = _select_proxy_frames(frames)
        self.assertTrue(any(frame["reason"] == "scroll_overlap" for frame in selected))
        self.assertEqual(selected[-1]["timestamp"], 4)
        self.assertTrue(any(frame["timestamp"] == 2 for frame in selected))

    def test_cache_rebuilds_missing_images_without_rescanning(self):
        metadata = {"stream_index": 1, "start": 2, "end": 200, "width": 320,
                    "height": 180, "duration": 198, "container_start": 0}
        blank = np.full((180, 320), 255, np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)

            def native(media, meta, timestamp, image):
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"image")

            with (patch("shad_helper.video.probe_video", return_value=metadata),
                  patch("shad_helper.video._proxy_frames", return_value=[(105.25, blank)]) as scan,
                  patch("shad_helper.video._native_frame", side_effect=native) as retrieve):
                first = select_frames(Path("lecture.mp4"), job, 105, 110, "first-hash")
                image = Path(first["items"][0]["image"])
                self.assertEqual(first["items"][0]["timestamp"], 105.25)
                image.unlink()
                second = select_frames(Path("lecture.mp4"), job, 105, 110, "first-hash")
                self.assertTrue(second["diagnostics"]["cache_hit"])
                self.assertEqual(scan.call_count, 1)
                self.assertEqual(retrieve.call_count, 2)
                self.assertEqual(first["items"][0]["id"], second["items"][0]["id"])
                with patch.dict(SELECTION_SETTINGS, version=999):
                    select_frames(Path("lecture.mp4"), job, 105, 110, "first-hash")
                self.assertEqual(scan.call_count, 2)
                select_frames(Path("lecture.mp4"), job, 105, 110, "second-hash")
                self.assertEqual(scan.call_count, 3)
                self.assertEqual(retrieve.call_count, 3)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_ffmpeg_proxy_retains_recording_timeline_with_nonzero_container_pts(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "offset.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                "testsrc2=size=320x180:rate=20:duration=2", "-c:v", "mpeg4",
                "-output_ts_offset", "5", str(media),
            ], check=True)
            metadata = probe_video(media)
            self.assertEqual(metadata["container_start"], 5)
            self.assertEqual(metadata["start"], 0)
            frames = list(_proxy_frames(media, metadata, 0.5, 1.5))
            self.assertEqual(len(frames), 2)
            self.assertAlmostEqual(frames[0][0], 0.5, places=4)
            self.assertAlmostEqual(frames[1][0], 1.0, places=4)
            self.assertEqual(frames[0][1].shape, (180, 320))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_ffmpeg_video_start_offset_is_not_removed_from_recording_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "delayed-video.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                "sine=duration=3", "-itsoffset", "1", "-f", "lavfi", "-i",
                "testsrc2=size=320x180:rate=20:duration=2", "-map", "1:v", "-map", "0:a",
                "-c:v", "mpeg4", "-c:a", "aac", "-fps_mode", "passthrough", str(media),
            ], check=True)
            metadata = probe_video(media)
            self.assertAlmostEqual(metadata["container_start"], 0)
            self.assertAlmostEqual(metadata["start"], 1)
            frames = list(_proxy_frames(media, metadata, 1.25, 2.25))
            self.assertEqual(len(frames), 2)
            self.assertAlmostEqual(frames[0][0], 1.25, places=4)
            self.assertAlmostEqual(frames[1][0], 1.75, places=4)


if __name__ == "__main__":
    unittest.main()
