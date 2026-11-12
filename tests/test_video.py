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
    _instructional_regions,
    _proxy_frames,
    _select_proxy_frames,
    packet_bounds,
    probe_video,
    select_frames,
)


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("shad_helper.frame_novelty._PersonSegmenter", return_value=
                                lambda image: np.zeros(image.shape[:2], dtype=bool)))

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

    @staticmethod
    def board_frame(step=0):
        gray = np.full((240, 400), 40, np.uint8)
        gray[20:180, 12:388] = 230
        gray[50:54, 45:105] = 50
        # Movement below the board, with more contrast than the writing.
        x = 15 + (step * 7) % 300
        gray[195:238, x:x + 50] = 110
        return gray

    def test_movement_outside_board_reduces_selection(self):
        diagnostics = {}
        frames = [(step / 2, self.board_frame(step)) for step in range(80)]
        selected, count = _select_proxy_frames(iter(frames), diagnostics)
        self.assertEqual(count, 80)
        with patch("shad_helper.video._instructional_regions", return_value=None):
            full_frame, _ = _select_proxy_frames(frames)
        self.assertLess(len(selected), len(full_frame))
        self.assertLessEqual(len(selected), 3)
        self.assertEqual(diagnostics, {"region_frames": 80, "full_frame_frames": 0})
        self.assertTrue(selected[0]["content_regions"])

    def test_settled_outside_label_survives_erasure(self):
        frames = []
        for step in range(10):
            gray = self.board_frame()
            if 2 <= step < 8:
                gray[205:217, 270:370] = 230
            frames.append((step / 2, gray))
        selected, _ = _select_proxy_frames(frames)
        self.assertTrue(any(1 <= frame["timestamp"] < 4 for frame in selected))
        self.assertTrue(any(frame["timestamp"] >= 4 for frame in selected))

    def test_final_outside_content_does_not_need_another_sample(self):
        first = self.board_frame()
        last = first.copy()
        last[207:211, 290:350] = 230
        selected, _ = _select_proxy_frames([(0, first), (0.5, last)])
        self.assertEqual(selected[-1]["timestamp"], 0.5)

    def test_clear_representative_is_scored_inside_board(self):
        frames = []
        for step in range(6):
            gray = self.board_frame()
            if step:
                gray[85:95, 70:170] = 20 if step == 2 else 36
            # A sharp moving pattern below the board must not outweigh the
            # clearer writing at t=1 when choosing a settled representative.
            gray[195:] = 40
            if step != 2:
                gray[195:, step % 2::2] = 120
            frames.append((step / 2, gray))
        selected, _ = _select_proxy_frames(frames)
        settled = [frame for frame in selected if frame["reason"] == "settled_change"]
        self.assertEqual([frame["timestamp"] for frame in settled], [1])

    def test_new_writing_and_erasure_survive_outside_motion(self):
        frames = []
        for step in range(9):
            gray = self.board_frame(step)
            if step in (1, 2):
                gray[90:94, 50:100] = 50
            frames.append((step / 2, gray))
        selected, _ = _select_proxy_frames(frames)
        self.assertIn((1, "before_erasure"),
                      [(frame["timestamp"], frame["reason"]) for frame in selected])
        self.assertGreaterEqual(selected[-1]["timestamp"], 1.5)

    def test_two_detected_boards_preserve_independent_intermediate_drawings(self):
        frames = []
        for step in range(11):
            gray = self.board_frame(step)
            gray[20:180, 190:210] = 40
            if 1 <= step < 5:
                gray[95:100, 40:140] = 50
            if step >= 5:
                gray[95:100, 260:360] = 50
            frames.append((step / 2, gray))
        selected, _ = _select_proxy_frames(frames)
        self.assertEqual(len(selected[0]["content_regions"]), 2)
        self.assertTrue(any(0.5 <= frame["timestamp"] < 2.5 for frame in selected))
        self.assertTrue(any(frame["timestamp"] >= 2.5 for frame in selected))

    def test_region_detection_keeps_dark_diagrams_and_margin_labels(self):
        gray = self.board_frame()
        gray[65:145, 240:320] = 10
        gray[18:22, 50:150] = 10
        regions = _instructional_regions(gray)
        self.assertIsNotNone(regions)
        self.assertTrue(regions[1][65:145, 240:320].all())
        self.assertTrue(regions[1][18:22, 50:150].all())

    def test_full_frame_slides_and_ambiguous_layouts_use_full_frame(self):
        slide = np.full((240, 400), 230, np.uint8)
        slide[30:180, 30:370] = 255
        small_panel = np.full((240, 400), 40, np.uint8)
        small_panel[20:200, 20:190] = 230
        dark_board = np.full((240, 400), 30, np.uint8)
        dark_board[50:54, 45:105] = 240
        for gray in (slide, small_panel, dark_board):
            with self.subTest():
                self.assertIsNone(_instructional_regions(gray))

    def test_layout_change_rechecks_content_outside_previous_regions(self):
        board = self.board_frame()
        slide = np.full((240, 400), 230, np.uint8)
        slide[210:218, 50:200] = 40
        frames = [(0, board), (0.5, board), (1, slide), (1.5, slide), (2, slide)]
        selected, _ = _select_proxy_frames(frames)
        self.assertTrue(any(frame["timestamp"] >= 1 for frame in selected))
        self.assertNotIn("content_regions", selected[-1])

    def test_motion_inside_board_is_not_silently_masked(self):
        frames = []
        for step in range(10):
            gray = self.board_frame(step)
            gray[80:160, 40 + step * 15:100 + step * 15] = 40
            frames.append((step / 2, gray))
        selected, _ = _select_proxy_frames(frames)
        self.assertGreater(len(selected), 1)

    def test_cache_rebuilds_missing_images_without_rescanning(self):
        metadata = {"stream_index": 1, "start": 2, "end": 200, "width": 320,
                    "height": 180, "duration": 198, "container_start": 0}
        blank = np.full((180, 320), 255, np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)

            def native(media, meta, timestamp, image):
                import cv2

                image.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image), blank)

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
