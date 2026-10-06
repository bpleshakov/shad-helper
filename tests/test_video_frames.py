import json
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from shad_helper import video_frames

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="Real frame-decoding tests require FFmpeg and ffprobe.",
)


@pytest.fixture(scope="module")
def variable_rate_video(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-frame-decoding")
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for position, color in enumerate(colors):
        Image.new("RGB", (640, 360), color).save(directory / f"{position}.png")
    manifest = directory / "frames.txt"
    manifest.write_text(
        "file '0.png'\nduration 0.12\n"
        "file '1.png'\nduration 0.28\n"
        "file '2.png'\nduration 0.44\n"
        "file '3.png'\nduration 0.16\n"
        "file '3.png'\n",
        encoding="utf-8",
    )
    video = directory / "variable-rate.mkv"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(manifest), "-fps_mode", "passthrough", "-c:v", "ffv1", "-pix_fmt",
         "bgr0", str(video)],
        check=True, capture_output=True,
    )
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "frame=best_effort_timestamp_time", "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    )
    timestamps = [float(frame["best_effort_timestamp_time"])
                  for frame in json.loads(probe.stdout)["frames"]]
    assert timestamps == pytest.approx([0, 0.12, 0.4, 0.84, 1])
    return video, {"width": 640, "height": 360}, timestamps, colors + [colors[-1]]


def test_previews_preserve_native_variable_timestamps_and_rgb(variable_rate_video):
    video, metadata, timestamps, colors = variable_rate_video
    frames = list(video_frames.iter_previews(video, metadata, 0, 1.04))
    assert [frame["timestamp"] for frame in frames] == pytest.approx(timestamps)
    assert len(frames) == len(colors)
    for frame, color in zip(frames, colors, strict=True):
        assert frame["rgb"].shape == (216, 384, 3)
        assert frame["rgb"].dtype == np.uint8
        assert np.all(frame["rgb"] == color)


@pytest.mark.parametrize(("start", "end"), [(0.13, 0.9), (0.401, 1), (0.12, 0.4)])
def test_preview_clip_bounds_use_source_timestamps(variable_rate_video, start, end):
    video, metadata, timestamps, colors = variable_rate_video
    expected = [(timestamp, color) for timestamp, color in zip(timestamps, colors, strict=True)
                if start <= timestamp < end]
    frames = list(video_frames.iter_previews(video, metadata, start, end))
    assert [frame["timestamp"] for frame in frames] == pytest.approx(
        [timestamp for timestamp, _ in expected]
    )
    for frame, (_, color) in zip(frames, expected, strict=True):
        assert np.all(frame["rgb"] == color)


def test_random_access_keeps_full_resolution_and_exact_rgb(variable_rate_video):
    video, _, timestamps, colors = variable_rate_video
    for timestamp, color in zip(timestamps, colors, strict=True):
        frame = video_frames.read_frame(video, timestamp)
        assert frame.shape == (360, 640, 3)
        assert frame.dtype == np.uint8
        assert frame.flags.owndata
        assert np.all(frame == color)


def test_full_frame_decode_errors_and_empty_seek(variable_rate_video, tmp_path):
    video, _, _, _ = variable_rate_video
    with pytest.raises(subprocess.CalledProcessError):
        video_frames.read_frame(tmp_path / "missing.mkv", 0)
    with pytest.raises(RuntimeError, match="No video frame could be decoded"):
        video_frames.read_frame(video, 10)


def test_preview_failure_closes_real_process(tmp_path, monkeypatch):
    processes = []
    popen = subprocess.Popen

    def tracked_popen(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(video_frames.subprocess, "Popen", tracked_popen)
    with pytest.raises(RuntimeError, match="FFmpeg page preview failed"):
        list(video_frames.iter_previews(tmp_path / "missing.mkv", {"width": 640, "height": 360},
                                        0, 1))
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].stdout.closed and processes[0].stderr.closed


def test_early_preview_close_reaps_real_process(variable_rate_video, monkeypatch):
    video, metadata, _, _ = variable_rate_video
    processes = []
    popen = subprocess.Popen

    def tracked_popen(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(video_frames.subprocess, "Popen", tracked_popen)
    previews = video_frames.iter_previews(video, metadata, 0, 1.04)
    assert next(previews)["timestamp"] == 0
    previews.close()
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].stdout.closed and processes[0].stderr.closed
