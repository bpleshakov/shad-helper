"""Decode page-selection previews without exporting intermediate images."""

import io
import queue
import re
import subprocess
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_frame(video: Path, timestamp: float) -> np.ndarray:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(timestamp),
         "-i", str(video), "-map", "0:v:0", "-frames:v", "1", "-f", "image2pipe",
         "-vcodec", "png", "pipe:1"],
        check=True, capture_output=True,
    )
    if not result.stdout:
        raise RuntimeError(f"No video frame could be decoded at {timestamp} seconds.")
    with Image.open(io.BytesIO(result.stdout)) as image:
        return np.asarray(image.convert("RGB")).copy()


def iter_previews(video: Path, metadata: dict, start: float, end: float, *,
                  width: int = 384, sample_seconds: float | None = None):
    """Yield actual PTS and native-cadence RGB previews with bounded pixel memory."""
    width = min(width, metadata["width"])
    height = max(2, round(metadata["height"] * width / metadata["width"] / 2) * 2)
    filters = f"scale={width}:{height}"
    if sample_seconds is not None:
        # Scene events supplement sampling; a short intervening slide is not skipped.
        filters += (f",select='isnan(prev_selected_t)+gte(t-prev_selected_t,{sample_seconds})"
                    "+gt(scene,0.12)'")
    filters += ",showinfo"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-ss", str(start),
               "-i", str(video), "-t", str(end - start), "-map", "0:v:0", "-an", "-sn", "-dn",
               "-vf", filters, "-fps_mode", "passthrough",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timestamps = queue.Queue()
    errors = deque(maxlen=20)

    def read_timestamps():
        for line in iter(process.stderr.readline, b""):
            text = line.decode("utf-8", errors="replace")
            match = re.search(r"\bn:\s*\d+.*?\bpts_time:([-\d.e+]+)", text)
            if match:
                timestamps.put(start + float(match.group(1)))
            elif "error" in text.lower():
                errors.append(text.strip())
        timestamps.put(None)

    reader = threading.Thread(target=read_timestamps, daemon=True)
    reader.start()
    previous = None
    exhausted = False
    try:
        while True:
            pixels = process.stdout.read(width * height * 3)
            if not pixels:
                exhausted = True
                break
            if len(pixels) != width * height * 3:
                raise RuntimeError("FFmpeg returned a partial page preview frame.")
            timestamp = timestamps.get(timeout=60)
            if timestamp is None:
                raise RuntimeError("FFmpeg did not provide a page preview timestamp.")
            rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            change = (1.0 if previous is None else
                      float(np.count_nonzero(cv2.absdiff(gray, previous) > 12) / gray.size))
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            previous = gray
            if start <= timestamp < end:
                yield {"timestamp": timestamp, "rgb": rgb, "change": change,
                       "sharpness": sharpness}
    finally:
        process.stdout.close()
        if not exhausted and process.poll() is None:
            process.terminate()
        process.wait()
        reader.join(timeout=5)
        process.stderr.close()
    if process.returncode:
        raise RuntimeError("FFmpeg page preview failed: " + "\n".join(errors))
