"""Bounded-memory video scanning and resumable native frame retrieval."""

import json
import math
import os
import queue
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from tqdm import tqdm

from .frame_novelty import filter_frames
from .jobs import checkpoint, fingerprint, read_json
from .run_diagnostics import timed_step

SELECTION_SETTINGS = {
    "version": 4,
    "proxy_width": 720,
    "sample_fps": 2,
    "pixel_threshold": 24,
    "minimum_change_fraction": 0.0008,
    "settled_change_fraction": 0.005,
    "large_change_fraction": 0.14,
    "settle_seconds": 1.0,
    "maximum_change_gap_seconds": 15.0,
    "scroll_viewport_fraction": 0.4,
    "scroll_response": 0.55,
    "region_thresholds": [128, 160, 180, 200, 220, 240],
    "region_minimum_area": 0.5,
    "region_minimum_span": 0.85,
    "region_minimum_fill": 0.8,
    "region_margin": 0.02,
}


def _number(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def packet_bounds(media: Path, stream: str | int, container_start: float) -> tuple[float, float]:
    """Read packet PTS bounds with constant memory for unreliable container metadata."""
    process = subprocess.Popen(
        ["ffprobe", "-v", "error", "-select_streams", str(stream), "-show_packets",
         "-show_entries", "packet=pts_time,duration_time", "-of", "compact=p=0:nk=0", str(media)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    first = last = None
    try:
        for line in process.stdout:
            fields = dict(field.split("=", 1) for field in line.strip().split("|") if "=" in field)
            pts = _number(fields.get("pts_time"))
            if pts is None:
                continue
            duration = max(0.0, _number(fields.get("duration_time"), 0.0))
            first = pts if first is None else min(first, pts)
            last = pts + duration if last is None else max(last, pts + duration)
        if process.wait():
            raise RuntimeError(f"FFprobe packet scan failed for stream {stream}.")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
    if first is None or last <= first:
        raise ValueError(f"Input has no valid packet timestamps for stream {stream}.")
    return first - container_start, last - container_start


@timed_step("video_probe")
def probe_video(media: Path) -> dict | None:
    """Return video bounds in seconds relative to the container's start PTS."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
         str(media)], check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if (stream.get("codec_type") != "video"
                or stream.get("disposition", {}).get("attached_pic")):
            continue
        container = data.get("format", {})
        origin = _number(container.get("start_time"), 0.0)
        start = _number(stream.get("start_time"), origin) - origin
        duration = _number(stream.get("duration"))
        if duration is None:
            duration = _number(container.get("duration"))
            if duration is not None:
                duration -= max(0.0, start)
        if set(container.get("format_name", "").split(",")) & {"matroska", "webm"}:
            start, end = packet_bounds(media, stream["index"], origin)
            duration = end - start
        if duration is None or duration <= 0:
            raise ValueError("Input has no valid video duration.")
        width, height = int(stream.get("width", 0)), int(stream.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError("Input has no valid video dimensions.")
        rotation = _number(stream.get("tags", {}).get("rotate"), 0.0)
        for side_data in stream.get("side_data_list", []):
            rotation = _number(side_data.get("rotation"), rotation)
        return {"stream_index": stream["index"], "start": start,
                "end": start + duration, "duration": duration, "width": width,
                "height": height, "container_start": origin, "rotation": rotation}
    return None


def _proxy_frames(media: Path, metadata: dict, start: float, end: float):
    """Yield actual source PTS and gray proxies without storing a proxy video."""
    import numpy as np

    width = min(SELECTION_SETTINGS["proxy_width"], metadata["width"])
    height = max(1, round(metadata["height"] * width / metadata["width"]))
    interval = 1 / SELECTION_SETTINGS["sample_fps"]
    filters = (f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval})',"
               f"scale={width}:{height},format=gray,showinfo")
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "info",
               "-ss", str(start), "-t", str(end - start), "-copyts", "-noautorotate",
               "-i", str(media), "-map", f"0:{metadata['stream_index']}",
               "-an", "-sn", "-dn", "-vf", filters, "-fps_mode", "passthrough",
               "-f", "rawvideo", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timestamps = queue.Queue()
    errors = deque(maxlen=8)

    def read_log():
        time_base = None
        for line in iter(process.stderr.readline, b""):
            line = line.decode("utf-8", errors="replace")
            base = re.search(r"config in time_base:\s*(\d+)/(\d+)", line)
            if base:
                time_base = int(base.group(1)) / int(base.group(2))
            match = re.search(r"\bpts_time:([\d.eE+-]+)", line)
            if "showinfo" in line and match:
                pts = re.search(r"\bpts:\s*(-?\d+)\s+pts_time:", line)
                source_time = (int(pts.group(1)) * time_base if pts and time_base is not None
                               else float(match.group(1)))
                timestamps.put(source_time - metadata["container_start"])
            elif "showinfo" not in line:
                errors.append(line.strip())
        timestamps.put(None)

    reader = threading.Thread(target=read_log, daemon=True)
    reader.start()
    try:
        size = width * height
        with tqdm(total=math.ceil(end - start), desc="Video selection", unit="s",
                  dynamic_ncols=True) as progress:
            while True:
                raw = process.stdout.read(size)
                if not raw:
                    break
                if len(raw) != size:
                    raise RuntimeError("FFmpeg returned an incomplete video proxy frame.")
                try:
                    timestamp = timestamps.get(timeout=30)
                except queue.Empty as error:
                    raise RuntimeError("Timed out waiting for FFmpeg source timestamps.") from error
                if timestamp is None:
                    raise RuntimeError("FFmpeg did not return source timestamps for proxy frames.")
                if start - 0.001 <= timestamp < end:
                    yield timestamp, np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
                    progress.update(max(0, min(progress.total, int(timestamp - start)) - progress.n))
            if process.wait():
                raise RuntimeError("FFmpeg video scan failed: " + " ".join(errors))
            progress.update(progress.total - progress.n)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        reader.join(timeout=5)
        process.stderr.close()


def _instructional_regions(gray):
    """Find broad, bright rectangular surfaces; ambiguous layouts use the full frame.

    Keep entire rectangles, including their dark writing, and all panels at a
    given threshold. A small bright patch alone is not enough to identify a board.
    Prefer the largest supported area so that darker panels are not cropped away.
    """
    import cv2
    import numpy as np

    height, width = gray.shape
    margin = max(2, round(min(height, width) * SELECTION_SETTINGS["region_margin"]))
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    candidates = []
    for threshold in SELECTION_SETTINGS["region_thresholds"]:
        surface = np.uint8(gray > threshold) * 255
        surface = cv2.morphologyEx(surface, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(surface, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (w < width * 0.2 or h < height * 0.4
                    or cv2.contourArea(contour) < w * h * SELECTION_SETTINGS["region_minimum_fill"]):
                continue
            regions.append((max(0, x - margin), max(0, y - margin),
                            min(width, x + w + margin), min(height, y + h + margin)))
        if not regions:
            continue
        mask = np.zeros(gray.shape, dtype=bool)
        for left, top, right, bottom in regions:
            mask[top:bottom, left:right] = True
        # A uniformly bright border indicates a full-frame slide/page, rather
        # than a board surrounded by a room. Do not crop its darker panels.
        if mask.mean() >= 0.95:
            if np.mean(border > threshold) >= 0.95:
                return None
            continue
        span = max(box[2] for box in regions) - min(box[0] for box in regions)
        if (mask.mean() >= SELECTION_SETTINGS["region_minimum_area"]
                and span >= width * SELECTION_SETTINGS["region_minimum_span"]):
            candidates.append((int(mask.sum()), [list(box) for box in sorted(regions)], mask))
    return max(candidates, key=lambda item: item[0])[1:] if candidates else None


def _difference(first, second, mask=None):
    import numpy as np

    difference = second.astype(np.int16) - first.astype(np.int16)
    threshold = SELECTION_SETTINGS["pixel_threshold"]
    added_pixels, removed_pixels = difference < -threshold, difference > threshold
    if mask is not None:
        added_pixels &= mask
        removed_pixels &= mask
    added = float(np.count_nonzero(added_pixels)) / difference.size
    removed = float(np.count_nonzero(removed_pixels)) / difference.size
    return added + removed, added, removed


def _scroll(first, second):
    import cv2
    import numpy as np

    (dx, dy), response = cv2.phaseCorrelate(first.astype(np.float32), second.astype(np.float32))
    if (response >= SELECTION_SETTINGS["scroll_response"]
            and abs(dx) <= first.shape[1] * 0.08 and abs(dy) >= 2):
        return dy
    return 0.0


def _select_proxy_frames(frames, diagnostics=None) -> tuple[list[dict], int]:
    """Retain changes conservatively; only constant-size gray frame state is kept."""
    import cv2

    selected = []
    minimum = SELECTION_SETTINGS["minimum_change_fraction"]
    anchor = previous = None
    quiet_since = None
    before_change = None
    scroll_distance = 0.0
    best_quiet = None
    previous_change = 0.0
    count = 0
    region_frames = 0

    def difference(first, second):
        # Compare the union: a newly exposed panel or a lost region must never
        # be hidden by the other frame's narrower detection.
        mask = first[3][1] | second[3][1] if first[3] is not None and second[3] is not None else None
        return _difference(first[1], second[1], mask)

    def retain(frame, reason):
        nonlocal anchor, scroll_distance
        timestamp, _gray, sharpness, regions = frame
        if selected and timestamp <= selected[-1]["timestamp"]:
            return
        selected.append({"timestamp": timestamp, "reason": reason, "sharpness": sharpness})
        if regions is not None:
            selected[-1]["content_regions"] = regions[0]
        anchor = frame
        scroll_distance = 0.0

    def retain_before_change(current):
        nonlocal before_change
        if before_change is not None:
            candidate, reason = before_change
            if (difference(candidate, current)[0] >= minimum
                    and difference(anchor, candidate)[0] >= minimum):
                retain(candidate, reason)
            before_change = None

    for timestamp, gray in frames:
        count += 1
        regions = _instructional_regions(gray)
        if regions is not None and previous is not None:
            # Geometry is only a hint. When the excluded area settles into a
            # new state, fall back to the whole frame and preserve that state.
            outside = ~regions[1]
            if (_difference(previous[1], gray, outside)[0] < minimum
                    and _difference(anchor[1], gray, outside)[0] >= minimum):
                regions = None
        region_frames += regions is not None
        detail = cv2.Laplacian(gray, cv2.CV_32F)
        sharpness = float(detail[regions[1]].var() if regions is not None else detail.var())
        frame = (timestamp, gray, sharpness, regions)
        if previous is None:
            retain(frame, "initial")
            previous = frame
            continue
        adjacent, added, removed = difference(previous, frame)
        accumulated = difference(anchor, frame)[0]
        scope_changed = (previous[3] is None) != (regions is None)
        if adjacent >= minimum or scope_changed:
            quiet_since = None
            best_quiet = None
        else:
            if quiet_since is None:
                quiet_since = timestamp
            if best_quiet is None or frame[2] >= best_quiet[2]:
                best_quiet = frame

        shift = (_scroll(previous[1], gray)
                 if adjacent >= SELECTION_SETTINGS["settled_change_fraction"] else 0.0)
        erasure = (not shift and removed >= minimum
                   and removed > 1.8 * max(added, minimum / 4))
        replacement = added >= minimum and removed >= minimum and previous_change < minimum
        transition = not shift and (
            adjacent >= SELECTION_SETTINGS["large_change_fraction"] or replacement
        )
        # Defer confirmation until the change settles or the time guard.
        # Lecturer movement must not trigger a new image every half-second.
        if ((erasure or transition)
                and difference(anchor, previous)[0] >= minimum):
            reason = "before_erasure" if erasure else "before_transition"
            if previous_change < SELECTION_SETTINGS["settled_change_fraction"]:
                retain_before_change(previous)
                retain(previous, reason)
                accumulated = difference(anchor, frame)[0]
            elif before_change is None:
                before_change = (previous, reason)

        if shift:
            scroll_distance = (scroll_distance + shift if scroll_distance * shift >= 0
                               else shift)
        if abs(scroll_distance) >= gray.shape[0] * SELECTION_SETTINGS["scroll_viewport_fraction"]:
            retain(frame, "scroll_overlap")
            before_change = None
        elif (quiet_since is not None
              and timestamp - quiet_since >= SELECTION_SETTINGS["settle_seconds"]
              and (accumulated >= SELECTION_SETTINGS["settled_change_fraction"]
                   or before_change is not None)):
            retain_before_change(best_quiet)
            if difference(anchor, best_quiet)[0] >= minimum:
                retain(best_quiet, "settled_change")
        elif (accumulated >= minimum
              and timestamp - anchor[0] + (timestamp - previous[0])
              >= SELECTION_SETTINGS["maximum_change_gap_seconds"]):
            retain_before_change(frame)
            retain(frame, "change_time_guard")
        previous = frame
        previous_change = adjacent

    if previous is not None:
        retain_before_change(previous)
        # The last sample cannot wait for another observation to confirm a new
        # state outside a detected surface.
        if _difference(anchor[1], previous[1])[0] >= minimum:
            retain(previous, "final_change")
    if diagnostics is not None:
        diagnostics.update(region_frames=region_frames, full_frame_frames=count - region_frames)
    return selected, count


def _native_frame(media: Path, metadata: dict, timestamp: float, image: Path) -> None:
    image.parent.mkdir(parents=True, exist_ok=True)
    # showinfo formats decimal PTS with limited precision; a millisecond margin
    # avoids rounding past the intended frame when seeking back to its source.
    temporary = image.with_name(f".{image.stem}.{os.getpid()}.tmp.png")
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
             "-ss", str(max(0.0, timestamp - 0.001)), "-noautorotate", "-i", str(media),
             "-map", f"0:{metadata['stream_index']}", "-frames:v", "1", str(temporary)],
            check=True,
        )
        if not temporary.is_file() or not temporary.stat().st_size:
            raise RuntimeError(f"FFmpeg did not retrieve the selected frame at {timestamp}.")
        temporary.replace(image)
    finally:
        temporary.unlink(missing_ok=True)


@timed_step("frame_selection")
def select_frames(media: Path, job: Path, start: float, end: float, media_hash: str,
                  metadata: dict | None = None) -> dict:
    """Select and cache native source images for a recording-relative interval."""
    media, job = Path(media).resolve(), Path(job).resolve()
    if not all(math.isfinite(value) for value in (start, end)) or not 0 <= start < end:
        raise ValueError("Video interval must have finite bounds with 0 <= start < end.")
    metadata = probe_video(media) if metadata is None else metadata
    if metadata is None or max(start, metadata["start"]) >= min(end, metadata["end"]):
        return {"items": [], "diagnostics": {"proxy_frames": 0, "selected_frames": 0}}
    if metadata.get("rotation", 0) % 360:
        raise ValueError("Rotated video streams are not supported; normalize their orientation first.")
    start, end = max(start, metadata["start"]), min(end, metadata["end"])
    inputs = {"media": media_hash, "start": start, "end": end,
              "metadata": metadata, "settings": SELECTION_SETTINGS}
    cache_path = job / "checkpoints" / "video" / "selection.json"
    cache_hit = cache_path.is_file() and read_json(cache_path).get("fingerprint") == fingerprint(inputs)

    def scan():
        began = time.monotonic()
        diagnostics = {}
        frames, count = _select_proxy_frames(_proxy_frames(media, metadata, start, end), diagnostics)
        return {"frames": frames, "proxy_frames": count, "seconds": time.monotonic() - began,
                **diagnostics}

    selection = checkpoint(job, "video/selection", inputs, scan)
    began = time.monotonic()
    items = []
    retrieved = 0
    with tqdm(selection["frames"], desc="Video frames", unit="frame",
              dynamic_ncols=True, disable=not selection["frames"]) as frames:
        for frame in frames:
            identifier = f"frame_{metadata['stream_index']}_{round(frame['timestamp'] * 1000000):012d}"
            image = job / "frames" / f"{identifier}_{media_hash[:12]}.png"
            if not image.is_file() or not image.stat().st_size:
                _native_frame(media, metadata, frame["timestamp"], image)
                retrieved += 1
            items.append({"id": identifier, "image": str(image), **frame})
    retrieval_seconds = time.monotonic() - began
    items, novelty = filter_frames(job, items) if items else (items, {})
    return {"items": items, "diagnostics": {
        "settings": dict(SELECTION_SETTINGS), "clip_start": start, "clip_end": end,
        "proxy_frames": selection["proxy_frames"], "selected_frames": len(items),
        "scan_seconds": 0.0 if cache_hit else selection["seconds"],
        "frame_retrieval_seconds": retrieval_seconds,
        "retrieved_frames": retrieved, "cache_hit": cache_hit,
        "region_frames": selection["region_frames"],
        "full_frame_frames": selection["full_frame_frames"],
        "novelty": novelty,
        "selection_note": "Board/screen regions when detected; full-frame fallback otherwise. "
                          "Person-aware novelty filtering runs before vision inference.",
    }}
