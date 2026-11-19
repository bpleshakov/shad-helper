"""Recover tall printed figures by joining verified overlapping scroll views."""

import math
import re
from pathlib import Path

import pymupdf

from .illustrations import illustration_index, native_frames
from .jobs import fingerprint, sha256_file, write_json

MAX_FRAME_SECONDS = 90
MAX_FRAMES = 24
MAX_CANDIDATES = 2


def _points(pixels, left, right):
    """Sample actual marks, not the mostly white margins of a notebook page."""
    samples = pixels.samples
    points = []
    for y in range(3, pixels.height, max(3, pixels.height // 160)):
        for x in range(left + 3, right, max(3, (right - left) // 200)):
            offset = (y * pixels.width + x) * 3
            color = samples[offset:offset + 3]
            if max(color) - min(color) >= 15 or 20 < sum(color) // 3 < 220:
                points.append((x, y, *color))
    return points


def _error(points, samples, width, height, shift):
    total = bad = count = 0
    for x, y, red, green, blue in points:
        translated = y - shift
        if 0 <= translated < height:
            offset = (translated * width + x) * 3
            difference = max(abs(red - samples[offset]), abs(green - samples[offset + 1]),
                             abs(blue - samples[offset + 2]))
            total += difference
            bad += difference > 25
            count += 1
    return (total / count, bad / count, count) if count else (255, 1, 0)


def _alignment(reference, other, left, right):
    """Integer translation only; the caller still needs semantic visual verification."""
    points = _points(reference, left, right)
    if len(points) < 200:
        return None
    sampled = points[::max(1, len(points) // 256)]
    samples = other.samples
    static = _error(sampled, samples, other.width, other.height, 0)
    if static[0] <= 8 and static[1] <= 0.08:
        return None  # A changed cursor, or a static frame, adds no source content.
    limit = reference.height * 3 // 5
    ranked = sorted((_error(sampled, samples, other.width, other.height, shift)[0], shift)
                    for shift in range(-limit, limit + 1) if abs(shift) >= 20)
    score, shift = ranked[0]
    if score > 12 or any(error <= score + 1 and abs(offset - shift) > 4
                        for error, offset in ranked[1:]):
        return None
    error, bad, count = _error(points, samples, other.width, other.height, shift)
    if error > 12 or bad > 0.10 or count < max(200, len(points) // 3):
        return None
    return shift, {"mean_max_channel_error": error, "mismatched_fraction": bad,
                   "informative_pixels": count, "overlap_height": reference.height - abs(shift)}


def _native(job, media_hash, visual):
    identifier = visual["id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", identifier):
        raise ValueError("Invalid native frame identity")
    path = job / "frames" / f"{identifier}_{media_hash[:12]}.png"
    if path.resolve() != path:
        raise ValueError("Native frame must stay within its managed path")
    pixels = pymupdf.Pixmap(path)
    if pixels.n != 3:
        raise ValueError("Scroll recovery requires RGB native frames")
    return path, pixels, {"visual_id": identifier, "timestamp": visual["timestamp"],
                          "path": path.relative_to(job).as_posix(),
                          "sha256": sha256_file(path), "size": [pixels.width, pixels.height]}


def _assemble(reference, other, viewport, shift):
    left, _, right, height = viewport
    top, bottom = min(0, shift), height + max(0, shift)
    assembled = pymupdf.Pixmap(pymupdf.csRGB,
                              pymupdf.IRect(left, top, right, bottom), False)
    assembled.copy(reference, pymupdf.IRect(viewport))
    other.set_origin(0, shift)
    extension = ([left, height, right, bottom] if shift > 0 else [left, top, right, 0])
    assembled.copy(other, pymupdf.IRect(extension))
    assembled.set_origin(0, 0)
    return assembled


def validate_stitched_source(job: Path, evidence: dict, metadata: dict, path: Path) -> None:
    """Verify current native sources, alignment, and every assembled pixel."""
    job = Path(job).resolve()
    try:
        provenance = metadata["stitch"]
        observation = illustration_index(evidence)[provenance["illustration_id"]]
        visuals = {item["id"]: item for item in native_frames(job, evidence)}
        sources = provenance["sources"]
        if provenance["version"] != 1 or len(sources) != 2:
            raise ValueError("Invalid scroll source provenance")
        native = [_native(job, evidence["media"]["sha256"], visuals[item["visual_id"]])
                  for item in sources]
        if [item[2] for item in native] != sources:
            raise ValueError("Scroll source no longer matches its native frame")
        reference, other = (item[1] for item in native)
        x0, _, x1, _ = observation["box"]
        viewport = [math.floor(x0 * reference.width), 0,
                    math.ceil(x1 * reference.width), reference.height]
        if (observation["source_kind"] != "printed"
                or sources[0]["visual_id"] != observation["visual_id"]
                or provenance["viewport"] != viewport
                or (other.width, other.height) != (reference.width, reference.height)
                or not 0 < abs(sources[0]["timestamp"] - sources[1]["timestamp"])
                <= MAX_FRAME_SECONDS):
            raise ValueError("Scroll source does not match its observation")
        alignment = _alignment(reference, other, viewport[0], viewport[2])
        if alignment != (provenance["translation_y"], provenance["alignment"]):
            raise ValueError("Scroll source alignment cannot be reproduced")
        expected = _assemble(reference, other, viewport, provenance["translation_y"])
        actual = pymupdf.Pixmap(path)
        if (metadata["visual_id"] != "stitched_" + fingerprint(provenance)[:24]
                or metadata["timestamp"] != observation["timestamp"]
                or metadata["size"] != [expected.width, expected.height]
                or [actual.width, actual.height] != metadata["size"]
                or actual.samples != expected.samples
                or sha256_file(path) != metadata["sha256"]):
            raise ValueError("Scroll source pixels do not match their provenance")
    except (KeyError, TypeError, OSError, RuntimeError, pymupdf.mupdf.FzErrorBase) as error:
        raise ValueError(f"Cannot validate scroll source: {error}") from error


def stitched_sources(job: Path, evidence: dict, observation: dict,
                     reference_path: Path) -> list[tuple[Path, dict]]:
    """Offer pixel-only scroll assemblies, with provenance, for independent crop review.

    Assemblies are candidates, never accepted crops. Only a vertical translation is
    allowed: reference pixels are retained throughout the overlap and the second
    frame supplies only previously absent rows. No interpolation or blending occurs.
    """
    if observation["source_kind"] != "printed":
        return []
    job = Path(job).resolve()
    media_hash = evidence.get("media", {}).get("sha256", "")
    if not re.fullmatch(r"[a-f0-9]{64}", media_hash):
        return []
    path, reference, reference_record = _native(job, media_hash, {
        "id": observation["visual_id"], "timestamp": observation["timestamp"]})
    if Path(reference_path).resolve() != path:
        raise ValueError("Scroll recovery reference differs from its native frame")
    x0, _, x1, _ = observation["box"]
    left, right = math.floor(x0 * reference.width), math.ceil(x1 * reference.width)
    if right - left < 128 or reference.height < 128:
        return []
    candidates = sorted((visual for visual in native_frames(job, evidence)
                         if 0 < abs(visual["timestamp"] - observation["timestamp"])
                         <= MAX_FRAME_SECONDS),
                        key=lambda visual: (abs(visual["timestamp"] - observation["timestamp"]),
                                            visual["timestamp"]))[:MAX_FRAMES]
    aligned = []
    for visual in candidates:
        try:
            _, other, other_record = _native(job, media_hash, visual)
        except (OSError, ValueError, RuntimeError, pymupdf.mupdf.FzErrorBase):
            continue
        if (other.width, other.height) != (reference.width, reference.height):
            continue
        alignment = _alignment(reference, other, left, right)
        if alignment is None:
            continue
        aligned.append((alignment, other, other_record))
    result, shifts = [], []
    for (shift, metrics), other, other_record in sorted(
            aligned, key=lambda item: abs(item[0][0]), reverse=True):
        if any(abs(shift - previous) < 20 for previous in shifts):
            continue
        viewport = [left, 0, right, reference.height]
        provenance = {"version": 1, "illustration_id": observation["id"],
                      "sources": [reference_record, other_record],
                      "viewport": viewport, "translation_y": shift, "alignment": metrics}
        identity = fingerprint(provenance)
        folder = job / "illustration-stitches" / identity
        if folder.resolve() != folder:
            raise ValueError("Scroll recovery path must not contain symlinks")
        folder.mkdir(parents=True, exist_ok=True)
        assembled = _assemble(reference, other, viewport, shift)
        output = folder / "frame.png"
        assembled.save(output)
        metadata = {"visual_id": "stitched_" + identity[:24],
                    "timestamp": observation["timestamp"], "sha256": sha256_file(output),
                    "size": [assembled.width, assembled.height], "stitch": provenance}
        write_json(folder / "source.json", metadata)
        result.append((output, metadata))
        shifts.append(shift)
        if len(result) == MAX_CANDIDATES:
            break
    return result
