import random
from copy import deepcopy

import pymupdf
import pytest

from shad_helper.illustration_stitching import stitched_sources, validate_stitched_source
from shad_helper.jobs import sha256_file


@pytest.fixture
def scrolling(tmp_path):
    job = tmp_path / "job"
    (job / "frames").mkdir(parents=True)
    page = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 256, 440), False)
    rng = random.Random(41)
    colors = {(x, y): tuple(rng.randrange(30, 220) for _ in range(3))
              for x in range(22) for y in range(34)}
    for y in range(page.height):
        for x in range(page.width):
            page.set_pixel(x, y, colors[x // 12, y // 13])
    observation = {"id": "v01_ill01", "source_kind": "printed", "visual_id": "v01",
                   "timestamp": 100, "box": [0.125, 0.2, 1, 0.95],
                   "description": "Матрица с подписями"}
    evidence = {"media": {"sha256": "a" * 64}, "visuals": []}

    def frame(identifier, offset, timestamp):
        pixels = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, offset, 256, offset + 220), False)
        pixels.copy(page, pixels.irect)
        pixels.set_origin(0, 0)
        path = job / "frames" / f"{identifier}_{'a' * 12}.png"
        pixels.save(path)
        evidence["visuals"].append({"id": identifier, "timestamp": timestamp, "status": "complete",
                                    "illustrations": [observation] if identifier == "v01" else []})
        return path

    return job, evidence, observation, page, frame


@pytest.mark.parametrize("shift", [40, -40])
def test_recovery_joins_only_missing_native_rows_and_verifies_provenance(scrolling, shift):
    job, evidence, observation, page, frame = scrolling
    first = max(0, -shift)
    reference = frame("v01", first, 100)
    other = frame("v02", first + shift, 101)
    original = [path.read_bytes() for path in (reference, other)]
    [(path, metadata)] = stitched_sources(job, evidence, observation, reference)
    pixels = pymupdf.Pixmap(path)
    expected = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(32, 0, 256, 260), False)
    expected.copy(page, expected.irect)
    assert pixels.samples == expected.samples
    assert metadata["stitch"]["translation_y"] == shift
    assert metadata["stitch"]["sources"][1]["visual_id"] == "v02"
    assert metadata["size"] == [224, 260]
    assert metadata["sha256"] == sha256_file(path)
    validate_stitched_source(job, evidence, metadata, path)
    assert [path.read_bytes() for path in (reference, other)] == original


@pytest.mark.parametrize("kind", ["blank", "static", "different_matrix", "scale"])
def test_rejects_uninformative_static_or_nonmatching_views(scrolling, kind):
    job, evidence, observation, _, frame = scrolling
    reference = frame("v01", 0, 100)
    other = frame("v02", 40 if kind != "static" else 0, 101)
    pixels = pymupdf.Pixmap(other)
    if kind == "blank":
        pixels.clear_with(255)
        pixels.save(reference)
        pixels.save(other)
    elif kind == "different_matrix":
        for y in range(140):
            for x in range(32, 256):
                red, green, blue = pixels.pixel(x, y)
                pixels.set_pixel(x, y, (255 - red, 255 - green, 255 - blue))
        pixels.save(other)
    elif kind == "scale":
        resized = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 256, 220), False)
        for y in range(220):
            for x in range(256):
                resized.set_pixel(x, y, pixels.pixel(x * 3 // 4, y * 3 // 4))
        resized.save(other)
    assert stitched_sources(job, evidence, observation, reference) == []


def test_tolerates_small_compression_noise_without_changing_reference_pixels(scrolling):
    job, evidence, observation, _, frame = scrolling
    reference = frame("v01", 0, 100)
    other = frame("v02", 40, 101)
    pixels = pymupdf.Pixmap(other)
    for y in range(220):
        for x in range(256):
            pixels.set_pixel(x, y, tuple(value + 2 for value in pixels.pixel(x, y)))
    pixels.save(other)
    [(path, metadata)] = stitched_sources(job, evidence, observation, reference)
    assert metadata["stitch"]["translation_y"] == 40
    actual = pymupdf.Pixmap(path)
    original = pymupdf.Pixmap(reference)
    for y in (0, 70, 219):
        for x in (0, 110, 223):
            assert actual.pixel(x, y) == original.pixel(x + 32, y)
    validate_stitched_source(job, evidence, metadata, path)


@pytest.mark.parametrize("tampering", ["source", "pixels", "viewport", "shift", "evidence", "hash"])
def test_rejects_changed_sources_or_unreproducible_provenance(scrolling, tampering):
    job, evidence, observation, _, frame = scrolling
    reference = frame("v01", 0, 100)
    other = frame("v02", 40, 101)
    [(path, metadata)] = stitched_sources(job, evidence, observation, reference)
    if tampering in {"source", "pixels"}:
        target = other if tampering == "source" else path
        pixels = pymupdf.Pixmap(target)
        pixels.set_pixel(10, 10, (0, 0, 0))
        pixels.save(target)
        if tampering == "pixels":
            metadata["sha256"] = sha256_file(path)  # Hash alone cannot attest correct assembly.
    elif tampering == "viewport":
        metadata["stitch"]["viewport"][0] += 1
    elif tampering == "shift":
        metadata["stitch"]["translation_y"] += 1
    elif tampering == "evidence":
        evidence["visuals"].pop()
    else:
        metadata["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        validate_stitched_source(job, evidence, metadata, path)


def test_prefers_larger_extensions_and_deduplicates_similar_translations(scrolling):
    job, evidence, observation, _, frame = scrolling
    reference = frame("v01", 0, 100)
    frame("v02", 40, 101)
    frame("v03", 41, 102)
    frame("v04", 100, 103)
    results = stitched_sources(job, evidence, observation, reference)
    assert [item[1]["stitch"]["translation_y"] for item in results] == [100, 41]


def test_does_not_join_drawings_distant_frames_or_wrong_reference(scrolling):
    job, evidence, observation, _, frame = scrolling
    reference = frame("v01", 0, 100)
    other = frame("v02", 40, 191)
    assert stitched_sources(job, evidence, observation, reference) == []
    drawn = deepcopy(observation)
    drawn["source_kind"] = "drawn"
    assert stitched_sources(job, evidence, drawn, reference) == []
    with pytest.raises(ValueError, match="reference"):
        stitched_sources(job, evidence, observation, other)
