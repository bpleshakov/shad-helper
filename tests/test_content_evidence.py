from pathlib import Path

from PIL import Image

from shad_helper.extraction import _prepare_visuals
from shad_helper.jobs import evidence_index, sha256_file
from shad_helper.notes import _evidence_image_paths, chapter_context


def test_shared_analysis_keeps_occurrences_and_prepares_one_canonical_crop(tmp_path):
    canonical = tmp_path / "first.png"
    original = tmp_path / "second.png"
    Image.new("RGB", (40, 20), "white").save(canonical)
    Image.new("RGB", (40, 20), "gray").save(original)
    shared = {"image": str(canonical), "analysis_id": "visual_1", "kind": "slide",
              "summary": "x + y", "diagram": "", "issues": [],
              "regions": [{"kind": "formula", "bbox": [0, 0, 0.5, 1]}]}
    visuals, crops = _prepare_visuals([
        {**shared, "id": "visual_1", "timestamp": 10, "original_image": str(canonical)},
        {**shared, "id": "visual_2", "timestamp": 1000, "original_image": str(original)},
    ], tmp_path)

    assert len(crops) == 1
    assert crops[0]["id"] == "visual_1_region_000"
    assert [v["timestamp"] for v in visuals] == [10, 1000]
    assert visuals[1]["original_image"] == "second.png"
    assert visuals[1]["original_sha256"] == sha256_file(original)
    assert visuals[1]["sha256"] == sha256_file(canonical)
    assert visuals[1]["regions"][0]["id"] == "visual_2_region_000"
    assert visuals[1]["regions"][0]["analysis_region_id"] == crops[0]["id"]
    assert visuals[1]["regions"][0]["image"] == visuals[0]["regions"][0]["image"]
    assert _evidence_image_paths(tmp_path, {"visuals": visuals}) == [
        canonical, Path(crops[0]["image"]), original]

    evidence = {"visuals": visuals, "sources": [], "segments": [], "issues": []}
    context = chapter_context(evidence, {"start": 950, "end": 1200})
    assert [v["id"] for v in context["visuals"]] == ["visual_1", "visual_2"]
    assert evidence_index(context)["visual_2_region_000"] == "Видео 00:16:40"


def test_canonical_frame_later_in_input_still_has_correct_crop_coordinates(tmp_path):
    image = tmp_path / "canonical.png"
    earlier = tmp_path / "earlier.png"
    Image.new("RGB", (100, 60), "red").save(image)
    Image.new("RGB", (40, 20), "blue").save(earlier)
    shared = {"image": str(image), "analysis_id": "later", "kind": "whiteboard",
              "regions": [{"kind": "text", "bbox": [0.5, 0, 1, 1]}]}
    visuals, crops = _prepare_visuals([
        {**shared, "id": "earlier", "timestamp": 1, "original_image": str(earlier)},
        {**shared, "id": "later", "timestamp": 2, "original_image": str(image)},
    ], tmp_path)
    with Image.open(crops[0]["image"]) as crop:
        assert crop.size == (50, 60)
        assert crop.getpixel((0, 0)) == (255, 0, 0)
    assert visuals[0]["image"] == "canonical.png"
    assert visuals[0]["original_image"] == "earlier.png"
