import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image, PngImagePlugin

from shad_helper import local_worker
from shad_helper.jobs import read_json, write_json


def image_item(tmp_path, identifier, *, color="white", kind=None, metadata=False):
    path = tmp_path / f"{identifier}.png"
    options = {}
    if metadata:
        info = PngImagePlugin.PngInfo()
        info.add_text("description", identifier)
        options["pnginfo"] = info
    Image.new("RGB", (40, 30), color).save(path, **options)
    return {"id": identifier, "image": str(path), "timestamp": int(identifier[-1]),
            **({"kind": kind} if kind else {})}


@pytest.fixture
def model(monkeypatch):
    loaded = Mock(return_value=(None, None))
    generate = Mock(return_value="recognized")
    monkeypatch.setattr(local_worker, "_vlm_model", loaded)
    monkeypatch.setattr(local_worker, "_generate", generate)
    return loaded, generate


def test_ocr_reuses_decoded_pixels_but_keeps_occurrence_ids(tmp_path, model):
    loaded, generate = model
    first = image_item(tmp_path, "crop1", kind="formula")
    second = image_item(tmp_path, "crop2", kind="formula", metadata=True)
    assert Path(first["image"]).read_bytes() != Path(second["image"]).read_bytes()
    request = {"job": str(tmp_path), "model": "test-ocr", "vision_mode": "content",
               "items": [first, second]}
    result = local_worker.run(request, "ocr")
    assert result["items"] == [{"id": "crop1", "text": "recognized"},
                               {"id": "crop2", "text": "recognized"}]
    assert loaded.call_count == generate.call_count == 1
    assert read_json(tmp_path / "raw/ocr/crop2.json")["raw"] == "recognized"
    assert read_json(tmp_path / "ocr-reuse.json")["reused_results"] == 1

    local_worker.run({**request, "model": "changed-model"}, "ocr")
    assert generate.call_count == 2
    local_worker.run({**request, "items": [{**second, "kind": "text"}]}, "ocr")
    assert generate.call_count == 3


def test_empty_ocr_is_not_shared_or_cached_on_resume(tmp_path, model):
    _, generate = model
    generate.return_value = "  "
    request = {"job": str(tmp_path), "model": "ocr", "vision_mode": "content",
               "items": [image_item(tmp_path, "crop1", kind="text"),
                         image_item(tmp_path, "crop2", kind="text")]}
    local_worker.run(request, "ocr")
    assert generate.call_count == 2
    local_worker.run(request, "ocr")
    assert generate.call_count == 4


def test_ocr_outer_checkpoint_invalidates_prompt_and_generation_settings(tmp_path, model, monkeypatch):
    _, generate = model
    request = {"job": str(tmp_path), "model": "ocr", "vision_mode": "content",
               "items": [image_item(tmp_path, "crop1", kind="formula")]}
    local_worker.run(request, "ocr")
    monkeypatch.setitem(local_worker.OCR_PROMPTS, "formula", "New formula instruction")
    local_worker.run(request, "ocr")
    assert generate.call_count == 2
    assert generate.call_args.args[3] == "New formula instruction"
    monkeypatch.setitem(local_worker.GENERATION_SETTINGS, "max_tokens", 2048)
    local_worker.run(request, "ocr")
    assert generate.call_count == 3
    assert generate.call_args.args[4] == 2048


def test_ocr_exif_orientation_changes_effective_pixels(tmp_path, model):
    _, generate = model
    rgb = Image.new("RGB", (40, 30), "white")
    rgb.putpixel((0, 0), (255, 0, 0))
    plain, rotated = tmp_path / "plain.png", tmp_path / "rotated.png"
    rgb.save(plain)
    exif = Image.Exif()
    exif[274] = 3
    rgb.save(rotated, exif=exif)
    local_worker.run({"job": str(tmp_path), "model": "ocr", "vision_mode": "content",
                      "items": [{"id": "plain", "kind": "text", "image": str(plain)},
                                {"id": "rotated", "kind": "text", "image": str(rotated)}]}, "ocr")
    assert generate.call_count == 2


def test_invalid_shared_ocr_cache_is_regenerated(tmp_path, model):
    _, generate = model
    request = {"job": str(tmp_path), "model": "ocr", "vision_mode": "content",
               "items": [image_item(tmp_path, "crop1", kind="text")]}
    local_worker.run(request, "ocr")
    cache = next((tmp_path / "checkpoints/ocr_content").glob("*.json"))
    data = read_json(cache)
    data["result"] = {"raw": 123}
    write_json(cache, data)
    result = local_worker.run({**request, "items": [image_item(tmp_path, "crop2", kind="text")]}, "ocr")
    assert generate.call_count == 2
    assert result["items"][0]["text"] == "recognized"


def test_existing_blank_occurrence_ocr_cache_is_retried(tmp_path, model):
    _, generate = model
    request = {"job": str(tmp_path), "model": "ocr", "vision_mode": "content",
               "items": [image_item(tmp_path, "crop1", kind="text")]}
    local_worker.run(request, "ocr")
    cache = tmp_path / "checkpoints/ocr/crop1.json"
    data = read_json(cache)
    data["result"]["text"] = ""
    write_json(cache, data)
    # Remove the shared success so the retried occurrence needs real recognition.
    next((tmp_path / "checkpoints/ocr_content").glob("*.json")).unlink()
    result = local_worker.run(request, "ocr")
    assert generate.call_count == 2
    assert result["items"][0]["text"] == "recognized"


def test_content_vision_reuses_analysis_and_retains_loaded_model(tmp_path, model, monkeypatch):
    from shad_helper import person_segmentation

    loaded, generate = model
    generate.return_value = json.dumps({"kind": "slide", "summary": "slide", "diagram": "",
                                       "regions": [], "issues": [],
                                       "lecture_surface_bbox": [0, 0, 1, 1], "lecturer_bbox": None})
    monkeypatch.setattr(person_segmentation, "PersonSegmenter", lambda: lambda pixels: None)
    first = image_item(tmp_path, "frame1")
    second = image_item(tmp_path, "frame2", metadata=True)
    third = image_item(tmp_path, "frame3", color="black")
    request = {"job": str(tmp_path), "model": "vision", "vision_mode": "content",
               "grouping_version": 1, "items": [first, second, third]}
    visuals = local_worker.run(request, "vision")["items"]
    assert loaded.call_count == 1
    assert generate.call_count == 2
    assert visuals[1]["image"] == first["image"]
    assert visuals[1]["original_image"] == second["image"]
    assert visuals[1]["timestamp"] == 2
    assert visuals[2]["image"] == third["image"]
    assert local_worker.CONTENT_VISION_PROMPT in generate.call_args.args
    assert read_json(tmp_path / "visual-grouping.json")["generated_calls"] == 2

    local_worker.run(request, "vision")
    assert loaded.call_count == 1
    assert generate.call_count == 2


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [False, 0, 1, 1], [0, 0, float("inf"), 1],
                                  [0, 0, 1.1, 1], "board"])
def test_surface_hints_are_strictly_validated(box):
    with pytest.raises(ValueError):
        local_worker.validate_visual_result({"kind": "whiteboard", "summary": "", "diagram": "",
                                            "regions": [], "issues": [], "lecture_surface_bbox": box})
