import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from shad_helper import (
    extraction,
    local_worker,
    page_pipeline,
    person_segmentation,
    whiteboard_pages,
)
from shad_helper.jobs import read_json, write_json


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    request = {"job": str(tmp_path), "video": str(video), "video_sha256": "video-hash",
               "model": "qwen", "metadata": {"width": 40, "height": 30},
               "start": 0, "end": 10, "vision_mode": "pages", "grouping_version": 1,
               "page_selection_version": 1, "items": []}
    pixels = np.full((30, 40, 3), 255, dtype=np.uint8)
    pixels[3:5, 3:25] = (180, 20, 20)
    decoded = Mock(side_effect=lambda video, timestamp: pixels.copy())
    monkeypatch.setattr(page_pipeline, "read_frame", decoded)
    previews = Mock(return_value=iter([]))
    monkeypatch.setattr(page_pipeline, "iter_previews", previews)

    def select(previews, *, classify, read_frame, **kwargs):
        classify(0)
        read_frame(8)
        return [{"timestamp": 8, "page_id": "page_00000", "page_start": 0, "page_end": 10}], {
            "native_mask_calls": 0, "input_count": 100, "pages": []}

    selected = Mock(side_effect=select)
    monkeypatch.setattr(whiteboard_pages, "select_pages", selected)
    loaded = Mock(return_value=(None, None))
    generated = Mock(return_value=json.dumps({
        "kind": "whiteboard", "summary": "final board", "regions": [], "diagram": "",
        "issues": [], "lecture_surface_bbox": [0, 0, 1, 1], "lecturer_bbox": None,
        "page_id": "invented", "page_start": 100, "page_end": 200,
    }))
    monkeypatch.setattr(local_worker, "_vlm_model", loaded)
    monkeypatch.setattr(local_worker, "_generate", generated)
    monkeypatch.setattr(person_segmentation, "PersonSegmenter", lambda **kwargs: lambda rgb: None)
    return request, selected, previews, decoded, loaded, generated


def test_pages_worker_loads_once_preserves_trusted_ranges_and_resumes(pipeline):
    request, selected, previews, decoded, loaded, generated = pipeline
    result = local_worker.run(request, "vision")["items"]
    assert selected.call_count == previews.call_count == loaded.call_count == 1
    assert generated.call_count == 2  # Bootstrap and one final board, same loaded model.
    assert len(result) == 1
    visual = result[0]
    assert (visual["timestamp"], visual["page_start"], visual["page_end"]) == (8, 0, 10)
    assert visual["page_id"] == "page_00000"
    assert Path(visual["image"]).name == "visual_00000.png"
    job = Path(request["job"])
    bootstrap = read_json(job / "checkpoints/vision/bootstrap_00000.json")["result"]
    assert "page_id" not in bootstrap
    assert Path(bootstrap["image"]) != Path(visual["image"])
    audit = read_json(job / "whiteboard-pages.json")
    assert audit["selected_video_frames"] == 1
    assert audit["bootstrap_generated_calls"] == 1
    assert audit["qwen_generated_calls"] == 2
    before = decoded.call_count
    resumed = local_worker.run(request, "vision")["items"]
    assert resumed == result
    assert selected.call_count == previews.call_count == loaded.call_count == 1
    assert generated.call_count == 2
    assert decoded.call_count == before

    Path(visual["image"]).unlink()
    Path(bootstrap["image"]).unlink()
    assert local_worker.run(request, "vision")["items"] == result
    assert selected.call_count == previews.call_count == 1
    assert generated.call_count == 2
    assert decoded.call_count == before + 2


def test_selector_policy_change_invalidates_selection_but_reuses_inference(pipeline, monkeypatch):
    request, selected, _, _, loaded, generated = pipeline
    local_worker.run(request, "vision")
    monkeypatch.setitem(whiteboard_pages.POLICY, "test_threshold", 42)
    local_worker.run(request, "vision")
    assert selected.call_count == 2
    assert loaded.call_count == 1
    assert generated.call_count == 2


def test_unexpected_final_media_type_falls_back_before_ocr(pipeline, monkeypatch):
    request, _, _, _, _, generated = pipeline
    board = json.loads(generated.return_value)
    generated.side_effect = [json.dumps(board), json.dumps({**board, "kind": "slide"}),
                             json.dumps({**board, "kind": "slide"})]
    scan = Mock(return_value=[{"timestamp": 4, "change": 1, "sharpness": 1}])
    monkeypatch.setattr(page_pipeline, "_scan_video", scan)
    result = local_worker.run(request, "vision")["items"]
    assert scan.call_count == 1
    assert len(result) == 1 and result[0]["timestamp"] == 4
    assert "page_id" not in result[0]
    assert result[0]["issues"]
    assert read_json(Path(request["job"]) / "whiteboard-pages.json")["final_type_fallback_pages"] == [
        "page_00000"]
    local_worker.run(request, "vision")
    assert scan.call_count == 1


def test_pages_ocr_reuses_content_cache_and_updates_page_audit(pipeline):
    request, _, _, _, loaded, generated = pipeline
    job = Path(request["job"])
    write_json(job / "whiteboard-pages.json", {"selected_video_frames": 1})
    path = job / "crop.png"
    Image.new("RGB", (20, 10), "white").save(path)
    generated.return_value = "x + y"
    result = local_worker.run({**request, "model": "ocr", "items": [
        {"id": "crop_1", "kind": "formula", "image": str(path)},
        {"id": "crop_2", "kind": "formula", "image": str(path)}]}, "ocr")
    assert [item["text"] for item in result["items"]] == ["x + y", "x + y"]
    assert loaded.call_count == generated.call_count == 1
    audit = read_json(job / "whiteboard-pages.json")
    assert audit["selected_video_frames"] == 1
    assert audit["ocr_generated_calls"] == 1 and audit["ocr_items"] == 2


@pytest.mark.parametrize("version", [None, True, 1.0, 0, 2])
def test_pages_worker_rejects_unsupported_selector_before_model_loading(pipeline, version):
    request, _, _, _, loaded, _ = pipeline
    with pytest.raises(ValueError, match="page selection version"):
        local_worker.run({**request, "page_selection_version": version}, "vision")
    loaded.assert_not_called()


def test_pages_extraction_does_not_export_state_candidates(tmp_path, monkeypatch):
    video, job = tmp_path / "lecture.mp4", tmp_path / "job"
    video.write_bytes(b"video")
    requests = {}
    monkeypatch.setattr(extraction.shutil, "which", lambda tool: f"/mock/{tool}")
    monkeypatch.setattr(extraction, "probe", lambda path: {
        "width": 40, "height": 30, "duration": 20})
    monkeypatch.setattr(extraction, "import_sources", lambda *args: [])
    monkeypatch.setattr(extraction, "_audio_chunks", lambda *args: [])
    monkeypatch.setattr(person_segmentation, "check_person_segmentation", lambda: None)
    scan, frames = Mock(), Mock()
    monkeypatch.setattr(extraction, "_scan_video", scan)
    monkeypatch.setattr(extraction, "_frames", frames)

    def worker(out, task, request):
        requests[task] = request
        if task == "vision":
            path = out / "frames/final.png"
            path.parent.mkdir(parents=True)
            Image.new("RGB", (40, 30), "white").save(path)
            return {"items": [{"id": "visual_00000", "timestamp": 18, "image": str(path),
                               "kind": "whiteboard", "summary": "", "diagram": "", "issues": [],
                               "regions": [], "page_id": "page_00000", "page_start": 5, "page_end": 20}]}
        return {"items": []}

    monkeypatch.setattr(extraction, "_worker", worker)
    evidence = read_json(extraction.extract(video, [], job, start=5, ocr_model="test-ocr"))
    scan.assert_not_called()
    frames.assert_not_called()
    assert requests["vision"]["video"] == str(video)
    assert (requests["vision"]["start"], requests["vision"]["end"]) == (5, 20)
    assert requests["vision"]["items"] == []
    assert "video" not in requests["ocr"]
    visual = evidence["visuals"][0]
    assert (visual["timestamp"], visual["page_start"], visual["page_end"]) == (18, 5, 20)
    assert visual["page_id"] == "page_00000"
