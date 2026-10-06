import io
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from tqdm import tqdm

from shad_helper import extraction, notes, sparse_pipeline, vision_mode
from shad_helper.jobs import read_json, write_json


def test_sparse_config_budget_defaults_to_clip_and_resume_preserves_it(tmp_path):
    config = vision_mode.resolve_vision_mode(tmp_path, "sparse", clip_seconds=3600)
    assert config == {"vision_mode": "sparse", "grouping_version": 1,
                      "sparse_selection_version": 1, "vision_budget_minutes": 30}
    original = (tmp_path / "extraction-config.json").read_bytes()
    assert vision_mode.resolve_vision_mode(tmp_path, clip_seconds=7200) == config
    assert (tmp_path / "extraction-config.json").read_bytes() == original
    assert vision_mode.resolve_vision_mode(tmp_path, vision_budget_minutes=45)["vision_budget_minutes"] == 45
    with pytest.raises(ValueError, match="new output directory"):
        vision_mode.resolve_vision_mode(tmp_path, "pages")


@pytest.mark.parametrize("budget", [True, 0, -1, float("nan"), float("inf"), "30"])
def test_invalid_budget_does_not_create_job(tmp_path, budget):
    out = tmp_path / "job"
    with pytest.raises(ValueError, match="budget"):
        vision_mode.resolve_vision_mode(out, "sparse", vision_budget_minutes=budget, clip_seconds=3600)
    assert not out.exists()


def test_budget_flag_on_old_modes_is_rejected_without_writes(tmp_path):
    with pytest.raises(ValueError, match="requires sparse"):
        vision_mode.resolve_vision_mode(tmp_path, "content", vision_budget_minutes=30)
    assert not (tmp_path / "extraction-config.json").exists()


def test_deferred_and_failed_visuals_bypass_full_image_ocr(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (20, 20), "white").save(path)
    raw = [{"id": status, "image": str(path), "timestamp": 3, "status": status,
            "regions": [], "issues": ["gap"], "source_id": "source"}
           for status in ["deferred", "failed"]]
    visuals, crops = extraction._prepare_visuals(raw, tmp_path)
    assert crops == []
    assert all(visual["regions"] == [] for visual in visuals)
    assert [visual["status"] for visual in visuals] == ["deferred", "failed"]
    assert all(visual["source_id"] == "source" for visual in visuals)


def test_deferred_page_gets_explicit_note_gap_and_chapter_guidance():
    visual = {"id": "page", "timestamp": 590, "page_id": "page_00000", "page_start": 0,
              "page_end": 600, "status": "deferred", "regions": [], "summary": ""}
    evidence = {"title": "Lecture", "video": {"duration": 600}, "visuals": [visual],
                "segments": [{"id": "speech", "start": 1, "end": 5, "text": "Speech"}],
                "issues": [], "sources": []}
    chapter = {"id": "ch1", "title": "Chapter", "start": 0, "end": 300}
    reply = {"title": "Lecture", "chapters": [{**chapter, "blocks": [
        {"id": "text", "kind": "text", "content": "Speech", "evidence_ids": ["speech"]}]}]}
    result = notes.validate_notes(reply, evidence)
    gap = result["chapters"][0]["blocks"][-1]
    assert gap["kind"] == "uncertain" and gap["evidence_ids"] == ["page"]
    assert "не обработаны" in gap["content"]
    assert notes.validate_notes(result, evidence) == result
    context = notes.chapter_context(evidence, chapter)
    prompt = notes._chapter_prompt(context, {"title": "Lecture"}, [])
    assert "status deferred" in prompt
    assert context["visuals"] == [visual]


def test_sparse_parent_passes_range_budget_and_never_dense_exports(tmp_path, monkeypatch):
    video, out = tmp_path / "video.mp4", tmp_path / "job"
    video.write_bytes(b"video")
    monkeypatch.setattr(extraction.shutil, "which", lambda name: name)
    monkeypatch.setattr(extraction, "probe", lambda path: {"duration": 3600, "width": 40, "height": 30})
    monkeypatch.setattr(extraction, "import_sources", lambda *args: [])
    monkeypatch.setattr(extraction, "_audio_chunks", lambda *args: [])
    scan, frames = Mock(), Mock()
    monkeypatch.setattr(extraction, "_scan_video", scan)
    monkeypatch.setattr(extraction, "_frames", frames)
    requests = {}

    def worker(job, task, request):
        requests[task] = request
        return {"items": []}

    monkeypatch.setattr(extraction, "_worker", worker)
    evidence = read_json(extraction.extract(video, [], out, vision_mode="sparse", ocr_model="test"))
    assert requests["vision"]["vision_budget_minutes"] == 30
    assert requests["vision"]["video"] == str(video)
    assert requests["vision"]["sparse_selection_version"] == 1
    assert "video" not in requests["ocr"]
    scan.assert_not_called()
    frames.assert_not_called()
    assert evidence["visual_processing"]["vision_mode"] == "sparse"
    timings = read_json(out / "extraction-timings.json")
    assert timings["audio_asr_seconds"] >= 0
    assert timings["vision_selection_export_analysis_seconds"] >= 0
    assert timings["ocr_preparation_conversion_analysis_seconds"] >= 0
    assert timings["qwen_cumulative_seconds"] == 0


def test_sparse_manifest_precedes_routing_and_resume_restores_missing_image(tmp_path, monkeypatch):
    from shad_helper import page_pipeline, sparse_selection

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    request = {"job": str(tmp_path), "video": str(video), "video_sha256": "hash",
               "metadata": {"width": 20, "height": 20}, "start": 0, "end": 10,
               "model": "qwen", "items": [], "vision_budget_minutes": 1}
    pixels = np.zeros((20, 20, 3), dtype=np.uint8)
    read = Mock(return_value=pixels)
    monkeypatch.setattr(sparse_pipeline, "read_frame", read)
    monkeypatch.setattr(page_pipeline, "read_frame", read)
    previews = Mock(side_effect=lambda *args, **kwargs: (sample for sample in []))
    monkeypatch.setattr(sparse_pipeline, "iter_previews", previews)

    def select(previews, *, route, **kwargs):
        route(2)
        return [{"timestamp": 8, "page_id": "page_00000", "page_start": 0,
                 "page_end": 10, "selection_role": "primary"}], {"selected_count": 1}

    monkeypatch.setattr(sparse_selection, "select_sparse", select)

    def route(timestamp, image):
        assert read_json(tmp_path / "sparse-selection.json")["stage"] == "provisional"
        return {"kind": "whiteboard"}

    routing = Mock(side_effect=route)
    items, _ = sparse_pipeline.prepare_sparse_images(request, routing, None)
    assert previews.call_count == 2 and routing.call_count == 1
    assert all(call.kwargs["sample_seconds"] == 0.5 for call in previews.call_args_list)
    assert read_json(tmp_path / "sparse-selection.json")["stage"] == "selected"
    before = read.call_count
    again, resumed = sparse_pipeline.prepare_sparse_images(request, routing, None)
    assert again == items and resumed["selection_reused"]
    assert previews.call_count == 2 and routing.call_count == 1 and read.call_count == before
    Path(items[0]["image"]).unlink()
    sparse_pipeline.prepare_sparse_images(request, routing, None)
    assert previews.call_count == 2 and routing.call_count == 1 and read.call_count == before + 1


@pytest.mark.parametrize("failure", [None, "decoder", "selector"])
def test_overview_progress_tracks_video_time_and_closes_on_failure(tmp_path, monkeypatch, failure):
    from shad_helper import page_pipeline, sparse_selection

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    request = {"job": str(tmp_path), "video": str(video), "video_sha256": "hash",
               "metadata": {"width": 20, "height": 20}, "start": 100, "end": 110,
               "model": "qwen", "items": []}
    pixels = np.full((20, 20, 3), 240, dtype=np.uint8)
    pixels[5:15, 5] = 30
    closed, bars = [], []
    output = io.StringIO()

    def previews(*args, **kwargs):
        try:
            for timestamp in (100, 104, 104.5, 109.5):
                yield {"timestamp": timestamp, "rgb": pixels, "sharpness": 10}
                if failure == "decoder" and timestamp == 104.5:
                    raise RuntimeError("decoder failed")
        finally:
            closed.append(True)

    def progress(**kwargs):
        bar = tqdm(**{**kwargs, "file": output, "dynamic_ncols": False,
                      "mininterval": 0, "miniters": 0})
        bar.close = Mock(wraps=bar.close)
        bars.append(bar)
        return bar

    def failed_selector(previews, **kwargs):
        for sample in previews:
            if sample["timestamp"] == 104.5:
                raise RuntimeError("selector failed")

    monkeypatch.setattr(sparse_pipeline, "iter_previews", Mock(side_effect=previews))
    monkeypatch.setattr(sparse_pipeline, "tqdm", progress)
    monkeypatch.setattr(sparse_pipeline, "read_frame", lambda *args: pixels)
    monkeypatch.setattr(page_pipeline, "read_frame", lambda *args: pixels)
    route = Mock(return_value={"kind": "whiteboard"})
    if failure:
        if failure == "selector":
            monkeypatch.setattr(sparse_selection, "select_sparse", failed_selector)
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            sparse_pipeline.prepare_sparse_images(request, route, None)
        assert bars[0].n < 10 and "100%" not in output.getvalue()
        assert closed == [True]
        assert not (tmp_path / "checkpoints/sparse-overview.json").exists()
    else:
        items, _ = sparse_pipeline.prepare_sparse_images(request, route, None)
        assert [item["timestamp"] for item in items] == [109.5]
        assert bars[0].total == bars[0].n == 10
        assert "40%" in output.getvalue() and "45%" in output.getvalue()
        assert "100%" in output.getvalue() and "video-s" in output.getvalue()
        assert closed == [True, True]
        assert sparse_pipeline.prepare_sparse_images(request, route, None)[0] == items
        assert sparse_pipeline.iter_previews.call_count == 2
    assert len(bars) == 1 and bars[0].close.called


def test_sparse_worker_parent_uses_watchdog(tmp_path, monkeypatch):
    from shad_helper import vision_budget

    watch = Mock(return_value={"items": []})
    monkeypatch.setattr(vision_budget, "run_budgeted_worker", watch)
    assert extraction._worker(tmp_path, "vision", {"vision_mode": "sparse"}) == {"items": []}
    assert watch.call_count == 1
    request = watch.call_args.args[2]
    assert request["job"] == str(tmp_path.resolve())
    assert watch.call_args.kwargs["request_path"] == tmp_path / "workers/vision.request.json"


def test_sparse_default_requires_both_quality_and_full_pilot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / ".shad-helper/vision-profile.json"
    write_json(path, {"version": 1, "qualified": True, "sparse_default": False})
    assert vision_mode.resolve_vision_mode(tmp_path / "first", clip_seconds=3600)["vision_mode"] == "pages"
    write_json(path, {"version": 1, "qualified": True, "sparse_default": True})
    assert vision_mode.resolve_vision_mode(tmp_path / "second", clip_seconds=3600)["vision_mode"] == "sparse"
