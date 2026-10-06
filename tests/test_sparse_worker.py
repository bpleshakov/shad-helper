import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from shad_helper import sparse_worker
from shad_helper.jobs import read_json, write_json


def response(raw, finish="stop"):
    return SimpleNamespace(text=json.dumps(raw) if isinstance(raw, dict) else raw,
                           finish_reason=finish, prompt_tokens=100, generation_tokens=50,
                           prompt_tps=200, generation_tps=25, peak_memory=1.5)


def analysis(**changes):
    return {"kind": "whiteboard", "summary": "board", "regions": [
        {"kind": "text", "bbox": [0.1, 0.1, 0.9, 0.9]}], "diagram": None,
        "issues": [], **changes}


@pytest.fixture
def worker(tmp_path, monkeypatch):
    from shad_helper import person_segmentation

    image = tmp_path / "frame.png"
    Image.new("RGB", (40, 30), "white").save(image)
    item = {"id": "v1", "image": str(image), "timestamp": 100,
            "page_id": "p1", "page_start": 0, "page_end": 200,
            "selection_issues": []}
    request = {"job": str(tmp_path), "model": "large", "start": 0, "end": 3600, "items": [],
               "sparse_selection_version": 1}
    module = ModuleType("shad_helper.sparse_pipeline")
    prepared = Mock(return_value=([item], {}))
    module.prepare_sparse_images = prepared
    monkeypatch.setitem(sys.modules, "shad_helper.sparse_pipeline", module)
    monkeypatch.setattr(person_segmentation, "PersonSegmenter", lambda **kwargs: None)
    loaded = Mock(return_value=(None, None))
    streamed = Mock(side_effect=lambda *args: iter([response(analysis())]))
    monkeypatch.setattr(sparse_worker, "_load", loaded)
    monkeypatch.setattr(sparse_worker, "_stream", streamed)
    return request, item, prepared, loaded, streamed


def test_success_telemetry_trusted_provenance_and_zero_cost_pixel_resume(worker):
    request, item, prepared, loaded, streamed = worker
    streamed.side_effect = lambda *args: iter([response(analysis(page_id="invented", page_start=999))])
    first = sparse_worker.run_sparse(request)
    assert first["items"][0]["page_id"] == "p1"
    assert first["items"][0]["page_start"] == 0
    assert first["items"][0]["status"] == "processed"
    assert first["audit"]["qwen_calls"][-1]["generation_tokens"] == 50
    before = read_json(prepared.call_args[0][0]["job"] + "/sparse-qwen-budget.json")["used_seconds"]
    # A different occurrence with identical pixels reuses the content result.
    second_item = {**item, "id": "v2", "timestamp": 250, "page_id": "p2", "page_start": 200}
    prepared.return_value = ([second_item], {})
    second = sparse_worker.run_sparse(request)
    assert loaded.call_count == streamed.call_count == 1
    assert second["items"][0]["id"] == "v2"
    assert second["items"][0]["page_id"] == "p2"
    assert second["audit"]["qwen_generated_calls"] == 0
    assert second["audit"]["qwen_budget"]["used_seconds"] == before


def test_exhausted_job_runs_selection_but_never_loads_or_invents_ocr(worker):
    request, _, prepared, loaded, streamed = worker
    write_json(request["job"] + "/sparse-qwen-budget.json",
               {"version": 1, "limit_seconds": 10, "used_seconds": 10,
                "active_started": None, "active_pid": None, "phase": None})
    result = sparse_worker.run_sparse(request)
    assert prepared.call_count == 1
    assert loaded.call_count == streamed.call_count == 0
    item = result["items"][0]
    assert item["status"] == "deferred"
    assert item["summary"] == ""
    assert item["regions"] == []
    assert result["audit"]["budget_exhausted"] is True


@pytest.mark.parametrize("output,finish", [("unfinished", "stop"), (analysis(), "length")])
def test_invalid_or_truncated_outputs_are_failed_and_not_cached(worker, output, finish):
    request, _, _, loaded, streamed = worker
    streamed.side_effect = lambda *args: iter([response(output, finish)])
    result = sparse_worker.run_sparse(request)
    assert result["items"][0]["status"] == "failed"
    assert result["items"][0]["regions"] == []
    assert not list((Path(request["job"]) / "checkpoints/sparse-content").glob("*.json"))
    sparse_worker.run_sparse(request)
    assert streamed.call_count == loaded.call_count == 2


def test_selection_warning_does_not_trigger_large_model_fallback(worker):
    request, item, prepared, loaded, _ = worker
    request["primary_model"] = "small"
    prepared.return_value = ([{**item, "selection_issues": ["selection uncertain"]}], {})
    result = sparse_worker.run_sparse(request)
    assert loaded.call_args_list[0].args == ("small",)
    assert loaded.call_count == 1
    assert result["items"][0]["issues"] == ["selection uncertain"]


def test_failed_primary_gets_single_larger_model_batch_after_temporal_coverage(worker):
    request, item, prepared, loaded, streamed = worker
    request["primary_model"] = "small"
    other = {**item, "id": "v2", "timestamp": 700}
    other_image = Path(request["job"]) / "other.png"
    Image.new("RGB", (40, 30), "red").save(other_image)
    other["image"] = str(other_image)
    prepared.return_value = ([item, other], {})
    streamed.side_effect = [iter([response("bad")]), iter([response(analysis())]),
                            iter([response(analysis())])]
    result = sparse_worker.run_sparse(request)
    assert [call.args[0] for call in loaded.call_args_list] == ["small", "large"]
    assert streamed.call_count == 3
    assert [item["status"] for item in result["items"]] == ["processed", "processed"]
    assert [call["id"] for call in result["audit"]["qwen_calls"] if call.get("phase") != "model_load"] == [
        "v1", "v2", "v1"]


def test_round_robin_covers_late_video_before_second_early_frame():
    items = [{"id": "early1", "timestamp": 10}, {"id": "early2", "timestamp": 20},
             {"id": "middle", "timestamp": 700}, {"id": "late", "timestamp": 2000}]
    assert [item["id"] for item in sparse_worker._ordered(items)] == [
        "early1", "middle", "late", "early2"]


def test_routing_is_compact_cached_and_uses_low_image_pixel_budget(worker):
    request, item, prepared, _, streamed = worker
    routed = {"kind": "whiteboard", "lecture_surface_bbox": [0, 0, 1, 1],
              "lecturer_bbox": None, "issues": []}
    streamed.side_effect = [iter([response(routed)]), iter([response(analysis())])]

    def prepare(request, route, segmenter):
        assert route(100, item["image"]) == routed
        return [item], {}

    prepared.side_effect = prepare
    sparse_worker.run_sparse(request)
    settings = streamed.call_args_list[0].args[-1]
    assert settings["max_tokens"] == 128
    assert settings["max_pixels"] == 262144
    assert streamed.call_args_list[1].args[-1]["max_tokens"] == 1024


def test_only_qualified_model_profile_changes_the_primary(worker):
    request, _, _, loaded, _ = worker
    path = request["job"] + "/sparse-model-profile.json"
    write_json(path, {"version": 1, "qualified": False, "primary_model": "small"})
    sparse_worker.run_sparse(request)
    assert loaded.call_args.args == ("large",)
    write_json(path, {"version": 1, "qualified": True, "primary_model": "small"})
    sparse_worker.run_sparse(request)
    assert loaded.call_args.args == ("small",)


def test_exhausted_budget_still_allows_successful_cache_hits(worker):
    request, _, _, loaded, streamed = worker
    sparse_worker.run_sparse(request)
    path = Path(request["job"]) / "sparse-qwen-budget.json"
    state = read_json(path)
    state["used_seconds"] = state["limit_seconds"]
    write_json(path, state)
    result = sparse_worker.run_sparse(request)
    assert result["items"][0]["status"] == "processed"
    assert loaded.call_count == streamed.call_count == 1
    assert result["audit"]["qwen_cache_hits"] == 1


def test_stream_deadline_closes_generator_and_does_not_cache_partial_output(worker, monkeypatch):
    from shad_helper import vision_budget

    request, _, _, _, streamed = worker
    request["vision_budget_minutes"] = 0.01
    clock = [100.0]
    monkeypatch.setattr(vision_budget.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(vision_budget.time, "time", lambda: clock[0])
    closed = []

    def stream(*args):
        try:
            yield response('{"kind":', None)
            clock[0] += 0.7
            yield response('"whiteboard"}', "stop")
        finally:
            closed.append(True)

    streamed.side_effect = stream
    result = sparse_worker.run_sparse(request)
    assert closed == [True]
    assert result["items"][0]["status"] == "deferred"
    assert not list((Path(request["job"]) / "checkpoints/sparse-content").glob("*.json"))
    assert result["audit"]["qwen_budget"]["used_seconds"] == pytest.approx(0.7)


def test_cancelled_escalation_does_not_claim_ambiguous_primary_is_complete(worker, monkeypatch):
    from shad_helper import vision_budget

    request, _, _, loaded, streamed = worker
    request.update(primary_model="small", vision_budget_minutes=0.01)
    clock = [100.0]
    monkeypatch.setattr(vision_budget.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(vision_budget.time, "time", lambda: clock[0])

    def load(model):
        if model == "large":
            clock[0] += 0.7
        return None, None

    loaded.side_effect = load
    streamed.side_effect = lambda *args: iter([response(analysis(issues=["unreadable formula"]))])
    result = sparse_worker.run_sparse(request)
    assert result["items"][0]["status"] == "deferred"
    assert result["items"][0]["regions"] == []
    assert streamed.call_count == 1


def test_native_unavailable_uses_cheap_selection_without_crashing(worker, monkeypatch):
    from shad_helper import person_segmentation

    request, _, prepared, _, _ = worker
    monkeypatch.setattr(person_segmentation, "PersonSegmenter", Mock(side_effect=RuntimeError("not installed")))
    result = sparse_worker.run_sparse(request)
    assert prepared.call_args.args[2] is None
    assert result["audit"]["native_segmentation_issue"] == "not installed"


@pytest.mark.parametrize("version", [None, True, 2])
def test_unsupported_version_is_rejected_before_model_loading(worker, version):
    request, _, prepared, loaded, _ = worker
    request["sparse_selection_version"] = version
    with pytest.raises(ValueError, match="Unsupported"):
        sparse_worker.run_sparse(request)
    loaded.assert_not_called()
    prepared.assert_not_called()


def test_complementary_views_are_processed_after_primary_coverage(worker):
    request, item, prepared, _, streamed = worker
    complement = {**item, "id": "c1", "timestamp": 90, "selection_role": "complementary"}
    late = {**item, "id": "late", "timestamp": 900}
    for record, color in [(complement, "red"), (late, "blue")]:
        image = Path(request["job"]) / f"{record['id']}.png"
        Image.new("RGB", (40, 30), color).save(image)
        record["image"] = str(image)
    prepared.return_value = ([item, complement, late], {})
    result = sparse_worker.run_sparse(request)
    calls = [call["id"] for call in result["audit"]["qwen_calls"] if call.get("phase") != "model_load"]
    assert calls == ["v1", "late", "c1"]
    assert streamed.call_count == 3


def test_workspace_profile_is_used_when_job_profile_is_absent(worker, monkeypatch):
    request, _, prepared, loaded, _ = worker
    monkeypatch.chdir(request["job"])
    write_json(Path(".shad-helper/vision-profile.json"),
               {"version": 1, "qualified": True, "primary_model": "qualified-small",
                "benchmark_fingerprint": "reviewed-holdout", "sparse_default": False})
    result = sparse_worker.run_sparse(request)
    assert loaded.call_args.args == ("qualified-small",)
    assert prepared.call_args.args[0]["primary_model"] == "qualified-small"
    assert read_json(Path("sparse-model-profile.json"))["qualified"] is True
    assert result["audit"]["benchmark_fingerprint"] == "reviewed-holdout"


def test_workspace_profile_changes_do_not_change_resumed_baseline_job(worker, monkeypatch):
    request, _, _, loaded, _ = worker
    monkeypatch.chdir(request["job"])
    sparse_worker.run_sparse(request)
    profile = read_json(Path("sparse-model-profile.json"))
    assert profile["primary_model"] == "large" and profile["qualified"] is False
    write_json(Path(".shad-helper/vision-profile.json"),
               {"version": 1, "qualified": True, "primary_model": "small"})
    result = sparse_worker.run_sparse(request)
    assert result["audit"]["primary_model"] == "large"
    assert loaded.call_count == 1


def test_model_cannot_supply_occurrence_provenance_or_crop_path_ids(worker):
    request, _, _, _, streamed = worker
    extra = {"analysis_id": "../../malicious", "source_id": "invented-source",
             "original_image": "/private/forged.png", "analysis_model": "invented-model",
             "sha256": "forged", "status": "deferred", "page_id": "invented-page"}
    streamed.side_effect = lambda *args: iter([response(analysis(**extra))])
    first = sparse_worker.run_sparse(request)["items"][0]
    assert first["id"] == "v1" and first["page_id"] == "p1" and first["status"] == "processed"
    assert first["analysis_model"] == "large"
    assert first["sha256"] != "forged"
    assert not ({"analysis_id", "source_id", "original_image"} & first.keys())
    # Cached raw replies receive the same whitelist on resume.
    assert sparse_worker.run_sparse(request)["items"][0] == first
