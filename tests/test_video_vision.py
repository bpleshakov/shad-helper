"""Native video batches preserve individual source frames and resumable evidence."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from shad_helper import video_vision, vision_worker

CONTENT = {"kind": "whiteboard", "text": "Определение", "formulas": [r"x_i = \theta"],
           "diagrams": [], "illustrations": [], "unreadable": []}


def response(indices, **metrics):
    return json.dumps({"observations": [{"frame_indices": indices, **CONTENT}]}), {
        "finish_reason": "stop", "generation_tokens": 100, **metrics,
    }


@pytest.fixture
def worker(tmp_path, monkeypatch):
    items = []
    for index in range(5):
        image = tmp_path / f"frame-{index}.png"
        Image.new("RGB", (320, 180), (index * 30, 0, 0)).save(image)
        items.append({"id": f"v{index}", "timestamp": 100 + index, "image": str(image)})
    runtime = Mock()
    runtime.infer_video.side_effect = lambda items, limit, **kwargs: response(list(range(len(items))))
    runtime.infer.return_value = json.dumps(CONTENT), {"finish_reason": "stop"}
    loader = Mock(return_value=runtime)
    monkeypatch.setattr(vision_worker, "_load_runtime", loader)
    monkeypatch.setattr(vision_worker, "_versions", lambda: {"mlx-vlm": "test"})
    request = {"job": str(tmp_path / "job"), "model": "test/model", "revision": "abc",
               "mode": "video", "items": items}
    return request, runtime, loader


def test_groups_bound_frame_count_time_span_and_geometry(worker):
    request, _, _ = worker
    items = request["items"]
    assert [len(batch) for batch in video_vision.group_frames(items)] == [4, 1]
    items[2]["timestamp"] = 120.1
    items[3]["timestamp"] = 121
    items[4]["timestamp"] = 122
    Image.new("RGB", (640, 360)).save(items[4]["image"])
    assert video_vision.group_frames(items) == [items[:2], items[2:4], items[4:]]
    assert video_vision.group_frames([]) == []


def test_native_video_input_keeps_pixels_and_irregular_source_timestamps(worker, monkeypatch):
    request, _, _ = worker
    items = request["items"][:2]
    items[1]["timestamp"] = 117.123456
    prepared = Mock(return_value={
        "video_grid_thw": np.array([[2, 12, 20]]), "input_ids": np.array([[7, 7]]),
        "pixel_values_videos": "video tensors", "attention_mask": "mask",
    })
    generated = Mock(return_value=SimpleNamespace(
        text="result", prompt_tokens=10, generation_tokens=20, prompt_tps=1,
        generation_tps=2, peak_memory=3, finish_reason="stop"))
    template = Mock(return_value="video prompt")
    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace(generate=generated))
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", SimpleNamespace(apply_chat_template=template))
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", SimpleNamespace(
        prepare_inputs=prepared, should_add_special_tokens=lambda *args: False))
    processor = SimpleNamespace(video_processor=SimpleNamespace(temporal_patch_size=2, patch_size=16))
    runtime = SimpleNamespace(processor=processor, model=SimpleNamespace(config=SimpleNamespace(
        model_type="qwen3_5_moe", video_token_index=7)))
    raw, metrics = video_vision.infer_video(runtime, items, 4096)
    assert raw == "result"
    inputs = prepared.call_args.kwargs
    assert "images" not in inputs
    frames, = inputs["videos"]
    metadata, = inputs["video_metadata"]
    times = [tick / metadata["fps"] for tick in metadata["frames_indices"]]
    assert frames.shape == (4, 3, 180, 320)
    assert times == [100, 100, 117.123456, 117.123456]
    for index, frame in enumerate(frames):
        with Image.open(items[index // 2]["image"]) as image:
            assert np.array_equal(frame.transpose(1, 2, 0), np.asarray(image))
    assert processor.video_processor.max_pixels == 4 * vision_worker.MAX_PIXELS
    assert template.call_args.kwargs["num_images"] == 0
    assert metrics["video_tokens"] == 2
    assert generated.call_args.kwargs["pixel_values_videos"] == "video tensors"
    assert generated.call_args.kwargs["mask"] == "mask"


def test_parser_expands_groups_in_source_order_with_independent_content():
    raw = json.dumps({"observations": [
        {"frame_indices": [2], **CONTENT, "text": "Новое"},
        {"frame_indices": [0, 1], **CONTENT},
    ]})
    contents = video_vision.parse_video_output(f"```json\n{raw}\n```", 3)
    assert [content["text"] for content in contents] == ["Определение", "Определение", "Новое"]
    contents[0]["formulas"].append("new")
    assert contents[1]["formulas"] == CONTENT["formulas"]


@pytest.mark.parametrize("indices", [[], [0], [0, 0], [0, 2], [-1, 0], [False, 1], [0.0, 1]])
def test_parser_requires_coverage_and_valid_unique_indices_per_observation(indices):
    with pytest.raises(ValueError):
        video_vision.parse_video_output(response(indices)[0], 2)


def test_parser_reuses_strict_visual_schema():
    observation = {"frame_indices": [0], **CONTENT, "formulas": ["\theta"]}
    with pytest.raises(ValueError, match="escape LaTeX"):
        video_vision.parse_video_output(json.dumps({"observations": [observation]}), 1)
    with pytest.raises(ValueError):
        video_vision.parse_video_output(json.dumps({"observations": [], "extra": True}), 0)


def test_parser_combines_two_boards_without_losing_formulas_or_frame_ownership():
    right_board = {**CONTENT, "text": "Правая доска", "formulas": ["y = 2", *CONTENT["formulas"]],
                   "illustrations": [{"source_kind": "drawn", "box": [0.6, 0.1, 0.9, 0.9],
                                      "description": "Схема справа"}]}
    raw = json.dumps({"observations": [
        {"frame_indices": [0, 1], **CONTENT},
        {"frame_indices": [1], **right_board},
    ]})
    first, second = video_vision.parse_video_output(raw, 2)
    assert first == CONTENT
    assert second["text"] == CONTENT["text"] + "\n\nПравая доска"
    assert second["formulas"] == [*CONTENT["formulas"], "y = 2"]
    assert second["illustrations"] == right_board["illustrations"]


def test_video_caches_batches_and_keeps_original_frame_metadata(worker):
    request, runtime, loader = worker
    first = vision_worker.run(request)
    assert [{key: item[key] for key in ("id", "timestamp", "image")}
            for item in first["items"]] == request["items"]
    assert all(item["content"] == CONTENT for item in first["items"])
    assert runtime.infer_video.call_count == 2
    runtime.infer.assert_not_called()
    loader.assert_called_once_with("test/model", "abc")
    runtime.reset_mock()
    loader.reset_mock()
    resumed = vision_worker.run(request)
    assert resumed["items"] == first["items"]
    assert resumed["diagnostics"]["cached_items"] == 5
    loader.assert_not_called()
    runtime.infer_video.assert_not_called()
    # A changed source timestamp invalidates its entire temporal context, not later batches.
    request["items"][1]["timestamp"] += 0.25
    updated = vision_worker.run(request)
    assert updated["diagnostics"]["fresh_items"] == 4
    assert updated["diagnostics"]["cached_items"] == 1
    runtime.infer_video.assert_called_once()


def test_frame_and_video_caches_do_not_mix(worker):
    request, runtime, _ = worker
    vision_worker.run({**request, "mode": "frames"})
    assert runtime.infer.call_count == 5
    result = vision_worker.run(request)
    assert result["diagnostics"]["fresh_items"] == 5
    assert runtime.infer_video.call_count == 2
    runtime.reset_mock()
    result = vision_worker.run({**request, "mode": "frames"})
    assert result["diagnostics"]["cached_items"] == 5
    runtime.infer.assert_not_called()


def test_incomplete_mapping_retries_and_failed_batch_is_not_cached(worker):
    request, runtime, _ = worker
    request["items"] = request["items"][:2]
    runtime.infer_video.side_effect = [response([0]), response([0])]
    failed = vision_worker.run(request)
    assert all(item["diagnostics"]["status"] == "incomplete" for item in failed["items"])
    assert all(item["content"]["illustrations"][0]["source_kind"] == "unknown"
               for item in failed["items"])
    assert [call.args[1] for call in runtime.infer_video.call_args_list] == [4096, 8192]
    runtime.infer_video.side_effect = [response([0], finish_reason="length"), response([0, 1])]
    repaired = vision_worker.run(request)
    assert all(item["diagnostics"]["status"] == "complete" for item in repaired["items"])
    assert repaired["diagnostics"]["fresh_items"] == 2


def test_interrupted_video_run_resumes_completed_batches(worker):
    request, runtime, _ = worker
    runtime.infer_video.side_effect = [response([0, 1, 2, 3]), RuntimeError("interrupted")]
    with pytest.raises(RuntimeError, match="interrupted"):
        vision_worker.run(request)
    runtime.infer_video.side_effect = [response([0])]
    runtime.infer_video.reset_mock()
    result = vision_worker.run(request)
    assert result["diagnostics"]["cached_items"] == 4
    assert result["diagnostics"]["fresh_items"] == 1
    runtime.infer_video.assert_called_once()
