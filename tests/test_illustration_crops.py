import json
from copy import deepcopy
from unittest.mock import Mock

import httpx
import pymupdf
import pytest

from shad_helper.illustration_crops import refine_crop, resolve_crop
from shad_helper.jobs import fingerprint, read_json, sha256_file, write_json
from shad_helper.openai_client import PlanClient


@pytest.fixture
def example(tmp_path):
    job = tmp_path / "job"
    frame = job / "frames" / f"v01_{'a' * 12}.png"
    frame.parent.mkdir(parents=True)
    pixels = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 120, 80), False)
    pixels.clear_with(255)
    for x in range(120):
        for y in range(80):
            pixels.set_pixel(x, y, (x, y, 42))
    pixels.save(frame)
    box = [20, 20, 100, 70]
    original = pymupdf.Pixmap(pixels.colorspace, pymupdf.IRect(box), False)
    original.copy(pixels, pymupdf.IRect(box))
    source = job / "illustrations" / "v01_ill01.png"
    source.parent.mkdir()
    original.save(source)
    observation = {"id": "v01_ill01", "source_kind": "printed",
                   "description": "График и легенда", "box": [1 / 6, 0.25, 5 / 6, 0.875],
                   "image_sha256": sha256_file(source)}
    evidence = {"media": {"sha256": "a" * 64}, "visuals": [
        {"id": "v01", "timestamp": 3, "status": "complete", "illustrations": [observation]}]}
    write_json(source.parent / "manifest.json", {"version": 1, "assets": {
        "v01_ill01": {"path": "illustrations/v01_ill01.png", "sha256": sha256_file(source)}}})
    block = {"illustration_id": "v01_ill01", "caption": "Зависимость y от x", "content": "График"}
    return job, evidence, block, frame, source


def client(box=(30, 25, 90, 60)):
    return Mock(model="test-model", complete=Mock(return_value={
        "status": "cropped", "box": list(box), "reason": "Оси и легенда сохранены."}))


def run(example, model=None, folder="archive"):
    job, evidence, block, _, _ = example
    return refine_crop(job, evidence, block, model or client(), job / folder)


@pytest.mark.parametrize("box", [(30, 25, 90, 60), (5, 10, 115, 78)])
def test_shrinks_or_expands_using_native_pixels_and_preserves_originals(example, box):
    job, evidence, block, frame, source = example
    original_evidence, original_block = deepcopy(evidence), deepcopy(block)
    original_assets = {path: path.read_bytes() for path in
                       (frame, source, source.parent / "manifest.json")}
    model = client(box)
    crop_id, retry, reason = run(example, model)
    assert crop_id and retry is False and reason == ""
    result = resolve_crop(job, evidence, "v01_ill01", crop_id)
    pixels = pymupdf.Pixmap(result)
    assert (pixels.width, pixels.height) == (box[2] - box[0], box[3] - box[1])
    assert pixels.pixel(0, 0) == (box[0], box[1], 42)
    assert pixels.pixel(pixels.width - 1, pixels.height - 1) == (box[2] - 1, box[3] - 1, 42)
    assert evidence == original_evidence and block == original_block
    assert all(path.read_bytes() == original for path, original in original_assets.items())
    assert resolve_crop(job, evidence, "v01_ill01") == source
    args, kwargs = model.complete.call_args
    context = json.loads(args[1])
    assert context["frame_size"] == [120, 80]
    assert context["source_box"] == evidence["visuals"][0]["illustrations"][0]["box"]
    assert [label for label, _ in kwargs["images"]] == ["FULL_FRAME", "CANDIDATE"]
    archive = job / "archive/attempt-01"
    assert (archive / "frame.png").read_bytes() == frame.read_bytes()
    assert (archive / "candidate.png").read_bytes() == source.read_bytes()
    assert (archive / "crop.png").read_bytes() == result.read_bytes()
    assert read_json(archive / "response.json") == model.complete.return_value
    assert read_json(archive / "crop.json")["box"] == list(box)


def test_cache_reuses_completed_crop_and_tracks_selected_figure_and_native_frame(example):
    job, evidence, block, frame, _ = example
    model = client()
    first = run(example, model)[0]
    assert run(example, model)[0] == first
    assert model.complete.call_count == 1
    assert read_json(job / "archive/attempt-02/status.json")["status"] == "cached"
    block["caption"] = "Изменённая подпись"
    second = run(example, model)[0]
    assert second != first
    pixels = pymupdf.Pixmap(frame)
    pixels.set_pixel(0, 0, (0, 0, 0))
    pixels.save(frame)
    assert run(example, model)[0] not in (first, second)
    assert model.complete.call_count == 3
    assert resolve_crop(job, evidence, "v01_ill01", first).is_file()


@pytest.mark.parametrize("box", [[1, 2, 3], [10.0, 20, 30, 40], [True, 2, 3, 4],
                                   [-1, 0, 30, 40], [0, 0, 121, 80], [30, 20, 30, 40]])
def test_invalid_boxes_do_not_publish_or_poison_cache(example, box):
    model = client(box)
    crop_id, retry, reason = run(example, model)
    assert crop_id == "" and retry is True and reason
    assert not list((example[0] / "illustration-crops").glob("*/crop.json"))
    model.complete.return_value["box"] = [20, 10, 90, 70]
    assert run(example, model)[0]
    assert model.complete.call_count == 2


def test_uncertain_crop_keeps_original_and_does_not_cache_success(example):
    model = client()
    model.complete.return_value = {"status": "uncertain", "box": [0, 0, 0, 0],
                                   "reason": "Лектор закрывает ось."}
    crop_id, retry, reason = run(example, model)
    assert crop_id == "" and retry is False and "Лектор" in reason
    assert not list((example[0] / "illustration-crops").glob("*/crop.json"))
    model.complete.return_value = client().complete.return_value
    assert run(example, model)[0]


def test_transport_failure_retries_and_restores_request_recorder(example):
    job, evidence, block, _, _ = example
    model = PlanClient.__new__(PlanClient)
    model.model = "test-model"
    original_recorder = Mock()
    model.request_recorder = original_recorder

    def fail(*args, **kwargs):
        model.request_recorder("POST", "https://example.invalid/responses", b'{"request":1}')
        raise RuntimeError("unavailable")

    model.complete = fail
    assert refine_crop(job, evidence, block, model, job / "archive")[0:2] == ("", True)
    assert model.request_recorder is original_recorder
    original_recorder.assert_not_called()
    archive = job / "archive/attempt-01"
    assert read_json(archive / "http-001.request.json") == {
        "method": "POST", "url": "https://example.invalid/responses", "body": "http-001.body.json"}
    assert (archive / "http-001.body.json").read_bytes() == b'{"request":1}'
    model.complete = client().complete
    assert refine_crop(job, evidence, block, model, job / "archive")[0]
    assert model.request_recorder is original_recorder


def test_missing_native_frame_keeps_original_without_network(example):
    example[3].unlink()
    model = client()
    assert run(example, model)[0:2] == ("", False)
    model.complete.assert_not_called()
    assert resolve_crop(example[0], example[1], "v01_ill01") == example[4]


@pytest.mark.parametrize("crop_id", ["../escape", "f" * 63, "g" * 64, "/tmp/crop", 123])
def test_rejects_noncontent_addressed_crop_references(example, crop_id):
    with pytest.raises(ValueError, match="Invalid illustration crop ID"):
        resolve_crop(example[0], example[1], "v01_ill01", crop_id)


@pytest.mark.parametrize("mutation", ["bytes", "binding", "box", "not_png", "symlink_image",
                                       "symlink_record", "symlink_folder"])
def test_resolver_rejects_tampering_and_symlinks(example, mutation):
    job, evidence, _, _, _ = example
    crop_id = run(example)[0]
    root = job / "illustration-crops" / crop_id
    record = read_json(root / "crop.json")
    if mutation == "bytes":
        (root / "crop.png").write_bytes(b"changed")
    elif mutation == "binding":
        record["inputs"]["illustration_id"] = "v01_ill02"
        other_id = fingerprint(record["inputs"])
        other = root.with_name(other_id)
        root.rename(other)
        root, crop_id = other, other_id
        write_json(root / "crop.json", record)
    elif mutation == "box":
        record["box"] = [0, 0, 5, 5]
        write_json(root / "crop.json", record)
    elif mutation == "not_png":
        (root / "crop.png").write_bytes(b"not png")
        record["image_sha256"] = sha256_file(root / "crop.png")
        write_json(root / "crop.json", record)
    elif mutation in ("symlink_image", "symlink_record"):
        path = root / ("crop.png" if mutation == "symlink_image" else "crop.json")
        other = job / path.name
        path.rename(other)
        path.symlink_to(other)
    elif mutation == "symlink_folder":
        other = job / "elsewhere"
        root.rename(other)
        root.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError):
        resolve_crop(job, evidence, "v01_ill01", crop_id)


def test_source_hash_binding_survives_valid_new_source_manifest(example):
    job, evidence, _, _, source = example
    crop_id = run(example)[0]
    pixels = pymupdf.Pixmap(source)
    pixels.clear_with(0)
    pixels.save(source)
    evidence["visuals"][0]["illustrations"][0]["image_sha256"] = sha256_file(source)
    manifest = read_json(source.parent / "manifest.json")
    manifest["assets"]["v01_ill01"]["sha256"] = sha256_file(source)
    write_json(source.parent / "manifest.json", manifest)
    with pytest.raises(ValueError, match="does not match its source"):
        resolve_crop(job, evidence, "v01_ill01", crop_id)


def test_native_frame_symlink_is_not_sent(example):
    job, _, _, frame, _ = example
    moved = job / "elsewhere.png"
    frame.rename(moved)
    frame.symlink_to(moved)
    model = client()
    assert run(example, model)[0] == ""
    model.complete.assert_not_called()


def test_raw_http_transport_error_is_reported(example):
    model = client()
    model.complete.side_effect = httpx.ConnectError("offline")
    assert run(example, model)[0:2] == ("", True)
