import json
import random
import shutil
from copy import deepcopy
from unittest.mock import Mock

import httpx
import pymupdf
import pytest

from shad_helper import crop_validation, illustration_crops, illustration_stitching
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
    model = Mock(model="test-model", complete=Mock(return_value={
        "status": "cropped", "box": list(box), "reason": "Оси и легенда сохранены."}))

    def complete(instructions, text, schema, **kwargs):
        if schema == crop_validation.CROP_VERIFICATION_SCHEMA:
            return verdict()
        if schema == illustration_crops.SOURCE_SELECTION_SCHEMA:
            return {"ranked_frame_ids": [item["visual_id"] for item in json.loads(text)["frames"]],
                    "reason": "Сопоставлены состояния и читаемость рисунков."}
        return model.complete.return_value

    model.complete.side_effect = complete
    return model


def verdict(check=None, defect="clipped_content", repair="recrop", status="fail"):
    checks = dict.fromkeys(("identity", "completeness", "readability", "cleanliness"), "pass")
    if check:
        checks[check] = status
    return {"verdict": status if check else "pass", "checks": checks,
            "defects": [defect] if check else [], "repair": repair if check else "none",
            "reason": "Подпись оси обрезана." if check else "Рисунок полный и соответствует кадру."}


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
    crop_id, retry, reason, unusable = run(example, model)
    assert crop_id and retry is False and reason == "" and not unusable
    result = resolve_crop(job, evidence, "v01_ill01", crop_id)
    pixels = pymupdf.Pixmap(result)
    assert (pixels.width, pixels.height) == (box[2] - box[0], box[3] - box[1])
    assert pixels.pixel(0, 0) == (box[0], box[1], 42)
    assert pixels.pixel(pixels.width - 1, pixels.height - 1) == (box[2] - 1, box[3] - 1, 42)
    assert evidence == original_evidence and block == original_block
    assert all(path.read_bytes() == original for path, original in original_assets.items())
    assert resolve_crop(job, evidence, "v01_ill01") == source
    args, kwargs = model.complete.call_args_list[0]
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
    assert model.complete.call_count == 2
    assert read_json(job / "archive/attempt-02/status.json")["status"] == "cached"
    block["caption"] = "Изменённая подпись"
    second = run(example, model)[0]
    assert second != first
    pixels = pymupdf.Pixmap(frame)
    pixels.set_pixel(0, 0, (0, 0, 0))
    pixels.save(frame)
    assert run(example, model)[0] not in (first, second)
    assert model.complete.call_count == 6
    assert resolve_crop(job, evidence, "v01_ill01", first).is_file()


@pytest.mark.parametrize("box", [[1, 2, 3], [10.0, 20, 30, 40], [True, 2, 3, 4],
                                   [-1, 0, 30, 40], [0, 0, 121, 80], [30, 20, 30, 40]])
def test_invalid_boxes_do_not_publish_or_poison_cache(example, box):
    model = client(box)
    crop_id, retry, reason, unusable = run(example, model)
    assert crop_id == "" and retry is True and reason and not unusable
    assert not list((example[0] / "illustration-crops").glob("*/crop.json"))
    model.complete.return_value["box"] = [20, 10, 90, 70]
    assert run(example, model)[0]
    assert model.complete.call_count == 3


@pytest.mark.parametrize("uncertainty", [
    "Лектор закрывает ось.",
    "Прокрутка обрезала нижний ряд графиков и подписи осей в исходном кадре.",
])
def test_uncertain_crop_keeps_original_and_does_not_cache_success(example, uncertainty):
    model = client()
    model.complete.return_value = {"status": "uncertain", "box": [0, 0, 0, 0],
                                   "reason": uncertainty}
    crop_id, retry, reason, unusable = run(example, model)
    assert crop_id == "" and retry is False and uncertainty in reason and not unusable
    assert resolve_crop(example[0], example[1], "v01_ill01", crop_id) == example[4]
    assert not list((example[0] / "illustration-crops").glob("*/crop.json"))
    model.complete.return_value = client().complete.return_value
    assert run(example, model)[0]


def test_unusable_fragment_is_explicit_and_preserves_source_for_later_recheck(example):
    model = client()
    reason = "В кадре остались только заголовок и край графика без учебного содержания."
    model.complete.return_value = {"status": "unusable", "box": [0, 0, 0, 0], "reason": reason}
    original = example[4].read_bytes()
    crop_id, retry, detail, unusable = run(example, model)
    assert crop_id == "" and not retry and unusable and reason in detail
    assert example[4].read_bytes() == original
    assert not list((example[0] / "illustration-crops").glob("*/crop.json"))
    assert read_json(example[0] / "archive/attempt-01/status.json")["status"] == "unusable"
    model.complete.return_value = client().complete.return_value
    assert run(example, model)[0]
    assert model.complete.call_count == 3


@pytest.mark.parametrize("box", [[0, 0, 0], [False, 0, 0, 0], [0.0, 0, 0, 0], [0, 0, 0, 1]])
def test_invalid_unusable_result_cannot_authorize_exclusion(example, box):
    model = client()
    model.complete.return_value = {"status": "unusable", "box": box, "reason": "Только заголовок."}
    crop_id, retry, reason, unusable = run(example, model)
    assert crop_id == "" and retry and reason and not unusable
    assert resolve_crop(example[0], example[1], "v01_ill01") == example[4]


def test_changed_crop_policy_rechecks_old_success_without_deleting_evidence(example, monkeypatch):
    job, evidence, _, _, source = example
    model = client()
    previous_id = run(example, model)[0]
    original = source.read_bytes()
    monkeypatch.setattr(illustration_crops, "CROP_INSTRUCTIONS",
                        illustration_crops.CROP_INSTRUCTIONS + "\nRecheck completeness.")
    model.complete.return_value = {
        "status": "uncertain", "box": [0, 0, 0, 0],
        "reason": "Полная иллюстрация не помещается в исходный кадр.",
    }
    assert run(example, model)[0:2] == ("", False)
    assert model.complete.call_count == 3
    assert source.read_bytes() == original
    assert resolve_crop(job, evidence, "v01_ill01", previous_id).is_file()
    assert read_json(job / "archive/attempt-02/status.json")["status"] == "uncertain"


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


def test_verifier_sees_actual_crop_without_selector_justification(example):
    job, _, _, frame, source = example
    model = client()
    selector_reason = "THIS IS DEFINITELY CORRECT"
    model.complete.return_value["reason"] = selector_reason
    original = {path: path.read_bytes() for path in (frame, source)}
    crop_id = run(example, model)[0]
    args, kwargs = model.complete.call_args_list[1]
    assert args[2] == crop_validation.CROP_VERIFICATION_SCHEMA
    context = json.loads(args[1])
    assert selector_reason not in args[1] and "repair_feedback" not in context
    assert [label for label, _ in kwargs["images"]] == ["FULL_FRAME", "MARKED_FRAME", "CROP"]
    images = dict(kwargs["images"])
    assert images["CROP"].read_bytes() == resolve_crop(job, example[1], "v01_ill01", crop_id).read_bytes()
    assert images["FULL_FRAME"].read_bytes() == original[frame]
    assert images["MARKED_FRAME"].read_bytes() != original[frame]
    assert all(path.read_bytes() == contents for path, contents in original.items())
    record = read_json(job / "illustration-crops" / crop_id / "crop.json")
    assert record["verification"]["result"] == verdict()
    assert record["verification"]["image_sha256"] == sha256_file(images["CROP"])


@pytest.mark.parametrize("check,defect", [("completeness", "clipped_content"),
                                        ("identity", "wrong_figure")])
def test_repair_uses_concrete_feedback_and_rechecks_new_pixels(example, check, defect):
    model = client()
    rejected = verdict(check, defect)
    corrected = {"status": "cropped", "box": [5, 10, 115, 78], "reason": "Подпись восстановлена."}
    model.complete.side_effect = [model.complete.return_value, rejected, corrected, verdict()]
    crop_id, retry, reason, unusable = run(example, model)
    assert crop_id and not retry and not reason and not unusable
    repair_args, repair_kwargs = model.complete.call_args_list[2]
    assert json.loads(repair_args[1])["repair_feedback"] == rejected
    assert dict(repair_kwargs["images"])["CANDIDATE"].read_bytes() == (
        example[0] / "archive/attempt-01/verification/crop.png").read_bytes()
    verifier_context = json.loads(model.complete.call_args_list[3].args[1])
    assert "repair_feedback" not in verifier_context and rejected["reason"] not in str(verifier_context)
    pixels = pymupdf.Pixmap(resolve_crop(example[0], example[1], "v01_ill01", crop_id))
    assert pixels.pixel(0, 0) == (5, 10, 42)
    assert (pixels.width, pixels.height) == (110, 68)
    assert len(list((example[0] / "illustration-crops").glob("*/crop.json"))) == 1
    assert read_json(example[0] / "archive/attempt-01/status.json")["repairs"] == 1
    archive = example[0] / "archive/attempt-01"
    assert sha256_file(archive / "crop.png") == read_json(archive / "crop.json")["image_sha256"]
    assert (archive / "initial-crop.png").read_bytes() == (archive / "verification/crop.png").read_bytes()


def test_exhausted_repairs_keep_source_without_publishing_wrong_crop(example):
    model = client()
    model.complete.side_effect = [model.complete.return_value, verdict("identity", "wrong_figure")] * 3
    crop_id, retry, reason, unusable = run(example, model)
    assert crop_id == "" and not retry and reason and not unusable
    assert model.complete.call_count == 6
    assert not list((example[0] / "illustration-crops").glob("*/crop.json"))
    assert resolve_crop(example[0], example[1], "v01_ill01") == example[4]
    assert read_json(example[0] / "archive/attempt-01/status.json")["repairs"] == 2


@pytest.mark.parametrize("status", ["fail", "uncertain"])
def test_no_supported_repair_stops_and_never_caches_acceptance(example, status):
    model = client()
    model.complete.side_effect = [model.complete.return_value,
                                  verdict("readability", "unreadable", "none", status)]
    assert run(example, model)[0:2] == ("", False)
    assert model.complete.call_count == 2
    assert not list((example[0] / "illustration-crops").glob("*/crop.json"))


@pytest.mark.parametrize("failure", [RuntimeError("offline"), {"verdict": "pass"},
                                    {**verdict(), "checks": {}}])
def test_unavailable_or_malformed_verifier_does_not_publish_and_can_retry(example, failure):
    model = client()
    model.complete.side_effect = [model.complete.return_value, failure]
    assert run(example, model)[0:2] == ("", True)
    assert not list((example[0] / "illustration-crops").glob("*/crop.json"))
    assert run(example, client())[0]


def add_frame(example, identifier, timestamp, shade=150, size=(140, 90)):
    job, evidence, _, _, _ = example
    path = job / "frames" / f"{identifier}_{'a' * 12}.png"
    pixels = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, *size), False)
    pixels.clear_with(shade)
    pixels.save(path)
    evidence["visuals"].append({"id": identifier, "timestamp": timestamp,
                                "status": "complete", "illustrations": []})
    return path


@pytest.mark.parametrize("initial", ["cropped", "uncertain"])
def test_nearby_frame_recovery_verifies_original_identity_and_records_provenance(example, initial):
    job, evidence, _, original, source = example
    alternate = add_frame(example, "v02", 8)
    snapshots = {path: path.read_bytes() for path in (original, alternate, source)}
    original_evidence = deepcopy(evidence)
    model = client()
    first = ([model.complete.return_value, verdict("completeness", repair="nearby_frame")]
             if initial == "cropped" else [{"status": "uncertain", "box": [0, 0, 0, 0],
                                            "reason": "График обрезан краем кадра."}])
    model.complete.side_effect = [
        {"ranked_frame_ids": ["v01", "v02"], "reason": "Одинаковый рисунок."}, *first,
        {"status": "cropped", "box": [10, 10, 135, 85], "reason": "В соседнем кадре рисунок полный."},
        verdict()]
    crop_id, retry, reason, unusable = run(example, model)
    assert crop_id and not retry and not reason and not unusable
    verifier = model.complete.call_args_list[-1]
    images = dict(verifier.kwargs["images"])
    assert images["ORIGINAL_FRAME"].read_bytes() == snapshots[original]
    assert images["FULL_FRAME"].read_bytes() == snapshots[alternate]
    assert json.loads(verifier.args[1])["frame"]["visual_id"] == "v02"
    record = read_json(job / "illustration-crops" / crop_id / "crop.json")
    assert record["frame"]["visual_id"] == "v02"
    assert record["frame"]["timestamp"] == 8
    assert record["frame"]["sha256"] == sha256_file(alternate)
    assert record["inputs"]["reference_frame"]["sha256"] == sha256_file(original)
    assert record["inputs"]["illustration_id"] == "v01_ill01"
    assert pymupdf.Pixmap(resolve_crop(job, evidence, "v01_ill01", crop_id)).width == 125
    assert evidence == original_evidence
    assert all(path.read_bytes() == contents for path, contents in snapshots.items())
    assert run(example, model)[0] == crop_id  # no remaining mocked responses: must reuse cache
    cached = job / "archive/attempt-02"
    assert (cached / "verification/frame.png").read_bytes() == snapshots[alternate]
    assert (cached / "verification/original-frame.png").read_bytes() == snapshots[original]
    assert sha256_file(cached / "crop.png") == read_json(cached / "crop.json")["image_sha256"]
    changed = pymupdf.Pixmap(alternate)
    changed.set_pixel(0, 0, (9, 9, 9))
    changed.save(alternate)
    assert run(example, client())[0] != crop_id
    assert resolve_crop(job, evidence, "v01_ill01", crop_id).is_file()  # archived snapshot


def test_nearby_wrong_stage_is_rejected_even_when_crop_is_complete(example):
    add_frame(example, "v02", 8)
    model = client()
    model.complete.side_effect = [
        {"ranked_frame_ids": ["v01", "v02"], "reason": "Одинаковый рисунок."},
        model.complete.return_value, verdict("readability", "occluded", "nearby_frame"),
        model.complete.return_value, verdict("identity", "wrong_stage", "none")]
    assert run(example, model)[0:2] == ("", False)
    assert model.complete.call_count == 5
    assert not list((example[0] / "illustration-crops").glob("*/crop.json"))


def test_nearby_selection_skips_missing_duplicates_symlinks_and_distant_frames(example):
    job, evidence, _, original, _ = example
    missing = add_frame(example, "missing", 3.1)
    missing.unlink()
    duplicate = add_frame(example, "duplicate", 3.2)
    duplicate.write_bytes(original.read_bytes())
    linked = add_frame(example, "linked", 3.3)
    linked.unlink()
    linked.symlink_to(original)
    corrupt = add_frame(example, "corrupt", 3.4)
    corrupt.write_bytes(b"not an image")
    add_frame(example, "earlier", 2, shade=100)
    add_frame(example, "later", 6, shade=200)
    add_frame(example, "distant", 34, shade=99)
    reference = illustration_crops._native_frame(job, "a" * 64, evidence["visuals"][0])[1]
    nearby = illustration_crops._nearby_frames(job, evidence, "a" * 64, reference)
    assert [frame[1]["visual_id"] for frame in nearby] == ["earlier", "later"]


def test_warning_flags_are_sent_to_verifier_but_do_not_reject_a_pass(example):
    pixels = pymupdf.Pixmap(example[3])
    pixels.set_pixel(0, 0, (255, 255, 255))
    pixels.save(example[3])
    model = client((0, 0, 120, 80))
    crop_id = run(example, model)[0]
    assert crop_id
    context = json.loads(model.complete.call_args_list[1].args[1])
    assert "frame_edge" in context["warnings"]
    assert "content_at_boundary" in context["warnings"]


def test_verifier_policy_change_invalidates_previous_pass(example, monkeypatch):
    model = client()
    first = run(example, model)[0]
    monkeypatch.setattr(crop_validation, "CROP_VERIFICATION_INSTRUCTIONS",
                        crop_validation.CROP_VERIFICATION_INSTRUCTIONS + "\nRecheck stage.")
    second = run(example, model)[0]
    assert second and first != second and model.complete.call_count == 4


@pytest.mark.parametrize("missing", ["all", "marked-frame.png", "verdict.json"])
def test_incomplete_verification_archive_is_rebuilt_instead_of_permanent_fallback(example, missing):
    model = client()
    crop_id = run(example, model)[0]
    archive = example[0] / "illustration-crops" / crop_id / "verification"
    if missing == "all":
        shutil.rmtree(archive)
    else:
        (archive / missing).unlink()
    assert run(example, model) == (crop_id, False, "", False)
    assert model.complete.call_count == 4
    assert (archive / "marked-frame.png").is_file()
    assert read_json(archive / "verdict.json") == verdict()


@pytest.mark.parametrize("mutation", ["verdict", "crop_hash", "frame_hash", "box", "frame"])
def test_resolver_requires_verified_pixels_and_frame_binding(example, mutation):
    job, evidence, _, _, _ = example
    crop_id = run(example)[0]
    path = job / "illustration-crops" / crop_id / "crop.json"
    record = read_json(path)
    if mutation == "verdict":
        record["verification"]["result"] = verdict("identity", "wrong_figure")
    elif mutation == "crop_hash":
        record["verification"]["image_sha256"] = "b" * 64
    elif mutation == "frame_hash":
        record["verification"]["frame_sha256"] = "b" * 64
    elif mutation == "box":
        record["verification"]["box"] = [0, 0, 120, 80]
    else:
        record["frame"]["visual_id"] = "wrong"
    write_json(path, record)
    with pytest.raises(ValueError, match="verification"):
        resolve_crop(job, evidence, "v01_ill01", crop_id)


def test_legacy_crop_assets_remain_resolvable_but_are_not_reused_as_verified(example):
    job, evidence, _, _, _ = example
    current_id = run(example)[0]
    root = job / "illustration-crops" / current_id
    record = read_json(root / "crop.json")
    record["version"] = 1
    record.pop("verification")
    record["inputs"] = {"illustration_id": "v01_ill01", "source_sha256": sha256_file(example[4]),
                        "frame_size": [120, 80], "policy": 1}
    legacy_id = fingerprint(record["inputs"])
    legacy = root.with_name(legacy_id)
    root.rename(legacy)
    write_json(legacy / "crop.json", record)
    assert resolve_crop(job, evidence, "v01_ill01", legacy_id).is_file()
    model = client()
    assert run(example, model)[0] == current_id
    assert model.complete.call_count == 2


def test_nearby_pool_covers_interval_instead_of_only_nearest_frames(example):
    job, evidence, _, _, _ = example
    evidence["visuals"][0]["timestamp"] = 50
    for index, timestamp in enumerate([20, 25, 35, 45, 48, 49, 51, 54, 65, 75, 80]):
        add_frame(example, f"v{timestamp}", timestamp, shade=index + 50)
    reference = illustration_crops._native_frame(job, "a" * 64, evidence["visuals"][0])[1]
    nearby = illustration_crops._nearby_frames(job, evidence, "a" * 64, reference)
    assert [item[1]["timestamp"] for item in nearby] == [49, 51, 35, 65, 25, 75]
    assert len(nearby) == illustration_crops.MAX_NEARBY_FRAMES


def test_ranks_clean_farther_frame_before_committing_to_original_crop(example):
    job, _, _, original, _ = example
    add_frame(example, "occluded_near", 5, shade=100)
    clean = add_frame(example, "clean_far", 20, shade=200)
    model = client()
    model.complete.side_effect = [
        {"ranked_frame_ids": ["clean_far", "v01"],
         "reason": "Чистый прямой захват той же стадии вместо бледной проекции."},
        model.complete.return_value, verdict()]
    crop_id, retry, reason, unusable = run(example, model)
    assert crop_id and not retry and not reason and not unusable
    record = read_json(job / "illustration-crops" / crop_id / "crop.json")
    assert record["frame"]["visual_id"] == "clean_far"
    assert record["frame"]["sha256"] == sha256_file(clean)
    ranking = model.complete.call_args_list[0]
    assert ranking.args[2] == illustration_crops.SOURCE_SELECTION_SCHEMA
    assert len(ranking.kwargs["images"]) == 4  # original, two alternatives, source crop
    verify = model.complete.call_args_list[-1]
    assert dict(verify.kwargs["images"])["ORIGINAL_FRAME"].read_bytes() == original.read_bytes()
    assert "бледной проекции" not in verify.args[1]  # independent verification
    assert run(example, model)[0] == crop_id  # ranking and pixels share cache identity
    assert model.complete.call_count == 3


def test_ranked_alternative_still_requires_independent_stage_verification(example):
    add_frame(example, "later_stage", 20)
    model = client()
    model.complete.side_effect = [
        {"ranked_frame_ids": ["later_stage"], "reason": "Чистый рисунок."},
        model.complete.return_value, verdict("identity", "wrong_stage", "none"),
        model.complete.return_value, verdict()]
    crop_id, retry, reason, unusable = run(example, model)
    assert crop_id and not retry and not reason and not unusable
    record = read_json(example[0] / "illustration-crops" / crop_id / "crop.json")
    assert record["frame"]["visual_id"] == "v01"
    assert model.complete.call_count == 5


def test_selected_panel_scope_reaches_ranking_crop_and_independent_verifier(example):
    job, evidence, block, _, _ = example
    evidence["visuals"][0]["illustrations"][0]["description"] = "Большая сетка гистограмм"
    block.update(caption="Доход и давность покупки", content="Две нижние панели Income и Recency.")
    add_frame(example, "scrolled", 8)
    model = client()
    model.complete.side_effect = [
        {"ranked_frame_ids": ["scrolled"], "reason": "Те же панели видны полностью."},
        model.complete.return_value, verdict()]

    crop_id, retry, reason, unusable = run(example, model)

    assert crop_id and not retry and not reason and not unusable
    expected = {key: block[key] for key in ("caption", "content")}
    for call in model.complete.call_args_list:
        context = json.loads(call.args[1])
        assert context["figure"] == expected
        assert context["description"] == "Большая сетка гистограмм"
    record = read_json(job / "illustration-crops" / crop_id / "crop.json")
    assert record["frame"]["visual_id"] == "scrolled"
    assert record["inputs"]["figure"] == expected


def test_crop_policy_version_invalidates_verified_crop(example, monkeypatch):
    model = client()
    previous = run(example, model)[0]
    monkeypatch.setattr(illustration_crops, "CROP_POLICY_VERSION",
                        illustration_crops.CROP_POLICY_VERSION + 1)

    current = run(example, model)[0]

    assert previous and current and current != previous
    assert model.complete.call_count == 4
    assert resolve_crop(example[0], example[1], "v01_ill01", previous).is_file()


def test_relaxed_policy_reuses_independently_verified_previous_policy_crop(example, monkeypatch):
    with monkeypatch.context() as old:
        old.setattr(illustration_crops, "CROP_POLICY_VERSION", 3)
        old.setattr(illustration_crops, "CROP_INSTRUCTIONS", "Previous stricter cropping instructions")
        old.setattr(illustration_crops, "SOURCE_SELECTION_INSTRUCTIONS", "Previous stricter selection")
        old.setattr(crop_validation, "CROP_VERIFICATION_INSTRUCTIONS", "Previous stricter verification")
        previous = run(example)[0]
    model = client()

    assert run(example, model) == (previous, False, "", False)

    model.complete.assert_not_called()
    record = read_json(example[0] / "archive/attempt-02/crop.json")
    assert record["inputs"]["policy"] == 3  # retain the actual verification provenance
    assert record["verification"]["result"]["verdict"] == "pass"


@pytest.mark.parametrize("changed", ["purpose", "caption", "content", "model", "frames", "schema"])
def test_previous_policy_crop_reuse_requires_same_task_and_evidence(example, monkeypatch, changed):
    with monkeypatch.context() as old:
        old.setattr(illustration_crops, "CROP_POLICY_VERSION", 3)
        previous = run(example)[0]
    model = client()
    if changed == "purpose":
        example[2]["kind"] = "diagram"
    elif changed in ("caption", "content"):
        example[2][changed] = "Другая учебная иллюстрация"
    elif changed == "model":
        model.model = "another-model"
    elif changed == "frames":
        add_frame(example, "another-frame", 8)
    else:
        monkeypatch.setattr(illustration_crops, "CROP_SCHEMA",
                            {**illustration_crops.CROP_SCHEMA, "description": "Updated schema"})

    current = run(example, model)[0]

    assert current and current != previous
    assert model.complete.called


@pytest.mark.parametrize("damage", ["failed_verification", "incomplete_selection", "changed_pixels"])
def test_previous_policy_crop_reuse_rejects_invalid_or_incomplete_artifacts(example, monkeypatch, damage):
    with monkeypatch.context() as old:
        old.setattr(illustration_crops, "CROP_POLICY_VERSION", 3)
        previous = run(example)[0]
    cache = example[0] / "illustration-crops" / previous
    record = read_json(cache / "crop.json")
    if damage == "failed_verification":
        record["verification"]["result"] = verdict("identity", "wrong_stage", "none")
    elif damage == "incomplete_selection":
        record["source_selection_complete"] = False
    else:
        image = pymupdf.Pixmap(cache / "crop.png")
        image.set_pixel(0, 0, (0, 0, 0))
        image.save(cache / "crop.png")
    write_json(cache / "crop.json", record)
    model = client()

    current = run(example, model)[0]

    assert current and current != previous
    assert model.complete.called


@pytest.mark.parametrize("identifiers", [["unknown"], ["v02", "v02"], [None], "v02"])
def test_invalid_frame_ranking_falls_back_to_independent_original_check(example, identifiers):
    add_frame(example, "v02", 8)
    model = client()
    model.complete.side_effect = [{"ranked_frame_ids": identifiers, "reason": "Выбор."},
                                  model.complete.return_value, verdict()]
    original = example[4].read_bytes()
    crop_id, retry, reason, unusable = run(example, model)
    assert crop_id and retry and not reason and not unusable
    assert example[4].read_bytes() == original
    record = read_json(example[0] / "illustration-crops" / crop_id / "crop.json")
    assert record["frame"]["visual_id"] == "v01"
    assert not record["source_selection_complete"]


def test_reconstruction_clutter_pass_is_not_reused_for_screenshot(example):
    job, evidence, block, _, source = example
    model = client()
    proposal = model.complete.return_value

    def complete(instructions, text, schema, **kwargs):
        context = json.loads(text)
        if schema == illustration_crops.CROP_SCHEMA:
            return proposal
        if context["purpose"] == "reconstruction":
            return verdict()  # hand beside the figure; all mathematical content visible
        return verdict("cleanliness", "unrelated_content", "none")

    model.complete.side_effect = complete
    block["kind"] = "diagram"
    crop_id = run(example, model)[0]
    assert crop_id
    record = read_json(job / "illustration-crops" / crop_id / "crop.json")
    assert record["inputs"]["purpose"] == "reconstruction"
    block["kind"] = "screenshot"
    assert run(example, model)[0:2] == ("", False)
    assert model.complete.call_count == 4
    assert resolve_crop(job, evidence, "v01_ill01") == source
    assert resolve_crop(job, evidence, "v01_ill01", crop_id).is_file()


def test_inferior_projection_recrops_same_frame_and_verifies_new_pixels(example):
    model = client()
    model.complete.side_effect = [
        model.complete.return_value, verdict("readability", "inferior_source", "recrop"),
        {"status": "cropped", "box": [5, 10, 60, 70], "reason": "Прямой захват той же фигуры."},
        verdict()]
    crop_id = run(example, model)[0]
    assert crop_id
    record = read_json(example[0] / "illustration-crops" / crop_id / "crop.json")
    assert record["box"] == [5, 10, 60, 70]
    assert json.loads(model.complete.call_args_list[2].args[1])["repair_feedback"]["defects"] == [
        "inferior_source"]


def test_unusable_alternate_does_not_authorize_excluding_original_observation(example):
    add_frame(example, "v02", 8)
    model = client()
    model.complete.side_effect = [
        {"ranked_frame_ids": ["v02"], "reason": "Возможный чистый кадр."},
        {"status": "unusable", "box": [0, 0, 0, 0], "reason": "Здесь графика нет."},
        {"status": "uncertain", "box": [0, 0, 0, 0], "reason": "Рисунок закрыт рукой."}]
    crop_id, retry, reason, unusable = run(example, model)
    assert not crop_id and not retry and reason and not unusable
    assert read_json(example[0] / "archive/attempt-01/status.json")["status"] == "uncertain"


def test_ranking_outage_preserves_verified_original_and_reruns_ranking_later(example):
    add_frame(example, "v02", 8)
    model = client()
    model.complete.side_effect = [RuntimeError("ranking offline"),
                                  model.complete.return_value, verdict()]
    crop_id, retry, reason, unusable = run(example, model)
    assert crop_id and retry and not reason and not unusable
    assert read_json(example[0] / "archive/attempt-01/source-selection/error.json")[
        "message"] == "ranking offline"
    recovery = client()
    assert run(example, recovery) == (crop_id, False, "", False)
    assert recovery.complete.call_count == 3  # did not reuse incomplete ranking success
    record = read_json(example[0] / "illustration-crops" / crop_id / "crop.json")
    assert record["source_selection_complete"]


def test_original_gets_last_attempt_after_repeated_recrops_of_ranked_alternate(example):
    add_frame(example, "v02", 8)
    model = client()
    model.complete.side_effect = [
        {"ranked_frame_ids": ["v02"], "reason": "Более чёткий кадр."},
        model.complete.return_value, verdict("completeness", repair="recrop"),
        model.complete.return_value, verdict("completeness", repair="recrop"),
        model.complete.return_value, verdict()]
    crop_id, retry, reason, unusable = run(example, model)
    assert crop_id and not retry and not reason and not unusable
    record = read_json(example[0] / "illustration-crops" / crop_id / "crop.json")
    assert record["frame"]["visual_id"] == "v01"
    assert model.complete.call_count == 7
    assert read_json(example[0] / "archive/attempt-01/status.json")["repairs"] == 2


@pytest.fixture
def scroll_example(example):
    _, evidence, _, frame, source = example
    page = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 256, 280), False)
    rng = random.Random(41)
    colors = {(x, y): tuple(rng.randrange(30, 220) for _ in range(3))
              for x in range(22) for y in range(22)}
    for y in range(page.height):
        for x in range(page.width):
            page.set_pixel(x, y, colors[x // 12, y // 13])
    for identifier, offset, timestamp in [("v01", 0, 3), ("v02", 40, 43)]:
        pixels = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, offset, 256, offset + 220), False)
        pixels.copy(page, pixels.irect)
        pixels.set_origin(0, 0)
        pixels.save(frame.with_name(f"{identifier}_{'a' * 12}.png"))
        if identifier == "v02":
            evidence["visuals"].append({"id": identifier, "timestamp": timestamp,
                                       "status": "complete", "illustrations": []})
    observation = evidence["visuals"][0]["illustrations"][0]
    observation["box"] = [0.125, 0.2, 1, 0.95]
    source_pixels = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(32, 44, 256, 209), False)
    source_pixels.copy(pymupdf.Pixmap(frame), source_pixels.irect)
    source_pixels.save(source)
    observation["image_sha256"] = sha256_file(source)
    write_json(source.parent / "manifest.json", {"version": 1, "assets": {
        "v01_ill01": {"path": "illustrations/v01_ill01.png", "sha256": sha256_file(source)}}})
    return example


def scroll_client(accepted=True):
    model = client()

    def complete(instructions, text, schema, **kwargs):
        context = json.loads(text)
        if schema == crop_validation.CROP_VERIFICATION_SCHEMA:
            return verdict() if accepted else verdict("identity", "wrong_stage", "none")
        if "stitch" in context["frame"]:
            width, height = context["frame_size"]
            return {"status": "cropped", "box": [0, 0, width, height], "reason": "Полный график."}
        return {"status": "uncertain", "box": [0, 0, 0, 0], "reason": "Подписи за границей кадра."}

    model.complete.side_effect = complete
    return model


def test_scroll_recovery_verifies_both_native_sources_and_reuses_success(scroll_example, monkeypatch):
    job, evidence, _, _, source = scroll_example
    original = {path: path.read_bytes() for path in [source, *(job / "frames").glob("*.png")]}
    model = scroll_client()

    crop_id, retry, reason, unusable = run(scroll_example, model)

    assert crop_id and not retry and not reason and not unusable
    assert model.complete.call_count == 3  # exhausted native view, assembled crop, independent check
    record = read_json(job / "illustration-crops" / crop_id / "crop.json")
    assert record["version"] == 3 and fingerprint(record["inputs"]) == crop_id
    assert record["inputs"]["stitched_frames"] == [record["frame"]]
    assert record["frame"]["size"] == [224, 260]
    for call in model.complete.call_args_list[1:]:
        images = dict(call.kwargs["images"])
        assert images["SOURCE_FRAME_1"].read_bytes() == original[job / "frames/v01_aaaaaaaaaaaa.png"]
        assert images["SOURCE_FRAME_2"].read_bytes() == original[job / "frames/v02_aaaaaaaaaaaa.png"]
        assert illustration_crops.STITCH_INSTRUCTIONS in call.args[0]
    assert "repair_feedback" not in json.loads(model.complete.call_args_list[-1].args[1])
    assert all(path.read_bytes() == saved for path, saved in original.items())
    assert resolve_crop(job, evidence, "v01_ill01", crop_id).is_file()
    generator = Mock(side_effect=AssertionError("Successful source must be reused before native calls"))
    monkeypatch.setattr(illustration_stitching, "stitched_sources", generator)
    cached_client = scroll_client()
    assert run(scroll_example, cached_client) == (crop_id, False, "", False)
    cached_client.complete.assert_not_called()
    generator.assert_not_called()


def test_scroll_recovery_does_not_publish_a_mixed_stage_verdict(scroll_example):
    model = scroll_client(accepted=False)

    crop_id, retry, reason, unusable = run(scroll_example, model)

    assert not crop_id and not retry and reason and not unusable
    assert model.complete.call_count == 3
    assert not list((scroll_example[0] / "illustration-crops").glob("*/crop.json"))


@pytest.mark.parametrize("failure", ["transport", "unusable"])
def test_scroll_recovery_skips_transient_errors_and_nonfigures(scroll_example, monkeypatch, failure):
    generator = Mock(side_effect=AssertionError("Recovery must not run"))
    monkeypatch.setattr(illustration_stitching, "stitched_sources", generator)
    model = client()
    model.complete.side_effect = (RuntimeError("offline") if failure == "transport" else None)
    model.complete.return_value = {"status": "unusable", "box": [0, 0, 0, 0], "reason": "Только код."}

    crop_id, retry, _, unusable = run(scroll_example, model)

    assert not crop_id and retry == (failure == "transport") and unusable == (failure == "unusable")
    generator.assert_not_called()


def test_scroll_recovery_bounds_independent_candidate_attempts(scroll_example, monkeypatch):
    original = illustration_stitching.stitched_sources
    monkeypatch.setattr(illustration_stitching, "stitched_sources",
                        lambda *args: original(*args) * 3)
    model = scroll_client(accepted=False)

    assert not run(scroll_example, model)[0]

    assert model.complete.call_count == 5  # one native call and two proposal/verification pairs


@pytest.mark.parametrize("changed", ["source", "assembly", "crop"])
def test_scroll_crop_resolver_rechecks_native_and_assembled_pixels(scroll_example, changed):
    job, evidence, _, _, _ = scroll_example
    crop_id = run(scroll_example, scroll_client())[0]
    cache = job / "illustration-crops" / crop_id
    path = {"source": job / "frames/v02_aaaaaaaaaaaa.png",
            "assembly": cache / "verification/frame.png", "crop": cache / "crop.png"}[changed]
    pixels = pymupdf.Pixmap(path)
    pixels.set_pixel(10, 10, (0, 0, 0))
    pixels.save(path)
    if changed == "crop":
        record = read_json(cache / "crop.json")
        record["image_sha256"] = record["verification"]["image_sha256"] = sha256_file(path)
        write_json(cache / "crop.json", record)  # matching hashes cannot authorize invented pixels

    with pytest.raises(ValueError):
        resolve_crop(job, evidence, "v01_ill01", crop_id)


@pytest.mark.parametrize("changed", ["caption", "instructions"])
def test_scroll_success_cache_requires_same_task_and_policy(scroll_example, monkeypatch, changed):
    previous = run(scroll_example, scroll_client())[0]
    if changed == "caption":
        scroll_example[2]["caption"] = "Другая группа панелей"
    else:
        monkeypatch.setattr(illustration_crops, "STITCH_INSTRUCTIONS",
                            illustration_crops.STITCH_INSTRUCTIONS + "\nCheck a revised rule.")
    model = scroll_client()

    current = run(scroll_example, model)[0]

    assert current and current != previous
    assert model.complete.call_count == 3
