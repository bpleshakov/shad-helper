import copy

import pytest
from PIL import Image

from shad_helper.illustrations import (
    illustration_index,
    materialize_illustrations,
    resolve_illustration,
)
from shad_helper.jobs import evidence_index, read_json, sha256_file, validate_evidence, write_json


@pytest.fixture
def cropped(tmp_path):
    frame = tmp_path / "frame.png"
    image = Image.new("RGB", (100, 80), "white")
    for x in range(20, 60):
        for y in range(8, 40):
            image.putpixel((x, y), (x, y, 0))
    image.save(frame)
    items = [{"id": "frame_1", "image": str(frame), "content": {"illustrations": [
        {"source_kind": "drawn", "box": [0.2, 0.1, 0.6, 0.5],
         "description": "Два объекта и стрелка"},
        {"source_kind": "printed", "box": [0.6, 0, 1, 1], "description": "График"},
    ]}}]
    records = materialize_illustrations(tmp_path, items)["frame_1"]
    evidence = {"schema_version": 5, "title": "Лекция", "media": {}, "segments": [],
                "visuals": [{"id": "frame_1", "timestamp": 25, "status": "complete",
                             "kind": "slides", "text": "", "formulas": [], "diagrams": [],
                             "unreadable": [], "illustrations": records}]}
    return tmp_path, items, evidence


def test_native_crop_pixels_stable_ids_and_provenance(cropped):
    job, items, evidence = cropped
    assert validate_evidence(evidence) == evidence
    records = evidence["visuals"][0]["illustrations"]
    assert [record["id"] for record in records] == ["frame_1_ill01", "frame_1_ill02"]
    path = resolve_illustration(job, evidence, records[0]["id"])
    with Image.open(path) as image:
        assert image.size == (46, 38)
        assert image.getpixel((0, 0)) == (255, 255, 255)
        assert image.getpixel((3, 3)) == (20, 8, 0)
        assert image.getpixel((42, 34)) == (59, 39, 0)
    assert records[0]["box"] == [0.17, 0.0625, 0.63, 0.5375]
    assert records[1]["box"] == [0.57, 0, 1, 1]
    assert materialize_illustrations(job, items)["frame_1"] == records
    assert illustration_index(evidence)[records[0]["id"]] == {
        **records[0], "visual_id": "frame_1", "timestamp": 25, "status": "complete"}
    assert records[0]["id"] in evidence_index(evidence)
    assert "00:00:25" in evidence_index(evidence)[records[0]["id"]]


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "hash", "manifest_hash",
                                       "absolute", "parent", "outside", "unknown", "not_png"])
def test_rejects_missing_corrupt_or_escaping_assets(cropped, mutation):
    job, _, evidence = cropped
    identifier = "frame_1_ill01"
    path = job / "illustrations" / f"{identifier}.png"
    manifest_path = job / "illustrations/manifest.json"
    manifest = read_json(manifest_path)
    asset = manifest["assets"][identifier]
    if mutation == "missing":
        path.unlink()
    elif mutation in ("corrupt", "not_png"):
        path.write_bytes(b"\x89PNG\r\n\x1a\ncorrupt" if mutation == "corrupt" else b"not png")
        digest = sha256_file(path)
        asset["sha256"] = digest
        evidence["visuals"][0]["illustrations"][0]["image_sha256"] = digest
    elif mutation == "hash":
        path.write_bytes(b"changed source")
    elif mutation == "manifest_hash":
        asset["sha256"] = "a" * 64
    elif mutation == "absolute":
        asset["path"] = str(path)
    elif mutation == "parent":
        asset["path"] = f"illustrations/../illustrations/{identifier}.png"
    elif mutation == "outside":
        target = job / "outside.png"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    elif mutation == "unknown":
        identifier = "frame_1_ill99"
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError):
        resolve_illustration(job, evidence, identifier)


@pytest.mark.parametrize("field,value", [
    ("id", "arbitrary"), ("source_kind", "other"), ("description", ""),
    ("image_sha256", "not-a-hash"), ("box", [0, 0, float("inf"), 1]),
    ("box", [0, 0, 1, 0]), ("path", "private/file.png"),
])
def test_v5_rejects_invalid_illustration_records(cropped, field, value):
    _, _, evidence = cropped
    evidence["visuals"][0]["illustrations"][0][field] = value
    with pytest.raises(ValueError):
        validate_evidence(evidence)


def test_v5_requires_illustration_list_but_v4_and_audio_remain_supported(cropped):
    _, _, evidence = cropped
    evidence["visuals"][0].pop("illustrations")
    with pytest.raises(ValueError):
        validate_evidence(evidence)
    evidence["schema_version"] = 4
    assert validate_evidence(evidence) == evidence
    audio = copy.deepcopy(evidence)
    audio.pop("visuals")
    audio["schema_version"] = 3
    assert validate_evidence(audio) == audio


def test_missing_crops_are_recreated_from_cached_observations(cropped):
    job, items, evidence = cropped
    path = resolve_illustration(job, evidence, "frame_1_ill01")
    path.unlink()
    materialize_illustrations(job, items)
    assert resolve_illustration(job, evidence, "frame_1_ill01").is_file()


@pytest.mark.parametrize("frame_id", ["frame.1", "_frame", "a" * 75])
def test_generated_ids_fit_notes_identifier_contract(cropped, frame_id):
    job, items, _ = cropped
    items[0]["id"] = frame_id
    with pytest.raises(ValueError, match="visual item ID"):
        materialize_illustrations(job, items)
