from pathlib import Path

import pytest
from PIL import Image

from shad_helper import extraction, person_segmentation
from shad_helper.jobs import checkpoint, evidence_index, read_json, sha256_file
from shad_helper.notes import chapter_context


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"mock video")
    out = tmp_path / "job"
    requests, events = {}, []
    monkeypatch.setattr(extraction.shutil, "which", lambda binary: f"/mock/{binary}")
    monkeypatch.setattr(extraction, "probe", lambda path: {
        "duration": 1200, "width": 40, "height": 20})
    monkeypatch.setattr(extraction, "import_sources", lambda paths, job: [])

    def chunks(*args):
        events.append("audio")
        return []

    def scan(*args):
        events.append("scan")
        return []

    def worker(job, task, request):
        events.append(task)
        requests[task] = request
        return {"items": []}

    monkeypatch.setattr(extraction, "_audio_chunks", chunks)
    monkeypatch.setattr(extraction, "_scan_video", scan)
    monkeypatch.setattr(extraction, "_frames", lambda *args: [])
    monkeypatch.setattr(extraction, "_worker", worker)
    monkeypatch.setattr(person_segmentation, "check_person_segmentation",
                        lambda: events.append("preflight"))
    return video, out, requests, events


@pytest.mark.parametrize("legacy", [False, True])
def test_default_mode_reaches_visual_workers_and_preserves_legacy_checkpoints(pipeline, legacy):
    video, out, requests, events = pipeline
    if legacy:
        # These fingerprints are the baseline inputs used before content grouping.
        inputs = {"video": sha256_file(video), "start": 0.0, "end": 1200}
        checkpoint(out, "keyframes", inputs, list)
        checkpoint(out, "frames", {**inputs, "selections": []}, list)

    result = read_json(extraction.extract(video, [], out, ocr_model="mock-ocr"))
    assert "vision_mode" not in requests["asr"]
    for task in ("vision", "ocr"):
        if legacy:
            assert "vision_mode" not in requests[task]
            assert "grouping_version" not in requests[task]
        else:
            assert requests[task]["vision_mode"] == "content"
            assert requests[task]["grouping_version"] == 1
    if legacy:
        assert "preflight" not in events
        assert "scan" not in events
        assert not (out / "extraction-config.json").exists()
        assert "visual_processing" not in result
    else:
        assert events.index("preflight") < events.index("audio") < events.index("asr")
        assert result["visual_processing"] == {
            "vision_mode": "content", "grouping_version": 1}
        assert read_json(out / "extraction-config.json") == result["visual_processing"]


def test_missing_native_support_fails_before_audio_or_model_work(pipeline, monkeypatch):
    video, out, requests, events = pipeline

    def unavailable():
        raise RuntimeError("Install Apple Vision bindings")

    monkeypatch.setattr(person_segmentation, "check_person_segmentation", unavailable)
    with pytest.raises(RuntimeError, match="Apple Vision bindings"):
        extraction.extract(video, [], out)
    assert events == []
    assert requests == {}
    assert not (out / "transcript.json").exists()
    assert not (out / "checkpoints").exists()


def test_explicit_baseline_new_job_skips_native_preflight(pipeline):
    video, out, requests, events = pipeline
    extraction.extract(video, [], out, vision_mode="baseline", ocr_model="mock-ocr")
    assert "preflight" not in events
    assert "vision_mode" not in requests["vision"]
    assert "vision_mode" not in requests["ocr"]
    assert read_json(out / "extraction-config.json")["vision_mode"] == "baseline"


def test_legacy_mode_switch_is_blocked_before_audio_or_preflight(pipeline):
    video, out, requests, events = pipeline
    out.mkdir()
    artifact = out / "transcript.srt"
    artifact.write_bytes(b"original transcript")
    with pytest.raises(ValueError, match="new output directory"):
        extraction.extract(video, [], out, vision_mode="content")
    assert requests == {}
    assert events == []
    assert artifact.read_bytes() == b"original transcript"
    assert not (out / "extraction-config.json").exists()


def test_alias_ocr_and_sources_keep_occurrence_times_without_future_content(pipeline, monkeypatch):
    video, out, requests, events = pipeline
    source = video.with_suffix(".pdf")
    source.write_bytes(b"mock source")
    formula = "x^2+y^2=z^2"

    def sources(paths, job):
        rendered = job / "source_assets" / "page.png"
        rendered.parent.mkdir()
        Image.new("RGB", (40, 20), "white").save(rendered)
        return [{"id": "source_1", "kind": "pdf", "source": source.name,
                 "location": "page 1", "text": f"${formula}$", "code": "",
                 "images": ["source_assets/page.png"],
                 "rendered_image": "source_assets/page.png"}]

    def frames(path, job, selections):
        directory = job / "frames"
        directory.mkdir()
        items = []
        for identifier, timestamp, color in (
                ("visual_00000", 10, "white"),
                ("visual_00001", 1000, "gray"),
                ("visual_00002", 1100, "black")):
            image = directory / f"{identifier}.png"
            Image.new("RGB", (40, 20), color).save(image)
            items.append({"id": identifier, "timestamp": timestamp, "image": str(image),
                          "sha256": sha256_file(image)})
        return items

    def worker(job, task, request):
        events.append(task)
        requests[task] = request
        if task == "asr":
            return {"items": []}
        if task == "ocr":
            return {"items": [{"id": item["id"], "text": (
                "future_formula" if item["id"].startswith("visual_00002") else formula)}
                for item in request["items"]]}
        assert task == "vision"
        items = request["items"]
        results = []
        for position, item in enumerate(items):
            canonical = items[0] if position == 1 else item
            results.append({**item, "image": canonical["image"],
                            "analysis_id": canonical["id"],
                            "original_image": item["image"], "kind": "whiteboard",
                            "summary": "", "diagram": "", "issues": [],
                            "regions": [{"kind": "formula", "bbox": [0, 0, 1, 1]}]})
        return {"items": results}

    monkeypatch.setattr(extraction, "import_sources", sources)
    monkeypatch.setattr(extraction, "_frames", frames)
    monkeypatch.setattr(extraction, "_worker", worker)
    evidence = read_json(extraction.extract(video, [source], out, ocr_model="mock-ocr"))

    first, alias, future, source_visual = evidence["visuals"]
    assert [visual["timestamp"] for visual in evidence["visuals"]] == [10, 1000, 1100, None]
    assert alias["analysis_id"] == first["id"]
    assert alias["image"] == first["image"]
    assert alias["original_image"] != alias["image"]
    assert alias["original_sha256"] == sha256_file(out / alias["original_image"])
    assert first["regions"][0]["text"] == alias["regions"][0]["text"] == formula
    assert first["regions"][0]["image"] == alias["regions"][0]["image"]
    assert len(requests["ocr"]["items"]) == 3
    assert {item["id"] for item in requests["ocr"]["items"]} == {
        "visual_00000_region_000", "visual_00002_region_000", "visual_source_1_region_000"}
    assert evidence["sources"][0]["matched_visual_ids"] == [first["id"], alias["id"]]

    context = chapter_context(evidence, {"start": 950, "end": 1020})
    assert [visual["id"] for visual in context["visuals"]] == [
        first["id"], alias["id"], source_visual["id"]]
    assert context["sources"][0]["id"] == "source_1"
    assert future not in context["visuals"]
    assert all(region["text"] != "future_formula" for visual in context["visuals"]
               for region in visual["regions"])
    assert evidence_index(context)[alias["regions"][0]["id"]] == "Видео 00:16:40"
    assert Path(requests["ocr"]["items"][0]["image"]).is_file()
