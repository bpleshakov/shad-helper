import json
import shutil
import subprocess
import sys
import wave
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from shad_helper import extraction, frame_novelty, local_worker, video, vision_worker
from shad_helper.illustrations import resolve_illustration
from shad_helper.jobs import (
    AUDIO_CONFIG,
    check_job,
    fingerprint,
    read_json,
    validate_evidence,
    write_json,
)
from shad_helper.vision_config import DEFAULT_VISION_MODEL, VISION_MODELS

CONTENT = {"kind": "whiteboard", "text": "Определение", "formulas": [r"x^T\theta"],
           "diagrams": [], "illustrations": [], "unreadable": []}


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    media, job = tmp_path / "lecture.mp4", tmp_path / "job"
    media.write_bytes(b"recording")
    monkeypatch.setattr(extraction.shutil, "which", lambda name: name)
    monkeypatch.setattr(extraction, "probe", lambda path: {"duration": 120})
    metadata = {"stream_index": 1, "start": 2, "end": 120, "duration": 118,
                "width": 320, "height": 180, "container_start": 5, "rotation": 0}
    probe_video = Mock(return_value=metadata)
    monkeypatch.setattr(extraction, "probe_video", probe_video)
    monkeypatch.setattr(video, "probe_video", probe_video)
    monkeypatch.setattr(frame_novelty, "_PersonSegmenter", lambda:
                        lambda image: np.zeros(image.shape[:2], dtype=bool))
    order = []
    proxy_frames = {}

    def chunks(media, out, start, end, media_hash, audio_start=0):
        audio = out / "audio/chunk_0000.wav"
        audio.parent.mkdir(exist_ok=True)
        audio.write_bytes(b"audio")
        return [{"id": "chunk_0000", "audio": str(audio), "offset": 100,
                 "core_start": start, "core_end": end}]

    monkeypatch.setattr(extraction, "_audio_chunks", chunks)
    transcribe = Mock(return_value={"segments": [
        {"start": 5.3, "end": 6.3, "text": "Теорема"}], "language": "ru"})
    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=transcribe))

    def asr(out, request):
        order.append("asr")
        return local_worker.run({"job": str(out), **request})

    monkeypatch.setattr(extraction, "_worker", asr)
    blank = np.full((180, 320), 255, np.uint8)
    changed = blank.copy()
    changed[20:80, 30:90] = 0

    def scan(*args):
        assert order[-1] == "asr"
        order.append("selection")
        frames = [(105.25, blank), (106.75, changed)]
        proxy_frames.update(frames)
        return frames

    scan = Mock(side_effect=scan)
    monkeypatch.setattr(video, "_proxy_frames", scan)

    def native(media, metadata, timestamp, image):
        image.parent.mkdir(exist_ok=True)
        Image.fromarray(proxy_frames[timestamp]).convert("RGB").save(image)

    native = Mock(side_effect=native)
    monkeypatch.setattr(video, "_native_frame", native)
    runtime = Mock()
    runtime.infer.return_value = (json.dumps(CONTENT, ensure_ascii=False), {
        "seconds": 1, "finish_reason": "stop", "generation_tokens": 20})
    runtime.infer_video.return_value = (json.dumps({"observations": [
        {"frame_indices": [0, 1], **CONTENT}]}), {"finish_reason": "stop"})
    loader = Mock(return_value=runtime)
    monkeypatch.setattr(vision_worker, "_load_runtime", loader)
    monkeypatch.setattr(vision_worker, "_versions", lambda: {"mlx-vlm": "test"})

    def vision(out, request):
        assert read_json(out / "transcript.json")[0]["text"] == "Теорема"
        assert read_json(out / "evidence.partial.json")["schema_version"] == 3
        assert read_json(out / "extraction-status.json")["state"] == "running"
        order.append("vision")
        return vision_worker.run({"job": str(out), **request})

    vision = Mock(side_effect=vision)
    monkeypatch.setattr(extraction, "_vision_worker", vision)
    return SimpleNamespace(media=media, job=job, order=order, transcribe=transcribe,
                           runtime=runtime, loader=loader, scan=scan, native=native,
                           vision=vision, probe_video=probe_video, proxy_frames=proxy_frames)


def run(pipeline, **kwargs):
    return read_json(extraction.extract(pipeline.media, pipeline.job, start=105, duration=3,
                                       **kwargs))


@pytest.mark.parametrize("alias,model,revision", [
    ("4b", "mlx-community/Qwen3.5-4B-4bit", "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"),
    ("9b", "mlx-community/Qwen3.5-9B-4bit", "8b2b98c00a6b4d291155e4890773ca8f769aee53"),
    ("35b", "mlx-community/Qwen3.5-35B-A3B-4bit", "1e20fd8d42056f870933bf98ca6211024744f7ec"),
])
def test_presets_load_pinned_models_sequentially_and_keep_source_timeline(
        pipeline, alias, model, revision):
    evidence = run(pipeline, vision_model=alias)
    assert DEFAULT_VISION_MODEL == "35b"
    pipeline.loader.assert_called_once_with(model, revision)
    assert pipeline.order == ["asr", "selection", "vision"]
    assert validate_evidence(evidence) == evidence
    assert evidence["schema_version"] == 5
    assert [item["timestamp"] for item in evidence["visuals"]] == [105.25, 106.75]
    assert evidence["segments"][0]["start"] == 105.3
    assert evidence["media"]["clip_start"] == 105
    assert all(set(item) == {"id", "timestamp", "status", "kind", "text", "formulas",
                             "diagrams", "illustrations", "unreadable"} for item in evidence["visuals"])
    assert read_json(pipeline.job / "extraction-config.json") == {
        "schema_version": 5, "pipeline": "audio_video", "vision_model": alias}
    diagnostics = read_json(pipeline.job / "extraction-diagnostics.json")
    assert diagnostics["evidence_fingerprint"] == fingerprint(evidence)
    assert diagnostics["models"]["vision"] == {"preset": alias, "model": model, "mode": "frames",
                                                "revision": revision}
    assert diagnostics["selection"]["clip_start"] == 105
    assert diagnostics["visuals"][0]["image"]
    assert diagnostics["visuals"][0]["reason"] == "initial"
    assert "raw" in diagnostics["visuals"][0]["diagnostics"]["attempts"][0]
    timings = read_json(pipeline.job / "extraction-timings.json")
    assert 0 <= timings["audio_asr_seconds"] <= timings["total_seconds"]
    assert 0 <= timings["vision_seconds"] <= timings["total_seconds"]


def test_video_mode_preserves_source_crops_and_notes_handoff(pipeline):
    from shad_helper.notes import generate_notes

    content = {**CONTENT, "illustrations": [
        {"source_kind": "drawn", "box": [0, 0, 1, 1], "description": "Схема"}]}
    pipeline.runtime.infer_video.return_value = (json.dumps({"observations": [
        {"frame_indices": [0, 1], **content}]}), {"finish_reason": "stop"})
    evidence = run(pipeline, vision_mode="video")
    assert validate_evidence(evidence) == evidence
    extraction.assert_extraction_complete(pipeline.job, evidence)
    check_job(pipeline.job)
    assert read_json(pipeline.job / "extraction-config.json")["vision_mode"] == "video"
    pipeline.runtime.infer.assert_not_called()
    pipeline.runtime.infer_video.assert_called_once()
    assert [item["timestamp"] for item in evidence["visuals"]] == [105.25, 106.75]
    hashes = []
    for item in evidence["visuals"]:
        illustration = item["illustrations"][0]
        assert illustration["id"] == f"{item['id']}_ill01"
        crop = resolve_illustration(pipeline.job, evidence, illustration["id"])
        with Image.open(crop) as image:
            assert np.array_equal(np.asarray(image.convert("L")),
                                  pipeline.proxy_frames[item["timestamp"]])
        hashes.append(illustration["image_sha256"])
    assert len(set(hashes)) == 2
    handoff = generate_notes(pipeline.job, pipeline.job / "auth", manual=True)
    assert read_json(handoff / "inputs/ch001/context.json")["visuals"] == evidence["visuals"]


def test_mode_switches_reuse_asr_and_frames_with_separate_vision_caches(pipeline):
    original = run(pipeline)
    video_evidence = run(pipeline, vision_mode="video")
    assert video_evidence == original
    assert run(pipeline, vision_mode="video") == video_evidence
    assert read_json(pipeline.job / "extraction-diagnostics.json")["vision"]["cached_items"] == 2
    assert run(pipeline) == original
    assert pipeline.transcribe.call_count == 1
    assert pipeline.scan.call_count == 1
    assert pipeline.native.call_count == 2
    assert pipeline.runtime.infer.call_count == 2
    assert pipeline.runtime.infer_video.call_count == 1
    assert "vision_mode" not in read_json(pipeline.job / "extraction-config.json")


def test_invalid_mode_is_rejected_before_creating_job(pipeline):
    with pytest.raises(ValueError, match="vision_mode"):
        run(pipeline, vision_mode="unknown")
    assert not pipeline.job.exists()
    pipeline.transcribe.assert_not_called()


@pytest.mark.parametrize("mode", ["frames", "video"])
def test_off_skips_all_video_work_and_dependencies(pipeline, mode):
    evidence = run(pipeline, vision_model="off", vision_mode=mode)
    assert evidence["schema_version"] == 3
    assert "visuals" not in evidence
    assert read_json(pipeline.job / "extraction-config.json") == AUDIO_CONFIG
    pipeline.probe_video.assert_not_called()
    pipeline.scan.assert_not_called()
    pipeline.vision.assert_not_called()
    pipeline.loader.assert_not_called()


@pytest.mark.parametrize("cover", [False, True])
def test_default_audio_or_attached_cover_stays_audio_only(pipeline, monkeypatch, cover):
    probe = video.probe_video
    # Use the real probe logic to verify attached images do not become lecture frames.
    streams = [{"index": 0, "codec_type": "audio"}]
    if cover:
        streams.append({"index": 1, "codec_type": "video", "disposition": {"attached_pic": 1}})
    monkeypatch.setattr(video.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=json.dumps({"streams": streams})))
    monkeypatch.setattr(extraction, "probe_video", ORIGINAL_PROBE_VIDEO)
    evidence = run(pipeline)
    assert evidence["schema_version"] == 3
    assert "visuals" not in evidence
    assert read_json(pipeline.job / "extraction-config.json") == AUDIO_CONFIG
    pipeline.scan.assert_not_called()
    pipeline.vision.assert_not_called()
    probe.assert_not_called()


ORIGINAL_PROBE_VIDEO = video.probe_video


def test_video_without_frames_keeps_v5_without_loading_vision(pipeline, monkeypatch):
    monkeypatch.setattr(extraction, "select_frames", lambda *args, **kwargs: {
        "items": [], "diagnostics": {"selected_frames": 0, "proxy_frames": 0}})
    evidence = run(pipeline)
    assert evidence["schema_version"] == 5
    assert evidence["visuals"] == []
    assert read_json(pipeline.job / "extraction-config.json")["vision_model"] == "35b"
    pipeline.loader.assert_not_called()
    pipeline.vision.assert_not_called()


def test_unknown_model_rejected_before_creating_job_or_probing(pipeline):
    with pytest.raises(ValueError, match="vision_model"):
        run(pipeline, vision_model="unknown")
    assert not pipeline.job.exists()
    pipeline.probe_video.assert_not_called()


def test_incomplete_frame_retained_and_only_failed_frame_retried(pipeline):
    success = pipeline.runtime.infer.return_value
    pipeline.runtime.infer.side_effect = [success, ("bad", {}), ("still bad", {})]
    evidence = run(pipeline)
    assert [i["status"] for i in evidence["visuals"]] == ["complete", "incomplete"]
    failed = evidence["visuals"][1]
    assert failed["timestamp"] == 106.75
    assert failed["text"] == ""
    assert failed["unreadable"] == []
    assert not (pipeline.job / "checkpoints/vision" / f"{failed['id']}.json").exists()
    diagnostics = read_json(pipeline.job / "extraction-diagnostics.json")
    assert len(diagnostics["visuals"][1]["diagnostics"]["attempts"]) == 2
    assert diagnostics["warnings"][0]["evidence_ids"] == [failed["id"]]
    pipeline.runtime.infer.side_effect = None
    pipeline.runtime.infer.reset_mock()
    repaired = run(pipeline)
    assert all(i["status"] == "complete" for i in repaired["visuals"])
    pipeline.runtime.infer.assert_called_once()
    assert pipeline.transcribe.call_count == 1
    assert pipeline.scan.call_count == 1


def test_model_switches_reuse_audio_and_frames_but_refresh_visuals(pipeline):
    for alias in ("35b", "9b", "4b"):
        run(pipeline, vision_model=alias)
        assert pipeline.loader.call_args.args == tuple(VISION_MODELS[alias].values())
    assert pipeline.transcribe.call_count == 1
    assert pipeline.scan.call_count == 1
    assert pipeline.native.call_count == 2
    assert pipeline.runtime.infer.call_count == 6
    run(pipeline, vision_model="4b")
    assert pipeline.loader.call_count == 3
    assert pipeline.runtime.infer.call_count == 6
    assert read_json(pipeline.job / "extraction-diagnostics.json")["vision"]["cached_items"] == 2
    # Switching off removes visual evidence while retaining all reusable artifacts.
    assert run(pipeline, vision_model="off")["schema_version"] == 3
    assert pipeline.runtime.infer.call_count == 6
    assert pipeline.transcribe.call_count == 1


def test_board_regions_reduce_native_retrieval_and_inference_and_resume(pipeline):
    def scan(*args):
        pipeline.order.append("selection")
        frames = []
        for step in range(80):
            gray = np.full((240, 400), 40, np.uint8)
            gray[20:180, 12:388] = 230
            gray[50:54, 45:105] = 50
            x = 15 + (step * 7) % 300
            gray[195:238, x:x + 50] = 110
            frames.append((105 + step / 2, gray))
        pipeline.proxy_frames.update(frames)
        return frames

    pipeline.probe_video.return_value.update(end=180, duration=178, width=400, height=240)
    pipeline.scan.side_effect = scan
    evidence = read_json(extraction.extract(pipeline.media, pipeline.job, start=105, duration=40))
    assert len(evidence["visuals"]) == 2
    assert pipeline.runtime.infer.call_count == 2
    assert pipeline.native.call_count == 2
    diagnostics = read_json(pipeline.job / "extraction-diagnostics.json")
    assert diagnostics["selection"]["region_frames"] == 80
    assert diagnostics["selection"]["full_frame_frames"] == 0
    assert diagnostics["visuals"][0]["content_regions"]
    assert "content_regions" not in evidence["visuals"][0]
    assert read_json(extraction.extract(pipeline.media, pipeline.job,
                                        start=105, duration=40)) == evidence
    assert pipeline.runtime.infer.call_count == 2
    assert pipeline.native.call_count == 2
    pipeline.scan.assert_called_once()


def test_novelty_filter_skips_inference_for_lecturer_motion_and_resumes(pipeline, monkeypatch):
    frames = []
    for step in range(80):
        gray = np.full((240, 400), 240, np.uint8)
        gray[25:30, 25:180] = 20
        gray[90:95, 25:180] = 20
        if step:
            x = 30 + (step * 7) % 260
            gray[50:, x:x + 60] = 80
        frames.append((105 + step / 2, gray))

    def scan(*args):
        pipeline.order.append("selection")
        pipeline.proxy_frames.update(frames)
        return frames

    factory = Mock(return_value=lambda image: np.all(image == 80, axis=2))
    monkeypatch.setattr(frame_novelty, "_PersonSegmenter", factory)
    pipeline.probe_video.return_value.update(end=180, duration=178, width=400, height=240)
    pipeline.scan.side_effect = scan
    candidates, _ = video._select_proxy_frames(frames)
    assert len(candidates) > 1
    evidence = read_json(extraction.extract(pipeline.media, pipeline.job, start=105, duration=40))
    assert len(evidence["visuals"]) == 1
    pipeline.runtime.infer.assert_called_once()
    # Native candidates remain available for inspection even when inference is skipped.
    assert pipeline.native.call_count == len(candidates)
    selection = read_json(pipeline.job / "extraction-diagnostics.json")["selection"]
    assert selection["novelty"]["suppressed_frames"] == len(candidates) - 1
    assert read_json(extraction.extract(pipeline.media, pipeline.job,
                                        start=105, duration=40)) == evidence
    assert read_json(pipeline.job / "extraction-diagnostics.json")["selection"]["novelty"]["cache_hit"]
    pipeline.runtime.infer.assert_called_once()
    factory.assert_called_once()


def test_illustration_crops_persist_and_legacy_jobs_upgrade_without_retranscription(pipeline):
    evidence = run(pipeline)
    legacy = {**evidence, "schema_version": 4}
    for visual in legacy["visuals"]:
        visual.pop("illustrations")
    write_json(pipeline.job / "evidence.json", legacy)
    write_json(pipeline.job / "extraction-config.json", {
        "schema_version": 4, "pipeline": "audio_video", "vision_model": "35b"})
    for checkpoint in (pipeline.job / "checkpoints/vision").glob("*.json"):
        value = read_json(checkpoint)
        value["fingerprint"] = "legacy prompt"
        write_json(checkpoint, value)
    content = {**CONTENT, "illustrations": [
        {"source_kind": "drawn", "box": [0.1, 0.1, 0.5, 0.9], "description": "Стрелка"},
        {"source_kind": "printed", "box": [0.5, 0.1, 0.9, 0.9], "description": "График"},
    ]}
    pipeline.runtime.infer.return_value = (json.dumps(content), {"finish_reason": "stop"})
    upgraded = run(pipeline)
    assert upgraded["schema_version"] == 5
    assert pipeline.transcribe.call_count == 1
    assert pipeline.scan.call_count == 1
    assert pipeline.native.call_count == 2
    for visual in upgraded["visuals"]:
        assert len(visual["illustrations"]) == 2
        for observation in visual["illustrations"]:
            path = resolve_illustration(pipeline.job, upgraded, observation["id"])
            with Image.open(path) as image:
                assert image.size == (148, 156)


def test_fatal_vision_error_preserves_partial_transcript_and_can_resume(pipeline):
    pipeline.loader.side_effect = RuntimeError("model unavailable")
    with pytest.raises(RuntimeError, match="model unavailable"):
        run(pipeline)
    assert not (pipeline.job / "evidence.json").exists()
    evidence = read_json(pipeline.job / "evidence.partial.json")
    assert evidence["schema_version"] == 3
    assert evidence["segments"][0]["text"] == "Теорема"
    assert (pipeline.job / "transcript.srt").exists()
    diagnostics = read_json(pipeline.job / "extraction-diagnostics.json")
    assert diagnostics["vision"] == {"status": "failed", "error": "model unavailable"}
    assert diagnostics["evidence_fingerprint"] == fingerprint(evidence)
    assert "vision_seconds" in read_json(pipeline.job / "extraction-timings.json")
    assert read_json(pipeline.job / "extraction-status.json")["state"] == "failed"
    with pytest.raises(ValueError, match="Extraction is incomplete"):
        extraction.assert_extraction_complete(pipeline.job, evidence)
    check_job(pipeline.job)
    pipeline.loader.side_effect = None
    resumed = run(pipeline)
    assert resumed["schema_version"] == 5
    extraction.assert_extraction_complete(pipeline.job, resumed)
    assert not (pipeline.job / "evidence.partial.json").exists()
    assert pipeline.transcribe.call_count == 1
    assert pipeline.scan.call_count == 1


def test_failed_video_rerun_keeps_completed_evidence_but_blocks_notes(pipeline):
    completed = run(pipeline)
    before = (pipeline.job / "evidence.json").read_bytes()
    pipeline.vision.side_effect = RuntimeError("interrupted rerun")
    with pytest.raises(RuntimeError, match="interrupted rerun"):
        run(pipeline, vision_model="9b")
    assert (pipeline.job / "evidence.json").read_bytes() == before
    assert read_json(pipeline.job / "evidence.partial.json")["segments"] == completed["segments"]
    with pytest.raises(ValueError, match="Extraction is incomplete"):
        extraction.assert_extraction_complete(pipeline.job, completed)


def test_interrupted_asr_rerun_cannot_look_complete(pipeline):
    completed = run(pipeline)
    before = (pipeline.job / "evidence.json").read_bytes()
    pipeline.transcribe.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run(pipeline, language="en")
    assert (pipeline.job / "evidence.json").read_bytes() == before
    with pytest.raises(ValueError, match="Extraction is incomplete"):
        extraction.assert_extraction_complete(pipeline.job, completed)


def test_completed_evidence_allows_transcript_corrections_but_not_stale_identity(pipeline):
    evidence = run(pipeline)
    status = read_json(pipeline.job / "extraction-status.json")
    assert status["evidence_fingerprint"] == fingerprint(evidence)
    evidence["segments"][0]["text"] = "Исправленная теорема"
    extraction.assert_extraction_complete(pipeline.job, evidence)
    evidence["media"]["clip_end"] += 1
    with pytest.raises(ValueError, match="does not match"):
        extraction.assert_extraction_complete(pipeline.job, evidence)


def test_legacy_audio_only_video_job_requires_visual_extraction(pipeline):
    evidence = run(pipeline)
    (pipeline.job / "extraction-status.json").unlink()
    extraction.assert_extraction_complete(pipeline.job, evidence)
    evidence.pop("visuals")
    evidence["schema_version"] = 3
    with pytest.raises(ValueError, match="Visual extraction is missing"):
        extraction.assert_extraction_complete(pipeline.job, evidence)
    disabled = run(pipeline, vision_model="off")
    extraction.assert_extraction_complete(pipeline.job, disabled)
    (pipeline.job / "extraction-status.json").unlink()
    (pipeline.job / "extraction-config.json").unlink()
    extraction.assert_extraction_complete(pipeline.job, disabled)


def test_changed_config_requires_new_completed_extraction(pipeline):
    evidence = run(pipeline)
    write_json(pipeline.job / "extraction-config.json", AUDIO_CONFIG)
    with pytest.raises(ValueError, match="does not match"):
        extraction.assert_extraction_complete(pipeline.job, evidence)


@pytest.mark.parametrize("start,duration,clip_end", [(0, None, 120), (90, 15, 105)])
def test_longer_video_retains_tail_and_only_transcribes_available_audio(
        pipeline, monkeypatch, start, duration, clip_end):
    monkeypatch.setattr(extraction, "probe", lambda media: {"duration": 60})
    chunks = Mock(return_value=[{"id": "chunk_0000", "audio": "audio.wav"}])
    asr = Mock(return_value={"items": []})
    selected = {"id": "tail", "image": "frame.png", "timestamp": clip_end - 1}
    selector = Mock(return_value={"items": [selected], "diagnostics": {"selected_frames": 1}})
    vision = Mock(return_value={"items": [{**selected, "content": CONTENT,
                                           "diagnostics": {"status": "complete", "attempts": []}}],
                                 "diagnostics": {"fresh_items": 1, "cached_items": 0}})
    monkeypatch.setattr(extraction, "_audio_chunks", chunks)
    monkeypatch.setattr(extraction, "_worker", asr)
    monkeypatch.setattr(extraction, "select_frames", selector)
    monkeypatch.setattr(extraction, "_vision_worker", vision)
    evidence = read_json(extraction.extract(pipeline.media, pipeline.job,
                                           start=start, duration=duration))
    assert evidence["media"]["duration"] == 120
    assert evidence["media"]["clip_start"] == start
    assert evidence["media"]["clip_end"] == clip_end
    assert evidence["visuals"][0]["timestamp"] == clip_end - 1
    assert selector.call_args.args[2:4] == (start, clip_end)
    if start == 0:
        assert chunks.call_args.args[2:4] == (0, 60)
        asr.assert_called_once()
    else:
        chunks.assert_not_called()
        asr.assert_not_called()
        assert evidence["segments"] == []
        assert read_json(pipeline.job / "transcript.json") == []
    assert evidence["schema_version"] == 5


def test_off_still_rejects_clip_beyond_audio_even_with_longer_video(pipeline, monkeypatch):
    monkeypatch.setattr(extraction, "probe", lambda media: {"duration": 60})
    with pytest.raises(ValueError, match="duration"):
        extraction.extract(pipeline.media, pipeline.job, start=90, duration=15, vision_model="off")
    pipeline.probe_video.assert_not_called()
    assert not pipeline.job.exists()


@pytest.mark.parametrize("stream_duration,expected_end", [("2", 7), (None, 8)])
def test_audio_probe_accounts_for_relative_start_without_double_counting_format_duration(
        monkeypatch, stream_duration, expected_end):
    monkeypatch.setattr(extraction.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=json.dumps({
                            "streams": [{"duration": stream_duration, "start_time": "15"}],
                            "format": {"start_time": "10", "duration": "8"}})))
    assert extraction.probe("delayed.mov") == {"duration": expected_end, "audio_start": 5}


def test_audio_preroll_never_produces_negative_ffmpeg_seek(tmp_path, monkeypatch):
    commands = []

    def ffmpeg(command, **kwargs):
        commands.append(command)
        (tmp_path / "audio/chunk_0000.wav").write_bytes(b"audio")

    monkeypatch.setattr(extraction.subprocess, "run", ffmpeg)
    chunks = extraction._audio_chunks(tmp_path / "lecture.mkv", tmp_path, 0, 2,
                                      "hash", audio_start=-0.02)
    assert chunks[0]["offset"] == 0
    assert commands[0][commands[0].index("-ss") + 1] == "0"


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="FFmpeg is required")
@pytest.mark.parametrize("container_start", [0, 10])
@pytest.mark.parametrize("container", ["mov", "mkv"])
def test_real_delayed_audio_keeps_recording_timestamps_and_overlap(
        tmp_path, monkeypatch, container_start, container):
    media, job = tmp_path / f"delayed.{container}", tmp_path / "job"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=white:s=32x32:r=2:d=8",
        "-itsoffset", "5", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=2",
        "-map", "0:v", "-map", "1:a", "-c:v", "mpeg4", "-c:a", "pcm_s16le",
        "-output_ts_offset", str(container_start), str(media),
    ], check=True)
    assert extraction.probe(media) == {"duration": 7, "audio_start": 5}
    transcribe = Mock(return_value={"segments": [{"start": 0, "end": 0.5, "text": "Теорема"}],
                                   "language": "ru"})
    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=transcribe))
    requests = []

    def worker(out, request):
        requests.append(request)
        return local_worker.run({"job": str(out), **request})

    monkeypatch.setattr(extraction, "_worker", worker)
    monkeypatch.setattr(extraction, "select_frames", lambda *args, **kwargs: {
        "items": [], "diagnostics": {"selected_frames": 0, "proxy_frames": 0}})
    evidence = read_json(extraction.extract(media, job))
    assert evidence["media"]["audio_start"] == 5
    assert evidence["media"]["duration"] == 8
    assert evidence["media"]["clip_start"] == 0
    assert evidence["segments"][0]["start"] == 5
    assert evidence["segments"][0]["end"] == 5.5
    chunk = requests[0]["items"][0]
    assert (chunk["offset"], chunk["core_start"], chunk["core_end"]) == (5, 5, 7)
    with wave.open(chunk["audio"]) as audio:
        assert audio.getnframes() / audio.getframerate() == 2
    assert read_json(job / "transcript.json")[0]["start"] == 5
    assert "00:00:05,000 --> 00:00:05,500" in (job / "transcript.srt").read_text()
    # The trailing visual-only interval must not transcribe earlier overlapping audio.
    requests.clear()
    tail = read_json(extraction.extract(media, tmp_path / "tail", start=7.5, duration=0.5))
    assert tail["media"]["duration"] == 8
    assert tail["segments"] == []
    assert requests == []


def test_vision_subprocess_uses_separate_worker_and_propagates_failure(tmp_path, monkeypatch):
    request = {**VISION_MODELS["35b"], "items": []}
    command = Mock(side_effect=subprocess.CalledProcessError(1, "vision"))
    monkeypatch.setattr(extraction.subprocess, "run", command)
    with pytest.raises(subprocess.CalledProcessError):
        extraction._vision_worker(tmp_path, request)
    assert command.call_args.args[0][:3] == [sys.executable, "-m", "shad_helper.vision_worker"]
    assert read_json(tmp_path / "workers/vision.request.json") == {"job": str(tmp_path), **request}
