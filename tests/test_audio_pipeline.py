import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from shad_helper import cli, extraction, local_worker, notes, rendering
from shad_helper.jobs import AUDIO_CONFIG, check_audio_job, fingerprint, read_json, write_json


def audio_evidence():
    return {"schema_version": 3, "title": "Лекция", "media": {"duration": 120},
            "segments": [{"id": "speech", "start": 0, "end": 120, "text": "Определение"}]}


@pytest.mark.parametrize("stream_duration,format_duration,expected", [
    ("12.5", "20", 12.5), (None, "20", 20), ("N/A", "20", 20),
])
def test_probe_selects_first_audio_stream_and_duration(monkeypatch, stream_duration,
                                                       format_duration, expected):
    command = Mock(return_value=SimpleNamespace(stdout=json.dumps({
        "streams": [{"duration": stream_duration}], "format": {"duration": format_duration}})))
    monkeypatch.setattr(extraction.subprocess, "run", command)
    assert extraction.probe(Path("lecture.m4a")) == {"duration": expected}
    arguments = command.call_args.args[0]
    assert arguments[arguments.index("-select_streams") + 1] == "a:0"
    assert "width" not in arguments[arguments.index("-show_entries") + 1]


@pytest.mark.parametrize("data,message", [
    ({"streams": [], "format": {"duration": "20"}}, "no audio stream"),
    ({"streams": [{}], "format": {}}, "valid audio duration"),
    *[({"streams": [{"duration": value}]}, "valid audio duration")
      for value in ("N/A", "nan", "inf", "0", "-1", "invalid")],
])
def test_probe_rejects_missing_audio_or_invalid_duration(monkeypatch, data, message):
    monkeypatch.setattr(extraction.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(data)))
    with pytest.raises(ValueError, match=message):
        extraction.probe(Path("lecture.mp4"))


@pytest.mark.parametrize("start,duration", [
    (-1, None), (120, None), (float("nan"), None), (float("inf"), None),
    (0, 0), (0, -1), (0, float("nan")), (0, float("inf")),
])
def test_invalid_intervals_create_no_job(tmp_path, monkeypatch, start, duration):
    monkeypatch.setattr(extraction.shutil, "which", lambda name: name)
    monkeypatch.setattr(extraction, "probe", lambda media: {"duration": 120})
    out = tmp_path / "job"
    with pytest.raises(ValueError):
        extraction.extract(tmp_path / "lecture.wav", out, start=start, duration=duration,
                           vision_model="off")
    assert not out.exists()


def worker_request(job, count=2):
    write_json(job / "extraction-config.json", AUDIO_CONFIG)
    items = []
    for number in range(count):
        audio = job / "audio" / f"chunk_{number:04d}.wav"
        audio.parent.mkdir(exist_ok=True)
        audio.write_bytes(f"audio {number}".encode())
        items.append({"id": f"chunk_{number:04d}", "audio": str(audio),
                      "offset": number * 600, "core_start": number * 600,
                      "core_end": (number + 1) * 600})
    return {"job": str(job), "model": "fake", "language": "ru", "items": items}


def test_asr_resume_reuses_only_completed_chunks(tmp_path, monkeypatch):
    request = worker_request(tmp_path)
    response = {"segments": [{"start": 0, "end": 2, "text": "Теорема"}], "language": "ru"}
    transcribe = Mock(side_effect=[response, RuntimeError("Interrupted"), response])
    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=transcribe))
    with pytest.raises(RuntimeError, match="Interrupted"):
        local_worker.run(request)
    assert (tmp_path / "checkpoints/asr/chunk_0000.json").exists()
    assert not (tmp_path / "checkpoints/asr/chunk_0001.json").exists()
    result = local_worker.run(request)
    assert len(result["items"]) == 2
    assert transcribe.call_count == 3
    assert local_worker.run(request) == result
    assert transcribe.call_count == 3
    assert transcribe.call_args.kwargs == {
        "path_or_hf_repo": "fake", "language": "ru", "task": "transcribe",
        "word_timestamps": True, "verbose": False}


@pytest.mark.parametrize("changed", ["audio", "model", "language"])
def test_asr_input_changes_invalidate_checkpoint(tmp_path, monkeypatch, changed):
    request = worker_request(tmp_path, count=1)
    transcribe = Mock(return_value={"segments": [], "language": "ru"})
    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=transcribe))
    local_worker.run(request)
    local_worker.run(request)
    assert transcribe.call_count == 1
    if changed == "audio":
        Path(request["items"][0]["audio"]).write_bytes(b"changed recording")
    else:
        request[changed] = "changed"
    local_worker.run(request)
    assert transcribe.call_count == 2


@pytest.mark.parametrize("operation", [
    "extract", "notes", "notes_rebuild", "render", "resolve", "review",
])
@pytest.mark.parametrize("with_config", [False, True])
@pytest.mark.parametrize("version", [1, 2])
def test_legacy_jobs_are_rejected_with_cleanup_only_on_rebuild(
        tmp_path, operation, with_config, version, capsys):
    data = ({"schema_version": 1, "video": {"duration": 120}, "segments": [],
             "visuals": [], "sources": [], "issues": []} if version == 1 else
            {**audio_evidence(), "schema_version": 2, "issues": [], "models": {"asr": "fake"}})
    write_json(tmp_path / "evidence.json", data)
    write_json(tmp_path / "checkpoints/notes/ch001.json", {"preserve": True})
    write_json(tmp_path / "notes.json", {"preserve": "previous notes"})
    if with_config:
        write_json(tmp_path / "extraction-config.json",
                   {"vision_mode": "pages"} if version == 1 else
                   {"schema_version": 2, "pipeline": "audio"})
    before = {path.relative_to(tmp_path): path.read_bytes()
              for path in tmp_path.rglob("*") if path.is_file()}
    if operation == "review":
        assert cli.main(["review", str(tmp_path)]) == 1
        assert "new output directory" in capsys.readouterr().err
    else:
        with pytest.raises(ValueError, match="new output directory"):
            if operation == "extract":
                extraction.extract(tmp_path / "lecture.wav", tmp_path)
            elif operation in ("notes", "notes_rebuild"):
                notes.generate_notes(tmp_path, tmp_path / "auth",
                                     rebuild=operation == "notes_rebuild")
            elif operation == "render":
                rendering.render(tmp_path)
            else:
                notes.resolve_issue(tmp_path, "issue1", "Checked")
    if operation == "notes_rebuild":
        before.pop(Path("checkpoints/notes/ch001.json"))
        before.pop(Path("notes.json"))
    assert {path.relative_to(tmp_path): path.read_bytes()
            for path in tmp_path.rglob("*") if path.is_file()} == before


def test_new_job_marker_cannot_adopt_legacy_evidence(tmp_path):
    write_json(tmp_path / "extraction-config.json", AUDIO_CONFIG)
    write_json(tmp_path / "evidence.json", {"schema_version": 1, "video": {}})
    with pytest.raises(ValueError, match="new output directory"):
        check_audio_job(tmp_path)


def test_unmarked_partial_job_is_not_overwritten(tmp_path):
    (tmp_path / "transcript.srt").write_text("Existing transcript")
    with pytest.raises(ValueError, match="new output directory"):
        check_audio_job(tmp_path)


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("rebuild", [False, True])
def test_silent_extraction_guard_clears_prior_outputs_only_on_rebuild(tmp_path, manual, rebuild):
    data = audio_evidence()
    data["segments"] = []
    write_json(tmp_path / "evidence.json", data)
    write_json(tmp_path / "extraction-config.json", AUDIO_CONFIG)
    inputs = {tmp_path / name: (tmp_path / name).read_bytes()
              for name in ("evidence.json", "extraction-config.json")}
    previous = {tmp_path / "checkpoints/notes/ch001.json": {"preserve": True},
                tmp_path / "notes.json": {"preserve": "previous notes"}}
    for path, value in previous.items():
        write_json(path, value)
    with pytest.raises(ValueError, match="No speech"):
        notes.generate_notes(tmp_path, tmp_path / "auth", manual=manual, rebuild=rebuild)
    assert all(path.read_bytes() == value for path, value in inputs.items())
    for path, value in previous.items():
        if rebuild:
            assert not path.exists()
        else:
            assert read_json(path) == value
    assert not (tmp_path / "outline.json").exists()
    run = tmp_path / "handoff"
    assert read_json(run / "run.json")["status"] == "failed"
    assert not (run / "inputs").exists()
    assert not (run / "runs").exists()


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="FFmpeg is required")
def test_real_audio_and_multistream_video_use_the_same_soundtrack(tmp_path, monkeypatch):
    audio = tmp_path / "lecture.wav"
    video = tmp_path / "lecture.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=16000:duration=2", str(audio)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(audio), "-f", "lavfi", "-i",
                    "sine=frequency=880:sample_rate=16000:duration=2", "-f", "lavfi", "-i",
                    "color=c=black:s=32x32:r=2:d=2", "-map", "2:v:0", "-map", "0:a:0",
                    "-map", "1:a:0", "-c:v", "mpeg4", "-c:a", "pcm_s16le", str(video)], check=True)
    transcribe = Mock(return_value={"segments": [
        {"start": 0.2, "end": 1.5, "text": "Теорема", "avg_logprob": -1.5,
         "words": [{"start": 0.2, "end": 1.5, "word": "Теорема", "probability": 0.1}]}],
        "language": "ru"})
    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=transcribe))
    monkeypatch.setattr(extraction, "_worker",
                        lambda job, request: local_worker.run({"job": str(job), **request}))
    pcm = []
    for media in (audio, video):
        job = tmp_path / media.suffix[1:]
        path = extraction.extract(media, job, start=0.5, duration=1, vision_model="off")
        evidence = read_json(path)
        assert set(evidence) == {"schema_version", "title", "media", "segments"}
        assert evidence["schema_version"] == 3
        assert all(set(segment) == {"id", "start", "end", "text"}
                   for segment in evidence["segments"])
        assert (evidence["media"]["clip_start"], evidence["media"]["clip_end"]) == (0.5, 1.5)
        assert evidence["segments"][0]["start"] == 0.5
        diagnostics = read_json(job / "extraction-diagnostics.json")
        assert set(diagnostics) == {"schema_version", "evidence_fingerprint", "models",
                                    "segments", "warnings"}
        assert diagnostics["schema_version"] == 1
        assert diagnostics["evidence_fingerprint"] == fingerprint(evidence)
        assert diagnostics["models"] == {"asr": extraction.ASR_MODEL}
        assert diagnostics["segments"] == [{
            "id": "segment_00000", "words": [
                {"start": 0.2, "end": 1.5, "word": "Теорема", "probability": 0.1}],
            "diagnostics": {"avg_logprob": -1.5}}]
        assert len(diagnostics["warnings"]) == 2
        assert all(set(warning) == {"id", "description", "evidence_ids"}
                   and warning["evidence_ids"] == ["segment_00000"]
                   for warning in diagnostics["warnings"])
        assert read_json(job / "transcript.json") == evidence["segments"]
        assert read_json(job / "extraction-config.json") == AUDIO_CONFIG
        assert "00:00:00,500 --> 00:00:01,500" in (job / "transcript.srt").read_text()
        with wave.open(str(job / "audio/chunk_0000.wav")) as chunk:
            assert (chunk.getnchannels(), chunk.getframerate()) == (1, 16000)
            pcm.append(chunk.readframes(chunk.getnframes()))
        extraction.extract(media, job, start=0.5, duration=1, vision_model="off")
        assert not (job / "frames").exists()
    assert pcm[0] == pcm[1]
    assert transcribe.call_count == 2


def test_silent_audio_retains_extraction_artifacts(tmp_path, monkeypatch):
    media = tmp_path / "lecture.wav"
    media.write_bytes(b"recording")
    out = tmp_path / "job"
    monkeypatch.setattr(extraction.shutil, "which", lambda name: name)
    monkeypatch.setattr(extraction, "probe", lambda path: {"duration": 120})
    monkeypatch.setattr(extraction, "_audio_chunks", lambda *args: [])
    monkeypatch.setattr(extraction, "_worker", lambda *args: {"items": []})
    evidence = read_json(extraction.extract(media, out, vision_model="off"))
    assert evidence["segments"] == []
    diagnostics = read_json(out / "extraction-diagnostics.json")
    assert diagnostics["segments"] == diagnostics["warnings"] == []
    assert diagnostics["evidence_fingerprint"] == fingerprint(evidence)
    assert read_json(out / "transcript.json") == []
    assert (out / "transcript.srt").read_text() == ""
    assert (out / "transcript.vtt").read_text() == "WEBVTT\n\n"
