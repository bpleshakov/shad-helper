"""Resumable local speech and video extraction in sequential model processes."""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from threading import Thread

from .illustrations import materialize_illustrations
from .jobs import (
    AUDIO_CONFIG,
    EVIDENCE_VERSION,
    VIDEO_EVIDENCE_VERSION,
    check_job,
    checkpoint,
    fingerprint,
    read_json,
    sha256_file,
    write_json,
)
from .run_diagnostics import record_run, timed_step
from .video import packet_bounds, probe_video, select_frames
from .vision_config import DEFAULT_VISION_MODEL, VISION_MODELS

ASR_MODEL = "mlx-community/whisper-large-v3-mlx"


def assert_extraction_complete(job: Path, evidence: dict) -> None:
    """Reject partial video inputs while allowing legacy jobs and transcript edits."""
    job = Path(job)
    config_path, status_path = job / "extraction-config.json", job / "extraction-status.json"
    config = read_json(config_path) if config_path.exists() else None
    if status_path.exists():
        status = read_json(status_path)
        if status.get("state") != "complete":
            raise ValueError("Extraction is incomplete; resume extraction before generating notes.")
        if (status.get("config_fingerprint") != fingerprint(config)
                or status.get("media") != {key: evidence.get("media", {}).get(key)
                                          for key in ("sha256", "clip_start", "clip_end")}):
            raise ValueError("Evidence does not match the completed extraction; resume extraction.")
        # The recorded evidence fingerprint describes the published snapshot. Do not
        # enforce it here: correcting transcript text in evidence.json is supported.
    if config and config.get("pipeline") == "audio_video" and "visuals" not in evidence:
        raise ValueError("Visual extraction is missing from evidence; resume extraction "
                         "before generating notes, or explicitly extract with vision off.")


@timed_step("media_probe")
def probe(media: Path) -> dict:
    import json

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=duration,start_time:format=duration,start_time,format_name", "-of", "json",
         str(media)],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError("Input has no audio stream.")
    stream, container = data["streams"][0], data.get("format", {})
    origin = container.get("start_time")
    origin = 0 if origin in (None, "N/A") else origin
    start = stream.get("start_time")
    start = origin if start in (None, "N/A") else start
    try:
        audio_start = float(start) - float(origin)
    except (TypeError, ValueError) as error:
        raise ValueError("Input has no valid audio start time.") from error
    if not math.isfinite(audio_start):
        raise ValueError("Input has no valid audio start time.")
    audio_start = max(0.0, audio_start)
    if set(container.get("format_name", "").split(",")) & {"matroska", "webm"}:
        audio_start, audio_end = packet_bounds(media, "a:0", float(origin))
        return {"duration": audio_end, **({"audio_start": audio_start} if audio_start else {})}
    duration = stream.get("duration")
    stream_duration = duration not in (None, "N/A")
    if duration in (None, "N/A"):
        duration = container.get("duration")
    try:
        duration = float(duration)
    except (TypeError, ValueError) as error:
        raise ValueError("Input has no valid audio duration.") from error
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Input has no valid audio duration.")
    duration = duration + audio_start if stream_duration else duration
    if duration <= audio_start:
        raise ValueError("Input has no valid audio duration.")
    return {"duration": duration, **({"audio_start": audio_start} if audio_start else {})}


def stitch_segments(chunks: list[dict]) -> list[dict]:
    """Assign overlapping audio words to their core interval, then restore PTS."""
    result = []
    for chunk in sorted(chunks, key=lambda value: value["core_start"]):
        offset, lower, upper = chunk["offset"], chunk["core_start"], chunk["core_end"]
        for segment in chunk["segments"]:
            words = []
            for word in segment.get("words", []):
                start, end = offset + word["start"], offset + word["end"]
                if lower <= (start + end) / 2 < upper:
                    words.append({**word, "start": start, "end": end})
            if segment.get("words"):
                if not words:
                    continue
                start, end = words[0]["start"], words[-1]["end"]
                text = "".join(word["word"] for word in words).strip()
            else:
                start, end = offset + segment["start"], offset + segment["end"]
                if not lower <= (start + end) / 2 < upper:
                    continue
                text = segment["text"].strip()
            if not text:
                continue
            result.append({"id": f"segment_{len(result):05d}", "start": max(lower, start),
                           "end": min(upper, end), "text": text, "words": words,
                           "diagnostics": {key: segment[key] for key in
                                           ("avg_logprob", "no_speech_prob", "compression_ratio") if key in segment}})
    return result


def write_subtitles(segments: list[dict], out: Path) -> None:
    def timestamp(seconds: float, separator: str) -> str:
        millis = max(0, round(seconds * 1000))
        hours, remainder = divmod(millis, 3600000)
        minutes, remainder = divmod(remainder, 60000)
        secs, fraction = divmod(remainder, 1000)
        return f"{hours:02}:{minutes:02}:{secs:02}{separator}{fraction:03}"

    for extension, separator, header in (("srt", ",", ""), ("vtt", ".", "WEBVTT\n\n")):
        cues = []
        for position, segment in enumerate(segments, 1):
            cues.append(f"{position}\n{timestamp(segment['start'], separator)} --> "
                        f"{timestamp(segment['end'], separator)}\n{segment['text']}\n")
        (out / f"transcript.{extension}").write_text(header + "\n".join(cues), encoding="utf-8")


def transcription_warnings(segment: dict) -> list[str]:
    diagnostics = segment.get("diagnostics", {})
    warnings = []
    if diagnostics.get("avg_logprob", 0) < -1:
        warnings.append("Whisper отметил низкую уверенность транскрипции; сверить с аудио.")
    if diagnostics.get("no_speech_prob", 0) > 0.6 and segment.get("text", "").strip():
        warnings.append("Whisper предполагает отсутствие речи, но вернул текст; возможна галлюцинация.")
    weak_words = [word["word"].strip() for word in segment.get("words", [])
                  if word.get("probability", 1) < 0.3]
    if weak_words:
        warnings.append("Низкая уверенность распознавания слов: " + ", ".join(weak_words[:5]) + ". Сверить с аудио.")
    words = re.findall(r"[^\W_]+", segment.get("text", "").casefold())
    repetitions = Counter(tuple(words[position:position + 4]) for position in range(len(words) - 3))
    if diagnostics.get("compression_ratio", 0) > 2.4 or max(repetitions.values(), default=0) >= 4:
        warnings.append("В транскрипции обнаружено сильное повторение; сверить с аудио.")
    return warnings


def _artifact_checkpoint(out: Path, key: str, inputs, action, artifact_paths):
    result = checkpoint(out, key, inputs, action)
    if any(not Path(path).exists() for path in artifact_paths(result)):
        (out / "checkpoints" / f"{key}.json").unlink(missing_ok=True)
        result = checkpoint(out, key, inputs, action)
    return result


@timed_step("speech_recognition")
def _worker(out: Path, request: dict) -> dict:
    request = {"job": str(out.resolve()), **request}
    request_path = out / "workers" / "asr.request.json"
    response_path = out / "workers" / "asr.response.json"
    write_json(request_path, request)
    subprocess.run([sys.executable, "-m", "shad_helper.local_worker",
                    "--request", str(request_path), "--output", str(response_path)], check=True)
    return read_json(response_path)


@timed_step("visual_extraction")
def _vision_worker(out: Path, request: dict, *, on_batch=None, log=None, on_status=None) -> dict:
    request = {"job": str(out.resolve()), **request}
    request_path = out / "workers" / "vision.request.json"
    response_path = out / "workers" / "vision.response.json"
    write_json(request_path, request)
    command = [sys.executable, "-m", "shad_helper.vision_worker",
               "--request", str(request_path), "--output", str(response_path)]
    if on_batch is None:
        subprocess.run(command, check=True)
    else:
        _stream_vision(command, on_batch, log, on_status)
    return read_json(response_path)


def _stream_vision(command, on_batch, log, on_status=None):
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, encoding="utf-8") as events:
        try:
            process = subprocess.Popen([*command, "--events-fd", str(write_fd)],
                                       pass_fds=(write_fd,), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, errors="replace",
                                       env={**os.environ, "PYTHONUNBUFFERED": "1", "TQDM_DISABLE": "1",
                                            "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
        finally:
            os.close(write_fd)

        def drain_logs():
            for line in process.stdout:
                log(line.rstrip("\r\n"))

        logs = Thread(target=drain_logs, name="vision-output")
        logs.start()
        try:
            for line in events:
                event = json.loads(line)
                if isinstance(event, list):
                    on_batch(event)
                elif on_status:
                    on_status(event["phase"], event["frame"])
            if code := process.wait():
                raise subprocess.CalledProcessError(code, command)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            logs.join()
            process.stdout.close()


@timed_step("audio_preparation")
def _audio_chunks(media: Path, out: Path, start: float, end: float, media_hash: str,
                  audio_start: float = 0) -> list[dict]:
    chunks = []
    for position in range(math.ceil((end - start) / 600)):
        lower, upper = start + position * 600, min(end, start + (position + 1) * 600)
        offset, audio_end = max(0, audio_start, lower - 5), min(end, upper + 5)
        path = out / "audio" / f"chunk_{position:04d}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)

        def action():
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(offset),
                            "-i", str(media), "-t", str(audio_end - offset), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", str(path)], check=True)
            return {"id": f"chunk_{position:04d}", "audio": str(path.resolve()), "offset": offset,
                    "core_start": lower, "core_end": upper}

        chunks.append(_artifact_checkpoint(out, f"audio/chunk_{position:04d}",
                                          {"media": media_hash, "lower": lower, "upper": upper,
                                           "offset": offset, "audio_end": audio_end}, action,
                                          lambda result: [result["audio"]]))
    return chunks


@record_run("extract", job_argument="out")
def extract(media: Path, out: Path, *, language: str = "ru", start: float = 0.0,
            duration: float | None = None, asr_model: str = ASR_MODEL,
            vision_model: str = DEFAULT_VISION_MODEL, vision_mode: str = "frames",
            chapters=None) -> Path:
    total_started = time.monotonic()
    media, out = Path(media).resolve(), Path(out).resolve()
    if vision_model != "off" and vision_model not in VISION_MODELS:
        raise ValueError("vision_model must be off, 4b, 9b, or 35b.")
    if vision_mode not in ("frames", "video"):
        raise ValueError("vision_mode must be frames or video.")
    check_job(out)
    for binary in ("ffmpeg", "ffprobe"):
        if not shutil.which(binary):
            raise RuntimeError(f"{binary} is required. Install FFmpeg with: brew install ffmpeg")
    metadata = probe(media)
    if not math.isfinite(start) or start < 0:
        raise ValueError("start must be within the recording duration.")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be positive.")
    audio_end = metadata["duration"]
    audio_start = metadata.get("audio_start", 0)
    video_metadata = probe_video(media) if vision_model != "off" else None
    if video_metadata is not None:
        metadata = {**metadata, "duration": max(audio_end, video_metadata["end"])}
    if start >= metadata["duration"]:
        raise ValueError("start must be within the recording duration.")
    end = metadata["duration"] if duration is None else min(metadata["duration"], start + duration)
    config = ({"schema_version": VIDEO_EVIDENCE_VERSION, "pipeline": "audio_video",
               "vision_model": vision_model} if video_metadata is not None else AUDIO_CONFIG)
    if video_metadata is not None and vision_mode == "video":
        config["vision_mode"] = vision_mode
    media_hash = sha256_file(media)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "extraction-config.json", config)
    extraction_status = {"schema_version": 1, "state": "running",
                         "config_fingerprint": fingerprint(config),
                         "media": {"sha256": media_hash, "clip_start": start, "clip_end": end}}
    write_json(out / "extraction-status.json", extraction_status)
    started = time.monotonic()
    asr_start, asr_end = max(start, audio_start), min(end, audio_end)
    chunks = (_audio_chunks(media, out, asr_start, asr_end, media_hash, audio_start)
              if asr_start < asr_end else [])
    asr = (_worker(out, {"model": asr_model, "language": language, "items": chunks})
           if chunks else {"items": []})
    transcribed = stitch_segments(asr["items"])
    warnings = [(segment["id"], description) for segment in transcribed
                for description in transcription_warnings(segment)]
    segments = [{key: segment[key] for key in ("id", "start", "end", "text")}
                for segment in transcribed]
    write_json(out / "transcript.json", segments)
    write_subtitles(segments, out)
    evidence = {"schema_version": EVIDENCE_VERSION, "title": media.stem,
                "media": {"path": str(media), "sha256": media_hash, **metadata,
                          "clip_start": start, "clip_end": end},
                "segments": segments}
    write_json(out / "evidence.partial.json", evidence)
    diagnostics = {
        "schema_version": 1, "evidence_fingerprint": fingerprint(evidence),
        "models": {"asr": asr_model},
        "segments": [{key: segment[key] for key in ("id", "words", "diagnostics")}
                     for segment in transcribed],
        "warnings": [{"id": f"issue_{position:04d}", "description": description,
                      "evidence_ids": [identifier]}
                     for position, (identifier, description) in enumerate(warnings)],
    }
    timings = {"scope": "current extraction invocation",
               "audio_asr_seconds": time.monotonic() - started}
    write_json(out / "extraction-diagnostics.json", diagnostics)
    write_json(out / "extraction-timings.json", timings)
    if video_metadata is not None:
        vision_started = time.monotonic()
        model = VISION_MODELS[vision_model]
        diagnostics["models"]["vision"] = {"preset": vision_model, "mode": vision_mode, **model}
        try:
            selection = select_frames(media, out, start, end, media_hash, metadata=video_metadata)
            diagnostics["selection"] = selection["diagnostics"]
            visuals, assets = [], {}
            request = {**model, "mode": vision_mode, "items": selection["items"]}
            if chapters is not None:
                chapters.start(evidence, request)

            def accept(batch):
                crops = materialize_illustrations(out, batch, assets=assets)
                clean = []
                for item in batch:
                    status = item["diagnostics"]["status"]
                    content = {**item["content"], "illustrations": crops[item["id"]]}
                    descriptions = content["unreadable"]
                    if status == "incomplete":
                        content["unreadable"] = []
                        descriptions = ["Не удалось получить корректное описание кадра."]
                    clean.append({"id": item["id"], "timestamp": item["timestamp"],
                                  "status": status, **content})
                    for description in descriptions:
                        diagnostics["warnings"].append({
                            "id": f"issue_{len(diagnostics['warnings']):04d}",
                            "description": description, "evidence_ids": [item["id"]]})
                visuals.extend(clean)
                if chapters is not None:
                    chapters.accept(clean, assets)

            if not selection["items"]:
                accept([])
                visual_result = {"items": [], "diagnostics": {
                    "status": "no_frames", "fresh_items": 0, "cached_items": 0}}
            elif chapters is None:
                visual_result = _vision_worker(out, request)
                accept(visual_result["items"])
            else:
                visual_result = _vision_worker(out, request, on_batch=accept, log=chapters.progress.log,
                                              on_status=chapters.vision_status)
            if [item["id"] for item in visuals] != [item["id"] for item in selection["items"]]:
                raise ValueError("Vision worker did not complete the selected frames in order")
            write_json(out / "illustrations" / "manifest.json", {"version": 1, "assets": assets})
            evidence.update(schema_version=VIDEO_EVIDENCE_VERSION, visuals=visuals)
            diagnostics["vision"] = visual_result["diagnostics"]
            diagnostics["visuals"] = [{key: value for key, value in item.items() if key != "content"}
                                      for item in visual_result["items"]]
            diagnostics["evidence_fingerprint"] = fingerprint(evidence)
        except BaseException as error:
            diagnostics["vision"] = {"status": "failed", "error": str(error)}
            write_json(out / "extraction-status.json", {**extraction_status, "state": "failed",
                                                       "error": str(error)})
            raise
        finally:
            timings.update(vision_seconds=time.monotonic() - vision_started,
                           total_seconds=time.monotonic() - total_started)
            write_json(out / "extraction-diagnostics.json", diagnostics)
            write_json(out / "extraction-timings.json", timings)
    # Publish only after all required stages finish. A failed rerun leaves the
    # previous completed evidence intact and its new transcript separately inspectable.
    write_json(out / "evidence.json", evidence)
    write_json(out / "extraction-status.json", {
        **extraction_status, "state": "complete", "evidence_fingerprint": fingerprint(evidence)})
    (out / "evidence.partial.json").unlink(missing_ok=True)
    return out / "evidence.json"
