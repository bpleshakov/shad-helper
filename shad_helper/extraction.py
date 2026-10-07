"""Resumable local speech extraction from audio files or video soundtracks."""

import math
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .jobs import (
    AUDIO_CONFIG,
    EVIDENCE_VERSION,
    check_audio_job,
    checkpoint,
    fingerprint,
    read_json,
    sha256_file,
    write_json,
)

ASR_MODEL = "mlx-community/whisper-large-v3-mlx"


def probe(media: Path) -> dict:
    import json

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=duration:format=duration", "-of", "json", str(media)],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError("Input has no audio stream.")
    duration = data["streams"][0].get("duration")
    if duration in (None, "N/A"):
        duration = data.get("format", {}).get("duration")
    try:
        duration = float(duration)
    except (TypeError, ValueError) as error:
        raise ValueError("Input has no valid audio duration.") from error
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Input has no valid audio duration.")
    return {"duration": duration}


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


def _worker(out: Path, request: dict) -> dict:
    request = {"job": str(out.resolve()), **request}
    request_path = out / "workers" / "asr.request.json"
    response_path = out / "workers" / "asr.response.json"
    write_json(request_path, request)
    subprocess.run([sys.executable, "-m", "shad_helper.local_worker",
                    "--request", str(request_path), "--output", str(response_path)], check=True)
    return read_json(response_path)


def _audio_chunks(media: Path, out: Path, start: float, end: float, media_hash: str) -> list[dict]:
    chunks = []
    for position in range(math.ceil((end - start) / 600)):
        lower, upper = start + position * 600, min(end, start + (position + 1) * 600)
        offset, audio_end = max(0, lower - 5), min(end, upper + 5)
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


def extract(media: Path, out: Path, *, language: str = "ru", start: float = 0.0,
            duration: float | None = None, asr_model: str = ASR_MODEL) -> Path:
    media, out = Path(media).resolve(), Path(out).resolve()
    check_audio_job(out)
    for binary in ("ffmpeg", "ffprobe"):
        if not shutil.which(binary):
            raise RuntimeError(f"{binary} is required. Install FFmpeg with: brew install ffmpeg")
    metadata = probe(media)
    if not math.isfinite(start) or start < 0 or start >= metadata["duration"]:
        raise ValueError("start must be within the audio duration.")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be positive.")
    end = metadata["duration"] if duration is None else min(metadata["duration"], start + duration)
    media_hash = sha256_file(media)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "extraction-config.json", AUDIO_CONFIG)
    started = time.monotonic()
    chunks = _audio_chunks(media, out, start, end, media_hash)
    asr = _worker(out, {"model": asr_model, "language": language, "items": chunks})
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
    write_json(out / "evidence.json", evidence)
    write_json(out / "extraction-diagnostics.json", {
        "schema_version": 1, "evidence_fingerprint": fingerprint(evidence),
        "models": {"asr": asr_model},
        "segments": [{key: segment[key] for key in ("id", "words", "diagnostics")}
                     for segment in transcribed],
        "warnings": [{"id": f"issue_{position:04d}", "description": description,
                      "evidence_ids": [identifier]}
                     for position, (identifier, description) in enumerate(warnings)],
    })
    write_json(out / "extraction-timings.json",
               {"scope": "current extraction invocation",
                "audio_asr_seconds": time.monotonic() - started})
    return out / "evidence.json"
