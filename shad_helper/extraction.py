"""Resumable local video, speech and visual evidence extraction."""

import math
import queue
import re
import shutil
import subprocess
import sys
import threading
from collections import Counter, deque
from pathlib import Path

from .jobs import checkpoint, read_json, sha256_file, write_json
from .sources import import_sources, match_sources

ASR_MODEL = "mlx-community/whisper-large-v3-mlx"
VLM_MODEL = "mlx-community/Qwen3.8-27B-4bit"
OCR_MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"


def probe(video: Path) -> dict:
    import json

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate:format=duration", "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError("Input has no video stream.")
    stream = data["streams"][0]
    return {"duration": float(data["format"]["duration"]),
            "width": int(stream["width"]), "height": int(stream["height"])}


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


def source_glossary(records: list[dict]) -> str | None:
    headings = []
    for record in records:
        heading = next((line.strip() for line in record.get("text", "").splitlines() if line.strip()), "")
        terms = re.findall(r"\b[A-Za-z_][A-Za-z_0-9]{2,}\b", record.get("code", ""))[:12]
        if heading or terms:
            headings.append((heading[:160] + " " + " ".join(terms)).strip())
    return ("Тема и термины лекции: " + "; ".join(dict.fromkeys(headings)))[:1000] if headings else None


def _artifact_checkpoint(out: Path, key: str, inputs, action, artifact_paths):
    result = checkpoint(out, key, inputs, action)
    if any(not Path(path).exists() for path in artifact_paths(result)):
        (out / "checkpoints" / f"{key}.json").unlink(missing_ok=True)
        result = checkpoint(out, key, inputs, action)
    return result


def select_keyframes(samples: list[dict], *, stable_seconds: float = 0.7,
                     change_threshold: float = 0.003, max_gap: float = 30.0) -> list[dict]:
    """Keep sharp stable states, their pre-change views, and bounded-gap evidence."""
    if not samples:
        return []
    selected = [samples[0]]
    run_start = 0
    for position in range(1, len(samples) + 1):
        boundary = position == len(samples) or samples[position]["change"] > change_threshold
        if not boundary:
            continue
        run = samples[run_start:position]
        if run and run[-1]["timestamp"] - run[0]["timestamp"] >= stable_seconds:
            selected.append(max(run, key=lambda sample: (sample["sharpness"], sample["timestamp"])))
            selected.append(run[-1])
        run_start = position
    # Continuous handwriting/person motion may have no stable plateau. Retain
    # the sharpest frame in each short window instead of dropping that interval.
    window_start = samples[0]["timestamp"]
    window = []
    for sample in samples:
        if sample["timestamp"] - window_start >= max_gap:
            if window:
                selected.append(max(window, key=lambda value: (value["sharpness"], value["timestamp"])))
            selected.append(sample)
            window_start = sample["timestamp"]
            window = []
        window.append(sample)
    if window:
        selected.append(max(window, key=lambda value: (value["sharpness"], value["timestamp"])))
    selected.append(samples[-1])
    return sorted({sample["timestamp"]: sample for sample in selected}.values(), key=lambda sample: sample["timestamp"])


def _scan_video(video: Path, metadata: dict, start: float, duration: float) -> list[dict]:
    import cv2
    import numpy as np

    width = min(384, metadata["width"])
    height = max(2, round(metadata["height"] * width / metadata["width"] / 2) * 2)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-ss", str(start), "-i", str(video),
               "-t", str(duration), "-an", "-sn", "-dn", "-vf", f"scale={width}:{height},showinfo",
               "-fps_mode", "passthrough", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timestamps = queue.Queue()
    errors = deque(maxlen=20)

    def read_timestamps():
        for line in iter(process.stderr.readline, b""):
            text = line.decode("utf-8", errors="replace")
            match = re.search(r"\bn:\s*\d+.*?\bpts_time:([-\d.e+]+)", text)
            if match:
                timestamps.put(start + float(match.group(1)))
            elif "error" in text.lower():
                errors.append(text.strip())
        timestamps.put(None)

    reader = threading.Thread(target=read_timestamps, daemon=True)
    reader.start()
    samples = []
    previous = None
    try:
        while True:
            pixels = process.stdout.read(width * height)
            if not pixels:
                break
            if len(pixels) != width * height:
                raise RuntimeError("FFmpeg returned a partial preview frame.")
            timestamp = timestamps.get(timeout=60)
            if timestamp is None:
                raise RuntimeError("FFmpeg did not provide a presentation timestamp for a preview frame.")
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width)
            difference = 1.0 if previous is None else float(np.count_nonzero(cv2.absdiff(frame, previous) > 12) / frame.size)
            sharpness = float(cv2.Laplacian(frame, cv2.CV_64F).var())
            samples.append({"timestamp": timestamp, "change": difference, "sharpness": sharpness})
            previous = frame
    finally:
        process.stdout.close()
        if process.poll() is None and sys.exc_info()[0]:
            process.terminate()
        process.wait()
        reader.join(timeout=5)
        process.stderr.close()
    if process.returncode:
        raise RuntimeError("FFmpeg preview failed: " + "\n".join(errors))
    return select_keyframes(samples)


def _worker(out: Path, task: str, request: dict) -> dict:
    request = {"job": str(out.resolve()), **request}
    request_path = out / "workers" / f"{task}.request.json"
    response_path = out / "workers" / f"{task}.response.json"
    write_json(request_path, request)
    subprocess.run([sys.executable, "-m", "shad_helper.local_worker", task,
                    "--request", str(request_path), "--output", str(response_path)], check=True)
    return read_json(response_path)


def _audio_chunks(video: Path, out: Path, start: float, end: float, video_hash: str) -> list[dict]:
    chunks = []
    for position in range(math.ceil((end - start) / 600)):
        lower, upper = start + position * 600, min(end, start + (position + 1) * 600)
        offset, audio_end = max(0, lower - 5), min(end, upper + 5)
        path = out / "audio" / f"chunk_{position:04d}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)

        def action():
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(offset),
                            "-i", str(video), "-t", str(audio_end - offset), "-vn", "-ac", "1", "-ar", "16000", str(path)], check=True)
            return {"id": f"chunk_{position:04d}", "audio": str(path.resolve()), "offset": offset,
                    "core_start": lower, "core_end": upper}

        chunks.append(_artifact_checkpoint(out, f"audio/chunk_{position:04d}",
                                          {"video": video_hash, "lower": lower, "upper": upper,
                                           "offset": offset, "audio_end": audio_end}, action,
                                          lambda result: [result["audio"]]))
    return chunks


def _frames(video: Path, out: Path, selections: list[dict]) -> list[dict]:
    items = []
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for position, selection in enumerate(selections):
        identifier = f"visual_{position:05d}"
        path = out / "frames" / f"{identifier}.png"
        path.unlink(missing_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(selection["timestamp"]),
                        "-i", str(video), "-frames:v", "1", str(path)], check=True)
        if not path.exists():
            # A final PTS can round just beyond the last decodable frame.
            continue
        items.append({"id": identifier, "timestamp": selection["timestamp"], "image": str(path.resolve()),
                      "sha256": sha256_file(path)})
    return items


def _prepare_visuals(raw_visuals: list[dict], out: Path) -> tuple[list[dict], list[dict]]:
    from PIL import Image

    visuals, crops, prepared_crops = [], [], set()
    (out / "crops").mkdir(parents=True, exist_ok=True)
    for raw in raw_visuals:
        path = Path(raw["image"])
        analysis_id = raw.get("analysis_id", raw["id"])
        visual = {"id": raw["id"], "timestamp": raw.get("timestamp"), "image": path.relative_to(out).as_posix(),
                  "sha256": sha256_file(path),
                  "summary": str(raw.get("summary", "")), "kind": raw.get("kind", "other"),
                  "diagram": str(raw.get("diagram", "")), "issues": list(raw.get("issues", [])), "regions": []}
        if raw.get("analysis_id"):
            original = Path(raw.get("original_image", raw["image"]))
            visual.update(analysis_id=analysis_id,
                          original_image=original.relative_to(out).as_posix(),
                          original_sha256=sha256_file(original))
        if raw.get("source_id"):
            visual["source_id"] = raw["source_id"]
        regions = raw.get("regions") or [{"kind": "text", "bbox": [0, 0, 1, 1]}]
        with Image.open(path) as image:
            if raw.get("analysis_id"):
                from PIL import ImageOps

                image = ImageOps.exif_transpose(image)
            for position, region in enumerate(regions):
                if not isinstance(region, dict):
                    visual["issues"].append("Модель вернула область без координат и типа.")
                    continue
                bbox = region.get("bbox", [])
                if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox):
                    visual["issues"].append("Модель не определила корректные координаты области.")
                    continue
                left, top, right, bottom = [min(1.0, max(0.0, float(value))) for value in bbox]
                if left >= right or top >= bottom:
                    visual["issues"].append("Модель определила пустую область.")
                    continue
                identifier = f"{visual['id']}_region_{position:03d}"
                analysis_region_id = f"{analysis_id}_region_{position:03d}"
                crop_path = out / "crops" / f"{analysis_region_id}.png"
                if analysis_region_id not in prepared_crops:
                    image.crop((int(left * image.width), int(top * image.height), math.ceil(right * image.width), math.ceil(bottom * image.height))).save(crop_path)
                kind = region.get("kind", "text")
                kind = kind if isinstance(kind, str) else "text"
                digest = sha256_file(crop_path)
                visual["regions"].append({"id": identifier, "kind": kind, "bbox": [left, top, right, bottom],
                                          "image": crop_path.relative_to(out).as_posix(), "sha256": digest, "text": ""})
                if raw.get("analysis_id"):
                    visual["regions"][-1]["analysis_region_id"] = analysis_region_id
                if analysis_region_id not in prepared_crops:
                    crops.append({"id": analysis_region_id, "kind": kind, "image": str(crop_path.resolve()), "sha256": digest})
                    prepared_crops.add(analysis_region_id)
            if not visual["regions"]:
                identifier = f"{visual['id']}_region_fallback"
                analysis_region_id = f"{analysis_id}_region_fallback"
                crop_path = out / "crops" / f"{analysis_region_id}.png"
                if analysis_region_id not in prepared_crops:
                    image.save(crop_path)
                digest = sha256_file(crop_path)
                visual["regions"].append({"id": identifier, "kind": "text", "bbox": [0, 0, 1, 1],
                                          "image": crop_path.relative_to(out).as_posix(), "sha256": digest, "text": ""})
                if raw.get("analysis_id"):
                    visual["regions"][-1]["analysis_region_id"] = analysis_region_id
                if analysis_region_id not in prepared_crops:
                    crops.append({"id": analysis_region_id, "kind": "text", "image": str(crop_path.resolve()), "sha256": digest})
                    prepared_crops.add(analysis_region_id)
        visuals.append(visual)
    return visuals, crops


def extract(video: Path, sources: list[Path], out: Path, *, language: str = "ru", start: float = 0.0,
            duration: float | None = None, asr_model: str = ASR_MODEL, vlm_model: str = VLM_MODEL,
            ocr_model: str = OCR_MODEL, vision_mode: str | None = None) -> Path:
    video, out = Path(video).resolve(), Path(out).resolve()
    for binary in ("ffmpeg", "ffprobe"):
        if not shutil.which(binary):
            raise RuntimeError(f"{binary} is required. Install FFmpeg with: brew install ffmpeg")
    metadata = probe(video)
    if not math.isfinite(start) or start < 0 or start >= metadata["duration"]:
        raise ValueError("start must be within the video duration.")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be positive.")
    from .vision_mode import resolve_vision_mode

    config = resolve_vision_mode(out, vision_mode)
    if config["vision_mode"] == "content":
        from .person_segmentation import check_person_segmentation

        check_person_segmentation()
    end = metadata["duration"] if duration is None else min(metadata["duration"], start + duration)
    out.mkdir(parents=True, exist_ok=True)
    video_hash = sha256_file(video)
    run_inputs = {"video": video_hash, "start": start, "end": end}
    if config["vision_mode"] == "content":
        run_inputs["vision"] = config
    source_inputs = {"format": 2, "files": [
        {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))} for path in sources]}
    records = _artifact_checkpoint(out, "sources", source_inputs, lambda: import_sources(sources, out),
                                   lambda records: [out / image for record in records
                                                    for image in record["images"] +
                                                    ([record["original"]] if record.get("original") else [])])
    chunks = _audio_chunks(video, out, start, end, video_hash)
    asr = _worker(out, "asr", {"model": asr_model, "language": language, "items": chunks,
                               "initial_prompt": source_glossary(records)})
    segments = stitch_segments(asr["items"])
    write_json(out / "transcript.json", segments)
    write_subtitles(segments, out)
    selections = checkpoint(out, "keyframes", run_inputs, lambda: _scan_video(video, metadata, start, end - start))
    images = _artifact_checkpoint(out, "frames", {**run_inputs, "selections": selections},
                                  lambda: _frames(video, out, selections),
                                  lambda records: [record["image"] for record in records])
    images.extend({"id": f"visual_{record['id']}", "source_id": record["id"], "timestamp": None,
                   "image": str(out / record["rendered_image"])} for record in records if record.get("rendered_image"))
    # Notebook raster outputs are already exact assets; parse them for meaning.
    for record in records:
        if record["kind"] == "ipynb":
            images.extend({"id": f"visual_{record['id']}_{position}", "source_id": record["id"], "timestamp": None,
                           "image": str(out / image)} for position, image in enumerate(record["images"])
                          if Path(image).suffix.lower() in (".png", ".jpg", ".jpeg"))
    visual_settings = config if config["vision_mode"] == "content" else {}
    raw_visuals = _worker(out, "vision", {"model": vlm_model, "items": images, **visual_settings})["items"]
    visuals, crops = _prepare_visuals(raw_visuals, out)
    resolved_ocr = ocr_model
    if crops and ocr_model == OCR_MODEL:
        destination = out.parent / ".models" / "PaddleOCR-VL-1.6-bf16"
        resolved_ocr = _worker(out, "convert", {"model": ocr_model, "destination": str(destination)})["model"]
    recognized = _worker(out, "ocr", {"model": resolved_ocr, "items": crops, **visual_settings})["items"]
    texts = {item["id"]: item["text"] for item in recognized}
    for visual in visuals:
        for region in visual["regions"]:
            region["text"] = texts.get(region.get("analysis_region_id", region["id"]), "")
            if visual_settings and not region["text"].strip():
                visual["issues"].append(
                    f"OCR вернул пустой ответ для области {region['id']}. Проверить видимое содержимое.")
    records = match_sources(records, visuals)
    warnings = [(visual["id"], description) for visual in visuals for description in visual["issues"]]
    warnings.extend((segment["id"], description) for segment in segments
                    for description in transcription_warnings(segment))
    unmatched = [record["id"] for record in records if not record["matched_visual_ids"]]
    issues = [{"id": f"issue_{position:04d}", "description": description,
               "evidence_ids": [identifier], "status": "open", "resolution": ""}
              for position, (identifier, description) in enumerate(warnings)]
    if unmatched:
        issues.append({"id": f"issue_{len(issues):04d}", "description":
                       f"Не удалось уверенно сопоставить {len(unmatched)} фрагментов исходных материалов с видео. Проверить их применимость к лекции.",
                       "evidence_ids": unmatched, "status": "open", "resolution": ""})
    evidence = {"schema_version": 1, "title": video.stem,
                "video": {"path": str(video), "sha256": video_hash, **metadata,
                          "clip_start": start, "clip_end": end},
                "segments": segments, "visuals": visuals, "sources": records, "issues": issues,
                "models": {"asr": asr_model, "vlm": vlm_model, "ocr": ocr_model, "ocr_runtime": resolved_ocr}}
    if visual_settings:
        evidence["visual_processing"] = config
    write_json(out / "evidence.json", evidence)
    return out / "evidence.json"
