"""Atomic JSON artifacts and input-addressed stage checkpoints."""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

EVIDENCE_VERSION = 2
AUDIO_CONFIG = {"schema_version": EVIDENCE_VERSION, "pipeline": "audio"}


def check_audio_job(job: Path) -> None:
    """Allow empty/new jobs or resumable audio jobs, without migrating old files."""
    job = Path(job)
    config = job / "extraction-config.json"
    if config.exists():
        if read_json(config) != AUDIO_CONFIG:
            raise ValueError("Unsupported job format; use a new output directory for audio extraction.")
    elif job.exists() and any(job.iterdir()):
        evidence = job / "evidence.json"
        if not evidence.is_file():
            raise ValueError("Existing job requires a new output directory for audio extraction.")
    evidence = job / "evidence.json"
    if evidence.exists():
        validate_audio_evidence(read_json(evidence))


def validate_audio_evidence(evidence: dict) -> dict:
    if (evidence.get("schema_version") != EVIDENCE_VERSION or "media" not in evidence
            or any(key in evidence for key in ("video", "visuals", "sources", "visual_processing"))):
        raise ValueError("Unsupported evidence format; use a new output directory for audio extraction.")
    return evidence


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(job: Path, key: str, inputs: Any, action: Callable[[], Any]) -> Any:
    """Only successful stages are cached; interrupted actions can be rerun."""
    if (not key or Path(key).is_absolute()
            or any(part in ("..", "") for part in key.split("/"))):
        raise ValueError("Invalid checkpoint key")
    path = Path(job) / "checkpoints" / f"{key}.json"
    signature = fingerprint(inputs)
    if path.exists():
        cached = read_json(path)
        if cached.get("fingerprint") == signature:
            print(f"Reusing {key}", flush=True)
            return cached["result"]
    print(f"Running {key}", flush=True)
    started = time.monotonic()
    result = action()
    write_json(path, {"fingerprint": signature, "seconds": time.monotonic() - started,
                      "result": result})
    return result


def evidence_index(evidence: dict) -> dict[str, str]:
    return {segment["id"]: f"Аудио {timestamp(segment['start'])}"
            for segment in evidence.get("segments", [])}


def timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"
