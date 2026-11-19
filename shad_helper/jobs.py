"""Atomic JSON artifacts and input-addressed stage checkpoints."""

import hashlib
import json
import math
import os
import re
import shutil
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable

EVIDENCE_VERSION = 3
VIDEO_EVIDENCE_VERSION = 5
AUDIO_CONFIG = {"schema_version": EVIDENCE_VERSION, "pipeline": "audio"}


def check_job(job: Path) -> None:
    """Allow new jobs and supported audio/video jobs, without adopting older formats."""
    job = Path(job)
    config = job / "extraction-config.json"
    if config.exists():
        saved = read_json(config)
        video_config = (isinstance(saved, dict)
                        and set(saved) in ({"schema_version", "pipeline", "vision_model"},
                                           {"schema_version", "pipeline", "vision_model", "vision_mode"})
                        and saved["schema_version"] in (4, VIDEO_EVIDENCE_VERSION)
                        and saved["pipeline"] == "audio_video"
                        and saved["vision_model"] in ("4b", "9b", "35b")
                        and saved.get("vision_mode", "frames") in ("frames", "video"))
        if saved != AUDIO_CONFIG and not video_config:
            raise ValueError("Unsupported job format; use a new output directory for extraction.")
    elif job.exists() and any(path.name != "run-diagnostics.json" or not path.is_file()
                              for path in job.iterdir()):
        evidence = job / "evidence.json"
        if not evidence.is_file():
            raise ValueError("Existing job requires a new output directory for extraction.")
    evidence = job / "evidence.json"
    if evidence.exists():
        validate_evidence(read_json(evidence))


def check_audio_job(job: Path) -> None:
    """Compatibility entry point for callers checking an existing audio job."""
    check_job(job)


def validate_audio_evidence(evidence: dict) -> dict:
    if (not isinstance(evidence, dict) or evidence.get("schema_version") != EVIDENCE_VERSION
            or set(evidence) != {"schema_version", "title", "media", "segments"}
            or not isinstance(evidence["media"], dict)
            or not isinstance(evidence["segments"], list)
            or any(not isinstance(segment, dict)
                   or set(segment) != {"id", "start", "end", "text"}
                   for segment in evidence["segments"])):
        raise ValueError("Unsupported evidence format; use a new output directory for audio extraction.")
    return evidence


def validate_evidence(evidence: dict) -> dict:
    if isinstance(evidence, dict) and evidence.get("schema_version") == EVIDENCE_VERSION:
        return validate_audio_evidence(evidence)
    message = "Unsupported evidence format; use a new output directory for extraction."
    if (not isinstance(evidence, dict)
            or evidence.get("schema_version") not in (4, VIDEO_EVIDENCE_VERSION)
            or set(evidence) != {"schema_version", "title", "media", "segments", "visuals"}
            or not isinstance(evidence["visuals"], list)):
        raise ValueError(message)
    validate_audio_evidence({**{key: value for key, value in evidence.items() if key != "visuals"},
                             "schema_version": EVIDENCE_VERSION})
    identifiers = [segment["id"] for segment in evidence["segments"]]
    visual_keys = {"id", "timestamp", "status", "kind", "text", "formulas",
                   "diagrams", "unreadable"}
    if evidence["schema_version"] == VIDEO_EVIDENCE_VERSION:
        visual_keys.add("illustrations")
    for visual in evidence["visuals"]:
        if (not isinstance(visual, dict)
                or set(visual) != visual_keys
                or not isinstance(visual["id"], str) or not visual["id"]
                or type(visual["timestamp"]) not in (int, float)
                or not math.isfinite(visual["timestamp"]) or visual["timestamp"] < 0
                or visual["status"] not in ("complete", "incomplete")
                or visual["kind"] not in ("slides", "whiteboard", "notebook", "other")
                or not isinstance(visual["text"], str)
                or not isinstance(visual["diagrams"], list)):
            raise ValueError(message)
        for key in ("formulas", "unreadable"):
            if (not isinstance(visual[key], list)
                    or not all(isinstance(value, str) for value in visual[key])):
                raise ValueError(message)
        for diagram in visual["diagrams"]:
            if (not isinstance(diagram, dict)
                    or set(diagram) != {"description", "labels", "relationships"}
                    or not isinstance(diagram["description"], str)
                    or any(not isinstance(diagram[key], list)
                           or not all(isinstance(value, str) for value in diagram[key])
                           for key in ("labels", "relationships"))):
                raise ValueError(message)
        identifiers.append(visual["id"])
        if evidence["schema_version"] == VIDEO_EVIDENCE_VERSION:
            from .illustrations import validate_illustration_observation

            illustrations = visual["illustrations"]
            if not isinstance(illustrations, list):
                raise ValueError(message)
            for position, illustration in enumerate(illustrations, 1):
                if (not isinstance(illustration, dict)
                        or set(illustration) != {"id", "source_kind", "box", "description",
                                                "image_sha256"}
                        or illustration["id"] != f"{visual['id']}_ill{position:02d}"
                        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", illustration["id"])
                        or not isinstance(illustration["image_sha256"], str)
                        or len(illustration["image_sha256"]) != 64
                        or any(char not in "0123456789abcdef"
                               for char in illustration["image_sha256"])):
                    raise ValueError(message)
                validate_illustration_observation({key: illustration[key] for key in
                                                  ("source_kind", "box", "description")})
                identifiers.append(illustration["id"])
    if not all(isinstance(identifier, str) and identifier for identifier in identifiers):
        raise ValueError(message)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Evidence IDs must be unique across speech and video.")
    return evidence


_verification_files = ContextVar("chapter_verification_files", default=None)


@contextmanager
def verification_files(job):
    """Collect the managed inputs actually used by a verified chapter checkpoint."""
    files = set()
    token = _verification_files.set((job, files))
    try:
        yield files
    finally:
        _verification_files.reset(token)


def _record_file(path):
    active = _verification_files.get()
    if active is not None:
        relative = Path(path).relative_to(active[0], walk_up=True)
        if relative.parts and relative.parts[0] in {"illustration-crops", "illustration-verification",
                                                   "illustration-stitches"}:
            if not relative.name.startswith("http-") and relative.name not in {"call.json", "status.json"}:
                active[1].add(relative)
            if relative.name in {"verdict.json", "crop.json"}:
                for name in ("preview.png", "source.png") if relative.name == "verdict.json" else ("crop.png",):
                    sibling = relative.with_name(name)
                    if (active[0] / sibling).is_file():
                        active[1].add(sibling)


def read_json(path: Path) -> Any:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _record_file(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
        _record_file(path)
    finally:
        temporary.unlink(missing_ok=True)


def remove_artifacts(job: Path, names: tuple[str, ...]) -> None:
    """Delete stage-owned outputs without following links outside the job."""
    job = Path(job).resolve()
    paths = [job / name for name in names]
    for path in paths:
        if not path.parent.resolve().is_relative_to(job):
            raise ValueError(f"Artifact directory escapes the job: {path}")
        if any(parent.is_symlink() for parent in path.parents
               if parent != job and parent.is_relative_to(job)):
            raise ValueError(f"Artifact directory has a symbolic-link parent: {path}")
    # Interrupted atomic JSON writes belong to the same stage as their target.
    paths += [temporary for path in paths if path.suffix == ".json"
              for temporary in path.parent.glob(f".{path.name}.*.tmp")]
    for path in paths:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def fingerprint(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    _record_file(path)
    return digest.hexdigest()


def checkpoint(job: Path, key: str, inputs: Any, action: Callable[[], Any], *,
               quiet: bool = False) -> Any:
    """Only successful stages are cached; interrupted actions can be rerun."""
    from .run_diagnostics import timed_step

    if (not key or Path(key).is_absolute()
            or any(part in ("..", "") for part in key.split("/"))):
        raise ValueError("Invalid checkpoint key")
    with timed_step(f"checkpoint/{key}") as step:
        path = Path(job) / "checkpoints" / f"{key}.json"
        signature = fingerprint(inputs)
        if path.exists():
            cached = read_json(path)
            if cached.get("fingerprint") == signature:
                if not quiet:
                    print(f"Reusing {key}", flush=True)
                step["status"] = "cached"
                return cached["result"]
        if not quiet:
            print(f"Running {key}", flush=True)
        started = time.monotonic()
        result = action()
        write_json(path, {"fingerprint": signature, "seconds": time.monotonic() - started,
                          "result": result})
        return result


def evidence_index(evidence: dict) -> dict[str, str]:
    return {**{segment["id"]: f"Аудио {timestamp(segment['start'])}"
               for segment in evidence.get("segments", [])},
            **{visual["id"]: f"Видео {timestamp(visual['timestamp'])}"
               for visual in evidence.get("visuals", [])},
            **{illustration["id"]: f"Иллюстрация, видео {timestamp(visual['timestamp'])}"
               for visual in evidence.get("visuals", [])
               for illustration in visual.get("illustrations", [])}}


def timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"
