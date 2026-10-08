"""Run the lecture stages sequentially, with a separate resumable job per input."""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

from .anki import generate_anki
from .extraction import extract
from .jobs import read_json
from .notes import generate_notes
from .rendering import render
from .vision_config import DEFAULT_VISION_MODEL


def _jobs(media: list[Path], out: Path) -> list[tuple[Path, Path]]:
    if not media:
        raise ValueError("At least one media file is required")
    jobs, sources, destinations = [], set(), set()
    out = Path(out).resolve()
    for source in media:
        source = Path(source).resolve()
        if not source.is_file():
            raise ValueError(f"Media file does not exist or is not a file: {source}")
        if source in sources:
            raise ValueError(f"Repeated media file: {source}")
        sources.add(source)
        stem = re.sub(r"[^\w.-]+", "-", source.stem).strip("._-")[:64].rstrip("._-")
        digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
        job = out / f"{stem or 'lecture'}-{digest}"
        if job in destinations:
            raise ValueError(f"Media files resolve to the same job directory: {job}")
        destinations.add(job)
        jobs.append((source, job))
    return jobs


def run_pipeline(media: list[Path], out: Path, state_dir: Path, *, language: str = "ru",
                 start: float = 0.0, duration: float | None = None,
                 vision_model: str = DEFAULT_VISION_MODEL, rebuild: bool = False,
                 proxy: str | None = None, anki_collection: Path | None = None) -> bool:
    jobs = _jobs(media, out)
    completed, next_position = 0, 1
    for number, (source, job) in enumerate(jobs, 1):
        label = f"[{number}/{len(jobs)}] {source} (job {job})"
        stage = "extract"
        try:
            print(f"{label}: {stage}", flush=True)
            extract(source, job, language=language, start=start, duration=duration,
                    vision_model=vision_model)
            stage = "notes"
            print(f"{label}: {stage}", flush=True)
            result = generate_notes(job, state_dir, rebuild=rebuild, proxy=proxy)
            if result.resolve() != job / "notes" / "main.tex":
                raise RuntimeError(f"Notes require manual completion: {result}")
            stage = "render"
            print(f"{label}: {stage}", flush=True)
            pdf = render(job)
            stage = "anki"
            print(f"{label}: {stage}", flush=True)
            package = generate_anki(job, state_dir, pdf=pdf, rebuild=rebuild, proxy=proxy,
                                    anki_collection=anki_collection,
                                    minimum_position=next_position)
            next_position = read_json(job / "anki" / "report.json")["end_position"] + 1
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError,
                ModuleNotFoundError) as exc:
            print(f"{label}: {stage} failed: {exc}", file=sys.stderr, flush=True)
            continue
        completed += 1
        print(f"{label}: completed — {package}", flush=True)
    print(f"Pipeline: {completed} completed, {len(jobs) - completed} failed.", flush=True)
    return completed == len(jobs)
