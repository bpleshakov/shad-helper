"""Generate a resumable, chronologically ordered Anki package from a finished PDF."""

import hashlib
import html
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import genanki
import pymupdf

from .anki_export import write_package
from .anki_figures import export_figure, load_figure_assets
from .anki_latex import render_latex_media
from .anki_models import get_models
from .anki_positions import collection_position
from .anki_prompt import INSTRUCTIONS, PROMPT_VERSION
from .anki_schema import SourceFigure, generation_schema, note_fields, validate_response
from .anki_sources import (
    card_records,
    load_lecture_sources,
    source_catalog,
    validate_card_sources,
    write_sources_report,
)
from .auth import PlanAccessError
from .jobs import fingerprint, read_json, remove_artifacts, write_json
from .openai_client import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    MAX_PDF_BYTES,
    PlanClient,
    ResponseError,
)
from .run_diagnostics import record_run, timed_step


def _figure_catalog(sources, assets):
    return [{"block_id": identifier, "kind": asset["kind"],
             **{key: sources["blocks"][identifier][key]
                for key in ("topic", "content", "page_start", "page_end")}}
            for identifier, asset in assets.items()]


def _validate_figure_sources(response, sources, assets):
    for note in response.notes:
        for figure in note.figures:
            if isinstance(figure, SourceFigure):
                if sources is None or figure.block_id not in assets:
                    raise ValueError("Figure must reference an original figure block from figure_catalog")
            elif sources is not None:
                raise ValueError("Lecture figures require block_id references, not PDF crop coordinates")


def _generate(client, pdf, page_count, sources, diagnostics, figure_assets):
    text = (f"Создай Anki-колоду по всему приложенному PDF ({page_count} страниц). "
            "Сохраняй порядок первого появления объектов и отмечай дополнения модели.")
    text += "\nКаталог блоков конспекта для привязки источников:\n" + json.dumps(
        source_catalog(sources), ensure_ascii=False)
    if sources is not None:
        text += "\nfigure_catalog:\n" + json.dumps(
            _figure_catalog(sources, figure_assets), ensure_ascii=False)
    for attempt in range(4):
        folder = diagnostics / f"attempt-{attempt + 1:02d}"
        try:
            reply = client.complete(INSTRUCTIONS, text,
                                    schema=generation_schema(source_figures=sources is not None), pdf=pdf)
            write_json(folder / "response.json", reply)
            response = validate_response(reply, page_count)
            validate_card_sources(response, page_count, sources)
            _validate_figure_sources(response, sources, figure_assets)
            write_json(folder / "status.json", {"status": "success"})
            return response
        except (PlanAccessError, ResponseError, ValueError) as exc:
            write_json(folder / "status.json", {"status": "failed", "error": str(exc),
                                               "error_type": type(exc).__name__})
            print(f"Anki generation failed: {exc}\nDiagnostics: {folder}", flush=True)
            if getattr(exc, "code", None) == "context_length_exceeded":
                raise RuntimeError("The complete PDF exceeds the model context limit; "
                                   "no partial Anki package was published.") from exc
            if (attempt == 3 or getattr(exc, "status", None) in (401, 403)
                    or getattr(exc, "code", None) == "model_unavailable"):
                raise
            delay = 5 * 2 ** attempt
            print(f"Retrying Anki generation ({attempt + 1}/3) in {delay} seconds.", flush=True)
            time.sleep(delay)
    raise AssertionError("Unreachable retry state")


def _identity(folder: Path, title: str) -> dict:
    path = folder / "identity.json"
    if path.exists():
        value = read_json(path)
        uuid.UUID(value["job_uuid"])
        if not isinstance(value["deck_id"], int) or not value["deck_name"].strip():
            raise ValueError("Invalid saved Anki deck identity")
        return value
    job_id = uuid.uuid4()
    name = re.sub(r"\s+", " ", title.replace("::", " — ")).strip()
    value = {"job_uuid": str(job_id), "deck_id": 1_000_000_000 + job_id.int % 1_000_000_000,
             "deck_name": f"{name} [{str(job_id)[:8]}]"}
    write_json(path, value)
    return value


def _field_html(value: str) -> str:
    # Anki turns <br> back into a newline before compiling LaTeX. A second
    # literal newline would insert an invalid paragraph break inside math.
    return html.escape(value, quote=False).replace("\n", "<br>")


def _publish(staging: Path, folder: Path) -> None:
    names = ("notes.json", "report.json", "sources.md", "deck.apkg")
    backup = staging / "previous"
    backup.mkdir()
    for name in names:
        if (folder / name).exists():
            shutil.copy2(folder / name, backup / name)
    published = []
    try:
        for name in names:
            os.replace(staging / name, folder / name)
            published.append(name)
    except OSError:
        for name in reversed(published):
            if (backup / name).exists():
                os.replace(backup / name, folder / name)
            else:
                (folder / name).unlink()
        raise


@timed_step("anki_note_preparation")
def _prepare_notes(response, document, media_dir, identity, pdf_hash, *, start_position=1,
                   job=None, figure_assets=None):
    models = get_models()
    notes, provenance, media = [], [], {}
    position = start_position
    exported_figures = {}
    for note in response.notes:
        plain_fields = note_fields(note)
        fields = {name: _field_html(value) for name, value in plain_fields.items()}
        for name in set(note.supplemented_fields):
            fields[name] += '<br><small>Дополнено моделью: содержит сведения сверх PDF.</small>'
        figure_records = []
        for figure in note.figures:
            source_record = {}
            if isinstance(figure, SourceFigure):
                asset = figure_assets[figure.block_id]
                if figure.block_id not in exported_figures:
                    exported_figures[figure.block_id] = export_figure(
                        job, asset, media_dir.parent / "figures")
                png = exported_figures[figure.block_id].read_bytes()
                source_record = {"source_asset": asset}
            else:
                page = document[figure.page - 1]
                left, top, right, bottom = figure.box
                rect = page.rect
                clip = pymupdf.Rect(left * rect.width, top * rect.height,
                                    right * rect.width, bottom * rect.height)
                png = page.get_pixmap(clip=clip, dpi=200, alpha=False).tobytes("png")
            filename = f"shad-{hashlib.sha256(png).hexdigest()}.png"
            path = media_dir / filename
            if filename not in media:
                path.write_bytes(png)
                media[filename] = path
            fields[figure.field] += f'<br><img src="{filename}">'
            figure_records.append({**figure.model_dump(), "file": filename, **source_record})
        model = models[note.note_type]
        values = [fields.get(field["name"], "") for field in model.fields]
        key = {"source": note.source.model_dump(), "type": note.note_type,
               "name": plain_fields.get("Name", plain_fields.get("Front"))}
        # Changing the math renderer must not duplicate previously exported notes.
        identity_key = {**key, "name": key["name"].replace("[$$]", r"\[")
                        .replace("[/$$]", r"\]").replace("[$]", r"\(")
                        .replace("[/$]", r"\)")}
        guid = genanki.guid_for(identity["job_uuid"], pdf_hash, fingerprint(identity_key))
        generated = {card.ord for card in genanki.Note(model=model, fields=values).cards}
        positions = {ordinal: position for ordinal in sorted(generated)}
        position += 1
        tags = ["shad_helper"]
        if note.supplemented_fields:
            tags.append("model_supplement")
        if note.review_flags:
            tags.append("needs_review")
        notes.append({"guid": guid, "model": note.note_type, "fields": values,
                      "positions": positions, "tags": tags})
        provenance.append({"guid": guid, **key, "positions": positions,
                           "supplemented_fields": note.supplemented_fields,
                           "review_flags": note.review_flags, "figures": figure_records})
    return notes, list(media.values()), provenance


@record_run("anki")
def generate_anki(job: Path, state_dir: Path, *, pdf: Path | None = None,
                  rebuild: bool = False, proxy: str | None = None,
                  anki_collection: Path | None = None, minimum_position: int = 1,
                  client=None) -> Path:
    job = Path(job)
    if rebuild:
        remove_artifacts(job, ("checkpoints/anki.json", "anki", "diagnostics/anki"))
    if type(minimum_position) is not int or minimum_position < 1:
        raise ValueError("minimum_position must be a positive integer")
    source = Path(pdf) if pdf is not None else job / "notes" / "main.pdf"
    if not source.is_file():
        raise ValueError(f"Anki requires a finished PDF: {source}. Run render or use --pdf.")
    with source.open("rb") as stream:
        data = stream.read(MAX_PDF_BYTES)
    if len(data) >= MAX_PDF_BYTES:
        raise ValueError("Anki PDF must be smaller than 50,000,000 bytes.")
    if not data.startswith(b"%PDF-"):
        raise ValueError("Anki input must be a PDF file.")
    # Fail before model generation if the live collection cannot be read safely.
    position_source = collection_position(anki_collection)
    pdf_hash = hashlib.sha256(data).hexdigest()
    lecture_sources = load_lecture_sources(job, source, pdf_hash)
    figure_assets = load_figure_assets(job, lecture_sources)
    folder = job / "anki"
    folder.mkdir(parents=True, exist_ok=True)
    model = client.model if client is not None else DEFAULT_MODEL
    signature = fingerprint({"pdf_sha256": pdf_hash, "model": model,
        "reasoning_effort": DEFAULT_REASONING_EFFORT, "prompt_version": PROMPT_VERSION,
        "instructions": INSTRUCTIONS,
        "schema": generation_schema(source_figures=lecture_sources is not None),
        "source_catalog": source_catalog(lecture_sources),
        "figure_catalog": _figure_catalog(lecture_sources, figure_assets)})
    cache_path = job / "checkpoints" / "anki.json"
    with pymupdf.open(stream=data, filetype="pdf") as document:
        if document.needs_pass or len(document) == 0:
            raise ValueError("Anki requires an unencrypted PDF with at least one page.")
        with tempfile.TemporaryDirectory(prefix=".build-", dir=folder) as temporary:
            staging = Path(temporary)
            # Freeze the source bytes so the request, provenance and crops always agree.
            frozen_pdf = staging / source.name
            frozen_pdf.write_bytes(data)
            with timed_step("anki_generation") as step:
                cached = read_json(cache_path) if cache_path.exists() else {}
                if cached.get("fingerprint") == signature:
                    print("Reusing anki generation", flush=True)
                    response = validate_response(cached["result"], len(document))
                    validate_card_sources(response, len(document), lecture_sources)
                    _validate_figure_sources(response, lecture_sources, figure_assets)
                    step["status"] = "cached"
                else:
                    print(f"Generating Anki cards from {len(document)} PDF pages", flush=True)
                    selected_client = client or PlanClient(state_dir, proxy=proxy)
                    diagnostics = job / "diagnostics" / "anki" / str(time.time_ns())
                    response = _generate(selected_client, frozen_pdf, len(document), lecture_sources,
                                         diagnostics, figure_assets)
                    write_json(cache_path, {"fingerprint": signature,
                                           "result": response.model_dump()})
            identity = _identity(folder, response.title)
            media_dir = staging / "media"
            media_dir.mkdir()
            notes, media, provenance = _prepare_notes(
                response, document, media_dir, identity, pdf_hash,
                job=job, figure_assets=figure_assets)
            with timed_step("anki_latex_rendering"):
                media.extend(render_latex_media(notes, media_dir))
            # Generation and TeX compilation can take minutes. Read the allocation
            # cursor again immediately before packaging, including current WAL data.
            position_source = collection_position(Path(position_source["collection_path"]))
            # Earlier packages in the same batch may not have been imported yet.
            start_position = max(position_source["next_position"], minimum_position)
            for index, (note, source_note) in enumerate(zip(notes, provenance, strict=True)):
                positions = {ordinal: start_position + index for ordinal in note["positions"]}
                note["positions"] = positions
                source_note["positions"] = positions
            with timed_step("anki_packaging"):
                write_package(staging / "deck.apkg", deck_id=identity["deck_id"],
                              deck_name=identity["deck_name"], notes=notes, media_files=media)
            report = {"schema_version": 2, **identity, "source_pdf": str(source.resolve()),
                "pdf_sha256": pdf_hash, "page_count": len(document), "model": model,
                "reasoning_effort": DEFAULT_REASONING_EFFORT, "prompt_version": PROMPT_VERSION,
                "math_format": "anki_latex",
                "position_source": position_source, "start_position": start_position,
                "end_position": start_position + len(notes) - 1,
                "note_count": len(notes), "card_count": sum(len(n["positions"]) for n in notes),
                "warnings": response.warnings, "notes": provenance,
                "cards": card_records(response, provenance, lecture_sources),
                "import_instructions": "Import deck.apkg. New card gather order: Ascending "
                    "position; New card sort order: Order gathered. Sibling cards share one "
                    "position. Positions start after the live collection allocation cursor "
                    "at export time. Reimport preserves existing positions and review schedules "
                    "and does not remove obsolete notes after PDF changes."}
            write_json(staging / "notes.json", response.model_dump())
            write_json(staging / "report.json", report)
            write_sources_report(staging / "sources.md", report)
            (folder / "media").mkdir(exist_ok=True)
            for path in media:
                os.replace(path, folder / "media" / path.name)
            _publish(staging, folder)
    print(f"Anki: {report['note_count']} notes, {report['card_count']} cards; "
          f"deck {identity['deck_name']}", flush=True)
    return folder / "deck.apkg"
