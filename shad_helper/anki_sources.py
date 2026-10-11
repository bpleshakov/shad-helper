"""Per-card PDF references and locally resolved lecture timestamps."""

import re
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from .anki_models import get_models
from .anki_schema import note_fields
from .jobs import fingerprint, read_json


def load_lecture_sources(job: Path, pdf: Path, pdf_hash: str) -> dict | None:
    # A separately supplied PDF has no implied relationship to this job's video.
    if pdf.resolve() != (job / "notes/main.pdf").resolve():
        return None
    path = job / "notes/source-map.json"
    if not path.exists():
        if (job / "notes/project.json").exists():
            raise ValueError("PDF source map is missing. Regenerate the notes project "
                             "from notes.json and run render before Anki generation.")
        return None
    mapping = read_json(path)
    if mapping.get("schema_version") != 1 or mapping.get("pdf_sha256") != pdf_hash:
        raise ValueError("PDF source map is stale; run render before Anki generation.")
    evidence_path = job / "evidence.json"
    evidence = read_json(evidence_path) if evidence_path.exists() else None
    if (fingerprint(evidence) if evidence is not None else None) != mapping["evidence_fingerprint"]:
        raise ValueError("Lecture evidence changed; regenerate and render the notes.")
    evidence = evidence or {}
    times = {item["id"]: {"start": item["start"], "end": item["end"]}
             for item in evidence.get("segments", [])}
    times.update({item["id"]: {"start": item["timestamp"], "end": item["timestamp"]}
                  for item in evidence.get("visuals", [])})
    return {"blocks": mapping["blocks"], "times": times,
            "media": evidence.get("media", {}).get("path")}


def source_catalog(sources: dict | None) -> list[dict]:
    # The model receives note content and PDF locations, never transcript text,
    # media paths, timestamps, extraction diagnostics, or review issues.
    return [{key: block[key] for key in ("id", "topic", "content", "page_start", "page_end")}
            for block in sources["blocks"].values()] if sources else []


def validate_card_sources(response, page_count: int, sources: dict | None) -> None:
    for note in response.notes:
        fields = note_fields(note)
        answers = {name for name, value in fields.items()
                   if name not in {"Front", "Name"} and value}
        counts = Counter(source.field for source in note.card_sources)
        if (len(note.card_sources) != len(answers)
                or counts.keys() != answers):
            name = fields.get("Name", fields.get("Front"))
            raise ValueError("Every generated answer field requires exactly one card source: "
                             f"{note.note_type} {name!r} (page {note.source.page}, "
                             f"order {note.source.order}); missing={sorted(answers - counts.keys())}; "
                             f"unexpected={sorted(counts.keys() - answers)}; "
                             f"duplicate={sorted(key for key, count in counts.items() if count > 1)}")
        for source in note.card_sources:
            if source.page > page_count:
                raise ValueError("Card source is outside the PDF")
            if sources is None:
                if source.block_ids:
                    raise ValueError("Card source block IDs require a lecture source map")
                continue
            if not source.block_ids or len(source.block_ids) != len(set(source.block_ids)):
                raise ValueError("Card sources require distinct supporting lecture block IDs")
            if set(source.block_ids) - sources["blocks"].keys():
                raise ValueError("Card source references an unknown lecture block")
            blocks = [sources["blocks"][identifier] for identifier in source.block_ids]
            if not any(block["page_start"] <= source.page <= block["page_end"] for block in blocks):
                raise ValueError("Card source page does not match its lecture blocks")
            for block in blocks:
                if (not block["evidence_ids"]
                        or set(block["evidence_ids"]) - sources["times"].keys()):
                    raise ValueError(f"Lecture block {block['id']} has no valid timed evidence")


def card_records(response, provenance: list[dict], sources: dict | None) -> list[dict]:
    models = get_models()
    cards = []
    for note, exported in zip(response.notes, provenance, strict=True):
        references = {source.field: source for source in note.card_sources}
        model = models[note.note_type]
        for ordinal, position in exported["positions"].items():
            template = model.templates[ordinal]
            # These fixed templates end with the single answer field.
            field = re.search(r"\{\{([^{}]+)\}\}\s*$", template["afmt"])[1]
            source = references[field]
            blocks = [sources["blocks"][identifier] for identifier in source.block_ids]
            evidence_ids = list(dict.fromkeys(identifier for block in blocks
                                              for identifier in block["evidence_ids"]))
            times = sorted(({"evidence_id": identifier, **sources["times"][identifier]}
                            for identifier in evidence_ids), key=lambda item: item["start"])
            cards.append({"guid": exported["guid"], "ordinal": ordinal,
                          "template": template["name"], "model": note.note_type,
                          "name": exported["name"], "field": field, "position": position,
                          "pdf_page": source.page, "pdf_location": source.location,
                          "blocks": [{key: block[key] for key in
                                      ("id", "topic", "anchor", "page_start", "page_end")}
                                     for block in blocks],
                          "media": sources["media"] if sources else None, "timestamps": times,
                          "supplemented": field in note.supplemented_fields,
                          "review_flags": note.review_flags})
    return cards


def _stamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    total, fraction = divmod(milliseconds, 1000)
    return f"{total // 3600:02}:{total // 60 % 60:02}:{total % 60:02}.{fraction:03}"


def _text(value: str) -> str:
    return re.sub(r"([\\`*_\[\]<>#|~$])", r"\\\1", " ".join(value.split()))


def write_sources_report(path: Path, report: dict) -> None:
    pdf_uri = Path(report["source_pdf"]).as_uri()
    lines = ["# Источники карточек Anki", "", _text(report["deck_name"]), "",
             f"Карточек: {report['card_count']}. Страницы PDF указаны с 1.", "",
             ("Тайм-коды взяты из исходных ссылок блоков конспекта на речь и кадры; "
              "для нескольких фрагментов перечислены все ссылки. "
              "Это интервалы источников блока, а не автоматически найденные границы фразы."), ""]
    for number, card in enumerate(report["cards"], 1):
        lines.extend([f"## {number}. {_text(card['name'])} — {_text(card['template'])}", "",
                      (f"- ID: {_text(card['guid'])}; шаблон: `{card['ordinal']}`; "
                       f"тип: {_text(card['model'])}; поле ответа: `{card['field']}`."),
                      f"- PDF: [страница {card['pdf_page']}]({pdf_uri}#page={card['pdf_page']}); "
                      + _text(card["pdf_location"])])
        for block in card["blocks"]:
            lines.append(f"- Блок [{block['id']}]({pdf_uri}#nameddest={quote(block['anchor'])}): "
                         f"{_text(block['topic'])}, страницы {block['page_start']}–{block['page_end']}.")
        if card["timestamps"]:
            for source in card["timestamps"]:
                stamp = _stamp(source["start"])
                if source["end"] != source["start"]:
                    stamp += "–" + _stamp(source["end"])
                if card["media"]:
                    uri = Path(card["media"]).expanduser().resolve().as_uri()
                    fragment = str(source["start"])
                    if source["end"] != source["start"]:
                        fragment += f",{source['end']}"
                    stamp = f"[{stamp}]({uri}#t={fragment})"
                lines.append(f"- Исходная запись: {stamp} (`{source['evidence_id']}`).")
        else:
            lines.append("- Тайм-код недоступен: у этого PDF нет связи с исходной записью.")
        if card["supplemented"]:
            lines.append("- **Дополнено моделью.** Ссылки указывают основу рассуждения; "
                         "добавленная часть не приписывается лектору.")
        if card["review_flags"]:
            lines.append("- Требует проверки: " + "; ".join(map(_text, card["review_flags"])))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
