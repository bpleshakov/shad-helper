"""Read lecture source files without executing them."""

import base64
import copy
import json
import re
import shutil
import subprocess
from pathlib import Path

from .jobs import sha256_file


def _relative(path: Path, out: Path) -> str:
    return path.relative_to(out).as_posix()


def _record(index: int, kind: str, path: Path, location: str, text: str) -> dict:
    return {
        "id": f"source_{index:04d}", "kind": kind, "source": path.name,
        "location": location, "text": text, "images": [], "matched_visual_ids": [],
    }


def _pdf(path: Path, out: Path, index: int, label: str | None = None) -> list[dict]:
    import pymupdf

    records = []
    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document):
            record = _record(index + len(records), "pdf", path, f"page {page_number + 1}", page.get_text(sort=True))
            if label:
                record["source"] = label
            image = out / "source_assets" / f"{record['id']}.png"
            page.get_pixmap(dpi=150).save(image)
            record["rendered_image"] = _relative(image, out)
            record["images"].append(record["rendered_image"])
            record["text_blocks"] = [
                {"bbox": list(block[:4]), "text": block[4]}
                for block in page.get_text("blocks", sort=True) if block[6] == 0
            ]
            records.append(record)
    return records


def _pptx(path: Path, out: Path, index: int) -> list[dict]:
    from pptx import Presentation

    presentation = Presentation(path)
    records = []
    for slide_number, slide in enumerate(presentation.slides):
        pieces = []

        def read_shapes(shapes):
            for shape in shapes:
                if hasattr(shape, "shapes"):
                    read_shapes(shape.shapes)
                if shape.has_text_frame:
                    pieces.append(shape.text)
                if shape.has_table:
                    pieces.extend("\t".join(cell.text for cell in row.cells) for row in shape.table.rows)

        read_shapes(slide.shapes)
        record = _record(index + len(records), "pptx", path, f"slide {slide_number + 1}", "\n\n".join(pieces))
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame
            record["speaker_notes"] = notes.text if notes else ""
        records.append(record)

    # python-pptx does not render equations, charts or SmartArt. Keep a faithful
    # rendered slide as evidence rather than treating extracted text as complete.
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    mac_soffice = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")
    if not soffice and mac_soffice.exists():
        soffice = str(mac_soffice)
    if not soffice:
        raise RuntimeError("PPTX import requires LibreOffice (soffice) to render slides, equations and diagrams.")
    render_dir = out / "source_assets" / f"pptx_{index:04d}"
    render_dir.mkdir(parents=True, exist_ok=True)
    profile = render_dir / "libreoffice_profile"
    pdf_path = render_dir / f"{path.stem}.pdf"
    pdf_path.unlink(missing_ok=True)
    subprocess.run(
        [soffice, f"-env:UserInstallation={profile.resolve().as_uri()}", "--headless",
         "--convert-to", "pdf", "--outdir", str(render_dir), str(path.resolve())],
        check=True, capture_output=True, text=True,
    )
    if not pdf_path.exists():
        raise RuntimeError(f"LibreOffice did not produce a rendered PDF for {path.name}.")
    rendered = _pdf(pdf_path, out, index, label=path.name)
    if len(rendered) != len(records):
        raise RuntimeError(f"Rendered slide count differs from the PPTX slide count for {path.name}.")
    for record, page in zip(records, rendered):
        record["images"] = page["images"]
        record["rendered_image"] = page["rendered_image"]
    return records


def _notebook(path: Path, out: Path, index: int) -> list[dict]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError("Notebook import supports the current nbformat v4 JSON format.")
    records = []
    for cell_number, cell in enumerate(notebook["cells"]):
        source = cell["source"]
        source = source if isinstance(source, str) else "".join(source)
        record = _record(index + len(records), "ipynb", path, f"cell {cell_number + 1}", source if cell["cell_type"] != "code" else "")
        record["cell_id"] = cell.get("id", str(cell_number))
        record["cell_type"] = cell["cell_type"]
        if cell["cell_type"] == "code":
            record["code"] = source
            record["execution_count"] = cell.get("execution_count")
            record["outputs"] = json.loads(json.dumps(cell.get("outputs", []), ensure_ascii=False))
            texts = []
            for output_number, output in enumerate(cell.get("outputs", [])):
                if output.get("text"):
                    text = output["text"]
                    texts.append(text if isinstance(text, str) else "".join(text))
                if output.get("output_type") == "error":
                    texts.append(f"{output.get('ename', '')}: {output.get('evalue', '')}")
                data = output.get("data", {})
                for mime in ("text/plain", "text/markdown", "text/latex", "text/html"):
                    value = data.get(mime)
                    if value:
                        texts.append(value if isinstance(value, str) else "".join(value))
                for mime, extension in (("image/svg+xml", "svg"), ("image/png", "png"), ("image/jpeg", "jpg")):
                    if mime not in data:
                        continue
                    asset = out / "source_assets" / f"{record['id']}_{output_number}.{extension}"
                    value = data[mime]
                    value = value if isinstance(value, str) else "".join(value)
                    asset.write_bytes(value.encode("utf-8") if extension == "svg" else base64.b64decode(value))
                    record["images"].append(_relative(asset, out))
            record["text"] = "\n\n".join(texts)
        record["attachments"] = json.loads(json.dumps(cell.get("attachments", {}), ensure_ascii=False))
        for attachment_number, bundle in enumerate(record["attachments"].values()):
            for mime, extension in (("image/svg+xml", "svg"), ("image/png", "png"), ("image/jpeg", "jpg")):
                if mime in bundle:
                    value = bundle[mime]
                    value = value if isinstance(value, str) else "".join(value)
                    asset = out / "source_assets" / f"{record['id']}_attachment_{attachment_number}.{extension}"
                    asset.write_bytes(value.encode("utf-8") if extension == "svg" else base64.b64decode(value))
                    record["images"].append(_relative(asset, out))
        records.append(record)
    return records


def import_sources(paths: list[Path], out: Path) -> list[dict]:
    """Import native text/code plus rendered evidence; all assets live under out."""
    out = Path(out)
    (out / "source_assets").mkdir(parents=True, exist_ok=True)
    originals = out / "source_originals"
    originals.mkdir(parents=True, exist_ok=True)
    records = []
    importers = {".pdf": _pdf, ".pptx": _pptx, ".ipynb": _notebook}
    for position, path in enumerate(paths):
        path = Path(path)
        importer = importers.get(path.suffix.lower())
        if not importer:
            raise ValueError(f"Unsupported lecture source: {path.name}; expected PDF, PPTX or IPYNB.")
        original = originals / f"{position:04d}_{path.name}"
        if path.resolve() != original.resolve():
            shutil.copy2(path, original)
        imported = importer(original, out, len(records))
        digest = sha256_file(original)
        for record in imported:
            record["source"] = path.name
            record["original"] = _relative(original, out)
            record["sha256"] = digest
        records.extend(imported)
    return records


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[^\W_]+", text.casefold(), re.UNICODE) if len(token) >= 3}


def _formula(text: str) -> str:
    text = re.sub(r"\s+", "", text).strip("$")
    for left, right in ((r"\(", r"\)"), (r"\[", r"\]")):
        if text.startswith(left) and text.endswith(right):
            text = text[len(left):-len(right)]
    return text if len(text) >= 5 else ""


def match_sources(records: list[dict], visuals: list[dict]) -> list[dict]:
    """Conservative text matching; unmatched evidence remains available."""
    result = copy.deepcopy(records)
    token_sets = [_tokens(record.get("text", "") + " " + record.get("code", "")) for record in result]
    formula_sets = [set() for _ in result]
    source_positions = {record["id"]: position for position, record in enumerate(result)}
    for visual in visuals:
        if visual.get("source_id") in source_positions:
            formula_sets[source_positions[visual["source_id"]]].update(
                _formula(region.get("text", "")) for region in visual.get("regions", [])
                if region.get("kind") == "formula" and _formula(region.get("text", "")))
    for position, record in enumerate(result):
        for formula in re.findall(r"\$\$?(.*?)\$\$?|\\\((.*?)\\\)|\\\[(.*?)\\\]", record.get("text", ""), re.DOTALL):
            value = _formula(next((part for part in formula if part), ""))
            if value:
                formula_sets[position].add(value)
    for record in result:
        record["matched_visual_ids"] = []
    for visual in visuals:
        if visual.get("source_id") or visual.get("timestamp") is None:
            continue
        text = visual.get("summary", "") + " " + " ".join(region.get("text", "") for region in visual.get("regions", []))
        tokens = _tokens(text)
        formulas = {_formula(region.get("text", "")) for region in visual.get("regions", [])
                    if region.get("kind") == "formula" and _formula(region.get("text", ""))}
        code_regions = {region.get("text", "").strip() for region in visual.get("regions", [])
                        if region.get("kind") == "code" and len(region.get("text", "").strip()) >= 20}
        scores = []
        for position, source_tokens in enumerate(token_sets):
            common = len(tokens & source_tokens)
            score = common / min(len(tokens), len(source_tokens)) if tokens and source_tokens and common >= 4 else 0
            if formulas & formula_sets[position] or result[position].get("code", "").strip() in code_regions:
                score = 1.0
            scores.append(score)
        if scores and max(scores) >= 0.35:
            winner = scores.index(max(scores))
            # Ties are ambiguous; do not fabricate a correspondence.
            if scores.count(scores[winner]) == 1:
                result[winner]["matched_visual_ids"].append(visual["id"])
    return result
