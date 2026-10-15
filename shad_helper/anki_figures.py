"""Export the managed figures used by lecture notes, without cropping PDF pages."""

import re
from pathlib import Path

from .jobs import read_json, sha256_file
from .rendering import render_diagram_preview

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")


def _asset_path(job: Path, relative: str) -> Path:
    root = job.resolve() / "notes"
    path = root / relative
    if (root.is_symlink() or Path(relative).is_absolute() or ".." in Path(relative).parts
            or path.is_symlink() or not path.resolve().is_relative_to(root.resolve())):
        raise ValueError("Figure asset escapes the notes project")
    if not path.is_file():
        raise ValueError(f"Figure asset is missing: {relative}; regenerate and render the notes")
    return path


def load_figure_assets(job: Path, sources: dict | None) -> dict[str, dict]:
    """Bind figure block IDs to portable assets from the rendered notes project."""
    if sources is None:
        return {}
    blocks = sources["blocks"]
    has_figures = any(block.get("illustration_id") or block.get("figure_asset")
                      or block.get("kind") in {"diagram", "screenshot"}
                      for block in blocks.values())
    if not (job / "notes/project.json").exists() and not has_figures:
        return {}
    manifest = read_json(_asset_path(job, "project.json"))
    if manifest.get("version") != 2:
        raise ValueError("Unsupported notes project for figure export")
    declared = {"diagram": set(manifest.get("tex_files", [])),
                "screenshot": set(manifest.get("image_files", []))}
    project_blocks = {block["id"]: block for block in manifest.get("source_blocks", [])}
    assets = {}
    for identifier, block in blocks.items():
        if not _ID.fullmatch(identifier):
            raise ValueError("Invalid figure block ID")
        illustration_id = block.get("illustration_id", "")
        if illustration_id and not _ID.fullmatch(illustration_id):
            raise ValueError("Invalid illustration ID")
        expected = {"diagram": f"figures/diagram-{identifier}.tex"}
        if illustration_id:
            image_id = identifier if block.get("illustration_crop_id") else illustration_id
            expected["screenshot"] = f"figures/illustration-{image_id}.png"
        candidates = [(kind, path) for kind, path in expected.items() if path in declared[kind]]
        is_figure = (illustration_id or block.get("figure_asset")
                     or block.get("kind") in {"diagram", "screenshot"})
        if not candidates and not is_figure:
            continue
        if len(candidates) != 1:
            raise ValueError(f"Figure {identifier} has missing or ambiguous project assets")
        kind, relative = candidates[0]
        if block.get("kind", kind) != kind or identifier not in project_blocks:
            raise ValueError(f"Figure {identifier} does not match the notes project")
        metadata = block.get("figure_asset")
        if metadata and metadata.get("path") != relative:
            raise ValueError(f"Figure {identifier} asset does not match the notes project")
        path = _asset_path(job, relative)
        digest = sha256_file(path)
        if metadata:
            if metadata.get("sha256") != digest:
                raise ValueError(f"Figure {identifier} asset changed; run render before Anki export")
        elif kind == "diagram" and path.read_text(encoding="utf-8").rstrip() != block["content"].rstrip():
            raise ValueError(f"Figure {identifier} drawing changed; run render before Anki export")
        # Legacy manifests identify assets by their declared generated filename.
        # New source maps also bind those bytes to the rendered PDF with a digest.
        assets[identifier] = {"block_id": identifier, "kind": kind,
                              "path": relative, "sha256": digest,
                              **{key: block[key] for key in
                                 ("illustration_id", "illustration_crop_id") if block.get(key)}}
    return assets


def export_figure(job: Path, descriptor: dict, staging_dir: Path) -> Path:
    """Copy an original PNG or render its accepted TikZ source as a standalone PNG."""
    path = _asset_path(job, descriptor["path"])
    digest = descriptor["sha256"]
    if sha256_file(path) != digest:
        raise ValueError(f"Figure {descriptor['block_id']} asset changed during Anki export")
    if staging_dir.is_symlink():
        raise ValueError("Figure export directory cannot be a symbolic link")
    staging_dir.mkdir(parents=True, exist_ok=True)
    if descriptor["kind"] == "diagram":
        return render_diagram_preview(path.read_text(encoding="utf-8"),
                                      staging_dir / f"diagram-{digest}")
    if descriptor["kind"] != "screenshot":
        raise ValueError("Unsupported figure asset kind")
    import pymupdf

    data = path.read_bytes()
    try:
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Not a PNG image")
        pixels = pymupdf.Pixmap(data)
        if pixels.width <= 0 or pixels.height <= 0:
            raise ValueError("Empty image")
    except (ValueError, RuntimeError, pymupdf.mupdf.FzErrorBase) as error:
        raise ValueError(f"Figure {descriptor['block_id']} has an invalid PNG asset") from error
    destination = staging_dir / f"figure-{digest}.png"
    if destination.is_symlink():
        raise ValueError("Figure export cannot overwrite a symbolic link")
    destination.write_bytes(data)
    return destination
