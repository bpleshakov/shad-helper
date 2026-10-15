from copy import deepcopy

import pymupdf
import pytest

from shad_helper import anki_figures, pdf_sources, rendering
from shad_helper.jobs import read_json, sha256_file, write_json


@pytest.fixture
def figure_job(tmp_path):
    figures = tmp_path / "notes/figures"
    figures.mkdir(parents=True)
    screenshot = figures / "illustration-shot.png"
    pixels = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 13, 7), False)
    pixels.clear_with(180)
    pixels.save(screenshot)
    content = "\\begin{tikzpicture}\n\\draw (0,0) -- (1,1);\n\\end{tikzpicture}"
    diagram = figures / "diagram-drawing.tex"
    diagram.write_text(content + "\n", encoding="utf-8")
    blocks = {
        "shot": {"id": "shot", "kind": "screenshot", "content": "Original screenshot",
                 "illustration_id": "frame_1_ill01", "illustration_crop_id": "crop_hash",
                 "figure_asset": {"path": "figures/illustration-shot.png",
                                  "sha256": sha256_file(screenshot)}},
        "drawing": {"id": "drawing", "kind": "diagram", "content": content,
                    "illustration_id": "frame_2_ill01",
                    "figure_asset": {"path": "figures/diagram-drawing.tex",
                                     "sha256": sha256_file(diagram)}},
    }
    manifest = {"version": 2, "tex_files": ["main.tex", "figures/diagram-drawing.tex"],
                "image_files": ["figures/illustration-shot.png"],
                "source_blocks": list(deepcopy(blocks).values())}
    write_json(tmp_path / "notes/project.json", manifest)
    return tmp_path, {"blocks": blocks}, manifest


def test_original_screenshot_bytes_export_without_pdf_or_raw_evidence(figure_job):
    job, sources, _ = figure_job
    assets = anki_figures.load_figure_assets(job, sources)
    exported = anki_figures.export_figure(job, assets["shot"], job / "staging")
    assert exported.read_bytes() == (job / "notes/figures/illustration-shot.png").read_bytes()
    assert assets["shot"]["illustration_id"] == "frame_1_ill01"
    assert assets["shot"]["illustration_crop_id"] == "crop_hash"
    assert pymupdf.Pixmap(exported).width == 13
    assert not (job / "evidence.json").exists()
    assert not (job / "notes/main.pdf").exists()


def test_diagram_exports_managed_latex_not_original_camera_crop(figure_job, monkeypatch):
    job, sources, _ = figure_job
    assets = anki_figures.load_figure_assets(job, sources)
    calls = []

    def render(content, folder):
        calls.append(content)
        folder.mkdir()
        preview = folder / "preview.png"
        preview.write_bytes(b"rendered diagram")
        return preview

    monkeypatch.setattr(anki_figures, "render_diagram_preview", render)
    exported = anki_figures.export_figure(job, assets["drawing"], job / "staging")
    assert calls == [(job / "notes/figures/diagram-drawing.tex").read_text()]
    assert exported.read_bytes() == b"rendered diagram"


def test_legacy_manifest_resolves_exact_declared_filenames(figure_job):
    job, sources, manifest = figure_job
    for block in sources["blocks"].values():
        block.pop("kind")
        block.pop("figure_asset")
    manifest["source_blocks"] = list(sources["blocks"].values())
    write_json(job / "notes/project.json", manifest)
    assets = anki_figures.load_figure_assets(job, sources)
    assert assets["shot"]["path"] == "figures/illustration-shot.png"
    assert assets["drawing"]["path"] == "figures/diagram-drawing.tex"


def test_legacy_unrefined_screenshot_uses_illustration_id(figure_job):
    job, sources, manifest = figure_job
    block = sources["blocks"]["shot"]
    block.pop("figure_asset")
    block.pop("illustration_crop_id")
    relative = "figures/illustration-frame_1_ill01.png"
    (job / "notes/figures/illustration-shot.png").rename(job / "notes" / relative)
    manifest["image_files"] = [relative]
    write_json(job / "notes/project.json", manifest)
    assert anki_figures.load_figure_assets(job, sources)["shot"]["path"] == relative


@pytest.mark.parametrize("kind", ["shot", "drawing"])
def test_missing_original_asset_fails_instead_of_pdf_fallback(figure_job, kind):
    job, sources, _ = figure_job
    (job / "notes" / sources["blocks"][kind]["figure_asset"]["path"]).unlink()
    with pytest.raises(ValueError, match="asset is missing"):
        anki_figures.load_figure_assets(job, sources)


def test_ambiguous_legacy_assets_are_rejected(figure_job):
    job, sources, manifest = figure_job
    sources["blocks"]["shot"].pop("kind")
    sources["blocks"]["shot"].pop("figure_asset")
    manifest["tex_files"].append("figures/diagram-shot.tex")
    write_json(job / "notes/project.json", manifest)
    with pytest.raises(ValueError, match="ambiguous"):
        anki_figures.load_figure_assets(job, sources)


def test_asset_modified_since_render_is_rejected(figure_job):
    job, sources, _ = figure_job
    (job / "notes/figures/illustration-shot.png").write_bytes(b"edited")
    with pytest.raises(ValueError, match="changed; run render"):
        anki_figures.load_figure_assets(job, sources)


def test_asset_modified_during_export_is_rejected(figure_job):
    job, sources, _ = figure_job
    assets = anki_figures.load_figure_assets(job, sources)
    (job / "notes/figures/illustration-shot.png").write_bytes(b"edited")
    with pytest.raises(ValueError, match="changed during Anki export"):
        anki_figures.export_figure(job, assets["shot"], job / "staging")


def test_corrupt_legacy_image_is_rejected_without_pdf_fallback(figure_job):
    job, sources, _ = figure_job
    sources["blocks"]["shot"].pop("figure_asset")
    (job / "notes/figures/illustration-shot.png").write_bytes(b"not a PNG")
    descriptor = anki_figures.load_figure_assets(job, sources)["shot"]
    with pytest.raises(ValueError, match="invalid PNG"):
        anki_figures.export_figure(job, descriptor, job / "staging")


def test_legacy_drawing_changed_since_source_map_is_rejected(figure_job):
    job, sources, _ = figure_job
    sources["blocks"]["drawing"].pop("figure_asset")
    (job / "notes/figures/diagram-drawing.tex").write_text("different drawing")
    with pytest.raises(ValueError, match="drawing changed"):
        anki_figures.load_figure_assets(job, sources)


@pytest.mark.parametrize("relative", ["../outside.png", "/tmp/outside.png"])
def test_descriptor_cannot_export_outside_project(figure_job, relative):
    job, sources, _ = figure_job
    descriptor = anki_figures.load_figure_assets(job, sources)["shot"]
    descriptor["path"] = relative
    with pytest.raises(ValueError, match="escapes"):
        anki_figures.export_figure(job, descriptor, job / "staging")


def test_symlink_asset_cannot_escape_project(figure_job):
    job, sources, _ = figure_job
    asset = job / "notes/figures/illustration-shot.png"
    outside = job / "outside.png"
    asset.rename(outside)
    asset.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        anki_figures.load_figure_assets(job, sources)


def test_nonlecture_and_nonfigure_sources_need_no_project(tmp_path):
    assert anki_figures.load_figure_assets(tmp_path, None) == {}
    assert anki_figures.load_figure_assets(tmp_path, {"blocks": {
        "prose": {"id": "prose", "content": "Explanation"}}}) == {}


def test_renderer_declares_figure_kind_and_asset_digest(tmp_path):
    write_json(tmp_path / "evidence.json", {"schema_version": 3, "title": "Lecture",
               "media": {}, "segments": []})
    notes = {"title": "Lecture", "chapters": [{"id": "ch1", "title": "Topic", "start": 0,
             "blocks": [{"id": "figure1", "kind": "diagram", "content": r"\draw (0,0)--(1,1);",
                         "caption": "A drawing"}]}]}
    root = rendering.write_project(tmp_path, notes).parent
    block = read_json(root / "project.json")["source_blocks"][0]
    assert block["kind"] == "diagram"
    assert block["figure_asset"] == {"path": "figures/diagram-figure1.tex",
                                     "sha256": sha256_file(root / "figures/diagram-figure1.tex")}


def test_source_map_hashes_actual_assets_after_manual_edit(figure_job, monkeypatch):
    job, sources, manifest = figure_job
    pdf = job / "notes/main.pdf"
    pdf.write_bytes(b"stub")
    path = job / "notes/figures/diagram-drawing.tex"
    path.write_text(r"\begin{tikzpicture}\draw (0,0)--(2,1);\end{tikzpicture}")
    manifest["evidence_fingerprint"] = "evidence"

    class Document:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __len__(self):
            return 1

        def resolve_names(self):
            return {f"shad-{mark}-{identifier}": {"page": 0}
                    for identifier in sources["blocks"] for mark in ("block", "end")}

    monkeypatch.setattr(pdf_sources.pymupdf, "open", lambda _: Document())
    pdf_sources.write_source_map(pdf, manifest)
    mapped = read_json(job / "notes/source-map.json")
    assert mapped["blocks"]["drawing"]["figure_asset"]["sha256"] == sha256_file(path)
    assert mapped["blocks"]["drawing"]["figure_asset"]["sha256"] != (
        manifest["source_blocks"][1]["figure_asset"]["sha256"])
    assert anki_figures.load_figure_assets(job, mapped)["drawing"]["sha256"] == sha256_file(path)
