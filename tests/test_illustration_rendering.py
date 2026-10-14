import shutil
import subprocess
import unicodedata

import pymupdf
import pytest

from shad_helper import rendering
from shad_helper.jobs import read_json, sha256_file, write_json


@pytest.fixture
def illustrated_job(tmp_path):
    crop = tmp_path / "illustrations/frame_1_ill01.png"
    crop.parent.mkdir()
    with pymupdf.open() as document:
        page = document.new_page(width=120, height=70)
        page.draw_line((10, 55), (105, 15), color=(0, 0, 1), width=2)
        page.insert_text((10, 20), "A")
        page.get_pixmap().save(crop)
    digest = sha256_file(crop)
    write_json(crop.parent / "manifest.json", {"version": 1, "assets": {
        "frame_1_ill01": {"path": "illustrations/frame_1_ill01.png", "sha256": digest}}})
    evidence = {"schema_version": 5, "title": "Лекция", "media": {"duration": 60},
                "segments": [], "visuals": [
                    {"id": "frame_1", "timestamp": 10, "status": "complete", "kind": "slides",
                     "text": "", "formulas": [], "diagrams": [], "unreadable": [],
                     "illustrations": [{"id": "frame_1_ill01", "source_kind": "printed",
                                        "box": [0, 0, 1, 1], "description": "Стрелка A",
                                        "image_sha256": digest}]}]}
    write_json(tmp_path / "evidence.json", evidence)
    notes = {"title": "Лекция", "language": "ru", "glossary": [], "issues": [],
             "chapters": [{"id": "ch001", "title": "График", "start": 0, "end": 60,
                           "blocks": [{"id": "ch001_b1", "kind": "screenshot",
                                       "content": "Стрелка A", "caption": "График & подпись",
                                       "illustration_id": "frame_1_ill01",
                                       "evidence_ids": ["frame_1"]}]}]}
    return tmp_path, evidence, notes


def test_screenshots_copy_only_declared_crop_and_keep_description(illustrated_job):
    job, _, notes = illustrated_job
    root = rendering.write_project(job, notes).parent
    manifest = read_json(root / "project.json")
    assert manifest["version"] == 2
    assert manifest["image_files"] == ["figures/illustration-frame_1_ill01.png"]
    assert (root / manifest["image_files"][0]).read_bytes() == (
        job / "illustrations/frame_1_ill01.png").read_bytes()
    rendering._validate_project(root, manifest)
    source = (root / "chapters/section-001.tex").read_text()
    assert r"\includegraphics[width=\linewidth,height=.7\textheight,keepaspectratio]" in source
    assert r"\caption{График \& подпись}" in source
    assert r"\hypertarget{shad-block-ch001_b1}{}" in source
    assert r"\hypertarget{shad-end-ch001_b1}{}" in source
    assert manifest["source_blocks"][0]["content"] == "Стрелка A"
    assert manifest["source_blocks"][0]["illustration_id"] == "frame_1_ill01"


def test_legacy_project_without_image_files_remains_valid(illustrated_job):
    job, _, notes = illustrated_job
    notes["chapters"][0]["blocks"][0].update(kind="text", illustration_id="")
    root = rendering.write_project(job, notes).parent
    manifest = read_json(root / "project.json")
    manifest.pop("image_files")
    rendering._validate_project(root, manifest)


def test_flagged_prose_uses_illustration_timestamp(illustrated_job):
    job, _, notes = illustrated_job
    notes["chapters"][0]["blocks"].append({
        "id": "ch001_b2", "kind": "text", "content": "Направление стрелки неясно.",
        "evidence_ids": ["frame_1", "frame_1_ill01"],
        "uncertain_fragments": [{"text": "Направление стрелки",
                                 "evidence_ids": ["frame_1_ill01", "frame_1"]}],
    })
    root = rendering.write_project(job, notes).parent
    rendering._validate_project(root, read_json(root / "project.json"))
    source = (root / "chapters/section-001.tex").read_text()
    assert r"\textcolor{blue!60!black}{Направление стрелки} неясно.\footnote{00:00:10}" in source
    assert source.count(r"\footnote") == 1


@pytest.mark.parametrize("identifier", ["unknown", "../frame_1_ill01", r"\input{secret}"])
def test_screenshot_rejects_unknown_or_unsafe_id(illustrated_job, identifier):
    job, _, notes = illustrated_job
    notes["chapters"][0]["blocks"][0]["illustration_id"] = identifier
    with pytest.raises(rendering.RenderError):
        rendering.write_project(job, notes)


@pytest.mark.parametrize("reference", ["../illustrations/frame_1_ill01.png",
                                      "/tmp/source.png", "figures/unknown.png",
                                      r"\text{figures/illustration-frame_1_ill01.png}"])
def test_manual_image_paths_must_be_declared_literals(illustrated_job, reference):
    job, _, notes = illustrated_job
    root = rendering.write_project(job, notes).parent
    source = root / "chapters/section-001.tex"
    source.write_text(source.read_text() + r"\includegraphics{" + reference + "}")
    with pytest.raises(rendering.RenderError):
        rendering._validate_project(root, read_json(root / "project.json"))


@pytest.mark.parametrize("problem", ["missing", "symlink", "not_png", "undeclared"])
def test_project_image_assets_are_validated(illustrated_job, problem):
    job, _, notes = illustrated_job
    root = rendering.write_project(job, notes).parent
    manifest = read_json(root / "project.json")
    image = root / manifest["image_files"][0]
    if problem == "undeclared":
        manifest.pop("image_files")
    else:
        image.unlink()
        if problem == "symlink":
            image.symlink_to(job / "illustrations/frame_1_ill01.png")
        elif problem == "not_png":
            image.write_bytes(b"not an image")
    with pytest.raises(rendering.RenderError):
        rendering._validate_project(root, manifest)


def test_diagram_cannot_embed_images_even_when_manually_edited(illustrated_job):
    job, _, notes = illustrated_job
    block = notes["chapters"][0]["blocks"][0]
    block.update(kind="diagram", content=r"\includegraphics{figures/a.png}")
    with pytest.raises(rendering.RenderError, match="includegraphics"):
        rendering.write_project(job, notes)
    with pytest.raises(rendering.RenderError, match="inline paths"):
        rendering._validate_commands(block["content"], diagram=True, project=True)


needs_tex = pytest.mark.skipif(not shutil.which("latexmk") or not shutil.which("lualatex"),
                               reason="requires TeX Live/MacTeX")


@needs_tex
def test_cropped_screenshot_pdf_is_portable_and_keeps_manual_edits(illustrated_job):
    job, _, notes = illustrated_job
    root = rendering.write_project(job, notes).parent
    source = root / "chapters/section-001.tex"
    manual = source.read_text() + "\nАвторская правка.\n"
    source.write_text(manual)
    shutil.rmtree(job / "illustrations")
    pdf = rendering.render(job)
    assert source.read_text() == manual
    with pymupdf.open(pdf) as document:
        assert any(page.get_images() for page in document)
        assert "Авторская правка" in "\n".join(page.get_text() for page in document)
    mapping = read_json(root / "source-map.json")["blocks"]["ch001_b1"]
    assert mapping["illustration_id"] == "frame_1_ill01"
    assert mapping["content"] == "Стрелка A"
    assert mapping["page_start"] == mapping["page_end"] == 1


@needs_tex
def test_chapter_validation_copies_assets_from_original_job(illustrated_job, chapter_latex_checker):
    job, evidence, notes = illustrated_job
    folder = job / "isolated-validation"
    chapter_latex_checker(notes, evidence, folder, asset_job=job)
    assert (folder / "notes/figures/illustration-frame_1_ill01.png").is_file()
    assert (folder / "notes/main.pdf").is_file()


@needs_tex
def test_actual_diagram_preview_is_tightly_cropped_and_isolated(tmp_path):
    preview = rendering.render_diagram_preview(
        r"\draw[->] (0,0)--(2,1) node[right] {$x$};", tmp_path)
    pixmap = pymupdf.Pixmap(preview)
    assert 100 < pixmap.width < 300
    assert 50 < pixmap.height < 200
    with pymupdf.open(tmp_path / "main.pdf") as document:
        assert len(document) == 1
        assert unicodedata.normalize("NFKC", document[0].get_text()).strip() == "x"
    assert (tmp_path / "figures/diagram-preview.tex").is_file()
    assert "caption" not in (tmp_path / "main.tex").read_text()


@needs_tex
def test_actual_pgfplots_preview_supports_page_relative_dimensions(tmp_path):
    preview = rendering.render_diagram_preview(
        r"\begin{tikzpicture}"
        r"\begin{axis}[width=.6\linewidth,height=.25\textheight,"
        r"xlabel={$x$},ylabel={$y$},domain=-1:1,samples=20]"
        r"\addplot {x*x};\end{axis}"
        r"\node at (.25\textwidth,0) {График};"
        r"\end{tikzpicture}", tmp_path)
    pixmap = pymupdf.Pixmap(preview)
    assert pixmap.width > 300
    assert pixmap.height > 200
    with pymupdf.open(tmp_path / "main.pdf") as document:
        assert "График" in document[0].get_text()
    with pytest.raises(rendering.RenderError, match="includegraphics"):
        rendering._validate_commands(r"\includegraphics[width=\linewidth]{image.png}", diagram=True)


@needs_tex
@pytest.mark.parametrize("content, message", [
    (r"\draw (0,0)--(30,1);", "overflow"),
    (r"\begin{tikzpicture}\end{tikzpicture}", "empty|one page"),
    (r"\draw (0,0)--(bad);", "compilation failed"),
])
def test_unusable_diagram_previews_retain_diagnostics(tmp_path, content, message):
    with pytest.raises(rendering.RenderError, match=message):
        rendering.render_diagram_preview(content, tmp_path)
    assert (tmp_path / "main.tex").is_file()
    assert (tmp_path / "compiler-output.log").is_file()
    assert not (tmp_path / "preview.png").exists()


@pytest.mark.parametrize("problem", ["multiple_pages", "clipped"])
def test_preview_rejects_invalid_painted_bounds(tmp_path, monkeypatch, problem):
    def compile_pdf(root):
        with pymupdf.open() as document:
            page = document.new_page(width=100, height=100)
            if problem == "clipped":
                page.draw_line((-10, 10), (50, 50), color=(0, 0, 0))
            else:
                page.insert_text((10, 30), "x")
                document.new_page(width=100, height=100)
            document.save(root / "main.pdf")
        (root / "compiler-output.log").write_text("")
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(rendering, "_run_latex", compile_pdf)
    with pytest.raises(rendering.RenderError, match="one page|clipped"):
        rendering.render_diagram_preview(r"\draw (0,0)--(1,1);", tmp_path)
    assert not (tmp_path / "preview.png").exists()
