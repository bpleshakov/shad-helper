import shutil
import subprocess
import unicodedata
from copy import deepcopy

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
    assert (r"\includegraphics[width={\ifdim 90bp<\linewidth 90bp\else\linewidth\fi},"
            r"height=.7\textheight,keepaspectratio]") in source
    assert r"\caption{График \& подпись}" in source
    assert r"\caption{График \& подпись}\label{fig:ch001_b1}" in source
    assert r"\begin{figure}[H]" in source
    assert r"\hypertarget{shad-block-ch001_b1}{}" in source
    assert r"\hypertarget{shad-end-ch001_b1}{}" in source
    assert manifest["source_blocks"][0]["content"] == "Стрелка A"
    assert manifest["source_blocks"][0]["illustration_id"] == "frame_1_ill01"


def test_large_screenshots_keep_linewidth_bound(illustrated_job):
    job, evidence, notes = illustrated_job
    image = job / "illustrations/frame_1_ill01.png"
    pixels = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 1600, 900), False)
    pixels.clear_with(255)
    pixels.save(image)
    digest = sha256_file(image)
    evidence["visuals"][0]["illustrations"][0]["image_sha256"] = digest
    write_json(job / "evidence.json", evidence)
    manifest = read_json(image.parent / "manifest.json")
    manifest["assets"]["frame_1_ill01"]["sha256"] = digest
    write_json(image.parent / "manifest.json", manifest)
    root = rendering.write_project(job, notes).parent
    rendering._validate_project(root, read_json(root / "project.json"))
    source = (root / "chapters/section-001.tex").read_text()
    assert r"width={\ifdim 1200bp<\linewidth 1200bp\else\linewidth\fi}" in source
    assert r"height=.7\textheight,keepaspectratio" in source


def test_direct_explanation_stays_with_figure_and_reserves_text_space(illustrated_job):
    job, _, notes = illustrated_job
    notes["chapters"][0]["blocks"].insert(0, {
        "id": "explanation", "kind": "text", "content": "На графике показана зависимость.",
        "figure_refs": ["ch001_b1"],
    })
    root = rendering.write_project(job, notes).parent
    rendering._validate_project(root, read_json(root / "project.json"))
    source = (root / "chapters/section-001.tex").read_text()
    assert r"\setbox\shadfiguregroup=\vbox\bgroup" in source
    assert source.index(r"\vbox\bgroup") < source.index("shad-block-explanation")
    assert source.index("shad-end-ch001_b1") < source.index(r"\egroup")
    assert r"\ht\shadfiguregroup+\dp\shadfiguregroup\relax>\textheight" in source
    assert r"\unvbox\shadfiguregroup" in source
    assert r"\box\shadfiguregroup" in source
    assert source.count(r"\label{fig:ch001_b1}") == 1
    assert r"height=.36\textheight,keepaspectratio" in source
    assert r"\ifdim 90bp<\linewidth 90bp\else\linewidth\fi" in source
    assert "minipage" not in source
    assert "nopagebreak" not in source


@pytest.mark.parametrize("kind", ["text", "definition", "theorem", "proof", "clarification"])
def test_measured_group_has_breakable_fallback_for_long_prose(illustrated_job, kind):
    job, _, notes = illustrated_job
    content = "Длинное объяснение учебного материала. " * 300
    notes["chapters"][0]["blocks"].insert(0, {
        "id": "explanation", "kind": kind, "content": content, "figure_refs": ["ch001_b1"],
    })
    root = rendering.write_project(job, notes).parent
    rendering._validate_project(root, read_json(root / "project.json"))
    source = (root / "chapters/section-001.tex").read_text()
    assert source.count(content) == 1
    assert r"\unvbox\shadfiguregroup" in source
    assert r"\box\shadfiguregroup" in source


def test_measured_group_does_not_capture_uncertainty_footnotes(illustrated_job):
    job, _, notes = illustrated_job
    notes["chapters"][0]["blocks"].insert(0, {
        "id": "explanation", "kind": "text", "content": "Сомнительная связь.",
        "evidence_ids": ["frame_1"], "figure_refs": ["ch001_b1"],
        "uncertain_fragments": [{"text": "Сомнительная", "evidence_ids": ["frame_1"]}],
    })
    root = rendering.write_project(job, notes).parent
    source = (root / "chapters/section-001.tex").read_text()
    assert r"\footnote{00:00:10}" in source
    assert r"\setbox" not in source


def test_reference_to_nonadjacent_figure_does_not_bind_unrelated_content(illustrated_job):
    job, _, notes = illustrated_job
    notes["chapters"][0]["blocks"][0:0] = [
        {"id": "explanation", "kind": "text", "content": "Ссылка на график.",
         "figure_refs": ["ch001_b1"]},
        {"id": "between", "kind": "text", "content": "Отдельное пояснение."},
    ]
    root = rendering.write_project(job, notes).parent
    source = (root / "chapters/section-001.tex").read_text()
    assert "nopagebreak" not in source
    assert r"height=.7\textheight,keepaspectratio" in source


def test_one_explanation_groups_first_figure_and_sizes_all_consecutive_references(illustrated_job):
    job, _, notes = illustrated_job
    blocks = notes["chapters"][0]["blocks"]
    blocks.append({**blocks[0], "id": "second_figure"})
    blocks.insert(0, {
        "id": "explanation", "kind": "text", "content": "Сравним два графика.",
        "figure_refs": ["ch001_b1", "second_figure"],
    })
    root = rendering.write_project(job, notes).parent
    rendering._validate_project(root, read_json(root / "project.json"))
    source = (root / "chapters/section-001.tex").read_text()
    assert source.count(r"\setbox\shadfiguregroup=\vbox\bgroup") == 1
    assert source.count(r"\egroup") == 1
    assert source.count(r"height=.36\textheight") == 2
    assert source.index("shad-end-ch001_b1") < source.index(r"\egroup")
    assert source.index(r"\egroup") < source.index("shad-block-second_figure")
    assert source.count(r"\label{fig:ch001_b1}") == 1
    assert source.count(r"\label{fig:second_figure}") == 1
    assert r"\unvbox\shadfiguregroup" in source


def test_unreferenced_figure_ends_the_directly_explained_group(illustrated_job):
    job, _, notes = illustrated_job
    blocks = notes["chapters"][0]["blocks"]
    blocks.extend([{**blocks[0], "id": "unreferenced"}, {**blocks[0], "id": "later"}])
    blocks.insert(0, {
        "id": "explanation", "kind": "text", "content": "Сравним графики.",
        "figure_refs": ["ch001_b1", "later"],
    })
    root = rendering.write_project(job, notes).parent
    source = (root / "chapters/section-001.tex").read_text()
    assert source.count(r"height=.36\textheight") == 1
    assert source.count(r"height=.7\textheight") == 2
    assert source.index(r"\egroup") < source.index("shad-block-unreferenced")


@pytest.fixture
def refined_crop(illustrated_job):
    from shad_helper.crop_validation import CROP_VERIFICATION_SCHEMA
    from shad_helper.illustration_crops import refine_crop, resolve_crop

    job, evidence, notes = illustrated_job
    evidence["media"]["sha256"] = "a" * 64
    write_json(job / "evidence.json", evidence)
    frames = job / "frames"
    frames.mkdir()
    shutil.copyfile(job / "illustrations/frame_1_ill01.png", frames / "frame_1_aaaaaaaaaaaa.png")

    def make_crop(box, name="crop"):
        class Client:
            def complete(self, instructions, text, schema, **kwargs):
                if schema == CROP_VERIFICATION_SCHEMA:
                    return {"verdict": "pass", "checks": {
                        check: "pass" for check in
                        ("identity", "completeness", "readability", "cleanliness")},
                        "defects": [], "repair": "none", "reason": "Полный выбранный рисунок."}
                return {"status": "cropped", "box": box,
                        "reason": "Keep only the selected drawing region."}

        crop_id, retry_needed, reason, unusable = refine_crop(
            job, evidence, {**notes["chapters"][0]["blocks"][0], "content": name},
            Client(), job / name)
        assert crop_id and not retry_needed and not reason and not unusable
        return crop_id, resolve_crop(job, evidence, "frame_1_ill01", crop_id)

    return make_crop


def test_refined_screenshot_copies_selected_pixels_and_keeps_crop_provenance(
        illustrated_job, refined_crop):
    job, _, notes = illustrated_job
    crop_id, selected = refined_crop([0, 0, 60, 70])
    selected_pixels = selected.read_bytes()
    notes["chapters"][0]["blocks"][0]["illustration_crop_id"] = crop_id
    root = rendering.write_project(job, notes).parent
    manifest = read_json(root / "project.json")
    assert manifest["image_files"] == ["figures/illustration-ch001_b1.png"]
    copied = root / manifest["image_files"][0]
    assert copied.read_bytes() == selected_pixels
    assert copied.read_bytes() != (job / "illustrations/frame_1_ill01.png").read_bytes()
    assert manifest["source_blocks"][0]["illustration_crop_id"] == crop_id
    shutil.rmtree(job / "illustrations")
    shutil.rmtree(job / "illustration-crops")
    shutil.rmtree(job / "frames")
    rendering._validate_project(root, manifest)
    assert copied.read_bytes() == selected_pixels


def test_two_refined_crops_of_one_illustration_do_not_overwrite_each_other(
        illustrated_job, refined_crop):
    job, _, notes = illustrated_job
    first_id, first = refined_crop([0, 0, 60, 70], "left-crop")
    second_id, second = refined_crop([60, 0, 120, 70], "right-crop")
    first_block = notes["chapters"][0]["blocks"][0]
    first_block["illustration_crop_id"] = first_id
    notes["chapters"][0]["blocks"].append({
        **first_block, "id": "ch001_b2", "illustration_crop_id": second_id})
    root = rendering.write_project(job, notes).parent
    manifest = read_json(root / "project.json")
    assert manifest["image_files"] == ["figures/illustration-ch001_b1.png",
                                       "figures/illustration-ch001_b2.png"]
    assert (root / manifest["image_files"][0]).read_bytes() == first.read_bytes()
    assert (root / manifest["image_files"][1]).read_bytes() == second.read_bytes()
    assert first.read_bytes() != second.read_bytes()
    rendering._validate_project(root, manifest)


@pytest.mark.parametrize("crop_id", ["missing", "a" * 64, "../crop", r"\input{secret}"])
def test_screenshot_rejects_invalid_derived_crop_reference(illustrated_job, crop_id):
    job, _, notes = illustrated_job
    notes["chapters"][0]["blocks"][0]["illustration_crop_id"] = crop_id
    with pytest.raises(rendering.RenderError):
        rendering.write_project(job, notes)


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
def test_unresolved_figure_stays_in_diagnostics_without_recovery_text_or_bad_pixels(
        illustrated_job, monkeypatch):
    from shad_helper import illustration_verification
    from shad_helper import notes as notes_module

    job, evidence, notes = illustrated_job
    chapter = notes["chapters"][0]
    chapter["blocks"][0]["evidence_ids"].append("frame_1_ill01")
    chapter["illustration_coverage"] = [
        {"illustration_id": "frame_1_ill01", "block_id": "ch001_b1"}]
    chapter["blocks"].insert(0, {"id": "explanation", "kind": "text", "content": "Связь на графике.",
                                 "evidence_ids": ["frame_1"], "figure_refs": ["ch001_b1"]})
    original = (job / "illustrations/frame_1_ill01.png").read_bytes()
    monkeypatch.setattr(illustration_verification, "refine_crop",
                        lambda *args: ("", False, "Подпись закрыта рукой.", False))
    accepted, retry = illustration_verification.finalize_illustrations(
        notes, evidence, job, job / "verification", None)
    accepted = notes_module.validate_notes(accepted, evidence, finalized=True)
    assert not retry and accepted["chapters"][0]["blocks"][0]["figure_refs"] == []
    assert accepted["chapters"][0]["illustration_coverage"] == [
        {"illustration_id": "frame_1_ill01", "block_id": "", "exclusion_reason": "unresolved"}]
    assert any(issue["status"] == "open" and "frame_1_ill01" in issue["evidence_ids"]
               for issue in accepted["issues"])
    rendering.write_project(job, accepted)
    pdf = rendering.render(job)
    with pymupdf.open(pdf) as document:
        text = unicodedata.normalize("NFKC", "\n".join(page.get_text() for page in document))
        assert "требует восстановления" not in text
        assert "Связь на графике." in text
        assert not any(page.get_images() for page in document)
    assert (job / "illustrations/frame_1_ill01.png").read_bytes() == original
    assert read_json(job / "notes/project.json")["image_files"] == []


@needs_tex
@pytest.mark.parametrize("refined", [False, True])
def test_cropped_screenshot_pdf_is_portable_and_keeps_manual_edits(
        illustrated_job, refined_crop, refined):
    job, _, notes = illustrated_job
    if refined:
        crop_id, _ = refined_crop([0, 0, 60, 70])
        notes["chapters"][0]["blocks"][0]["illustration_crop_id"] = crop_id
    root = rendering.write_project(job, notes).parent
    source = root / "chapters/section-001.tex"
    manual = source.read_text() + "\nАвторская правка.\n"
    source.write_text(manual)
    shutil.rmtree(job / "illustrations")
    shutil.rmtree(job / "frames")
    if refined:
        shutil.rmtree(job / "illustration-crops")
    pdf = rendering.render(job)
    assert source.read_text() == manual
    with pymupdf.open(pdf) as document:
        assert any(page.get_images() for page in document)
        assert "Авторская правка" in "\n".join(page.get_text() for page in document)
    mapping = read_json(root / "source-map.json")["blocks"]["ch001_b1"]
    assert mapping["illustration_id"] == "frame_1_ill01"
    assert mapping["content"] == "Стрелка A"
    assert mapping["page_start"] == mapping["page_end"] == 1
    if refined:
        assert mapping["illustration_crop_id"] == crop_id


@needs_tex
def test_chapter_validation_copies_assets_from_original_job(illustrated_job, chapter_latex_checker):
    job, evidence, notes = illustrated_job
    folder = job / "isolated-validation"
    chapter_latex_checker(notes, evidence, folder, asset_job=job)
    assert (folder / "notes/figures/illustration-frame_1_ill01.png").is_file()


@needs_tex
def test_actual_oversized_prose_figure_group_unboxes_without_duplicate_numbering(illustrated_job):
    job, _, notes = illustrated_job
    explanation = ("GROUPBEGIN "
                   + "Объяснение зависимости показывает характер изменения величин. " * 160
                   + " GROUPEND")
    notes["chapters"][0]["blocks"].insert(0, {
        "id": "explanation", "kind": "text", "content": explanation,
        "evidence_ids": ["frame_1"], "figure_refs": ["ch001_b1"],
    })
    root = rendering.write_project(job, notes).parent
    pdf = rendering.render(job)
    report = read_json(root / "render-report.json")
    assert report["overfull"] == []
    assert report["missing_glyphs"] == []
    assert report["missing_fonts"] == []
    with pymupdf.open(pdf) as document:
        assert len(document) > 1
        text = unicodedata.normalize("NFKC", "\n".join(page.get_text() for page in document))
        assert text.count("GROUPBEGIN") == text.count("GROUPEND") == 1
        assert text.count("Рис. 1:") == 1
        assert "Рис. 2:" not in text
        assert sum(len(page.get_image_info()) for page in document) == 1
    assert (root / "main.aux").read_text().count(r"\newlabel{fig:ch001_b1}") == 1
    mapping = read_json(root / "source-map.json")["blocks"]
    assert mapping["explanation"]["page_start"] < mapping["explanation"]["page_end"]
    assert mapping["ch001_b1"]["page_start"] == mapping["ch001_b1"]["page_end"]


def test_chapter_validation_omits_only_external_figure_refs_without_mutation(
    illustrated_job, chapter_latex_checker, monkeypatch,
):
    job, evidence, notes = illustrated_job
    notes["chapters"][0]["blocks"].insert(0, {
        "id": "explanation", "kind": "text", "content": "Сравнение с предыдущей схемой.",
        "evidence_ids": ["frame_1"], "figure_refs": ["prior_figure", "ch001_b1"],
    })
    original = deepcopy(notes)

    def compiler(root):
        (root / "main.pdf").write_bytes(b"compiler placeholder")
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(rendering, "_run_latex", compiler)
    folder = job / "isolated-validation"
    chapter_latex_checker(notes, evidence, folder, asset_job=job)
    source = (folder / "notes/chapters/section-001.tex").read_text()
    assert r"\ref{fig:ch001_b1}" in source
    assert "prior_figure" not in source
    assert notes == original
    with pytest.raises(rendering.RenderError, match="unique figure block IDs"):
        rendering.write_project(job, notes)
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
