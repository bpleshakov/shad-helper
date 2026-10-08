from __future__ import annotations

import html
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from shad_helper import anki_latex
from shad_helper.anki_latex import _extract_latex, normalize_latex, render_latex_media
from shad_helper.anki_models import get_models


def test_normalize_preserves_matrix_rows_and_latex_source():
    source = (r"Матрица \(A\in\mathbb{R}^{2\times 2}\). "
              r"\[A=\begin{pmatrix}1&2\\[2pt]3&4\end{pmatrix}\]")
    expected = (r"Матрица [$]A\in\mathbb{R}^{2\times 2}[/$]. "
                r"[$$]A=\begin{pmatrix}1&2\\[2pt]3&4\end{pmatrix}[/$$]")
    assert normalize_latex(source) == expected
    assert normalize_latex(expected) == expected


def test_normalize_full_latex_with_internal_math_delimiters():
    text = r"[LATEX]Пусть \(x\in\mathds{R}\). Тогда \[x^2\geq 0\].[/LATEX]"
    assert normalize_latex(text) == text.replace("LATEX", "latex")


@pytest.mark.parametrize("source", [
    r"\(x", r"x\)", r"\[x\)", r"\(a\(b\)\)", r"\)x\(",
    "[$]x[/$$]", "[/latex]x[latex]", "[$]a[$$]b[/$$][/$]", "[$$] [/$$]",
    r"[latex]\(x[/latex]", r"[latex]\[x\)[/latex]",
])
def test_normalize_rejects_broken_or_nested_delimiters(source):
    with pytest.raises(ValueError, match="LaTeX"):
        normalize_latex(source)


@pytest.mark.parametrize("source", [
    r"[latex]\input{/etc/passwd}[/latex]",
    r"[$]\write18{touch bad}[/$]",
    r"[$]\csname input\endcsname[/$]",
    r"[$]\catcode`x=0[/$]",
    r"[$]\def\a{b}[/$]",
    r"[$]\special{psfile=/etc/passwd}[/$]",
    r"[$]\begin{document}x\end{document}[/$]",
    r"[$]^^5cinput[/$]",
])
def test_normalize_rejects_file_access_and_tex_metaprogramming(source):
    with pytest.raises(ValueError):
        normalize_latex(source)


def test_extracted_bodies_and_filenames_match_native_anki(tmp_path):
    collection_module = pytest.importorskip("anki.collection")
    collection = collection_module.Collection(str(tmp_path / "collection.anki2"))
    fields = [
        "[latex]one<br>and<div>two[/latex]",
        "[$]<b>hello</b>&nbsp; world[/$]",
        "[$$]math &amp; stuff[/$$]",
        "[latex]<!-- hidden --><style>x</style><script>y</script>shown[/latex]",
        "[$$]" + html.escape(r"A=\begin{pmatrix}1&2\\3&4\end{pmatrix}") + "[/$$]",
        "[$$]" + html.escape(r"\text{Если }x<y,\quad a>b") + "<br>\nz=1[/$$]",
        r"[latex]Теорема: \(x=1\).[/latex]",
    ]
    try:
        for field in fields:
            result = collection._backend.extract_latex(text=field, svg=False, expand_clozes=False)
            assert _extract_latex(field) == [(item.filename, item.latex_body)
                                             for item in result.latex]
    finally:
        collection.close()


def test_renderer_visits_all_fields_deduplicates_and_uses_original_preamble(tmp_path, monkeypatch):
    models = get_models()
    notes = []
    for number, (name, model) in enumerate(models.items()):
        fields = [""] * len(model.fields)
        fields[0] = "[$]x[/$]"
        fields[-1] = f"[$$]y={number}[/$$]"
        notes.append({"model": name, "fields": fields})
    calls = []

    def run(command, **kwargs):
        work = kwargs["cwd"]
        calls.append((command, kwargs, (work / "tmp.tex").read_text()))
        if Path(command[0]).name == "dvipng":
            (work / "tmp.png").write_bytes(b"\x89PNG\r\n\x1a\ncontent")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(anki_latex.shutil, "which", lambda name, **kwargs: f"/tex/bin/{name}")
    monkeypatch.setattr(anki_latex.subprocess, "run", run)
    paths = render_latex_media(notes, tmp_path / "media")
    assert len(paths) == 5
    assert len(calls) == 10
    assert len({path.name for path in paths}) == 5
    assert all(path.is_file() for path in paths)
    model = models["Basic"]
    assert calls[0][2] == f"{model.latex_pre}\n$x$\n{model.latex_post}"
    for command, kwargs, _ in calls:
        assert kwargs["timeout"] == 30
        assert kwargs["env"]["openin_any"] == kwargs["env"]["openout_any"] == "p"
        assert kwargs["cwd"] != tmp_path / "media"
        if Path(command[0]).name == "latex":
            assert "-no-shell-escape" in command
            assert "-halt-on-error" in command


def test_renderer_validates_all_formulas_before_any_process(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Unsafe TeX must be rejected before any compilation")

    monkeypatch.setattr(anki_latex.subprocess, "run", unexpected)
    notes = [{"model": "Basic", "fields": ["[$]x[/$]", r"[latex]\input{/etc/passwd}[/latex]"]}]
    with pytest.raises(ValueError, match="input"):
        render_latex_media(notes, tmp_path)


def test_plain_text_does_not_require_tex_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(anki_latex.shutil, "which", lambda *args, **kwargs: None)
    assert render_latex_media([{"model": "Basic", "fields": ["Термин", "Определение"]}],
                              tmp_path) == []
    with pytest.raises(RuntimeError, match="latex and dvipng"):
        render_latex_media([{"model": "Basic", "fields": ["[$]x[/$]", ""]}], tmp_path)


@pytest.mark.parametrize("failure", ["timeout", "compiler", "missing_png"])
def test_renderer_reports_failures_without_publishing_broken_media(tmp_path, monkeypatch, failure):
    def run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=int(failure == "compiler"), stdout="TeX error", stderr="")

    monkeypatch.setattr(anki_latex.shutil, "which", lambda name, **kwargs: f"/tex/bin/{name}")
    monkeypatch.setattr(anki_latex.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Anki LaTeX render"):
        render_latex_media([{"model": "Basic", "fields": ["[$]x[/$]", ""]}], tmp_path)
    assert not list(tmp_path.glob("*.png"))


@pytest.mark.skipif(
    not all(shutil.which(name, path=os.environ.get("PATH", "") + ":/Library/TeX/texbin")
            for name in ["latex", "dvipng"]),
    reason="TeX Live is needed for the real Anki media compilation check",
)
def test_real_russian_latex_and_matrix_render_with_snapshot_preamble(tmp_path):
    fields = [
        r"[$]x\in\mathds{R},\quad x<2[/$]",
        r"[$$]\text{Матрица } A=\begin{pmatrix}1&2\\[2pt]3&4\end{pmatrix}[/$$]",
        r"[latex]Если \(x\in\mathbb{R}\), то \[x^2\geq 0.\][/latex]",
    ]
    escaped = [html.escape(field, quote=False) for field in fields]
    paths = render_latex_media([{"model": "Theorem Full proof", "fields": escaped}], tmp_path)
    assert len(paths) == len(fields)
    for path, field in zip(paths, escaped, strict=True):
        assert path.name == _extract_latex(field)[0][0]
        picture = pymupdf.Pixmap(str(path))
        assert picture.width > 20 and picture.height > 15
        assert picture.alpha
        assert any(picture.samples[3::4])

    collection_module = pytest.importorskip("anki.collection")
    from anki.config import Config
    from anki.latex import render_latex_returning_errors

    collection = collection_module.Collection(str(tmp_path / "collection.anki2"))
    try:
        collection.set_config_bool(Config.Bool.RENDER_LATEX, False)
        for path in paths:
            collection.media.write_data(path.name, path.read_bytes())
        model = get_models()["Theorem Full proof"].to_json(0, 1)
        for field, path in zip(escaped, paths, strict=True):
            rendered, errors = render_latex_returning_errors(field, model, collection)
            assert errors == []
            assert path.name in rendered
            assert "[latex]" not in rendered and "[$" not in rendered
    finally:
        collection.close()
