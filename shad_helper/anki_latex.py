"""Canonical Anki LaTeX and prebuilt media using the saved note types' settings."""

from __future__ import annotations

import hashlib
import html
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .anki_models import get_models
from .rendering import RenderError, _validate_commands

_DELIMITERS = re.compile(r"\[/?(?:latex|\$\$?)\]|\\[()[\]]", re.IGNORECASE)
_OPEN = {r"\(": (r"\)", "[$]", "[/$]"),
         r"\[": (r"\]", "[$$]", "[/$$]"),
         "[$]": ("[/$]", "[$]", "[/$]"),
         "[$$]": ("[/$$]", "[$$]", "[/$$]"),
         "[latex]": ("[/latex]", "[latex]", "[/latex]")}
_LATEX = re.compile(
    r"\[latex\](.+?)\[/latex\]|\[\$\](.+?)\[/\$\]|\[\$\$\](.+?)\[/\$\$\]",
    re.IGNORECASE | re.DOTALL,
)
_HTML = re.compile(r"<!--.*?-->|<style.*?>.*?</style>|<script.*?>.*?</script>|<.*?>",
                   re.IGNORECASE | re.DOTALL)
_ROW_INTERVAL = re.compile(r"(?<!\\)(\\\\\*?)(\s*)(\[([^\[\]\r\n]*,[^\[\]\r\n]*)\])")
_ROW_LENGTH = re.compile(
    r"\s*(?:[+-]\s*)*(?:\d+(?:[.,]\d*)?|[.,]\d+)\s*"
    r"(?:true\s*)?(?:pt|pc|in|bp|cm|mm|dd|cc|sp|ex|em)\s*", re.IGNORECASE,
)
# Harmless amsmath/amssymb/dsfont commands available in the existing preamble.
_EXTRA_COMMANDS = re.compile(
    r"\\(?:mathds|mathfrak|textnormal|pmb|xrightarrow|xleftarrow|lnot|not|"
    r"smallsetminus|nsubseteq|nsupseteq|subsetneqq|supsetneqq|because|therefore|"
    r"vdash|dashv|vDash|Vdash|models|nmid|nparallel|overleftrightarrow|"
    r"underleftarrow|underrightarrow|underleftrightarrow)(?![A-Za-z@])"
)


def _validate_body(body: str) -> None:
    # Extend the existing allowlist for these harmless constructs, without changing
    # the source that is hashed or compiled. File access and macro definitions stay blocked.
    checked = _EXTRA_COMMANDS.sub(r"\\mathrm", body)
    checked = re.sub(r"\\(begin|end)\s*\{(?:displaymath|math)\}",
                     r"\\\1{equation}", checked)
    try:
        _validate_commands(checked)
    except RenderError as exc:
        raise ValueError(str(exc)) from exc


def _brace_row_intervals(body: str) -> str:
    def replace(match):
        row, whitespace, interval, bounds = match.groups()
        # TeX accepts decimal commas in lengths, e.g. \\[1,2pt].
        if _ROW_LENGTH.fullmatch(bounds):
            return match.group()
        return row + whitespace + "{" + interval + "}"

    return _ROW_INTERVAL.sub(replace, body)


def normalize_latex(text: str) -> str:
    """Convert delimiters and brace row-leading intervals without changing their meaning."""
    parts = []
    previous = 0
    active = None
    inner = None
    body_start = 0
    for match in _DELIMITERS.finditer(text):
        token = match.group().lower()
        if token.startswith("\\"):
            preceding = len(text[:match.start()]) - len(text[:match.start()].rstrip("\\"))
            if preceding % 2:
                continue  # The second backslash of a matrix row break is not a delimiter.
        parts.append(text[previous:match.start()])
        previous = match.end()
        if active == "[latex]" and token.startswith("\\"):
            if token in _OPEN and inner is None:
                inner = token
            elif inner is not None and token == _OPEN[inner][0]:
                inner = None
            else:
                raise ValueError("Unbalanced or nested LaTeX delimiters")
            parts.append(match.group())
        elif active is None and token in _OPEN:
            active = token
            body_start = match.end()
            parts.append(_OPEN[token][1])
        elif active is not None and token == _OPEN[active][0] and inner is None:
            body = text[body_start:match.start()]
            if not body.strip():
                raise ValueError("LaTeX blocks cannot be empty")
            _validate_body(body)
            parts.append(_OPEN[active][2])
            active = None
        else:
            raise ValueError("Unbalanced or nested LaTeX delimiters")
    if active is not None:
        raise ValueError("Unbalanced LaTeX delimiters")
    parts.append(text[previous:])
    return _LATEX.sub(lambda match: _brace_row_intervals(match.group()), "".join(parts))


def _extract_latex(field: str) -> list[tuple[str, str]]:
    """Match Anki's body transformations; its media name hashes the body only."""
    extracted = []
    for match in _LATEX.finditer(field):
        full, inline, display = match.groups()
        if full is not None:
            body = full
        elif inline is not None:
            body = f"${inline}$"
        else:
            body = r"\begin{displaymath}" + display + r"\end{displaymath}"
        body = re.sub(r"<br( /)?>|<div>", "\n", body, flags=re.IGNORECASE)
        body = _HTML.sub("", body)
        if "&" in body:
            body = html.unescape(body).replace("\xa0", " ")
        filename = "latex-" + hashlib.sha1(body.encode("utf-8")).hexdigest() + ".png"
        extracted.append((filename, body))
    return extracted


def render_latex_media(notes: list[dict], media_dir: Path) -> list[Path]:
    """Pre-render unique field formulas to Anki's expected PNG media filenames."""
    models = get_models()
    formulas = {}
    for note in notes:
        model = models[note["model"]]
        for field in note["fields"]:
            for filename, body in _extract_latex(field):
                _validate_body(body)
                formulas.setdefault(filename, (body, model.latex_pre, model.latex_post))
    if not formulas:
        return []
    search_path = os.environ.get("PATH", "") + os.pathsep + "/Library/TeX/texbin"
    latex = shutil.which("latex", path=search_path)
    dvipng = shutil.which("dvipng", path=search_path)
    if not latex or not dvipng:
        raise RuntimeError("Anki LaTeX media requires latex and dvipng (TeX Live/MacTeX).")
    environment = {**os.environ, "PATH": search_path, "openin_any": "p", "openout_any": "p"}
    media_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, (body, preamble, footer) in formulas.items():
        with tempfile.TemporaryDirectory(prefix="shad-anki-latex-") as temporary:
            work = Path(temporary)
            (work / "tmp.tex").write_text(f"{preamble}\n{body}\n{footer}", encoding="utf-8")
            commands = [
                [latex, "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "tmp.tex"],
                [dvipng, "-bg", "Transparent", "-D", "200", "-T", "tight",
                 "tmp.dvi", "-o", "tmp.png"],
            ]
            for command in commands:
                try:
                    result = subprocess.run(command, cwd=work, env=environment, timeout=30,
                                            capture_output=True, text=True, check=False)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    raise RuntimeError(f"Anki LaTeX rendering failed for {filename}: {exc}") from exc
                if result.returncode:
                    detail = (result.stdout + result.stderr)[-2000:]
                    raise RuntimeError(f"Anki LaTeX rendering failed for {filename}:\n{detail}")
            png = work / "tmp.png"
            if not png.is_file() or not png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError(f"Anki LaTeX renderer produced no PNG for {filename}")
            path = media_dir / filename
            shutil.copyfile(png, path)
            paths.append(path)
    return paths
