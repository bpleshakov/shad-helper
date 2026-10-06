"""Editable Russian LaTeX notes with confined assets and PDF quality reports."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

from .jobs import fingerprint, read_json


class RenderError(RuntimeError):
    """A note project cannot safely be rendered, or its compiler failed."""


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
_COMMAND = re.compile(r"\\([A-Za-z@]+|.)", re.DOTALL)
_MAX_TEX_BYTES = 2_000_000
_MATH_COMMANDS = set(
    "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa "
    "lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi "
    "psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega "
    "frac dfrac tfrac cfrac binom dbinom sqrt sum prod coprod int iint iiint oint "
    "lim limsup liminf inf sup min max log ln exp sin cos tan cot sec csc sinh cosh "
    "tanh arcsin arccos arctan det gcd Pr arg ker dim mod bmod pmod "
    "left right middle big Big bigg Bigg bigl bigr Bigl Bigr biggl biggr Biggl Biggr "
    "langle rangle lvert rvert lVert rVert vert Vert lceil rceil lfloor rfloor "
    "infty partial nabla ell hbar imath jmath Re Im emptyset varnothing "
    "in notin ni subset supset subseteq supseteq subsetneq supsetneq "
    "cup cap setminus union times cdot div pm mp ast star circ bullet oplus otimes "
    "le leq ge geq neq ne approx sim simeq equiv cong propto ll gg "
    "to mapsto rightarrow leftarrow leftrightarrow Rightarrow Leftarrow "
    "Leftrightarrow longrightarrow longleftarrow Longrightarrow Longleftrightarrow "
    "uparrow downarrow updownarrow iff implies forall exists nexists neg land lor "
    "wedge vee top bot perp parallel angle triangle square checkmark "
    "ldots cdots vdots ddots dots dot ddot hat widehat bar overline underline "
    "vec overrightarrow overbrace underbrace overset underset stackrel "
    "mathrm mathbf mathit mathsf mathtt mathcal mathbb mathscr boldsymbol "
    "operatorname text textrm textsf texttt textbf textit mathop limits nolimits "
    "quad qquad hspace vspace phantom hphantom vphantom substack label ref eqref "
    "nonumber notag tag underbracket overbracket boxed displaystyle textstyle "
    "scriptstyle scriptscriptstyle mathrm mathnormal cases begin end".split()
)
_DRAW_COMMANDS = set(
    "draw node path coordinate fill filldraw clip shade shadedraw foreach "
    "tikz tikzset scope addplot addlegendentry legend nextgroupplot "
    "pgfplotsset pgfmathsetmacro pgfmathparse pgfmathresult pgfmathprintnumber "
    "pgfmathsetlengthmacro".split()
)
_PROJECT_COMMANDS = set(
    "documentclass usepackage setmainfont setmonofont setmathfont "
    "setdefaultlanguage setotherlanguage usetikzlibrary usepgfplotslibrary "
    "title author date maketitle tableofcontents newpage clearpage section "
    "subsection subsubsection paragraph par noindent centering caption "
    "footnote footnotesize scriptsize small normalsize large Large "
    "textbackslash textasciitilde textasciicircum textcolor includegraphics "
    "input IfFileExists VerbatimInput pagestyle fancyhf fancyfoot renewcommand "
    "headrulewidth thepage linewidth textwidth medskip bigskip item "
    "hypersetup emergencystretch newif ifnotesdraft notesdrafttrue fi".split()
)
_SYMBOL_COMMANDS = set("\\{}%&#$_~^ ,;:![]()") | {"\n", "\r"}
_MATH_ENVIRONMENTS = set(
    "equation equation* "
    "align align* aligned alignedat gather gather* gathered multline multline* "
    "split cases matrix pmatrix bmatrix Bmatrix vmatrix Vmatrix smallmatrix "
    "array".split()
)
_DRAW_ENVIRONMENTS = {"tikzpicture", "scope", "axis", "groupplot"}
_ENVIRONMENTS = _MATH_ENVIRONMENTS | _DRAW_ENVIRONMENTS | {"document", "figure", "quote", "center", "itemize", "enumerate", "description"}
_PACKAGES = set(
    "geometry fontspec polyglossia unicode-math amsmath amssymb graphicx tikz "
    "pgfplots fvextra fancyhdr xcolor hyperref microtype".split()
)
_ESCAPE = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$",
    "&": r"\&", "#": r"\#", "%": r"\%", "_": r"\_",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}
_SOURCE_DIRS = ("evidence", "sources", "frames", "crops", "source_assets", "source_originals", "raw", "audio", "workers", "checkpoints")
_DRAFT_START = "% shad-helper: draft footer begin"
_DRAFT_END = "% shad-helper: draft footer end"


def escape_text(value: str) -> str:
    return "".join(_ESCAPE.get(character, character) for character in value)


def _prose(value: str) -> str:
    """Escape prose while retaining the only supported inline math delimiter."""
    parts = []
    previous = 0
    for match in re.finditer(r"\\\((.*?)\\\)", value, re.DOTALL):
        plain = value[previous:match.start()]
        if r"\(" in plain or r"\)" in plain:
            raise RenderError("Unbalanced inline math delimiters")
        _validate_commands(match.group(1))
        parts.extend((escape_text(plain), match.group(0)))
        previous = match.end()
    remainder = value[previous:]
    if r"\(" in remainder or r"\)" in remainder:
        raise RenderError("Unbalanced inline math delimiters")
    parts.append(escape_text(remainder))
    return "".join(parts)


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise RenderError(f"Invalid project identifier: {value!r}")
    return value


def _confined(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise RenderError(f"Unconfined asset reference: {relative}")
    current = root
    for component in Path(relative).parts:
        current = current / component
        if current.is_symlink():
            raise RenderError(f"Project assets cannot be symbolic links: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise RenderError(f"Asset escapes the notes project: {relative}")
    return path


def _uncomment(source: str) -> str:
    lines = []
    for line in source.splitlines():
        for index, character in enumerate(line):
            if character == "%":
                slashes = len(line[:index]) - len(line[:index].rstrip("\\"))
                if slashes % 2 == 0:
                    line = line[:index]
                    break
        lines.append(line)
    return "\n".join(lines)


def _validate_commands(source: str, *, diagram: bool = False, project: bool = False) -> None:
    if len(source.encode("utf-8")) > _MAX_TEX_BYTES or "^^" in source:
        raise RenderError("TeX source is too large or contains character-code escapes")
    source = _uncomment(source)
    allowed = _MATH_COMMANDS | _SYMBOL_COMMANDS
    if diagram or project:
        allowed |= _DRAW_COMMANDS
    if project:
        allowed |= _PROJECT_COMMANDS
    commands = _COMMAND.findall(source)
    if len(commands) > 40_000:
        raise RenderError("Too many TeX commands")
    unknown = sorted(set(commands) - allowed)
    if unknown:
        raise RenderError(f"Unsupported or unsafe TeX commands: {', '.join(unknown)}")
    environments = re.findall(r"\\(?:begin|end)\s*\{([^{}]*)\}", source)
    if len(environments) != len(re.findall(r"\\(?:begin|end)\b", source)):
        raise RenderError("Use literal braced environment names")
    allowed_environments = _ENVIRONMENTS if project else _MATH_ENVIRONMENTS | (_DRAW_ENVIRONMENTS if diagram else set())
    for environment in environments:
        if environment not in allowed_environments:
            raise RenderError(f"Unsupported TeX environment: {environment}")
    if diagram and re.search(r"\b(?:file|table|graphics|shell|external)\b", source, re.I):
        raise RenderError("Diagrams must use inline paths, expressions, or coordinates")


def _write(root: Path, relative: str, content: str) -> None:
    path = _confined(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _stamp(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02}:{total // 60 % 60:02}:{total % 60:02}"


def _evidence(ids: list[str], index: dict) -> str:
    if not ids:
        return ""
    descriptions = []
    for evidence_id in ids:
        entry = index.get(evidence_id, evidence_id)
        if isinstance(entry, dict):
            entry = entry.get("description") or entry.get("text") or evidence_id
        descriptions.append(escape_text(str(entry)))
    return "\\footnote{Источники: " + "; ".join(descriptions) + "}"


def _illustration_tex(identifier: str) -> str:
    fallback = r"\begin{quote}\textit{Иллюстрация " + escape_text(identifier) + r": изображение ещё не добавлено.}\end{quote}"
    for extension in reversed(("png", "jpg", "pdf")):
        reference = f"figures/illustrations/{identifier}.{extension}"
        fallback = "\\IfFileExists{" + reference + "}{\\includegraphics[width=0.9\\linewidth]{" + reference + "}}{" + fallback + "}"
    return fallback + "\n"


def _draft_footer(identifiers: list[str], review: list[dict]) -> str:
    marker = r"\scriptsize ЧЕРНОВИК - требуется проверка"
    if review:
        return r"\fancyfoot[L]{" + marker + "}\n"
    if not identifiers:
        return ""
    lines = [r"\newif\ifnotesdraft"]
    for identifier in identifiers:
        condition = r"\notesdrafttrue"
        for extension in reversed(("png", "jpg", "pdf")):
            condition = "\\IfFileExists{figures/illustrations/" + identifier + "." + extension + "}{}{ " + condition + "}"
        lines.append(condition)
    lines.append(r"\fancyfoot[L]{\ifnotesdraft " + marker + r"\fi}")
    return "\n".join(lines) + "\n"


def _marked_footer(identifiers: list[str], review: list[dict]) -> str:
    return _DRAFT_START + "\n" + _draft_footer(identifiers, review) + _DRAFT_END + "\n"


def _review_flags(notes: dict) -> list[dict]:
    flags = [{"id": block["id"], "description": block["content"]}
             for chapter in notes.get("chapters", []) for block in chapter.get("blocks", [])
             if block["kind"] == "uncertain"]
    flags.extend({"id": issue["id"], "description": issue["description"]}
                 for issue in notes.get("issues", []) if issue["status"] != "resolved")
    return flags


def _review_section(notes: dict) -> str:
    lines = []
    if notes.get("issues"):
        lines.append(r"\section{Вопросы и замечания}")
        for issue in notes["issues"]:
            status = "Решено" if issue["status"] == "resolved" else "Требует проверки"
            lines.append("\\paragraph{" + escape_text(issue["id"]) + ": " + status + "} " + _prose(issue["description"]) + _evidence(issue.get("evidence_ids", []), notes.get("evidence_index", {})))
            if issue.get("resolution"):
                lines.append("\\noindent\\textbf{Уточнение:} " + _prose(issue["resolution"]))
    return "\n\n".join(lines) + "\n"


def _evidence_fingerprint(job: Path) -> str | None:
    path = job / "evidence.json"
    return fingerprint(read_json(path)) if path.exists() else None


_PREAMBLE = r"""\documentclass[a4paper,11pt]{article}
\usepackage[margin=23mm]{geometry}
\usepackage{fontspec}
\setmainfont{CMU Serif}
\setmonofont{CMU Typewriter Text}
\usepackage{polyglossia}
\setdefaultlanguage{russian}
\setotherlanguage{english}
\usepackage{amsmath}
\usepackage{unicode-math}
\setmathfont{Latin Modern Math}
\usepackage{graphicx,tikz,pgfplots,fvextra,fancyhdr,xcolor,hyperref,microtype}
\usetikzlibrary{arrows.meta,positioning,calc,shapes.geometric,fit,matrix}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\hypersetup{hidelinks}
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[R]{\thepage}
\renewcommand{\headrulewidth}{0pt}
\emergencystretch=2em
"""


def write_project(job: Path, notes: dict) -> Path:
    """Create editable sources; evidence images are never note illustrations.

    Call this only when replacing the note project is intentional. ``render``
    preserves these sources, including subsequent manual corrections.
    """
    root = job.resolve() / "notes"
    if root.is_symlink() or not root.resolve().is_relative_to(job.resolve()):
        raise RenderError("The notes directory escapes the job")
    index = notes.get("evidence_index", {})
    tex_files = ["main.tex", "review.tex"]
    code_files = []
    illustrations = notes.get("illustrations", [])
    declared_ids = set()
    for illustration in illustrations:
        identifier = _identifier(illustration["id"])
        if identifier in declared_ids or illustration.get("path"):
            raise RenderError("Illustrations require unique IDs and must be imported with images_dir")
        declared_ids.add(identifier)
        _write(root, f"figures/illustration-{identifier}.tex", _illustration_tex(identifier))
        tex_files.append(f"figures/illustration-{identifier}.tex")
    chapter_sources = []
    block_ids = set()
    chapter_ids = set()
    review = _review_flags(notes)
    for chapter in notes.get("chapters", []):
        chapter_id = _identifier(chapter["id"])
        if chapter_id in chapter_ids:
            raise RenderError("Chapter IDs must be unique")
        chapter_ids.add(chapter_id)
        lines = ["\\section{" + escape_text(chapter["title"]) + "}",
                 "\\noindent\\textit{Видео: " + _stamp(chapter["start"]) + " - " + _stamp(chapter["end"]) + "}\\par\n"]
        for block in chapter.get("blocks", []):
            block_id = _identifier(block["id"])
            if block_id in block_ids:
                raise RenderError("Block IDs must be unique")
            block_ids.add(block_id)
            content = block["content"]
            source = _evidence(block.get("evidence_ids", []), index)
            kind = block["kind"]
            if kind == "math":
                _validate_commands(content)
                display_environment = re.match(r"\s*\\begin\s*\{(?:equation|align|gather|multline)\*?\}", content)
                if content.lstrip().startswith(r"\[") or display_environment:
                    lines.append(content + source + "\n")
                else:
                    lines.append("\\[\n" + content + "\n\\]" + source + "\n")
            elif kind == "diagram":
                _validate_commands(content, diagram=True)
                if not content.lstrip().startswith((r"\begin{tikzpicture}", r"\tikz")):
                    content = "\\begin{tikzpicture}\n" + content + "\n\\end{tikzpicture}"
                relative = f"figures/diagram-{block_id}.tex"
                _write(root, relative, content + "\n")
                tex_files.append(relative)
                lines.append("\\begin{figure}[htbp]\n\\centering\n\\input{" + relative + "}\n\\caption{" + escape_text(block.get("caption", "Схема")) + "}\n\\end{figure}\n" + source)
            elif kind == "code":
                relative = f"code/{block_id}.txt"
                _write(root, relative, content)
                code_files.append(relative)
                lines.append("\\VerbatimInput[breaklines,breakanywhere,fontsize=\\small,frame=single]{" + relative + "}" + source + "\n")
            elif kind in {"clarification", "uncertain"}:
                label = "Пояснение" if kind == "clarification" else "Требует проверки"
                lines.append("\\begin{quote}\\textbf{" + label + ".} " + _prose(content) + source + "\\end{quote}\n")
            elif kind == "text":
                lines.append(_prose(content) + source + "\n")
            else:
                raise RenderError(f"Unsupported note block kind: {kind}")
        for illustration in illustrations:
            if illustration["chapter_id"] == chapter_id:
                relative = f"figures/illustration-{illustration['id']}.tex"
                lines.append("\\begin{figure}[htbp]\n\\centering\n\\input{" + relative + "}\n\\caption{" + escape_text(illustration.get("caption", "Иллюстрация")) + "}\n\\end{figure}\n")
        relative = f"chapters/{chapter_id}.tex"
        _write(root, relative, "\n\n".join(lines) + "\n")
        tex_files.append(relative)
        chapter_sources.append("\\input{" + relative + "}")
    if any(item["chapter_id"] not in chapter_ids for item in illustrations):
        raise RenderError("An illustration refers to a nonexistent chapter")
    appendix = []
    if notes.get("glossary"):
        appendix.append(r"\section{Глоссарий}")
        for entry in notes["glossary"]:
            appendix.append("\\paragraph{" + _prose(entry["term"]) + "} " + _prose(entry["definition"]))
    _write(root, "review.tex", _review_section(notes))
    main = _PREAMBLE + _marked_footer(sorted(declared_ids), review) + "\\title{" + escape_text(notes["title"]) + "}\n\\author{}\n\\date{}\n\\begin{document}\n\\maketitle\n\\tableofcontents\n\\newpage\n" + "\n".join(chapter_sources + appendix + [r"\input{review.tex}"]) + "\n\\end{document}\n"
    _write(root, "main.tex", main)
    _write(root, "illustrations.json", json.dumps(illustrations, ensure_ascii=False, indent=2) + "\n")
    manifest = {"version": 1, "draft": bool(review or declared_ids), "tex_files": tex_files,
                "code_files": code_files, "illustration_ids": sorted(declared_ids), "review": review,
                "issues": notes.get("issues", []), "evidence_fingerprint": _evidence_fingerprint(job)}
    _write(root, "project.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return root / "main.tex"


def refresh_review(job: Path, notes: dict) -> Path:
    """Refresh explicit review metadata while retaining edited note content.

    The caller verifies the project was current before resolving evidence issues.
    This operation then binds it to the post-resolution evidence fingerprint.
    It updates only review.tex, project.json, and the marked footer in main.tex.
    """
    root = job.resolve() / "notes"
    if root.is_symlink() or not root.resolve().is_relative_to(job.resolve()):
        raise RenderError("The notes directory escapes the job")
    manifest = read_json(_confined(root, "project.json"))
    main_path = _confined(root, "main.tex")
    main = main_path.read_text(encoding="utf-8")
    if ("review.tex" not in manifest["tex_files"] or main.count(_DRAFT_START) != 1
            or main.count(_DRAFT_END) != 1):
        raise RenderError("Review refresh requires the generated appendix/footer markers; regenerate or import notes before refreshing review")
    pattern = re.escape(_DRAFT_START) + r".*?" + re.escape(_DRAFT_END) + r"\n?"
    if not re.search(pattern, main, flags=re.DOTALL):
        raise RenderError("Review footer markers are out of order; regenerate or import notes")
    review = _review_flags(notes)
    appendix = _review_section(notes)
    _validate_commands(appendix, project=True)
    identifiers = manifest["illustration_ids"]
    footer = _marked_footer(identifiers, review)
    main = re.sub(pattern, lambda _: footer, main, count=1, flags=re.DOTALL)
    missing = any(not any(_confined(root, f"figures/illustrations/{identifier}.{extension}").is_file()
                          for extension in ("png", "jpg", "pdf")) for identifier in identifiers)
    manifest.update(review=review, issues=notes.get("issues", []), draft=bool(review or missing),
                    evidence_fingerprint=_evidence_fingerprint(job))
    _write(root, "review.tex", appendix)
    _write(root, "main.tex", main)
    _write(root, "project.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return main_path


def _validate_project(root: Path, manifest: dict) -> None:
    tex_files = set(manifest["tex_files"])
    code_files = set(manifest["code_files"])
    image_files = {f"figures/illustrations/{_identifier(identifier)}.{extension}"
                   for identifier in manifest["illustration_ids"] for extension in ("png", "jpg", "pdf")}
    reference_commands = {"input": tex_files, "VerbatimInput": code_files,
                          "includegraphics": image_files, "IfFileExists": image_files}
    total = 0
    for relative in tex_files:
        if not relative.endswith(".tex"):
            raise RenderError("Only .tex files may appear in the TeX manifest")
        path = _confined(root, relative)
        source = path.read_text(encoding="utf-8")
        total += len(source.encode("utf-8"))
        _validate_commands(source, diagram=relative.startswith("figures/diagram-"), project=True)
        clean = _uncomment(source)
        if relative != "main.tex" and re.search(r"\\(?:documentclass|usepackage|setmainfont|renewcommand)\b", clean):
            raise RenderError("Preamble commands are allowed only in main.tex")
        for command, allowed in reference_commands.items():
            matches = list(re.finditer(r"\\" + command + r"\s*(?:\[[^\]\n]*\]\s*)?\{([^{}]*)\}", clean))
            if len(matches) != len(re.findall(r"\\" + command + r"\b", clean)):
                raise RenderError(f"Use literal braced paths for {command}")
            for match in matches:
                reference = match.group(1)
                if reference not in allowed:
                    raise RenderError(f"Undeclared asset reference: {reference}")
                _confined(root, reference)
        for package_group in re.findall(r"\\usepackage\s*(?:\[[^\]]*\])?\{([^{}]*)\}", clean):
            if set(package_group.split(",")) - _PACKAGES:
                raise RenderError("Unsupported LaTeX package")
        fixed_groups = {"documentclass": {"article"}, "usetikzlibrary": {"arrows.meta", "positioning", "calc", "shapes.geometric", "fit", "matrix"},
                        "usepgfplotslibrary": {"groupplots"}, "setdefaultlanguage": {"russian"}, "setotherlanguage": {"english"}}
        for command, allowed in fixed_groups.items():
            matches = re.findall(r"\\" + command + r"\s*(?:\[[^\]\n]*\]\s*)?\{([^{}]*)\}", clean)
            if len(matches) != len(re.findall(r"\\" + command + r"\b", clean)) or any(set(group.split(",")) - allowed for group in matches):
                raise RenderError(f"Unsupported {command} configuration")
        for command, expected in (("setmainfont", "CMU Serif"), ("setmonofont", "CMU Typewriter Text"), ("setmathfont", "Latin Modern Math")):
            fonts = re.findall(r"\\" + command + r"\s*\{([^{}]*)\}", clean)
            if len(fonts) != len(re.findall(r"\\" + command + r"\b", clean)) or any(font != expected for font in fonts):
                raise RenderError(f"Use the configured {expected} font")
        if len(re.findall(r"\\renewcommand\s*\{\\headrulewidth\}\s*\{0pt\}", clean)) != len(re.findall(r"\\renewcommand\b", clean)):
            raise RenderError("Only the page header rule may be redefined")
        if len(re.findall(r"\\newif\s*\\ifnotesdraft\b", clean)) != len(re.findall(r"\\newif\b", clean)):
            raise RenderError("Only the note draft flag may be declared")
        if re.search(r"\\(?:addplot|path|draw|fill|filldraw)\b[^;]*\b(?:file|table|graphics|shell|external)\b", clean, re.I):
            raise RenderError("Plots cannot load external data files")
    if total > 4 * _MAX_TEX_BYTES or "main.tex" not in tex_files:
        raise RenderError("Invalid or oversized TeX project")
    for relative in code_files:
        if not _confined(root, relative).is_file() or not relative.startswith("code/"):
            raise RenderError("Missing or invalid verbatim code asset")


def _image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_illustrations(job: Path, root: Path, identifiers: list[str], images_dir: Path | None) -> list[str]:
    missing = []
    evidence_roots = [(job / directory).resolve() for directory in _SOURCE_DIRS]
    source_hashes = None

    def reject_source_copy(path):
        nonlocal source_hashes
        if source_hashes is None:
            source_hashes = {_image_hash(source) for directory in evidence_roots if directory.is_dir()
                             for source in directory.rglob("*") if source.is_file() and source.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}}
        if _image_hash(path) in source_hashes:
            raise RenderError("A source evidence image was copied as a note illustration")

    for identifier in identifiers:
        candidates = []
        if images_dir is not None:
            candidates = [images_dir / f"{identifier}.{ext}" for ext in ("png", "jpg", "pdf") if (images_dir / f"{identifier}.{ext}").is_file()]
        if len(candidates) > 1:
            raise RenderError(f"Multiple illustrations supplied for {identifier}")
        if candidates:
            candidate = candidates[0].resolve()
            if any(candidate.is_relative_to(evidence) for evidence in evidence_roots):
                raise RenderError("Source evidence images cannot be used as finished-note illustrations")
            reject_source_copy(candidate)
            _check_image(candidate)
            destination = _confined(root, f"figures/illustrations/{identifier}{candidate.suffix}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if candidate != destination.resolve():
                shutil.copyfile(candidate, destination)
            for extension in ("png", "jpg", "pdf"):
                stale = _confined(root, f"figures/illustrations/{identifier}.{extension}")
                if stale != destination and stale.exists():
                    stale.unlink()
        existing = [_confined(root, f"figures/illustrations/{identifier}.{ext}") for ext in ("png", "jpg", "pdf")]
        existing = [path for path in existing if path.is_file()]
        for path in existing:
            reject_source_copy(path)
            _check_image(path)
        if not existing:
            missing.append(identifier)
    return missing


def _check_image(path: Path) -> None:
    # Decode only explicitly imported assets; raw evidence is not a fallback.
    if path.stat().st_size > 50_000_000:
        raise RenderError(f"Illustration is too large: {path.name}")
    prefix = path.read_bytes()[:8]
    expected = {".png": b"\x89PNG\r\n\x1a\n", ".jpg": b"\xff\xd8", ".pdf": b"%PDF-"}
    if path.suffix not in expected or not prefix.startswith(expected[path.suffix]):
        raise RenderError(f"Invalid illustration file: {path.name}")
    try:
        import pymupdf
        with pymupdf.open(path) as document:
            if document.page_count != 1:
                raise RenderError(f"Illustrations must contain one page: {path.name}")
            if document[0].rect.width * document[0].rect.height > 50_000_000:
                raise RenderError(f"Illustration dimensions are too large: {path.name}")
            document[0].get_pixmap(matrix=pymupdf.Matrix(0.25, 0.25))
    except ImportError as error:
        raise RenderError("PyMuPDF is required to validate illustrations") from error
    except RenderError:
        raise
    except Exception as error:
        raise RenderError(f"Cannot decode illustration {path.name}: {error}") from error


def _qa(pdf: Path) -> int:
    try:
        import pymupdf
    except ImportError as error:
        raise RenderError("PyMuPDF is required for PDF page quality previews") from error
    directory = _confined(pdf.parent, "qa")
    directory.mkdir(exist_ok=True)
    for stale in directory.glob("page-*.png"):
        _confined(pdf.parent, str(stale.relative_to(pdf.parent))).unlink()
    with pymupdf.open(pdf) as document:
        for page_number, page in enumerate(document, 1):
            destination = _confined(pdf.parent, f"qa/page-{page_number:03}.png")
            page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(destination)
        return document.page_count


def render(job: Path, *, images_dir: Path | None = None) -> Path:
    """Compile existing sources without replacing manual TeX corrections."""
    root = job.resolve() / "notes"
    if root.is_symlink() or not root.resolve().is_relative_to(job.resolve()):
        raise RenderError("The notes directory escapes the job")
    manifest = json.loads(_confined(root, "project.json").read_text(encoding="utf-8"))
    report = {"draft": bool(manifest.get("review") or manifest["illustration_ids"]), "review": manifest.get("review", []), "missing_illustrations": [],
              "missing_fonts": [], "missing_glyphs": [], "overfull": [], "status": "failed"}
    report_path = _confined(root, "render-report.json")
    log_path = _confined(root, "compiler-output.log")
    try:
        if manifest.get("evidence_fingerprint") != _evidence_fingerprint(job):
            raise RenderError("Evidence changed after these notes were generated; regenerate notes or import corrected notes before rendering")
        _validate_project(root, manifest)
        report["missing_illustrations"] = _import_illustrations(job.resolve(), root, manifest["illustration_ids"], images_dir)
        report["draft"] = bool(report["review"] or report["missing_illustrations"])
        if not shutil.which("latexmk") or not shutil.which("lualatex"):
            raise RenderError("Rendering requires latexmk and LuaLaTeX (for example, MacTeX)")
        for extension in ("pdf", "log", "aux", "out", "toc", "fls", "fdb_latexmk", "synctex.gz"):
            _confined(root, f"main.{extension}")
        environment = os.environ.copy()
        environment.update({"openin_any": "p", "openout_any": "p"})
        with log_path.open("w", encoding="utf-8") as compiler_output:
            completed = subprocess.run(
                ["latexmk", "-norc", "-lualatex", "-latexoption=-no-shell-escape",
                 "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "main.tex"],
                cwd=root, env=environment, stdout=compiler_output, stderr=subprocess.STDOUT,
                timeout=600, check=False,
            )
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if (root / "main.log").is_file():
            log += "\n" + (root / "main.log").read_text(encoding="utf-8", errors="replace")
        report["missing_fonts"] = sorted(set(re.findall(r".*(?:font.*(?:not found|cannot be found)|fontspec.*error).*", log, re.I)))
        report["missing_glyphs"] = sorted(set(re.findall(r".*Missing character:.*", log)))
        report["overfull"] = sorted(set(re.findall(r".*Overfull \\[hv]box.*", log)))
        pdf = _confined(root, "main.pdf")
        if completed.returncode or not pdf.is_file():
            raise RenderError(f"LuaLaTeX compilation failed; inspect {log_path}")
        report["pages"] = _qa(pdf)
        report["status"] = "rendered"
        return pdf
    except (OSError, subprocess.TimeoutExpired) as error:
        report["error"] = str(error)
        raise RenderError(f"Rendering failed; compiler logs are retained: {error}") from error
    except RenderError as error:
        report["error"] = str(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
