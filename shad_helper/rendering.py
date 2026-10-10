"""Editable Russian LaTeX notes with confined assets and PDF quality reports."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from .jobs import check_job, fingerprint, read_json, write_json
from .schemas import NoteBlock


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
    "cup cap setminus union times cdot div pm mp ast star circ bullet oplus otimes colon "
    "le leq ge geq neq ne approx sim simeq equiv cong propto ll gg "
    "to mapsto rightarrow leftarrow leftrightarrow Rightarrow Leftarrow "
    "Leftrightarrow longrightarrow longleftarrow Longrightarrow Longleftrightarrow "
    "uparrow downarrow updownarrow iff implies forall exists nexists neg land lor "
    "wedge vee top bot perp parallel mid angle triangle square checkmark "
    "ldots cdots vdots ddots dots dot ddot hat widehat tilde widetilde bar overline underline "
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
    "textbackslash textasciitilde textasciicircum textcolor "
    "input VerbatimInput pagestyle fancyhf fancyfoot renewcommand "
    "headrulewidth thepage linewidth textwidth medskip bigskip item "
    "hypersetup emergencystretch newtheorem theoremstyle".split()
)
_SYMBOL_COMMANDS = set("\\{}%&#$_~^ ,;:![]()|") | {"\n", "\r"}
_MATH_ENVIRONMENTS = set(
    "equation equation* "
    "align align* aligned alignedat gather gather* gathered multline multline* "
    "split cases matrix pmatrix bmatrix Bmatrix vmatrix Vmatrix smallmatrix "
    "array".split()
)
_DRAW_ENVIRONMENTS = {"tikzpicture", "scope", "axis", "groupplot"}
_ENVIRONMENTS = _MATH_ENVIRONMENTS | _DRAW_ENVIRONMENTS | {"document", "figure", "quote", "center", "itemize", "enumerate", "description", "definition", "theorem", "proof"}
_PACKAGES = set(
    "geometry fontspec polyglossia unicode-math amsmath amssymb amsthm tikz "
    "pgfplots fvextra fancyhdr xcolor hyperref microtype".split()
)
_ESCAPE = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$",
    "&": r"\&", "#": r"\#", "%": r"\%", "_": r"\_",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}
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


def _evidence_times(job: Path) -> dict[str, float]:
    path = job / "evidence.json"
    if not path.exists():
        return {}
    evidence = read_json(path)
    return {**{item["id"]: item["start"] for item in evidence.get("segments", [])},
            **{item["id"]: item["timestamp"] for item in evidence.get("visuals", [])}}


def _timestamp_footnote(evidence_ids: list[str], times: dict[str, float],
                        fallback: float, *, precise: bool) -> str:
    missing = set(evidence_ids) - times.keys()
    if precise and missing:
        raise RenderError("Uncertain fragments reference evidence without timestamps: "
                          + ", ".join(sorted(missing)))
    seconds = sorted({int(times[identifier]) for identifier in evidence_ids if identifier in times})
    stamps = [_stamp(second) for second in seconds] or [_stamp(fallback)]
    label = "–".join((stamps[0], stamps[-1])) if len(stamps) > 3 else ", ".join(stamps)
    return "\\footnote{" + label + "}"


def _marked_prose(block: dict, times: dict[str, float], fallback: float) -> str:
    fragments = block.get("uncertain_fragments", [])
    if not fragments and block["kind"] != "uncertain":
        return _prose(block["content"])
    if fragments:
        try:
            NoteBlock.model_validate(block)
        except ValueError as error:
            raise RenderError(f"Invalid uncertain fragments: {error}") from error
    paragraphs = []
    for paragraph in re.split(r"\n[ \t]*\n", block["content"]):
        if not paragraph.strip():
            continue
        selected = sorted((paragraph.index(fragment["text"]), fragment)
                          for fragment in fragments if fragment["text"] in paragraph)
        if not selected and (fragments or block["kind"] != "uncertain"):
            paragraphs.append(_prose(paragraph))
            continue
        parts = []
        evidence_ids = []
        previous = 0
        for start, fragment in selected:
            parts.extend((_prose(paragraph[previous:start]),
                          "\\textcolor{blue!60!black}{" + _prose(fragment["text"]) + "}"))
            evidence_ids.extend(fragment["evidence_ids"])
            previous = start + len(fragment["text"])
        if selected:
            parts.append(_prose(paragraph[previous:]))
        else:
            parts.append("\\textcolor{blue!60!black}{" + _prose(paragraph) + "}")
            evidence_ids = block.get("evidence_ids", [])
        parts.append(_timestamp_footnote(evidence_ids, times, fallback, precise=bool(fragments)))
        paragraphs.append("".join(parts))
    return "\n\n".join(paragraphs)


def _draft_footer(review: list[dict]) -> str:
    if review:
        return r"\fancyfoot[L]{\scriptsize ЧЕРНОВИК - требуется проверка}" + "\n"
    return ""


def _marked_footer(review: list[dict]) -> str:
    return _DRAFT_START + "\n" + _draft_footer(review) + _DRAFT_END + "\n"


def _review_flags(notes: dict) -> list[dict]:
    flags = [{"id": block["id"], "description": block["content"]}
             for chapter in notes.get("chapters", []) for block in chapter.get("blocks", [])
             if block["kind"] == "uncertain" or block.get("uncertain_fragments")]
    flags.extend({"id": issue["id"], "description": issue["description"]}
                 for issue in notes.get("issues", []) if issue["status"] != "resolved")
    return flags


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
\usepackage{amsthm}
\newtheorem{theorem}{Теорема}[section]
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Определение}
\usepackage{unicode-math}
\setmathfont{Latin Modern Math}
\usepackage{tikz,pgfplots,fvextra,fancyhdr,xcolor,hyperref,microtype}
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
    """Create editable sources from evidence-grounded notes.

    Call this only when replacing the note project is intentional. ``render``
    preserves these sources, including subsequent manual corrections.
    """
    check_job(job)
    root = job.resolve() / "notes"
    if root.is_symlink() or not root.resolve().is_relative_to(job.resolve()):
        raise RenderError("The notes directory escapes the job")
    tex_files = ["main.tex"]
    code_files = []
    section_sources = []
    block_ids = set()
    chapter_ids = set()
    topics = {}
    review = _review_flags(notes)
    evidence_times = _evidence_times(job)
    for chapter in notes.get("chapters", []):
        chapter_id = _identifier(chapter["id"])
        if chapter_id in chapter_ids:
            raise RenderError("Chapter IDs must be unique")
        chapter_ids.add(chapter_id)
        for block in chapter.get("blocks", []):
            block_id = _identifier(block["id"])
            if block_id in block_ids:
                raise RenderError("Block IDs must be unique")
            block_ids.add(block_id)
            title = " ".join((block.get("topic") or "").split()) or " ".join(chapter["title"].split())
            _, blocks = topics.setdefault(title.casefold(), (title, []))
            blocks.append((chapter, block))
    for number, (title, blocks) in enumerate(topics.values(), 1):
        lines = ["\\section{" + escape_text(title) + "}"]
        for chapter, block in blocks:
            block_id = block["id"]
            content = block["content"]
            kind = block["kind"]
            lines.append(f"% shad-helper: block {block_id}")
            if kind == "math":
                _validate_commands(content)
                display_environment = re.match(r"\s*\\begin\s*\{(?:equation|align|gather|multline)\*?\}", content)
                if content.lstrip().startswith(r"\[") or display_environment:
                    lines.append(content + "\n")
                else:
                    lines.append("\\[\n" + content + "\n\\]\n")
            elif kind == "diagram":
                _validate_commands(content, diagram=True)
                if not content.lstrip().startswith((r"\begin{tikzpicture}", r"\tikz")):
                    content = "\\begin{tikzpicture}\n" + content + "\n\\end{tikzpicture}"
                relative = f"figures/diagram-{block_id}.tex"
                _write(root, relative, content + "\n")
                tex_files.append(relative)
                lines.append("\\begin{figure}[htbp]\n\\centering\n\\input{" + relative + "}\n\\caption{" + escape_text(block.get("caption", "Схема")) + "}\n\\end{figure}\n")
            elif kind == "code":
                relative = f"code/{block_id}.txt"
                _write(root, relative, content)
                code_files.append(relative)
                lines.append("\\VerbatimInput[breaklines,breakanywhere,fontsize=\\small,frame=single]{" + relative + "}\n")
            elif kind in {"text", "clarification", "uncertain"}:
                lines.append(_marked_prose(block, evidence_times, chapter["start"]) + "\n")
            elif kind in {"definition", "theorem", "proof"}:
                heading = "[Доказательство]" if kind == "proof" else ""
                lines.append("\\begin{" + kind + "}" + heading + "\n"
                             + _marked_prose(block, evidence_times, chapter["start"])
                             + "\n\\end{" + kind + "}\n")
            else:
                raise RenderError(f"Unsupported note block kind: {kind}")
        relative = f"chapters/section-{number:03}.tex"
        _write(root, relative, "\n\n".join(lines) + "\n")
        tex_files.append(relative)
        section_sources.append("\\input{" + relative + "}")
    appendix = []
    if notes.get("glossary"):
        appendix.append(r"\section{Глоссарий}")
        for entry in notes["glossary"]:
            appendix.append("\\paragraph{" + _prose(entry["term"]) + "} " + _prose(entry["definition"]))
    _confined(root, "review.tex").unlink(missing_ok=True)
    main = _PREAMBLE + _marked_footer(review) + "\\begin{document}\n\\begin{center}\\Large " + escape_text(notes["title"]) + "\\end{center}\n" + "\n".join(section_sources + appendix) + "\n\\end{document}\n"
    _write(root, "main.tex", main)
    manifest = {"version": 2, "draft": bool(review), "tex_files": tex_files,
                "code_files": code_files, "review": review,
                "issues": notes.get("issues", []), "evidence_fingerprint": _evidence_fingerprint(job)}
    _write(root, "project.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return root / "main.tex"


def refresh_review(job: Path, notes: dict) -> Path:
    """Refresh explicit review metadata while retaining edited note content.

    The caller verifies the project matches the current evidence before refresh.
    It updates only project.json and the marked footer in main.tex.
    """
    check_job(job)
    root = job.resolve() / "notes"
    if root.is_symlink() or not root.resolve().is_relative_to(job.resolve()):
        raise RenderError("The notes directory escapes the job")
    manifest = read_json(_confined(root, "project.json"))
    if manifest.get("version") != 2:
        raise RenderError("Unsupported note project; use a new output directory.")
    main_path = _confined(root, "main.tex")
    main = main_path.read_text(encoding="utf-8")
    if main.count(_DRAFT_START) != 1 or main.count(_DRAFT_END) != 1:
        raise RenderError("Review refresh requires the generated footer markers; regenerate or import notes before refreshing review")
    pattern = re.escape(_DRAFT_START) + r".*?" + re.escape(_DRAFT_END) + r"\n?"
    if not re.search(pattern, main, flags=re.DOTALL):
        raise RenderError("Review footer markers are out of order; regenerate or import notes")
    review = _review_flags(notes)
    footer = _marked_footer(review)
    main = re.sub(pattern, lambda _: footer, main, count=1, flags=re.DOTALL)
    manifest.update(review=review, issues=notes.get("issues", []), draft=bool(review),
                    evidence_fingerprint=_evidence_fingerprint(job))
    _write(root, "main.tex", main)
    _write(root, "project.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return main_path


def _validate_project(root: Path, manifest: dict) -> None:
    tex_files = set(manifest["tex_files"])
    code_files = set(manifest["code_files"])
    reference_commands = {"input": tex_files, "VerbatimInput": code_files}
    total = 0
    for relative in tex_files:
        if not relative.endswith(".tex"):
            raise RenderError("Only .tex files may appear in the TeX manifest")
        path = _confined(root, relative)
        source = path.read_text(encoding="utf-8")
        total += len(source.encode("utf-8"))
        _validate_commands(source, diagram=relative.startswith("figures/diagram-"), project=True)
        clean = _uncomment(source)
        if relative != "main.tex" and re.search(r"\\(?:documentclass|usepackage|setmainfont|renewcommand|newtheorem|theoremstyle)\b", clean):
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
        declarations = re.findall(
            r"\\newtheorem(?:\{theorem\}\{Теорема\}\[section\]|\{definition\}\[theorem\]\{Определение\})", clean)
        if len(declarations) != len(re.findall(r"\\newtheorem\b", clean)):
            raise RenderError("Only the configured theorem and definition environments may be declared")
        if len(re.findall(r"\\theoremstyle\{definition\}", clean)) != len(re.findall(r"\\theoremstyle\b", clean)):
            raise RenderError("Only the configured definition style is supported")
        if re.search(r"\\(?:addplot|path|draw|fill|filldraw)\b[^;]*\b(?:file|table|graphics|shell|external)\b", clean, re.I):
            raise RenderError("Plots cannot load external data files")
    if total > 4 * _MAX_TEX_BYTES or "main.tex" not in tex_files:
        raise RenderError("Invalid or oversized TeX project")
    for relative in code_files:
        if not _confined(root, relative).is_file() or not relative.startswith("code/"):
            raise RenderError("Missing or invalid verbatim code asset")


def _run_latex(root: Path) -> subprocess.CompletedProcess:
    if not shutil.which("latexmk") or not shutil.which("lualatex"):
        raise RenderError("Rendering requires latexmk and LuaLaTeX (for example, MacTeX)")
    environment = os.environ.copy()
    environment.update({"openin_any": "p", "openout_any": "p"})
    with (root / "compiler-output.log").open("w", encoding="utf-8") as output:
        return subprocess.run(
            ["latexmk", "-norc", "-lualatex", "-latexoption=-no-shell-escape",
             "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "main.tex"],
            cwd=root, env=environment, stdout=output, stderr=subprocess.STDOUT,
            timeout=600, check=False,
        )


def validate_chapter_latex(notes: dict, evidence: dict, folder: Path) -> None:
    """Compile a chapter locally; retain sources and logs without PDF previews."""
    chapter = notes["chapters"][0]
    context = dict(evidence, segments=[segment for segment in evidence["segments"]
                                      if any(segment["id"] in block["evidence_ids"]
                                             for block in chapter["blocks"])])
    write_json(folder / "evidence.json", context)
    try:
        root = write_project(folder, notes).parent
        _validate_project(root, read_json(root / "project.json"))
        completed = _run_latex(root)
        if completed.returncode or not (root / "main.pdf").is_file():
            log_path = root / "compiler-output.log"
            log = log_path.read_text(encoding="utf-8", errors="replace")
            error = re.search(r"(?:\./)?((?:chapters|figures)/[\w-]+\.tex|main\.tex):(\d+): ([^\n]+)", log)
            detail = "LuaLaTeX compilation failed"
            if error:
                source, line, message = error.groups()
                detail = f"{source}:{line}: {message}"
                markers = re.findall(r"^% shad-helper: block ([\w-]+)$",
                                     "\n".join((root / source).read_text().splitlines()[:int(line)]),
                                     re.MULTILINE)
                if markers:
                    detail = f"block {markers[-1]}: {detail}"
                elif source.startswith("figures/diagram-"):
                    detail = f"block {Path(source).stem.removeprefix('diagram-')}: {detail}"
            raise RenderError(f"{detail}; inspect {log_path}")
    except (OSError, subprocess.TimeoutExpired, RenderError) as error:
        raise RenderError(f"Chapter {chapter['id']} LaTeX validation failed: {error}") from error


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


def render(job: Path) -> Path:
    """Compile existing sources without replacing manual TeX corrections."""
    check_job(job)
    root = job.resolve() / "notes"
    if root.is_symlink() or not root.resolve().is_relative_to(job.resolve()):
        raise RenderError("The notes directory escapes the job")
    manifest = json.loads(_confined(root, "project.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 2:
        raise RenderError("Unsupported note project; use a new output directory.")
    report = {"draft": bool(manifest.get("review")), "review": manifest.get("review", []),
              "missing_fonts": [], "missing_glyphs": [], "overfull": [], "status": "failed"}
    report_path = _confined(root, "render-report.json")
    log_path = _confined(root, "compiler-output.log")
    try:
        if manifest.get("evidence_fingerprint") != _evidence_fingerprint(job):
            raise RenderError("Evidence changed after these notes were generated; regenerate notes or import corrected notes before rendering")
        _validate_project(root, manifest)
        for extension in ("pdf", "log", "aux", "out", "toc", "fls", "fdb_latexmk", "synctex.gz"):
            _confined(root, f"main.{extension}")
        completed = _run_latex(root)
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if (root / "main.log").is_file():
            log += "\n" + (root / "main.log").read_text(encoding="utf-8", errors="replace")
        report["missing_fonts"] = sorted({line for line in re.findall(
            r".*(?:font.*(?:not found|cannot be found)|fontspec.*error).*", log, re.IGNORECASE)
            if "Font names database not found" not in line})
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
