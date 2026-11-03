"""Editable Russian LaTeX notes with confined assets and PDF quality reports."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

from .jobs import check_job, fingerprint, read_json, sha256_file, write_json
from .schemas import NoteBlock


class RenderError(RuntimeError):
    """A note project cannot safely be rendered, or its compiler failed."""


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
_COMMAND = re.compile(r"\\([A-Za-z@]+|.)", re.DOTALL)
_MAX_TEX_BYTES = 2_000_000
# Public formula commands from core LaTeX's fontmath.ltx (fontdef.dtx), TeX's
# eight math atom classes, and amsmath with its amstext/amsopn/amsbsy dependencies.
# Package internals and declarations are not formula content; keep those out.
_MATH_COMMANDS = set(
    "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa "
    "lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi "
    "psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega "
    "frac dfrac tfrac cfrac genfrac binom dbinom tbinom sqrt sqrtsign leftroot uproot "
    "sum prod coprod int intop smallint iint iiint iiiint idotsint oint ointop "
    "bigvee bigwedge biguplus bigcap bigcup bigotimes bigoplus bigodot bigsqcup "
    "lim limsup liminf inf sup min max log ln exp sin cos tan cot sec csc sinh cosh "
    "tanh coth arcsin arccos arctan det gcd Pr arg ker dim deg hom lg "
    "injlim projlim varinjlim varprojlim varliminf varlimsup mod bmod pmod pod "
    "left right middle big Big bigg Bigg bigl bigr Bigl Bigr biggl biggr Biggl Biggr "
    "bigm Bigm biggm Biggm "
    "langle rangle lvert rvert lVert rVert vert Vert lceil rceil lfloor rfloor "
    "lbrace rbrace lgroup rgroup lmoustache rmoustache arrowvert Arrowvert bracevert backslash "
    "infty partial nabla ell hbar imath jmath Re Im emptyset varnothing aleph wp prime surd "
    "flat natural sharp clubsuit diamondsuit heartsuit spadesuit "
    "in notin ni owns subset supset subseteq supseteq subsetneq supsetneq sqsubseteq sqsupseteq "
    "cup cap setminus union times cdot div pm mp ast star circ bullet oplus otimes colon "
    "sqcup sqcap uplus amalg diamond wr odot oslash ominus bigcirc dagger ddagger "
    "triangleleft triangleright bigtriangleup bigtriangledown varbigtriangleup varbigtriangledown "
    "le leq ge geq neq ne approx sim simeq equiv cong propto ll gg doteq asymp "
    "prec succ preceq succeq dashv vdash models smile frown bowtie "
    "to gets mapsto rightarrow leftarrow leftrightarrow Rightarrow Leftarrow "
    "Leftrightarrow longrightarrow longleftarrow longleftrightarrow longmapsto "
    "Longrightarrow Longleftarrow Longleftrightarrow hookrightarrow hookleftarrow "
    "xrightarrow xleftarrow relbar Relbar joinrel lhook rhook mapstochar "
    "nearrow searrow nwarrow swarrow leftharpoonup leftharpoondown "
    "rightharpoonup rightharpoondown rightleftharpoons "
    "uparrow downarrow updownarrow Uparrow Downarrow Updownarrow "
    "iff implies impliedby And forall exists nexists neg lnot not land lor "
    "wedge vee top bot perp parallel mid angle triangle square checkmark "
    "ldots cdots vdots ddots dots dotsb dotsc dotsi dotsm dotso hdots hdotsfor ldotp cdotp "
    "dot ddot dddot ddddot hat widehat tilde widetilde bar overline underline "
    "acute grave breve check mathring skew Hat Check Tilde Acute Grave Dot Ddot Breve Bar Vec "
    "varGamma varDelta varTheta varLambda varXi varPi varSigma varUpsilon varPhi varPsi varOmega "
    "vec overrightarrow overleftarrow overleftrightarrow underrightarrow underleftarrow "
    "underleftrightarrow overbrace underbrace overset underset overunderset stackrel sideset "
    "rightarrowfill leftarrowfill downbracefill upbracefill braceld bracerd bracelu braceru "
    "mathrm mathbf mathit mathsf mathtt mathcal mathbb mathscr boldsymbol pmb "
    "mathdollar mathsterling mathparagraph mathsection mathunderscore mathellipsis "
    "operatorname operatornamewithlimits operatorfont text textrm textsf texttt textbf textit "
    "mathord mathop mathbin mathrel mathopen mathclose mathpunct mathinner limits nolimits displaylimits "
    "quad qquad hspace vspace phantom hphantom vphantom substack label ref eqref "
    "thinspace medspace thickspace negthinspace negmedspace negthickspace mspace smash nobreakdash "
    "nonumber notag tag thetag underbracket overbracket boxed displaystyle textstyle "
    "intertext shoveleft shoveright raisetag displaybreak allowdisplaybreaks "
    "scriptstyle scriptscriptstyle mathrm mathnormal cases begin end".split()
)
_DRAW_COMMANDS = set(
    "draw node path coordinate fill filldraw clip shade shadedraw foreach "
    "tikz tikzset scope addplot addlegendentry legend nextgroupplot "
    "pgfplotsset pgfmathsetmacro pgfmathparse pgfmathresult pgfmathprintnumber "
    "pgfmathsetlengthmacro linewidth textwidth textheight".split()
)
_PROJECT_COMMANDS = set(
    "documentclass usepackage setmainfont setmonofont setmathfont "
    "setdefaultlanguage setotherlanguage usetikzlibrary usepgfplotslibrary "
    "title author date maketitle tableofcontents newpage clearpage section "
    "subsection subsubsection paragraph par noindent centering caption "
    "footnote footnotesize scriptsize small normalsize large Large "
    "textbackslash textasciitilde textasciicircum textcolor "
    "input VerbatimInput includegraphics pagestyle fancyhf fancyfoot renewcommand "
    "headrulewidth thepage linewidth textwidth textheight medskip bigskip item "
    "hypersetup hypertarget emergencystretch newtheorem theoremstyle ifdim else fi "
    "newsavebox shadfiguregroup setbox vbox bgroup egroup hsize dimexpr ht dp relax unvbox box".split()
)
_SYMBOL_COMMANDS = set("\\{}%&#$_~^ ,;:![]()|") | {"\n", "\r"}
_MATH_ENVIRONMENTS = set(
    "equation equation* "
    "align align* aligned alignedat gather gather* gathered multline multline* "
    "alignat alignat* xalignat xalignat* xxalignat flalign flalign* subequations subarray "
    "split cases matrix pmatrix bmatrix Bmatrix vmatrix Vmatrix smallmatrix "
    "array".split()
)
_DISPLAY_ENVIRONMENT = r"\{(?:(?:equation|align|alignat|xalignat|flalign|gather|multline)\*?|xxalignat|subequations)\}"
# These displays consume a column-count argument before their math body. Source
# anchors must follow that argument, otherwise TeX reads the anchor as the count.
_DISPLAY_BEGIN = (r"\\begin\s*\{(?:alignat\*?|xalignat\*?|xxalignat)\}"
                  r"(?:\s|%[^\n]*\n)*(?:\{[^{}]+\}|[0-9])"
                  r"|\\begin\s*" + _DISPLAY_ENVIRONMENT)
_DRAW_ENVIRONMENTS = {"tikzpicture", "scope", "axis", "groupplot"}
_ENVIRONMENTS = _MATH_ENVIRONMENTS | _DRAW_ENVIRONMENTS | {"document", "figure", "quote", "center", "itemize", "enumerate", "description", "definition", "theorem", "proof"}
_PACKAGES = set(
    "geometry fontspec polyglossia unicode-math amsmath amssymb amsthm tikz "
    "pgfplots fvextra fancyhdr xcolor hyperref microtype graphicx float".split()
)
_PROSE_KINDS = {"text", "definition", "theorem", "proof", "clarification", "uncertain"}
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
    if diagram and (set(commands) & {"input", "VerbatimInput", "includegraphics"}
                    or re.search(r"\b(?:file|table|graphics|shell|external)\b", source, re.IGNORECASE)):
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
            **{item["id"]: item["timestamp"] for item in evidence.get("visuals", [])},
            **{illustration["id"]: visual["timestamp"]
               for visual in evidence.get("visuals", [])
               for illustration in visual.get("illustrations", [])}}


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
    references = block.get("figure_refs", [])
    suffix = (" (см. рис. " + ", ".join("\\ref{fig:" + identifier + "}"
                                      for identifier in references) + ")") if references else ""
    fragments = block.get("uncertain_fragments", [])
    if not fragments and block["kind"] != "uncertain":
        return _prose(block["content"]) + suffix
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
    return "\n\n".join(paragraphs) + suffix


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
\usepackage{tikz,pgfplots,fvextra,fancyhdr,xcolor,hyperref,microtype,graphicx,float}
\usetikzlibrary{arrows.meta,positioning,calc,shapes.geometric,fit,matrix}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\hypersetup{hidelinks}
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[R]{\thepage}
\renewcommand{\headrulewidth}{0pt}
\emergencystretch=2em
\newsavebox{\shadfiguregroup}
"""


def write_project(job: Path, notes: dict, *, asset_job: Path | None = None) -> Path:
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
    image_files = []
    section_sources = []
    block_ids = set()
    chapter_ids = set()
    sections = []
    ordered_blocks = []
    source_blocks = []
    review = _review_flags(notes)
    evidence_times = _evidence_times(job)
    for chapter in sorted(notes.get("chapters", []), key=lambda chapter: chapter["start"]):
        chapter_id = _identifier(chapter["id"])
        if chapter_id in chapter_ids:
            raise RenderError("Chapter IDs must be unique")
        chapter_ids.add(chapter_id)
        for block in chapter.get("blocks", []):
            block_id = _identifier(block["id"])
            if block_id in block_ids:
                raise RenderError("Block IDs must be unique")
            block_ids.add(block_id)
            ordered_blocks.append((chapter, block))
    figure_ids = {block["id"] for _, block in ordered_blocks
                  if block["kind"] in {"diagram", "screenshot"}}
    for _, block in ordered_blocks:
        references = block.get("figure_refs", [])
        if not isinstance(references, list):
            raise RenderError("Figure references must be a list")
        for identifier in references:
            _identifier(identifier)
        if (len(set(references)) != len(references) or set(references) - figure_ids
                or (references and block["kind"] not in _PROSE_KINDS)):
            raise RenderError("Figure references must be unique figure block IDs on prose blocks")
    for index, (chapter, block) in enumerate(ordered_blocks):
        heading_block, heading_chapter = block, chapter
        if block["kind"] in {"diagram", "screenshot"}:
            following = next(
                ((candidate_chapter, candidate) for candidate_chapter, candidate
                 in ordered_blocks[index + 1:] if candidate["kind"] not in {"diagram", "screenshot"}),
                None)
            heading_chapter, heading_block = following or (chapter, {"topic": ""})
            following_topic = (" ".join((heading_block.get("topic") or "").split())
                               or " ".join(heading_chapter["title"].split()))
            figure_topic = " ".join((block.get("topic") or "").split())
            if sections and (following is None
                             or figure_topic.casefold() != following_topic.casefold()):
                sections[-1][1].append((chapter, block))
                continue
        title = (" ".join((heading_block.get("topic") or "").split())
                 or " ".join(heading_chapter["title"].split()))
        # Figures share a neighboring substantive topic without changing chronology.
        if not sections or sections[-1][0].casefold() != title.casefold():
            sections.append((title, []))
        sections[-1][1].append((chapter, block))
    for number, (title, blocks) in enumerate(sections, 1):
        lines = ["\\section{" + escape_text(title) + "}"]
        grouped_figure = None
        explained_figures = set()
        for position, (chapter, block) in enumerate(blocks):
            block_id = block["id"]
            content = block["content"]
            kind = block["kind"]
            if kind in _PROSE_KINDS:
                following_figures = []
                for following in range(position + 1, len(blocks)):
                    candidate = blocks[following][1]
                    if (candidate["kind"] not in {"diagram", "screenshot"}
                            or candidate["id"] not in block.get("figure_refs", [])):
                        break
                    following_figures.append(following)
                explained_figures.update(following_figures)
                if (following_figures and kind != "uncertain"
                        and not block.get("uncertain_fragments")):
                    grouped_figure = following_figures[0]
                    lines.append("\\par\\setbox\\shadfiguregroup=\\vbox\\bgroup\n"
                                 "\\hsize=\\linewidth\n")
            lines.append(f"% shad-helper: block {block_id}")
            if kind == "math":
                _validate_commands(content)
                display_environment = re.match(r"\s*\\begin\s*" + _DISPLAY_ENVIRONMENT, content)
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
                lines.append("\\begin{figure}[H]\n\\centering\n\\input{" + relative + "}\n\\caption{" + escape_text(block.get("caption", "Схема")) + "}\\label{fig:" + block_id + "}\n\\end{figure}\n")
            elif kind == "screenshot":
                import pymupdf

                from .illustration_crops import resolve_crop

                illustration_id = _identifier(block.get("illustration_id", ""))
                crop_id = block.get("illustration_crop_id", "")
                try:
                    evidence = read_json(job / "evidence.json")
                    source = resolve_crop(asset_job or job, evidence, illustration_id, crop_id)
                except (OSError, ValueError) as error:
                    raise RenderError(f"Cannot resolve illustration {illustration_id}: {error}") from error
                relative = f"figures/illustration-{block_id if crop_id else illustration_id}.png"
                destination = _confined(root, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if relative not in image_files:
                    image_files.append(relative)
                # 96 dpi gives small native crops a natural display size; larger
                # images still fit the available line width and page height.
                natural_width = pymupdf.Pixmap(source).width * 0.75
                width = ("{\\ifdim " + f"{natural_width:g}bp" + "<\\linewidth "
                         + f"{natural_width:g}bp" + "\\else\\linewidth\\fi}")
                # Reserve space for an immediately preceding explanation. This
                # affects only tall figures; native-size and width bounds remain.
                height = ".36" if position in explained_figures else ".7"
                lines.append("\\begin{figure}[H]\n\\centering\n"
                             "\\includegraphics[width=" + width
                             + ",height=" + height + "\\textheight,keepaspectratio]{"
                             + relative + "}\n\\caption{" + escape_text(block.get("caption", "Иллюстрация"))
                             + "}\\label{fig:" + block_id + "}\n\\end{figure}\n")
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
            start = f"\\hypertarget{{shad-block-{block_id}}}{{}}"
            end = f"\\hypertarget{{shad-end-{block_id}}}{{}}"
            if kind in {"diagram", "screenshot"}:
                lines[-1] = lines[-1].replace("\\centering\n", "\\centering\n" + start, 1)
                lines[-1] = lines[-1].replace("\\end{figure}", end + "\\end{figure}", 1)
            elif kind in {"definition", "theorem", "proof"}:
                lines[-1] = lines[-1].replace("\n", "\n" + start, 1)
                lines[-1] = lines[-1].replace("\n\\end{", end + "\n\\end{", 1)
            elif kind == "math":
                lines[-1] = re.sub(r"\\\[|" + _DISPLAY_BEGIN,
                                   lambda match, start=start: match[0] + start, lines[-1], count=1)
                lines[-1] = re.sub(r"\\(?:\]|end\s*" + _DISPLAY_ENVIRONMENT + r")\s*$",
                                   lambda match, end=end: end + match[0], lines[-1])
            elif kind == "code":
                lines[-1] = lines[-1].replace("[breaklines,", "[formatcom={" + start
                                             + "},breaklines,", 1).rstrip() + end + "\n"
            else:
                lines[-1] = start + lines[-1].rstrip() + end + "\n"
            if position == grouped_figure:
                # Typeset only once so figure numbers, labels and anchors stay
                # consistent. Oversized groups are unboxed into breakable content.
                lines.append("\\egroup\n"
                             "\\ifdim\\dimexpr\\ht\\shadfiguregroup+\\dp\\shadfiguregroup"
                             "\\relax>\\textheight\n\\unvbox\\shadfiguregroup\n"
                             "\\else\n\\box\\shadfiguregroup\n\\fi\n")
                grouped_figure = None
            source_blocks.append({"id": block_id, "kind": kind, "topic": title, "content": content,
                                  "evidence_ids": block.get("evidence_ids", []),
                                  **({"figure_asset": {"path": relative,
                                                       "sha256": sha256_file(_confined(root, relative))}}
                                     if kind in {"diagram", "screenshot"} else {}),
                                  **({"figure_refs": block["figure_refs"]}
                                     if block.get("figure_refs") else {}),
                                  **({"illustration_id": block["illustration_id"]}
                                     if block.get("illustration_id") else {}),
                                  **({"illustration_crop_id": block["illustration_crop_id"]}
                                     if block.get("illustration_crop_id") else {})})
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
                "code_files": code_files, "image_files": image_files, "review": review,
                "source_blocks": source_blocks,
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
    image_files = set(manifest.get("image_files", []))
    reference_commands = {"input": tex_files, "VerbatimInput": code_files,
                          "includegraphics": image_files}
    total = 0
    for relative in tex_files:
        if not relative.endswith(".tex"):
            raise RenderError("Only .tex files may appear in the TeX manifest")
        path = _confined(root, relative)
        source = path.read_text(encoding="utf-8")
        total += len(source.encode("utf-8"))
        _validate_commands(source, diagram=relative.startswith("figures/diagram-"), project=True)
        clean = _uncomment(source)
        if relative != "main.tex" and re.search(r"\\(?:documentclass|usepackage|setmainfont|renewcommand|newtheorem|theoremstyle|newsavebox)\b", clean):
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
        if len(re.findall(r"\\newsavebox\{\\shadfiguregroup\}", clean)) != len(re.findall(r"\\newsavebox\b", clean)):
            raise RenderError("Only the configured figure layout box may be declared")
        if re.search(r"\\(?:addplot|path|draw|fill|filldraw)\b[^;]*\b(?:file|table|graphics|shell|external)\b", clean, re.I):
            raise RenderError("Plots cannot load external data files")
    if total > 4 * _MAX_TEX_BYTES or "main.tex" not in tex_files:
        raise RenderError("Invalid or oversized TeX project")
    for relative in code_files:
        if not _confined(root, relative).is_file() or not relative.startswith("code/"):
            raise RenderError("Missing or invalid verbatim code asset")
    for relative in image_files:
        if not re.fullmatch(r"figures/illustration-[A-Za-z0-9][A-Za-z0-9_-]{0,79}\.png", relative):
            raise RenderError("Invalid illustration image asset")
        path = _confined(root, relative)
        if not path.is_file():
            raise RenderError(f"Missing illustration image asset: {relative}")
        with path.open("rb") as source:
            if source.read(8) != b"\x89PNG\r\n\x1a\n":
                raise RenderError(f"Illustration image asset is not a PNG: {relative}")


def _run_latex(root: Path, *, rebuild: bool = False) -> subprocess.CompletedProcess:
    if not shutil.which("latexmk") or not shutil.which("lualatex"):
        raise RenderError("Rendering requires latexmk and LuaLaTeX (for example, MacTeX)")
    environment = os.environ.copy()
    environment.update({"openin_any": "p", "openout_any": "p"})
    with (root / "compiler-output.log").open("w", encoding="utf-8") as output:
        return subprocess.run(
            ["latexmk", "-norc", *(["-gg"] if rebuild else []),
             "-lualatex", "-latexoption=-no-shell-escape",
             "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "main.tex"],
            cwd=root, env=environment, stdout=output, stderr=subprocess.STDOUT,
            timeout=600, check=False,
        )


def validate_chapter_latex(notes: dict, evidence: dict, folder: Path, *,
                           asset_job: Path | None = None) -> None:
    """Compile a chapter locally; retain sources and logs without PDF previews."""
    # Notes validation checks cross-chapter references against accepted prior blocks.
    # Their final numbering exists only in the merged project, so omit those refs
    # from this isolated compile without changing the accepted notes.
    notes = deepcopy(notes)
    local_figures = {block["id"] for chapter in notes["chapters"] for block in chapter["blocks"]
                     if block["kind"] in {"diagram", "screenshot"}}
    for chapter in notes["chapters"]:
        for block in chapter["blocks"]:
            if block.get("figure_refs"):
                block["figure_refs"] = [identifier for identifier in block["figure_refs"]
                                        if identifier in local_figures]
    chapter = notes["chapters"][0]
    context = dict(evidence, segments=[segment for segment in evidence["segments"]
                                      if any(segment["id"] in block["evidence_ids"]
                                             for block in chapter["blocks"])])
    write_json(folder / "evidence.json", context)
    try:
        root = write_project(folder, notes, asset_job=asset_job).parent
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


def diagram_preview_fingerprint() -> str:
    """Invalidate visual verdicts when diagram rendering settings change."""
    return fingerprint({"preamble": _PREAMBLE, "policy": 1, "compiler": "latexmk-lualatex",
                        "shell_escape": False, "page_style": "empty", "padding": 6, "scale": 2})


def render_diagram_preview(content: str, folder: Path) -> Path:
    """Compile only the drawing and crop its painted bounds for visual comparison."""
    import pymupdf

    _validate_commands(content, diagram=True)
    if not content.lstrip().startswith((r"\begin{tikzpicture}", r"\tikz")):
        content = "\\begin{tikzpicture}\n" + content + "\n\\end{tikzpicture}"
    folder = Path(folder)
    if folder.is_symlink():
        raise RenderError("The diagram preview directory cannot be a symbolic link")
    folder.mkdir(parents=True, exist_ok=True)
    source = "figures/diagram-preview.tex"
    _write(folder, source, content + "\n")
    _write(folder, "main.tex", _PREAMBLE + "\\pagestyle{empty}\n\\begin{document}\n"
           "\\noindent\\input{" + source + "}\n\\end{document}\n")
    manifest = {"version": 2, "tex_files": ["main.tex", source], "code_files": [],
                "image_files": []}
    _write(folder, "project.json", json.dumps(manifest, indent=2) + "\n")
    _validate_project(folder, manifest)
    for extension in ("pdf", "log", "aux", "out", "toc", "fls", "fdb_latexmk", "synctex.gz"):
        _confined(folder, f"main.{extension}")
    pdf = _confined(folder, "main.pdf")
    preview = _confined(folder, "preview.png")
    _confined(folder, "compiler-output.log")
    pdf.unlink(missing_ok=True)
    preview.unlink(missing_ok=True)
    _confined(folder, "main.fdb_latexmk").unlink(missing_ok=True)
    try:
        completed = _run_latex(folder)
        log = _confined(folder, "compiler-output.log").read_text(encoding="utf-8", errors="replace")
        if (folder / "main.log").is_file():
            log += "\n" + (folder / "main.log").read_text(encoding="utf-8", errors="replace")
        if completed.returncode or not pdf.is_file():
            raise RenderError(f"Diagram LaTeX compilation failed; inspect {folder / 'compiler-output.log'}")
        if re.search(r"Overfull \\[hv]box|Missing character:", log):
            raise RenderError("Diagram preview has overflow or missing glyphs")
        with pymupdf.open(pdf) as document:
            if len(document) != 1:
                raise RenderError("Diagram preview must occupy exactly one page")
            page = document[0]
            bounds = pymupdf.Rect()
            for kind, box, *_ in page.get_bboxlog():
                if kind != "ignore-text":
                    rectangle = pymupdf.Rect(box)
                    if rectangle.is_empty:
                        continue
                    if rectangle.is_infinite or not rectangle.is_valid:
                        raise RenderError("Diagram preview has invalid painted bounds")
                    bounds |= rectangle
            if bounds.is_empty:
                raise RenderError("Diagram preview is empty")
            if not page.rect.contains(bounds):
                raise RenderError("Diagram preview is clipped by the page")
            bounds = pymupdf.Rect(bounds.x0 - 6, bounds.y0 - 6,
                                 bounds.x1 + 6, bounds.y1 + 6) & page.rect
            page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=bounds, alpha=False).save(preview)
        return preview
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired,
            pymupdf.mupdf.FzErrorBase) as error:
        if isinstance(error, RenderError):
            raise
        raise RenderError(f"Diagram preview failed; compiler sources and logs are retained: {error}") from error


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


def render(job: Path, *, rebuild: bool = False) -> Path:
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
        _confined(root, "source-map.json").unlink(missing_ok=True)
        for extension in ("pdf", "log", "aux", "out", "toc", "fls", "fdb_latexmk", "synctex.gz"):
            _confined(root, f"main.{extension}")
        completed = _run_latex(root, rebuild=True) if rebuild else _run_latex(root)
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
        if "source_blocks" in manifest:
            from .pdf_sources import write_source_map

            try:
                write_source_map(pdf, manifest)
            except ValueError as error:
                raise RenderError(str(error)) from error
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
