import re
import shutil
from copy import deepcopy

import pytest

from shad_helper import rendering
from shad_helper.anki_latex import normalize_latex

EXPRESSIONS = [
    r"\hat\theta\xrightarrow{P}\theta",
    r"\hat\theta_n\xrightarrow[n\to\infty]{P}\theta\xleftarrow[m\to\infty]{P}\hat\theta_m",
    r"\genfrac{[}{]}{0pt}{}{n}{k}+\tbinom{n}{k}+\cfrac[l]{1}{1+\cfrac[r]{1}{x}}",
    r"a_1+\dotsb+a_n,\quad a_1,\dotsc,a_n,\quad A_1\dotsm A_n,\quad\int\dotsi\int f,\quad\dotso",
    r"\iint_D f\,dx\,dy+\iiint_D f+\iiiint_D f+\idotsint_D f",
    r"\operatorname*{arg\,min}_{x}f(x),\quad\injlim A_n,\quad\projlim A_n",
    r"\varlimsup_{n}a_n,\quad\varliminf_{n}a_n,\quad\varinjlim A_n,\quad\varprojlim A_n",
    r"\dddot{x}+\ddddot{x}+\mathring{x}+\pmb{x}+\boldsymbol{\theta}",
    (r"\overleftarrow{AB}+\overleftrightarrow{AB}+\underleftarrow{AB}"
     r"+\underrightarrow{AB}+\underleftrightarrow{AB}"),
    r"a\thinspace b\medspace c\thickspace d\negthinspace e\negmedspace f\negthickspace g\mspace{2mu}h",
    r"\sideset{_a^b}{_c^d}\sum_{i=1}^n x_i+\sum\displaylimits_{i=1}^n\smash[b]{x_i}",
    r"\lvert x\rvert\bigm|\lVert y\rVert\Bigm|x\biggm|y\Biggm|z,\quad\text{условие}",
    r"\sum_{\begin{subarray}{c}i<j\\j<n\end{subarray}}a_{ij}",
]
DISPLAY_BODIES = {
    "alignat": ("{2}", r"a&=b&\quad c&=d\\e&=f&g&=h"),
    "alignat*": ("{2}", r"a&=b&\quad c&=d\\e&=f&g&=h"),
    "flalign": ("", r"a&=b&&\\c&=d&&"),
    "flalign*": ("", r"a&=b&&\\c&=d&&"),
    "xalignat": ("{2}", r"a&=b&c&=d\\e&=f&g&=h"),
    "xalignat*": ("{2}", r"a&=b&c&=d\\e&=f&g&=h"),
    "xxalignat": ("{2}", r"a&=b&c&=d\\e&=f&g&=h"),
    "subequations": ("", r"\begin{align}a&=b\\c&=d\end{align}"),
}


def notes_for(blocks):
    return {"title": "Математические обозначения", "language": "ru", "glossary": [],
            "issues": [], "chapters": [{"id": "ch001", "title": "Обозначения",
                "start": 0, "end": 20, "blocks": [
                    {"id": f"block-{index}", "kind": kind, "content": content,
                     "evidence_ids": ["s1"]}
                    for index, (kind, content) in enumerate(blocks)]}]}


def display_formula(environment):
    argument, body = DISPLAY_BODIES[environment]
    return rf"\begin{{{environment}}}{argument}{body}\end{{{environment}}}"


@pytest.mark.parametrize("expression", EXPRESSIONS)
def test_public_amsmath_formulas_share_notes_and_anki_validation(expression):
    rendering._validate_commands(expression)
    prose = rf"Условие \({expression}\)."
    assert rendering._prose(prose) == prose
    for opening, closing in [("[$]", "[/$]"), ("[$$]", "[/$$]")]:
        field = opening + expression + closing
        assert normalize_latex(field).encode() == field.encode()


@pytest.mark.parametrize("environment", DISPLAY_BODIES)
def test_display_environments_keep_delimiters_and_required_arguments(tmp_path, environment):
    formula = display_formula(environment)
    root = rendering.write_project(tmp_path, notes_for([("math", formula)])).parent
    source = (root / "chapters/section-001.tex").read_text()
    assert r"\[" not in source and r"\]" not in source
    assert r"\hypertarget{shad-block-block-0}{}" in source
    assert r"\hypertarget{shad-end-block-0}{}" in source
    clean = re.sub(r"\\hypertarget\{shad-(?:block|end)-[^}]+\}\{\}", "", source)
    assert formula in clean
    if DISPLAY_BODIES[environment][0]:
        assert re.search(r"\\begin\{" + re.escape(environment) + r"\}\s*\{2\}", source)
    field = f"[latex]{formula}[/latex]"
    assert normalize_latex(field) == field


def test_subarray_is_kept_inside_display_math(tmp_path):
    formula = r"\begin{subarray}{c}i<j\\j<n\end{subarray}"
    root = rendering.write_project(tmp_path, notes_for([("math", formula)])).parent
    source = (root / "chapters/section-001.tex").read_text()
    assert source.index(r"\[") < source.index(formula) < source.index(r"\]")


@pytest.mark.parametrize("environment", ["alignat", "alignat*", "xalignat", "xalignat*", "xxalignat"])
def test_alignment_count_after_comment_precedes_source_anchor(tmp_path, environment):
    formula = display_formula(environment).replace("{2}", "% two column pairs\n{2}", 1)
    root = rendering.write_project(tmp_path, notes_for([("math", formula)])).parent
    source = (root / "chapters/section-001.tex").read_text()
    assert "% two column pairs\n{2}\\hypertarget{shad-block-block-0}{}" in source


@pytest.mark.parametrize("formula", [
    r"\input{secret}", r"\write18{echo unsafe}", r"\csname input\endcsname{secret}",
    r"\xrightarrowCustom{P}", r"\operatornameCustom{x}", r"\math@cr", r"\x@protect",
])
def test_new_math_support_does_not_allow_unsafe_custom_or_internal_commands(formula):
    with pytest.raises(rendering.RenderError, match="Unsupported or unsafe"):
        rendering._validate_commands(formula)
    with pytest.raises(ValueError, match="Unsupported or unsafe"):
        normalize_latex(f"[$]{formula}[/$]")


@pytest.mark.skipif(not shutil.which("latexmk") or not shutil.which("lualatex"),
                    reason="requires TeX Live/MacTeX")
def test_amsmath_commands_and_display_environments_compile_with_real_lualatex(
    tmp_path, chapter_latex_checker,
):
    blocks = [("text", rf"Сходимость \({EXPRESSIONS[0]}\).")]
    blocks.extend(("math", expression) for expression in EXPRESSIONS)
    blocks.extend(("math", display_formula(environment)) for environment in DISPLAY_BODIES)
    blocks.append(("math", display_formula("alignat").replace("{2}", "% columns\n{2}", 1)))
    notes = notes_for(blocks)
    original = deepcopy(notes)
    evidence = {"schema_version": 3, "title": notes["title"], "media": {"duration": 20},
                "segments": [{"id": "s1", "start": 0, "end": 20, "text": "Обозначения"}]}
    chapter_latex_checker(notes, evidence, tmp_path / "check")
    pdf = tmp_path / "check/notes/main.pdf"
    assert pdf.read_bytes().startswith(b"%PDF") and pdf.stat().st_size > 1000
    assert notes == original
    log = (pdf.parent / "compiler-output.log").read_text()
    assert "Missing character:" not in log
