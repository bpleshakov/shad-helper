"""Snapshot of the user's four existing Anki note types, including their ordinals."""

from __future__ import annotations

import genanki

_CSS = """.card {
    font-family: arial;
    font-size: 20px;
    text-align: center;
    color: black;
    background-color: white;
}
"""
_LATEX_PRE = r"""\documentclass[12pt]{article}
\special{papersize=3in,5in}
\usepackage{amssymb,amsmath}
\usepackage{dsfont}
\usepackage{mathtext,mathrsfs}
\usepackage{yhmath}
\usepackage[T1,T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[english,russian]{babel}
\usepackage[all]{xy}
\pagestyle{empty}
\setlength{\parindent}{0in}
\begin{document}
"""
_STEP_FIELDS = [f"Step {number}" for number in range(1, 16)]


class _SnapshotModel(genanki.Model):
    def __init__(self, model_id, name, fields, templates):
        super().__init__(
            model_id, name, fields=[{"name": field} for field in fields],
            templates=templates, css=_CSS, latex_pre=_LATEX_PRE,
            latex_post=r"\end{document}",
        )
    def to_json(self, timestamp, deck_id):
        result = super().to_json(timestamp, deck_id)
        # Import should reuse, rather than update, the user's existing note types.
        result["mod"] = 0
        return result


def _template(name: str, front: str, answer: str) -> dict[str, str]:
    return {"name": name, "qfmt": front,
            "afmt": "{{FrontSide}}\n\n<hr id=answer>\n\n{{" + answer + "}}"}


def get_models() -> dict[str, genanki.Model]:
    """Return fresh models with the live collection's IDs and original schema."""
    statement = _template(
        "Statement", "{{Name}}<br>\nСформулировать теорему.", "Statement",
    )
    definition_steps = [
        _template(field, "{{#" + field + "}}\n{{Name}}<br>\n{{Definition steps}}<br>\n"
                  f"Пункт {number}.\n" + "{{/" + field + "}}", field)
        for number, field in enumerate(_STEP_FIELDS, 1)
    ]
    proof_steps = [
        _template(field, "{{#" + field + "}}\n{{Name}}<br>\n{{Statement}}<br><br>\n"
                  "Шаги доказательства: <br>\n{{Steps to prove}}<br>\n"
                  f"Доказать шаг {number}.\n" + "{{/" + field + "}}", field)
        for number, field in enumerate(_STEP_FIELDS, 1)
    ]
    models = [
        _SnapshotModel(
            1725350504365, "Basic", ["Front", "Back"],
            [_template("Card 1", "{{Front}}", "Back")],
        ),
        _SnapshotModel(
            1789601952753, "Multistep definition", ["Name", "Definition steps", *_STEP_FIELDS],
            [*definition_steps, _template(
                "Statement", "{{Name}}<br>\nСформулировать пункты определения.", "Definition steps",
            )],
        ),
        _SnapshotModel(
            1742373872065, "Theorem Full proof", ["Name", "Statement", "Full proof"],
            [statement.copy(), _template(
                "Full proof", "{{#Full proof}}\n{{Name}}<br>\n{{Statement}}<br>\n"
                "Полное доказательство.\n{{/Full proof}}", "Full proof",
            )],
        ),
        _SnapshotModel(
            1781264227127, "Theorem step by step",
            ["Name", "Statement", "Steps to prove", *_STEP_FIELDS],
            [statement.copy(), *proof_steps, _template(
                "Steps to prove", "{{#Steps to prove}}\n{{Name}}<br>\n{{Statement}} <br><br>\n"
                "Назвать шаги для доказательства теоремы.\n{{/Steps to prove}}", "Steps to prove",
            )],
        ),
    ]
    return {model.name: model for model in models}
