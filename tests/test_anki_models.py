import unittest

import genanki

from shad_helper.anki_models import get_models


class AnkiModelTests(unittest.TestCase):
    def setUp(self):
        self.models = get_models()

    def note(self, name, values):
        model = self.models[name]
        return genanki.Note(model=model,
                           fields=[values.get(field["name"], "") for field in model.fields])

    def test_snapshot_preserves_collection_ids_and_field_ordinals(self):
        steps = [f"Step {number}" for number in range(1, 16)]
        expected = {
            "Basic": (1725350504365, ["Front", "Back"]),
            "Multistep definition": (1789601952753, ["Name", "Definition steps", *steps]),
            "Theorem Full proof": (1742373872065, ["Name", "Statement", "Full proof"]),
            "Theorem step by step": (1781264227127, ["Name", "Statement", "Steps to prove", *steps]),
        }
        self.assertEqual(set(self.models), set(expected))
        for name, (model_id, fields) in expected.items():
            with self.subTest(model=name):
                model = self.models[name]
                self.assertEqual(model.model_id, model_id)
                self.assertEqual([field["name"] for field in model.fields], fields)

    def test_snapshot_preserves_original_template_ordinals(self):
        steps = [f"Step {number}" for number in range(1, 16)]
        expected = {
            "Basic": ["Card 1"],
            "Multistep definition": [*steps, "Statement"],
            "Theorem Full proof": ["Statement", "Full proof"],
            "Theorem step by step": ["Statement", *steps, "Steps to prove"],
        }
        for name, templates in expected.items():
            with self.subTest(model=name):
                self.assertEqual([item["name"] for item in self.models[name].templates], templates)

    def test_only_nonempty_optional_fields_generate_cards(self):
        examples = [
            ("Basic", {"Front": "Матрица", "Back": "Прямоугольная таблица"}, [0]),
            ("Theorem Full proof", {"Name": "Теорема", "Statement": "Утверждение"}, [0]),
            ("Theorem Full proof", {"Name": "Теорема", "Statement": "Утверждение",
                                    "Full proof": "Доказательство"}, [0, 1]),
            ("Multistep definition", {"Name": "Объект", "Definition steps": "1. A; 2. B",
                                      "Step 1": "A", "Step 2": "B"}, [0, 1, 15]),
            ("Theorem step by step", {"Name": "Супремум", "Statement": "Утверждение",
                                      "Steps to prove": "1. Существование; 2. Единственность",
                                      "Step 1": "Доказательство 1", "Step 2": "Доказательство 2"},
             [0, 1, 2, 16]),
            ("Theorem step by step", {"Name": "Теорема", "Statement": "Утверждение"}, [0]),
        ]
        for name, values, expected in examples:
            with self.subTest(model=name, fields=list(values)):
                self.assertEqual([card.ord for card in self.note(name, values).cards], expected)

    def test_fifteenth_step_generates_its_original_ordinal(self):
        for name, expected in (("Multistep definition", [14, 15]),
                               ("Theorem step by step", [0, 15])):
            with self.subTest(model=name):
                note = self.note(name, {"Name": "Объект", "Step 15": "Ответ"})
                self.assertEqual([card.ord for card in note.cards], expected)

    def test_step_fronts_show_statement_and_complete_plan(self):
        for template in self.models["Theorem step by step"].templates[1:16]:
            front = template["qfmt"]
            self.assertIn("{{Name}}", front)
            self.assertIn("{{Statement}}", front)
            self.assertIn("{{Steps to prove}}", front)
            self.assertTrue(front.startswith("{{#" + template["name"] + "}}"))
            self.assertTrue(front.endswith("{{/" + template["name"] + "}}"))

    def test_exported_models_do_not_claim_newer_schema_or_styling(self):
        for model in self.models.values():
            with self.subTest(model=model.name):
                data = model.to_json(1_900_000_000, 1234)
                self.assertEqual(data["mod"], 0)
                self.assertEqual(data["req"], model._req)
                self.assertIn("font-family: arial;", data["css"])
                self.assertIn(r"\usepackage[english,russian]{babel}", data["latexPre"])
                self.assertEqual(data["latexPost"], r"\end{document}")
                self.assertFalse(data["latexsvg"])
                self.assertFalse(any("<script" in template["qfmt"]
                                     for template in data["tmpls"]))

    def test_get_models_does_not_share_mutable_models(self):
        self.models["Basic"].fields[0]["name"] = "Changed"
        self.models["Theorem Full proof"].templates[0]["qfmt"] = "Changed"
        fresh = get_models()
        self.assertEqual(fresh["Basic"].fields[0]["name"], "Front")
        self.assertEqual(fresh["Theorem Full proof"].templates[0]["qfmt"],
                         "{{Name}}<br>\nСформулировать теорему.")
