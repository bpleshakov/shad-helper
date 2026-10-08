import base64
import copy
import json
import sqlite3
import tempfile
import unittest
import zipfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import genanki
import pymupdf
from anki.collection import Collection
from anki.import_export_pb2 import ImportAnkiPackageOptions, ImportAnkiPackageRequest

from shad_helper.anki import _prepare_notes
from shad_helper.anki_export import write_package
from shad_helper.anki_models import get_models
from shad_helper.anki_schema import validate_response
from shad_helper.jobs import fingerprint


def example_notes():
    return [
        {
            "guid": "basic-formula",
            "model": "Basic",
            "fields": [
                "Верхняя грань множества",
                r'[$]\sup A = a[/$]. [$$]a \geq x[/$$] <img src="supremum.png">',
            ],
            "positions": {0: 1501},
        },
        {
            "guid": "complex-definition",
            "model": "Multistep definition",
            "fields": [
                "Нормированное пространство",
                "Векторное пространство; норма.",
                "Векторное пространство над полем действительных чисел.",
                r"Норма удовлетворяет [$]\|x\| \geq 0[/$] и остальным аксиомам.",
            ] + [""] * 13,
            "positions": {15: 1502, 0: 1502, 1: 1502},
        },
        {
            "guid": "short-proof",
            "model": "Theorem Full proof",
            "fields": [
                "Единственность предела",
                "Сходящаяся последовательность имеет единственный предел.",
                "По неравенству треугольника расстояние между пределами равно нулю.",
            ],
            "positions": {0: 1503, 1: 1503},
        },
        {
            "guid": "supremum-theorem",
            "model": "Theorem step by step",
            "fields": [
                "Существование и единственность супремума",
                ("Непустое ограниченное сверху подмножество действительных чисел "
                 "имеет единственный супремум."),
                "1. Доказать существование. 2. Доказать единственность.",
                "Использовать полноту действительных чисел.",
                "Две точные верхние грани не могут быть строго больше друг друга.",
            ] + [""] * 13,
            "positions": {0: 1504, 16: 1504, 1: 1504, 2: 1504},
        },
        {
            "guid": "alternative-proofs",
            "model": "Theorem step by step",
            "fields": [
                "Альтернатива Фредгольма",
                "Условие разрешимости линейной системы через сопряжённую систему.",
                "1. Подсчёт размерностей. 2. Ортогональное дополнение. 3. Ранг матрицы.",
                "Первое доказательство: применить теорему о ранге и дефекте.",
                "Второе доказательство: использовать ортогональное дополнение ядра.",
                "Третье доказательство: сравнить ранги матрицы и расширенной матрицы.",
            ] + [""] * 12,
            "positions": {0: 1505, 16: 1505, 1: 1505, 2: 1505, 3: 1505},
        },
    ]


class AnkiImportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.collection = Collection(str(self.root / "collection.anki2"))
        self.addCleanup(self.collection.close)
        # The user's Basic has a known ID; remove the empty stock Basic to model
        # that existing collection without a second, unrelated name collision.
        self.collection.models.remove(self.collection.models.by_name("Basic")["id"])
        self.models = get_models()
        self.notes = example_notes()
        self.media = self.root / "supremum.png"
        self.media.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
            "/x8AAwMCAO+aV1sAAAAASUVORK5CYII="
        ))
        self.media_files = [self.media]
        # Small pre-rendered fixtures exercise native Anki LaTeX image lookup
        # without running external TeX commands during importer tests.
        for note in self.notes:
            for field in note["fields"]:
                for latex in self.latex_images(field):
                    image = self.root / latex.filename
                    image.write_bytes(self.media.read_bytes())
                    self.media_files.append(image)

    def latex_images(self, text):
        return self.collection._backend.extract_latex(
            text=text, svg=False, expand_clozes=False,
        ).latex

    def existing_new_card(self):
        model = self.collection.models.by_name("Basic (type in the answer)")
        note = self.collection.new_note(model)
        note["Front"] = "Ранее добавленный материал"
        note["Back"] = "Существующий ответ"
        self.collection.add_note(note, self.collection.decks.id("Existing notes"))
        card = note.cards()[0]
        card.due = 1500
        self.collection.update_card(card)
        return card

    def package(self, *, name="Лекция — job-a", notes=None, deck_id=1824000001):
        output = self.root / f"{deck_id}.apkg"
        write_package(
            output,
            deck_id=deck_id,
            deck_name=name,
            notes=self.notes if notes is None else notes,
            media_files=self.media_files,
        )
        return output

    def import_package(self, path, *, with_scheduling=True):
        return self.collection.import_anki_package(ImportAnkiPackageRequest(
            package_path=str(path),
            options=ImportAnkiPackageOptions(with_scheduling=with_scheduling),
        ))

    def positions(self):
        return {
            (guid, ordinal): due
            for guid, ordinal, due in self.collection.db.all(
                "select n.guid, c.ord, c.due from cards c join notes n on n.id = c.nid"
            )
        }

    def test_real_import_generates_only_filled_cards_in_lecture_order(self):
        existing = self.existing_new_card()
        self.import_package(self.package(), with_scheduling=False)
        expected = {
            (note["guid"], ordinal): position
            for note in self.notes
            for ordinal, position in note["positions"].items()
        }
        expected[existing.note().guid, existing.ord] = 1500
        self.assertEqual(self.positions(), expected)
        self.assertEqual(self.collection.note_count(), 6)
        self.assertEqual(self.collection.card_count(), 16)
        self.assertEqual(self.collection.get_card(existing.id).due, 1500)
        self.assertEqual(self.collection.db.scalar("select count(*) from cards where type != 0"), 0)
        for name, model in self.models.items():
            imported = self.collection.models.get(model.model_id)
            self.assertEqual(imported["name"], name)
            self.assertEqual([field["name"] for field in imported["flds"]],
                             [field["name"] for field in model.fields])
            self.assertEqual([template["name"] for template in imported["tmpls"]],
                             [template["name"] for template in model.templates])

    def test_imported_math_media_and_local_step_context_render(self):
        self.import_package(self.package())
        for media in self.media_files:
            imported_media = Path(self.collection.media.dir()) / media.name
            self.assertEqual(imported_media.read_bytes(), media.read_bytes())
        basic_card_id = self.collection.db.scalar(
            "select c.id from cards c join notes n on n.id = c.nid where n.guid = ?",
            "basic-formula",
        )
        basic = self.collection.get_card(basic_card_id)
        self.assertIn("Верхняя грань множества", basic.question())
        self.assertIn(r"[$]\sup A = a[/$]", basic.note()["Back"])
        self.assertIn(r"[$$]a \geq x[/$$]", basic.note()["Back"])
        with patch("anki.latex._save_latex_image", side_effect=AssertionError(
            "The imported package must supply pre-rendered LaTeX media"
        )):
            answer = basic.answer()
        self.assertNotIn("[$]", answer)
        self.assertNotIn("[$$]", answer)
        for latex in self.latex_images(self.notes[0]["fields"][1]):
            self.assertIn(latex.filename, answer)
        self.assertIn('<img src="supremum.png">', answer)
        step_card_id = self.collection.db.scalar(
            "select c.id from cards c join notes n on n.id = c.nid "
            "where n.guid = ? and c.ord = 2",
            "supremum-theorem",
        )
        step = self.collection.get_card(step_card_id)
        self.assertIn(self.notes[3]["fields"][0], step.question())
        self.assertIn(self.notes[3]["fields"][1], step.question())
        self.assertIn(self.notes[3]["fields"][2], step.question())
        self.assertIn(self.notes[3]["fields"][4], step.answer())

    def test_reimport_reuses_existing_models_and_preserves_templates_and_progress(self):
        existing = self.existing_new_card()
        package = self.package()
        self.import_package(package)
        original_ids = self.collection.db.all("select id, nid, ord from cards order by id")
        model_count = len(self.collection.models.all_names_and_ids())
        expected_styles = {}
        for model in self.models.values():
            local = self.collection.models.get(model.model_id)
            local["css"] += "\n/* local collection customization */"
            local["tmpls"][0]["qfmt"] += '<div class="local-customization">Моя версия</div>'
            self.collection.models.update_dict(local)
            expected_styles[model.model_id] = (local["css"], local["tmpls"][0]["qfmt"])

        card = self.collection.get_card(self.collection.db.scalar(
            "select c.id from cards c join notes n on n.id = c.nid where n.guid = ?",
            "basic-formula",
        ))
        card.type = 2
        card.queue = 2
        card.due = self.collection.sched.today + 12
        card.ivl = 12
        card.factor = 2500
        card.reps = 7
        card.lapses = 1
        self.collection.update_card(card)
        schedule = (card.type, card.queue, card.due, card.ivl, card.factor, card.reps, card.lapses)

        self.import_package(package)
        self.assertEqual(self.collection.db.all("select id, nid, ord from cards order by id"),
                         original_ids)
        self.assertEqual(self.collection.note_count(), 6)
        self.assertEqual(self.collection.get_card(existing.id).due, 1500)
        self.assertEqual(self.collection.get_card(existing.id).type, 0)
        self.assertEqual(len(self.collection.models.all_names_and_ids()), model_count)
        for model_id, (css, question) in expected_styles.items():
            model = self.collection.models.get(model_id)
            self.assertEqual(model["css"], css)
            self.assertEqual(model["tmpls"][0]["qfmt"], question)
        card = self.collection.get_card(card.id)
        self.assertEqual(
            (card.type, card.queue, card.due, card.ivl, card.factor, card.reps, card.lapses),
            schedule,
        )

    def test_same_lecture_title_in_distinct_jobs_imports_to_distinct_decks(self):
        first = self.notes[0]
        second = {**first, "guid": "other-job-basic"}
        self.import_package(self.package(name="Лекция — job-a", notes=[first]))
        self.import_package(self.package(
            name="Лекция — job-b", notes=[second], deck_id=1824000002,
        ))
        decks = [self.collection.decks.by_name(name) for name in ("Лекция — job-a", "Лекция — job-b")]
        self.assertNotEqual(decks[0]["id"], decks[1]["id"])
        self.assertEqual(self.collection.note_count(), 2)
        for deck in decks:
            self.assertEqual(self.collection.db.scalar("select count(*) from cards where did = ?",
                                                       deck["id"]), 1)

    def test_native_latex_reexport_updates_existing_mathjax_note_without_duplicates(self):
        identity = {"job_uuid": "22c3f9a1-df67-4a8c-a43b-dbbcecf3a3ca"}
        pdf_hash = "unchanged-pdf"
        front = r"Определение \(x\)"
        source = {"page": 1, "order": 1}
        legacy_key = {"source": source, "type": "Basic", "name": front}
        old_guid = genanki.guid_for(identity["job_uuid"], pdf_hash, fingerprint(legacy_key))
        old_note = {"guid": old_guid, "model": "Basic", "fields": [front, "Ответ"],
                    "positions": {0: 1501}}
        # Older modification time lets Anki's normal "if newer" import setting
        # replace the fields deterministically, without a test sleep.
        with patch("time.time", return_value=1_700_000_000):
            old_package = self.package(notes=[old_note])
        self.import_package(old_package)
        original_ids = self.collection.db.all("select id, nid, ord from cards")

        response = validate_response({"title": "Лекция", "notes": [{
            "note_type": "Basic", "front": front, "back": "Ответ", "source": source,
        }]}, page_count=1)
        with pymupdf.open() as document:
            document.new_page()
            notes, _, _ = _prepare_notes(response, document, self.root, identity, pdf_hash,
                                         start_position=1502)
        self.import_package(self.package(notes=notes))

        self.assertEqual(self.collection.note_count(), 1)
        self.assertEqual(self.collection.card_count(), 1)
        self.assertEqual(self.collection.db.all("select id, nid, ord from cards"), original_ids)
        imported = self.collection.get_note(original_ids[0][1])
        self.assertEqual(imported.guid, old_guid)
        self.assertEqual(imported["Front"], "Определение [$]x[/$]")
        self.assertEqual(self.collection.get_card(original_ids[0][0]).due, 1501)

    def test_package_next_position_follows_last_note_position(self):
        with zipfile.ZipFile(self.package()) as package:
            database = self.root / "packaged.anki2"
            database.write_bytes(package.read("collection.anki2"))
        with closing(sqlite3.connect(database)) as connection:
            config = json.loads(connection.execute("select conf from col").fetchone()[0])
            self.assertEqual(config["nextPos"], 1506)

    def test_invalid_note_positions_are_rejected_without_replacing_package(self):
        package = self.package()
        original = package.read_bytes()
        invalid = [
            (0, {0: 0}, "positive integers"),
            (0, {0: True}, "positive integers"),
            (0, {0: 1501.0}, "positive integers"),
            (1, {15: 1502, 0: 1503, 1: 1502}, "Sibling cards"),
            (1, {15: 1501, 0: 1501, 1: 1501}, "unique consecutive"),
            (4, {0: 1506, 16: 1506, 1: 1506, 2: 1506, 3: 1506}, "unique consecutive"),
        ]
        for index, positions, message in invalid:
            with self.subTest(index=index, positions=positions):
                notes = copy.deepcopy(self.notes)
                notes[index]["positions"] = positions
                with self.assertRaisesRegex(ValueError, message):
                    self.package(notes=notes)
                self.assertEqual(package.read_bytes(), original)
