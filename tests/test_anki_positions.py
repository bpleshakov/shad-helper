import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shad_helper.anki_positions import AnkiPositionError, collection_position


class AnkiPositionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()

    def collection(self, path=None, *, modern=True, next_position=1, cards=()):
        path = path or self.root / "collection.anki2"
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE cards (type integer, queue integer, due integer, "
                               "odue integer, odid integer)")
            connection.executemany("INSERT INTO cards VALUES (?, ?, ?, ?, ?)", cards)
            connection.execute("CREATE TABLE col (conf text)")
            if modern:
                connection.execute("CREATE TABLE config (key text PRIMARY KEY, val blob)")
                connection.execute("INSERT INTO config VALUES ('nextPos', ?)",
                                   (json.dumps(next_position).encode(),))
                connection.execute("INSERT INTO col VALUES ('')")
            else:
                connection.execute("INSERT INTO col VALUES (?)",
                                   (json.dumps({"nextPos": next_position}),))
        return path

    def test_suspended_buried_and_filtered_new_positions_are_included(self):
        cases = [
            ((0, 0, 20, 0, 0), 20), ((0, -1, 40, 0, 0), 40),
            ((0, -2, 60, 0, 0), 60), ((0, -3, 80, 0, 0), 80),
            ((0, 4, -1000, 100, 15), 100), ((0, 4, 10000, 0, 15), 0),
        ]
        for index, (card, maximum) in enumerate(cases):
            with self.subTest(card=card):
                path = self.collection(self.root / f"case-{index}.anki2", cards=[
                    card, (1, 1, 1900000000, 0, 0), (2, 2, 20000, 0, 0),
                ])
                result = collection_position(path)
                self.assertEqual(result, {"collection_path": str(path), "max_position": maximum,
                                          "next_position": maximum + 1})

    def test_modern_next_position_retains_already_allocated_positions(self):
        path = self.collection(next_position=200, cards=[(0, 0, 100, 0, 0)])
        self.assertEqual(collection_position(path)["next_position"], 200)

    def test_legacy_configuration_is_supported(self):
        path = self.collection(modern=False, next_position=200, cards=[(0, 0, 100, 0, 0)])
        self.assertEqual(collection_position(path)["next_position"], 200)

    def test_empty_valid_collection_starts_at_one(self):
        path = self.collection(next_position=0)
        self.assertEqual(collection_position(path)["max_position"], 0)
        self.assertEqual(collection_position(path)["next_position"], 1)

    def test_wal_commits_are_visible_and_read_does_not_change_collection(self):
        path = self.collection(next_position=10, cards=[(0, 0, 5, 0, 0)])
        with sqlite3.connect(path) as writer:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute("UPDATE cards SET due=400")
            writer.execute("UPDATE config SET val=? WHERE key='nextPos'", (b"500",))
            writer.commit()
            self.assertTrue(Path(str(path) + "-wal").exists())
            before = path.read_bytes(), Path(str(path) + "-wal").read_bytes()
            result = collection_position(path)
            self.assertEqual(result["max_position"], 400)
            self.assertEqual(result["next_position"], 500)
            self.assertEqual(before, (path.read_bytes(), Path(str(path) + "-wal").read_bytes()))

    def test_unique_profile_is_discovered(self):
        path = self.collection(self.root / "Library/Application Support/Anki2/User 1/collection.anki2")
        with patch("shad_helper.anki_positions.Path.home", return_value=self.root):
            self.assertEqual(collection_position()["collection_path"], str(path))

    def test_missing_and_ambiguous_profiles_require_explicit_path(self):
        with patch("shad_helper.anki_positions.Path.home", return_value=self.root):
            with self.assertRaisesRegex(AnkiPositionError, "No local Anki collection"):
                collection_position()
            for profile in ["First", "Second"]:
                self.collection(self.root / "Library/Application Support/Anki2" / profile
                                / "collection.anki2")
            with self.assertRaisesRegex(AnkiPositionError, "Multiple Anki collections"):
                collection_position()

    def test_missing_explicit_path_is_not_created(self):
        path = self.root / "missing.anki2"
        with self.assertRaisesRegex(AnkiPositionError, "does not exist"):
            collection_position(path)
        self.assertFalse(path.exists())

    def test_invalid_database_and_configuration_fail_without_default_position(self):
        path = self.root / "invalid.anki2"
        path.write_text("not a database")
        with self.assertRaisesRegex(AnkiPositionError, "Cannot read Anki collection"):
            collection_position(path)
        path = self.collection()
        for invalid in [b"broken", b"null", b"-1", b"true", b'"500"']:
            with self.subTest(value=invalid):
                with sqlite3.connect(path) as connection:
                    connection.execute("UPDATE config SET val=?", (invalid,))
                with self.assertRaises(AnkiPositionError):
                    collection_position(path)

    def test_missing_config_tables_fail_without_default_position(self):
        path = self.collection()
        with sqlite3.connect(path) as connection:
            connection.execute("DROP TABLE config")
            connection.execute("DROP TABLE col")
        with self.assertRaisesRegex(AnkiPositionError, "configuration is missing"):
            collection_position(path)
