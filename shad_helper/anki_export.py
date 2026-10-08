"""Portable APKG creation; positions change only the package's scratch database."""

import json
import os
import tempfile
from contextlib import closing
from pathlib import Path

import genanki

from .anki_models import get_models


class _PositionedPackage(genanki.Package):
    def __init__(self, deck, media_files, positions):
        super().__init__(deck, media_files=media_files)
        self.positions = positions

    def write_to_db(self, cursor, timestamp, id_gen):
        super().write_to_db(cursor, timestamp, id_gen)
        rows = cursor.execute(
            "SELECT cards.id, notes.guid, cards.ord FROM cards "
            "JOIN notes ON notes.id = cards.nid").fetchall()
        expected = {(guid, ordinal): position for guid, values in self.positions.items()
                    for ordinal, position in values.items()}
        if {(guid, ordinal) for _, guid, ordinal in rows} != set(expected):
            raise ValueError("Exported cards do not match their planned positions")
        note_positions = []
        for values in self.positions.values():
            if not values or any(type(position) is not int or position < 1
                                 for position in values.values()):
                raise ValueError("New-card positions must be positive integers")
            if len(set(values.values())) != 1:
                raise ValueError("Sibling cards must share their note's position")
            note_positions.append(next(iter(values.values())))
        if not note_positions:
            raise ValueError("Anki package requires at least one note with cards")
        ordered = sorted(note_positions)
        if ordered != list(range(ordered[0], ordered[0] + len(ordered))):
            raise ValueError("Distinct notes must have unique consecutive positions")
        cursor.executemany("UPDATE cards SET due=? WHERE id=?",
                           [(expected[guid, ordinal], cid) for cid, guid, ordinal in rows])
        config = json.loads(cursor.execute("SELECT conf FROM col").fetchone()[0])
        config["nextPos"] = ordered[-1] + 1
        cursor.execute("UPDATE col SET conf=?", (json.dumps(config),))

    def write_to_file(self, file, timestamp=None):
        # genanki leaves its scratch SQLite file behind; confine it to a temporary folder.
        with tempfile.TemporaryDirectory(prefix="shad-anki-db-") as folder:
            import itertools
            import sqlite3
            import time
            import zipfile

            stamp = time.time() if timestamp is None else timestamp
            database = Path(folder) / "collection.anki2"
            with closing(sqlite3.connect(database)) as connection:
                self.write_to_db(connection.cursor(), stamp,
                                 itertools.count(int(stamp * 1000)))
                connection.commit()
            with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED) as package:
                package.write(database, "collection.anki2")
                media = dict(enumerate(sorted(self.media_files, key=lambda path: path.name)))
                package.writestr("media", json.dumps({i: path.name for i, path in media.items()}))
                for i, path in media.items():
                    package.write(path, str(i))


def write_package(output: Path, *, deck_id: int, deck_name: str,
                  notes: list[dict], media_files: list[Path]) -> None:
    models = get_models()
    deck = genanki.Deck(deck_id, deck_name)
    positions = {}
    for model in models.values():
        deck.add_model(model)
    for item in notes:
        if item["guid"] in positions:
            raise ValueError("Duplicate Anki note GUID")
        positions[item["guid"]] = item["positions"]
        deck.add_note(genanki.Note(model=models[item["model"]], fields=item["fields"],
                                  guid=item["guid"], tags=item.get("tags", [])))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".deck-", suffix=".apkg", dir=output.parent)
    os.close(fd)
    try:
        _PositionedPackage(deck, media_files, positions).write_to_file(temporary)
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
