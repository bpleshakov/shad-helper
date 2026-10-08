"""Read the next new-card position from the current local Anki collection."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path


class AnkiPositionError(RuntimeError):
    """The local collection's allocation position cannot be read reliably."""


def _collection_path(path: Path | None) -> Path:
    if path is not None:
        selected = path.expanduser().resolve()
        if not selected.is_file():
            raise AnkiPositionError(f"Anki collection does not exist: {selected}")
        return selected
    root = Path.home() / "Library/Application Support/Anki2"
    candidates = sorted({item.resolve() for item in root.glob("*/collection.anki2")
                         if item.is_file()})
    if not candidates:
        raise AnkiPositionError("No local Anki collection found; specify --anki-collection")
    if len(candidates) != 1:
        raise AnkiPositionError("Multiple Anki collections found; specify --anki-collection")
    return candidates[0]


def _configured_next_position(connection: sqlite3.Connection) -> int:
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    value = None
    configured = False
    if "config" in tables:
        row = connection.execute("SELECT val FROM config WHERE key = 'nextPos'").fetchone()
        if row is not None:
            value = json.loads(row[0])
            configured = True
    if not configured:
        row = connection.execute("SELECT conf FROM col").fetchone() if "col" in tables else None
        if row and row[0]:
            config = json.loads(row[0])
            if not isinstance(config, dict):
                raise AnkiPositionError("Anki collection configuration must be a JSON object")
            value = config.get("nextPos", 0)
        elif "config" in tables:
            value = 0  # Anki's default for a valid collection with no nextPos setting.
        else:
            raise AnkiPositionError("Anki collection configuration is missing")
    if type(value) is not int or value < 0:
        raise AnkiPositionError("Anki nextPos must be a nonnegative integer")
    return value


def collection_position(path: Path | None = None) -> dict[str, str | int]:
    """Read a consistent WAL-aware snapshot; this does not reserve any positions."""
    selected = _collection_path(path)
    try:
        with closing(sqlite3.connect(selected.as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            # type, unlike queue, retains new-card status when suspended or buried.
            # Filtered decks temporarily replace due; odue retains the original position.
            row = connection.execute(
                "SELECT max(CASE WHEN odid != 0 THEN odue ELSE due END) "
                "FROM cards WHERE type = 0"
            ).fetchone()
            maximum = row[0] if row[0] is not None else 0
            if type(maximum) is not int:
                raise AnkiPositionError("Anki new-card positions must be integers")
            maximum = max(0, maximum)
            next_position = max(1, maximum + 1, _configured_next_position(connection))
    except (sqlite3.Error, ValueError, TypeError) as error:
        raise AnkiPositionError(f"Cannot read Anki collection {selected}: {error}") from error
    return {"collection_path": str(selected), "max_position": maximum,
            "next_position": next_position}
