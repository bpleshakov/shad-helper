"""PDF-to-Anki stage checks with real PDF crops and portable package inspection."""

import copy
import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import genanki
import pymupdf
import pytest

from shad_helper import anki
from shad_helper.anki_schema import validate_response
from shad_helper.auth import PlanAccessError
from shad_helper.jobs import fingerprint, read_json, write_json
from shad_helper.openai_client import DEFAULT_MODEL, ResponseError


def write_pdf(path, title="PDF source"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as document:
        page = document.new_page(width=200, height=100)
        page.draw_rect((0, 0, 100, 100), color=None, fill=(1, 0, 0))
        page.draw_rect((100, 0, 200, 100), color=None, fill=(0, 0, 1))
        page.insert_text((12, 18), title, fontsize=8)
        document.new_page(width=200, height=100).insert_text((12, 18), "Second page")
        path.write_bytes(document.tobytes())
    return path


def make_job(path):
    write_json(path / "evidence.json", {
        "schema_version": 3, "title": "PRIVATE_EVIDENCE_TITLE", "media": {"duration": 120},
        "segments": [{"id": "s1", "start": 0, "end": 120,
                      "text": "PRIVATE_EVIDENCE_TEXT"}],
    })
    write_pdf(path / "notes" / "main.pdf")
    return path


@pytest.fixture
def job(tmp_path):
    return make_job(tmp_path / "lecture")


@pytest.fixture
def response():
    return {"title": "Математическая лекция", "notes": [
        {"note_type": "Basic", "front": "Положительные числа",
         "back": "Строка & смысл\n" + r"\(0 < x < y \Rightarrow x^2 < y^2\)",
         "source": {"page": 1, "order": 1}},
        {"note_type": "Multistep definition", "name": "Нормированное пространство",
         "overview": "1. Векторное пространство. 2. Норма.",
         "steps": ["Задать сложение и умножение на скаляр.", "Перечислить аксиомы нормы."],
         "source": {"page": 1, "order": 2}},
        {"note_type": "Theorem Full proof", "name": "Единственность предела",
         "statement": "Предел сходящейся последовательности единственен.",
         "proof": "Применить неравенство треугольника к двум пределам.",
         "source": {"page": 1, "order": 3}},
        {"note_type": "Theorem step by step", "name": "Супремум",
         "statement": "У непустого ограниченного сверху множества есть единственный супремум.",
         "plan": "1. Существование. 2. Единственность.",
         "steps": ["Применить полноту действительных чисел.",
                   "Сравнить две наименьшие верхние грани."],
         "source": {"page": 2, "order": 1}},
    ], "warnings": []}


@pytest.fixture
def client(response):
    return SimpleNamespace(model=DEFAULT_MODEL, complete=Mock(return_value=response))


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    sleeper = Mock()
    monkeypatch.setattr(anki.time, "sleep", sleeper)
    return sleeper


@pytest.fixture(autouse=True)
def collection_reader(monkeypatch, tmp_path):
    reader = Mock(return_value={"collection_path": str(tmp_path / "collection.anki2"),
                                "max_position": 0, "next_position": 1})
    monkeypatch.setattr(anki, "collection_position", reader)
    return reader


@pytest.fixture(autouse=True)
def no_latex_compile(monkeypatch):
    renderer = Mock(return_value=[])
    monkeypatch.setattr(anki, "render_latex_media", renderer)
    return renderer


def inspect_package(path):
    with zipfile.ZipFile(path) as package:
        media_map = json.loads(package.read("media"))
        media = {name: package.read(index) for index, name in media_map.items()}
        connection = sqlite3.connect(":memory:")
        try:
            connection.deserialize(package.read("collection.anki2"))
            rows = connection.execute(
                "SELECT n.guid,n.flds,c.ord,c.due,c.type,c.queue,n.tags "
                "FROM cards c JOIN notes n ON n.id=c.nid ORDER BY c.due"
            ).fetchall()
        finally:
            connection.close()
    return rows, media


def test_stage_exports_all_types_in_pdf_order_with_shared_note_positions(job, response, client):
    response["notes"] = [response["notes"][i] for i in (3, 1, 0, 2)]
    package = anki.generate_anki(job, job / "auth", client=client)
    rows, _ = inspect_package(package)
    assert {(fields.split("\x1f")[0], ordinal): due
            for _, fields, ordinal, due, *_ in rows} == {
        ("Положительные числа", 0): 1,
        ("Нормированное пространство", 15): 2,
        ("Нормированное пространство", 0): 2,
        ("Нормированное пространство", 1): 2,
        ("Единственность предела", 0): 3, ("Единственность предела", 1): 3,
        ("Супремум", 0): 4, ("Супремум", 16): 4,
        ("Супремум", 1): 4, ("Супремум", 2): 4,
    }
    assert all((row[4], row[5]) == (0, 0) for row in rows)
    report = read_json(job / "anki" / "report.json")
    assert (report["note_count"], report["card_count"], report["page_count"]) == (4, 10, 2)
    assert [(note["source"]["page"], note["source"]["order"]) for note in report["notes"]] == [
        (1, 1), (1, 2), (1, 3), (2, 1),
    ]


def test_row_intervals_are_repaired_in_fields_and_cache_before_compilation(
        job, response, client, no_latex_compile):
    numeric = r"[$$]\begin{cases}1,&x>0,\\[-1,1],&x=0.\end{cases}[/$$]"
    symbolic = r"[$$]\begin{cases}\lambda,&x>0,\\[-\lambda,\lambda],&x=0.\end{cases}[/$$]"
    response["notes"][0]["back"] = numeric
    response["notes"][3]["steps"][0] = symbolic
    package = anki.generate_anki(job, job / "auth", client=client)
    rows, _ = inspect_package(package)
    guids = {row[0] for row in rows}
    assert any(r"\\{[-1,1]}" in row[1] for row in rows)
    assert any(r"\\{[-\lambda,\lambda]}" in row[1] for row in rows)
    compiled_notes = no_latex_compile.call_args.args[0]
    assert r"\\{[-1,1]}" in compiled_notes[0]["fields"][1]
    cached = read_json(job / "checkpoints/anki.json")
    assert r"\\{[-1,1]}" in cached["result"]["notes"][0]["back"]
    assert r"\\{[-\lambda,\lambda]}" in cached["result"]["notes"][3]["steps"][0]

    # Older cached responses get the same repair without another model request.
    cached["result"]["notes"][0]["back"] = numeric
    cached["result"]["notes"][3]["steps"][0] = symbolic
    write_json(job / "checkpoints/anki.json", cached)
    client.complete.reset_mock()
    client.complete.side_effect = AssertionError("Unexpected model request")
    package = anki.generate_anki(job, job / "auth", client=client)
    client.complete.assert_not_called()
    rows, _ = inspect_package(package)
    assert {row[0] for row in rows} == guids
    assert any(r"\\{[-1,1]}" in row[1] for row in rows)
    assert any(r"\\{[-\lambda,\lambda]}" in row[1] for row in rows)


def test_stage_positions_follow_existing_collection_and_record_source(
        job, client, collection_reader, tmp_path):
    collection = tmp_path / "selected-profile" / "collection.anki2"
    source = {"collection_path": str(collection), "max_position": 3882, "next_position": 3883}
    collection_reader.return_value = source

    package = anki.generate_anki(job, job / "auth", client=client, anki_collection=collection)

    rows, _ = inspect_package(package)
    positions = {}
    for guid, _, _, due, *_ in rows:
        positions.setdefault(guid, set()).add(due)
    assert sorted(positions.values(), key=min) == [{3883}, {3884}, {3885}, {3886}]
    report = read_json(job / "anki" / "report.json")
    assert report["position_source"] == source
    assert (report["start_position"], report["end_position"]) == (3883, 3886)
    assert [set(note["positions"].values()) for note in report["notes"]] == [
        {3883}, {3884}, {3885}, {3886},
    ]
    assert collection_reader.call_args.args == (collection,)


def test_latex_conversion_preserves_previously_exported_note_identity(job, response, client):
    old_front = r"Почему \(x\) удовлетворяет \[x^2 \geq 0\]?"
    response["notes"][0]["front"] = old_front
    package = anki.generate_anki(job, job / "auth", client=client)
    report = read_json(job / "anki" / "report.json")
    original_key = {"source": {"page": 1, "order": 1}, "type": "Basic", "name": old_front}
    original_guid = genanki.guid_for(report["job_uuid"], report["pdf_sha256"],
                                    fingerprint(original_key))
    first = inspect_package(package)[0][0]
    assert first[0] == original_guid
    assert first[1].split("\x1f")[0] == r"Почему [$]x[/$] удовлетворяет [$$]x^2 \geq 0[/$$]?"


def test_missing_collection_aborts_before_model_without_position_fallback(
        job, client, collection_reader):
    collection_reader.side_effect = ValueError("No Anki collection found")

    with pytest.raises(ValueError, match="No Anki collection found"):
        anki.generate_anki(job, job / "auth", client=client)

    client.complete.assert_not_called()
    assert not (job / "anki" / "deck.apkg").exists()
    assert not (job / "checkpoints" / "anki.json").exists()


def test_positions_refresh_after_generation_and_latex_rendering(job, client, collection_reader):
    initial = collection_reader.return_value
    latest = {**initial, "max_position": 4200, "next_position": 4201}
    collection_reader.side_effect = [initial, latest]

    package = anki.generate_anki(job, job / "auth", client=client)

    assert {row[3] for row in inspect_package(package)[0]} == {4201, 4202, 4203, 4204}
    report = read_json(job / "anki" / "report.json")
    assert report["position_source"] == latest
    assert (report["start_position"], report["end_position"]) == (4201, 4204)


@pytest.mark.parametrize(("minimum_position", "expected_start"), [
    (4000, 4201), (5000, 5000),
])
def test_position_floor_respects_refreshed_collection_without_changing_source(
        job, client, collection_reader, minimum_position, expected_start):
    initial = {**collection_reader.return_value, "max_position": 3882, "next_position": 3883}
    latest = {**initial, "max_position": 4200, "next_position": 4201}
    collection_reader.side_effect = [initial, latest]

    package = anki.generate_anki(job, job / "auth", client=client,
                                 minimum_position=minimum_position)

    rows, _ = inspect_package(package)
    by_note = {}
    for guid, _, _, due, *_ in rows:
        by_note.setdefault(guid, set()).add(due)
    assert sorted(by_note.values(), key=min) == [
        {position} for position in range(expected_start, expected_start + 4)]
    report = read_json(job / "anki" / "report.json")
    assert report["position_source"] == latest
    assert report["position_source"]["next_position"] == 4201
    assert (report["start_position"], report["end_position"]) == (
        expected_start, expected_start + 3)
    assert [set(note["positions"].values()) for note in report["notes"]] == [
        {position} for position in range(expected_start, expected_start + 4)]
    assert collection_reader.call_count == 2


@pytest.mark.parametrize("minimum_position", [True, False, 0, -1, 1.5, "3", None])
def test_invalid_position_floor_aborts_before_model_without_export(
        job, client, collection_reader, minimum_position):
    with pytest.raises(ValueError, match="minimum_position"):
        anki.generate_anki(job, job / "auth", client=client,
                           minimum_position=minimum_position)

    client.complete.assert_not_called()
    collection_reader.assert_not_called()
    assert not (job / "anki").exists()
    assert not (job / "checkpoints" / "anki.json").exists()


def test_source_pdf_bytes_are_authoritative_and_frozen_for_model(job, client, tmp_path):
    original = write_pdf(tmp_path / "manual.pdf", "Manually corrected PDF")
    source_bytes = original.read_bytes()
    (job / "evidence.json").write_text("PRIVATE_INVALID_EVIDENCE_JSON")
    (job / "extraction-config.json").write_text("PRIVATE_OBSOLETE_EXTRACTION_CONFIG")
    (job / "notes.json").write_text("PRIVATE_INVALID_NOTES_JSON")
    write_json(job / "review.json", {"issues": ["PRIVATE_REVIEW_ISSUES"]})
    captured = {}
    reply = client.complete.return_value

    def complete(instructions, text, *, schema, pdf):
        captured.update(instructions=instructions, text=text, schema=schema,
                        data=Path(pdf).read_bytes(), filename=Path(pdf).name)
        assert Path(pdf) != original
        original.write_bytes(b"Source changed during the model request")
        return reply

    client.complete.side_effect = complete
    anki.generate_anki(job, job / "auth", pdf=original, client=client)
    assert captured["data"] == source_bytes
    assert captured["filename"] == "manual.pdf"
    assert "PRIVATE_" not in json.dumps(captured, default=str)
    report = read_json(job / "anki" / "report.json")
    assert report["pdf_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert report["source_pdf"] == str(original.resolve())
    assert (job / "evidence.json").read_text() == "PRIVATE_INVALID_EVIDENCE_JSON"
    assert (job / "extraction-config.json").read_text() == "PRIVATE_OBSOLETE_EXTRACTION_CONFIG"


def test_standalone_pdf_job_can_be_repeated_offline_and_rebuilt(tmp_path, client, monkeypatch):
    source = write_pdf(tmp_path / "standalone.pdf")
    job = tmp_path / "new-job"
    assert not job.exists()
    package = anki.generate_anki(job, tmp_path / "auth", pdf=source, client=client)
    identity = read_json(job / "anki" / "identity.json")
    monkeypatch.setattr(anki, "PlanClient", Mock(side_effect=AssertionError("Unexpected auth")))

    assert anki.generate_anki(job, tmp_path / "missing-auth", pdf=source) == package
    assert anki.generate_anki(job, tmp_path / "auth", pdf=source,
                              client=client, rebuild=True) == package

    assert client.complete.call_count == 2
    assert read_json(job / "anki" / "identity.json") == identity
    assert {path.name for path in job.iterdir()} == {"anki", "checkpoints"}


def test_cache_repackages_without_model_or_auth_and_keeps_identity(job, client, monkeypatch):
    package = anki.generate_anki(job, job / "auth", client=client)
    first_guids = {row[0] for row in inspect_package(package)[0]}
    identity = read_json(job / "anki" / "identity.json")
    package.unlink()
    monkeypatch.setattr(anki, "PlanClient", Mock(side_effect=AssertionError("Unexpected auth")))

    regenerated = anki.generate_anki(job, job / "unavailable-auth")

    assert regenerated.is_file()
    assert {row[0] for row in inspect_package(regenerated)[0]} == first_guids
    assert read_json(job / "anki" / "identity.json") == identity
    client.complete.assert_called_once()


@pytest.mark.parametrize("change", ["rebuild", "pdf", "model", "prompt"])
def test_generation_restarts_when_required_but_reuses_job_identity(
        job, client, monkeypatch, change):
    anki.generate_anki(job, job / "auth", client=client)
    identity = read_json(job / "anki" / "identity.json")
    previous_hash = read_json(job / "anki" / "report.json")["pdf_sha256"]
    if change == "pdf":
        write_pdf(job / "notes" / "main.pdf", "Corrected material")
    elif change == "model":
        client.model = "another-available-model"
    elif change == "prompt":
        monkeypatch.setattr(anki, "PROMPT_VERSION", anki.PROMPT_VERSION + 1)

    anki.generate_anki(job, job / "auth", client=client, rebuild=change == "rebuild")

    assert client.complete.call_count == 2
    assert read_json(job / "anki" / "identity.json") == identity
    report = read_json(job / "anki" / "report.json")
    assert (report["pdf_sha256"] != previous_hash) is (change == "pdf")


def test_same_title_in_different_jobs_has_distinct_decks_and_guids(job, client, tmp_path):
    second = make_job(tmp_path / "other-lecture")
    first_package = anki.generate_anki(job, job / "auth", client=client)
    second_package = anki.generate_anki(second, second / "auth", client=client)
    first = read_json(job / "anki" / "identity.json")
    other = read_json(second / "anki" / "identity.json")
    for field in ("job_uuid", "deck_id", "deck_name"):
        assert first[field] != other[field]
    assert {row[0] for row in inspect_package(first_package)[0]}.isdisjoint(
        {row[0] for row in inspect_package(second_package)[0]})


@pytest.mark.parametrize("kind", ["missing", "invalid", "corrupt", "encrypted", "oversized"])
def test_bad_source_fails_before_model_and_does_not_publish(job, client, monkeypatch, kind):
    source = job / "notes" / "main.pdf"
    if kind == "missing":
        source.unlink()
    elif kind == "invalid":
        source.write_bytes(b"This is not a PDF")
    elif kind == "corrupt":
        source.write_bytes(b"%PDF-1.7\ntruncated")
    elif kind == "encrypted":
        with pymupdf.open() as document:
            document.new_page()
            source.write_bytes(document.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                owner_pw="owner", user_pw="reader"))
    else:
        monkeypatch.setattr(anki, "MAX_PDF_BYTES", source.stat().st_size)
    with pytest.raises((ValueError, RuntimeError)):
        anki.generate_anki(job, job / "auth", client=client)
    client.complete.assert_not_called()
    assert not (job / "anki" / "deck.apkg").exists()
    assert not (job / "checkpoints" / "anki.json").exists()


@pytest.mark.parametrize("failure", [
    "invalid_json", "response_error", "package_error", "latex_error", "collection_refresh_error",
])
def test_failed_regeneration_keeps_previous_published_artifacts(
        job, client, monkeypatch, no_retry_wait, collection_reader, failure):
    anki.generate_anki(job, job / "auth", client=client)
    previous = {path.name: path.read_bytes() for path in (job / "anki").iterdir()
                if path.is_file()}
    client.complete.reset_mock()
    if failure == "invalid_json":
        client.complete.return_value = {"title": "Invalid", "notes": []}
        expected_calls = 4
    elif failure == "response_error":
        client.complete.side_effect = ResponseError("Invalid response JSON")
        expected_calls = 4
    elif failure == "latex_error":
        monkeypatch.setattr(anki, "render_latex_media", Mock(side_effect=RuntimeError("TeX failed")))
        expected_calls = 1
    elif failure == "collection_refresh_error":
        collection_reader.side_effect = [collection_reader.return_value,
                                         RuntimeError("Collection is no longer available")]
        expected_calls = 1
    else:
        monkeypatch.setattr(anki, "write_package", Mock(side_effect=OSError("Disk unavailable")))
        expected_calls = 1

    with pytest.raises((ValueError, RuntimeError, OSError)):
        anki.generate_anki(job, job / "auth", client=client, rebuild=True)

    assert client.complete.call_count == expected_calls
    assert {path.name: path.read_bytes() for path in (job / "anki").iterdir()
            if path.is_file()} == previous
    assert not list((job / "anki").glob(".build-*"))
    assert [call.args[0] for call in no_retry_wait.call_args_list] == (
        [5, 10, 20] if expected_calls == 4 else [])


@pytest.mark.parametrize("existing", [False, True])
def test_failed_final_package_publication_rolls_back_sidecars(tmp_path, monkeypatch, existing):
    folder, staging = tmp_path / "anki", tmp_path / "staging"
    folder.mkdir()
    staging.mkdir()
    names = ("notes.json", "report.json", "deck.apkg")
    for name in names:
        (staging / name).write_bytes(f"new {name}".encode())
        if existing:
            (folder / name).write_bytes(f"old {name}".encode())
    original_replace = anki.os.replace
    failures = []

    def fail_package_once(source, destination):
        if Path(source) == staging / "deck.apkg" and not failures:
            failures.append(destination)
            raise OSError("Package replacement failed")
        return original_replace(source, destination)

    monkeypatch.setattr(anki.os, "replace", fail_package_once)
    with pytest.raises(OSError, match="Package replacement failed"):
        anki._publish(staging, folder)

    assert failures == [folder / "deck.apkg"]
    for name in names:
        if existing:
            assert (folder / name).read_bytes() == f"old {name}".encode()
        else:
            assert not (folder / name).exists()


@pytest.mark.parametrize("error", [
    ResponseError("Too much context", code="context_length_exceeded"),
    PlanAccessError("Unauthorized", status=401),
    PlanAccessError("Forbidden", status=403),
    PlanAccessError("Unavailable", code="model_unavailable"),
])
def test_terminal_model_errors_do_not_retry(job, client, no_retry_wait, error):
    client.complete.side_effect = error
    with pytest.raises(RuntimeError):
        anki.generate_anki(job, job / "auth", client=client)
    client.complete.assert_called_once()
    no_retry_wait.assert_not_called()
    assert not (job / "anki" / "deck.apkg").exists()


def test_valid_reply_after_retry_is_cached_and_published(job, client, response, no_retry_wait):
    client.complete.side_effect = [{"notes": []}, response]
    assert anki.generate_anki(job, job / "auth", client=client).is_file()
    assert client.complete.call_count == 2
    no_retry_wait.assert_called_once_with(5)
    assert len(read_json(job / "checkpoints" / "anki.json")["result"]["notes"]) == 4


def test_pdf_crop_pixels_math_and_supplements_survive_packaging(job, client, response):
    response["notes"][0]["figures"] = [
        {"field": "Back", "page": 1, "box": [0, 0, 0.5, 1]},
    ]
    response["notes"][2]["supplemented_fields"] = ["Full proof"]
    response["notes"][2]["review_flags"] = ["Проверить реконструированное рассуждение."]
    response["warnings"] = ["Есть восстановленное доказательство."]
    package = anki.generate_anki(job, job / "auth", client=client)
    rows, media = inspect_package(package)
    assert len(media) == 1
    name, image = next(iter(media.items()))
    pixmap = pymupdf.Pixmap(image)
    assert 277 <= pixmap.width <= 279 and 277 <= pixmap.height <= 279
    assert pixmap.pixel(pixmap.width // 2, pixmap.height // 2) == (255, 0, 0)
    assert (job / "anki" / "media" / name).read_bytes() == image
    back = rows[0][1].split("\x1f")[1]
    assert f'<img src="{name}">' in back
    assert "Строка &amp; смысл<br>\n" in back
    assert r"[$]0 &lt; x &lt; y \Rightarrow x^2 &lt; y^2[/$]" in back
    proof = next(row for row in rows if row[1].startswith("Единственность предела\x1f"))
    assert "Дополнено моделью: содержит сведения сверх PDF." in proof[1]
    assert "model_supplement" in proof[6] and "needs_review" in proof[6]
    report = read_json(job / "anki" / "report.json")
    assert report["warnings"] == response["warnings"]
    assert report["notes"][0]["figures"][0]["file"] == name
    assert report["notes"][2]["supplemented_fields"] == ["Full proof"]
    assert report["notes"][2]["review_flags"] == response["notes"][2]["review_flags"]
    assert read_json(job / "anki" / "notes.json")["notes"][0]["back"] == (
        "Строка & смысл\n" + r"[$]0 < x < y \Rightarrow x^2 < y^2[/$]")


@pytest.mark.parametrize(("index", "update"), [
    (0, {"source": {"page": 3, "order": 1}}),
    (0, {"source": {"page": 1, "order": 2}}),
    (0, {"source": {"page": 1, "order": 0}}),
    (0, {"front": "   "}),
    (0, {"back": r"\(x"}),
    (0, {"back": r"\)x\("}),
    (0, {"back": r"\(x + \[y\]\)"}),
    (0, {"back": r"\(x\] + \[y\)"}),
    (0, {"back": "[$]x"}),
    (0, {"back": "[/$]x[$]"}),
    (0, {"back": "[$]x + [$$]y[/$$][/$]"}),
    (0, {"back": "[$]x[/$$] + [$$]y[/$]"}),
    (1, {"steps": ["Шаг"] * 16}),
    (1, {"steps": ["Первый шаг", "   "]}),
    (2, {"proof": None}),
    (2, {"supplemented_fields": ["Back"]}),
    (0, {"figures": [{"field": "Back", "page": 3, "box": [0, 0, 1, 1]}]}),
    (0, {"figures": [{"field": "Step 1", "page": 1, "box": [0, 0, 1, 1]}]}),
    (0, {"figures": [{"field": "Back", "page": 1, "box": [0.6, 0, 0.2, 1]}]}),
    (0, {"figures": [{"field": "Back", "page": 1, "box": [0, 0, 1.1, 1]}]}),
])
def test_invalid_model_fields_and_provenance_are_rejected(response, index, update):
    response["notes"][index].update(copy.deepcopy(update))
    with pytest.raises(ValueError):
        validate_response(response, page_count=2)


def test_consecutive_inline_and_display_math_convert_to_native_anki_latex(response):
    text = r"Пусть \(x > 0\). Тогда \[x^2 > 0.\]"
    response["notes"][0]["back"] = text
    assert validate_response(response, page_count=2).notes[0].back == (
        "Пусть [$]x > 0[/$]. Тогда [$$]x^2 > 0.[/$$]")
    assert response["notes"][0]["back"] == text


@pytest.mark.parametrize(("index", "field"), [
    (0, "front"), (0, "back"),
    (1, "name"), (1, "overview"), (1, "steps"),
    (2, "name"), (2, "statement"), (2, "proof"),
    (3, "name"), (3, "statement"), (3, "plan"), (3, "steps"),
])
def test_every_note_content_field_normalizes_legacy_math(response, index, field):
    text = r"Объект \(x\) и формула \[x^2\]."
    expected = "Объект [$]x[/$] и формула [$$]x^2[/$$]."
    response["notes"][index][field] = [text] if field == "steps" else text

    normalized = validate_response(response, page_count=2).notes[index]

    assert getattr(normalized, field) == ([expected] if field == "steps" else expected)


def test_missing_proof_with_review_flag_exports_statement_without_empty_proof_card(
        job, response, client):
    response["notes"] = [response["notes"][2]]
    response["notes"][0].update(proof=None, review_flags=["Доказательство не восстановлено."])
    rows, _ = inspect_package(anki.generate_anki(job, job / "auth", client=client))
    assert len(rows) == 1
    assert rows[0][2:4] == (0, 1)
    assert "needs_review" in rows[0][6]
