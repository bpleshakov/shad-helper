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
from shad_helper.anki_schema import generation_schema, validate_response
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
    result = {"title": "Математическая лекция", "notes": [
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
    fields = [("Back",), ("Definition steps", "Step 1", "Step 2"),
              ("Statement", "Full proof"), ("Statement", "Steps to prove", "Step 1", "Step 2")]
    for note, answers in zip(result["notes"], fields, strict=True):
        note["card_sources"] = [{"field": field, "page": note["source"]["page"],
                                 "location": f"Фрагмент: {field}", "block_ids": []}
                                for field in answers]
    return result


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


def sourced_reply(response):
    reply = copy.deepcopy(response)
    for note in reply["notes"]:
        sources = {item["field"]: {key: value for key, value in item.items() if key != "field"}
                   for item in note.pop("card_sources")}
        for name, field in [("back", "Back"), ("overview", "Definition steps"),
                            ("statement", "Statement"), ("proof", "Full proof"),
                            ("plan", "Steps to prove")]:
            if note.get(name) is not None:
                note[name] = {"text": note[name], "source": sources[field]}
        if "steps" in note:
            note["steps"] = [{"text": step, "source": sources[f"Step {i}"]}
                             for i, step in enumerate(note["steps"], 1)]
    return reply


def test_generation_schema_requires_sources_inside_answers_and_steps():
    definitions = generation_schema()["$defs"]
    answer = {"$ref": "#/$defs/SourcedText"}
    for kind, fields in {"BasicNote": ["back"], "DefinitionNote": ["overview"],
                         "ProofNote": ["statement"], "StepsNote": ["statement", "plan"]}.items():
        properties = definitions[kind]["properties"]
        assert "card_sources" not in properties
        assert all(properties[field] == answer for field in fields)
        if "steps" in properties:
            assert properties["steps"]["items"] == answer
    assert definitions["ProofNote"]["properties"]["proof"] == {
        "anyOf": [answer, {"type": "null"}]}
    assert definitions["SourcedText"]["required"] == ["text", "source"]
    assert "field" not in definitions["AnswerSource"]["properties"]


def test_sourced_generation_exports_all_card_types_without_mutating_raw_reply(job, response, client):
    reply = sourced_reply(response)
    original = copy.deepcopy(reply)
    client.complete.return_value = reply
    rows, _ = inspect_package(anki.generate_anki(job, job / "auth", client=client))
    assert reply == original
    assert len(rows) == 10
    notes = read_json(job / "anki/notes.json")["notes"]
    assert [[source["field"] for source in note["card_sources"]] for note in notes] == [
        ["Back"], ["Definition steps", "Step 1", "Step 2"], ["Statement", "Full proof"],
        ["Statement", "Steps to prove", "Step 1", "Step 2"]]
    assert len(read_json(job / "anki/report.json")["cards"]) == len(rows)
    assert client.complete.call_args.kwargs["schema"] == generation_schema()


def test_missing_nested_source_is_rejected_and_raw_reply_is_retained(job, response, client):
    reply = sourced_reply(response)
    del reply["notes"][0]["back"]["source"]
    client.complete.return_value = reply
    with pytest.raises(ValueError, match="source"):
        anki.generate_anki(job, job / "auth", client=client)
    attempts = sorted((job / "diagnostics/anki").glob("*/attempt-*/response.json"))
    assert len(attempts) == 4
    assert all(read_json(path) == reply for path in attempts)
    assert all(read_json(path.with_name("status.json"))["status"] == "failed" for path in attempts)
    assert not (job / "anki/deck.apkg").exists()


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
    assert {(card["guid"], card["ordinal"], card["position"]) for card in report["cards"]} == {
        (row[0], row[2], row[3]) for row in rows}
    markdown = (job / "anki/sources.md").read_text()
    assert markdown.count("\n## ") == len(rows) == len(report["cards"])
    assert markdown.count("Тайм-код недоступен") == 10
    assert "#page=2" in markdown


def mapped_lecture(job, response):
    evidence = read_json(job / "evidence.json")
    evidence["media"]["path"] = str(job / "PRIVATE_RECORDING.mp4")
    evidence["segments"] = [
        {"id": "s1", "start": 61.25, "end": 67.5, "text": "PRIVATE_SPEECH"},
        {"id": "s2", "start": 112.125, "end": 119.75, "text": "PRIVATE_SPEECH"},
    ]
    evidence["visuals"] = [{"id": "v1", "timestamp": 113.5, "text": "PRIVATE_FRAME"}]
    evidence["issues"] = ["PRIVATE_DIAGNOSTICS"]
    write_json(job / "evidence.json", evidence)
    mapping = {"schema_version": 1, "evidence_fingerprint": fingerprint(evidence),
               "pdf_sha256": hashlib.sha256((job / "notes/main.pdf").read_bytes()).hexdigest(),
               "blocks": {}}
    for page, identifiers in [(1, ["s1"]), (2, ["v1", "s2", "s2"])]:
        mapping["blocks"][f"b{page}"] = {
            "id": f"b{page}", "topic": f"Тема {page}", "content": f"Текст блока {page}",
            "anchor": f"shad-block-b{page}", "evidence_ids": identifiers,
            "page_start": page, "page_end": page, "issues": ["PRIVATE_BLOCK_ISSUES"],
        }
    write_json(job / "notes/source-map.json", mapping)
    for note in response["notes"]:
        for reference in note["card_sources"]:
            reference["block_ids"] = [f"b{reference['page']}"]
    return mapping


def test_report_resolves_each_proof_step_locally_and_matches_exported_cards(job, client, response):
    mapped_lecture(job, response)
    response["notes"] = [response["notes"][3]]
    note = response["notes"][0]
    note["card_sources"][2].update(page=1, block_ids=["b1"])
    note["card_sources"][1]["block_ids"] = ["b1", "b2"]
    note["supplemented_fields"] = ["Step 2"]
    note["review_flags"] = ["Проверить дополнение"]
    client.complete.return_value = sourced_reply(response)
    package = anki.generate_anki(job, job / "auth", client=client)
    rows, _ = inspect_package(package)
    cards = read_json(job / "anki/report.json")["cards"]
    assert {(card["guid"], card["ordinal"]) for card in cards} == {
        (row[0], row[2]) for row in rows}
    by_field = {card["field"]: card for card in cards}
    assert by_field["Step 1"]["timestamps"] == [
        {"evidence_id": "s1", "start": 61.25, "end": 67.5}]
    assert by_field["Step 2"]["timestamps"] == [
        {"evidence_id": "s2", "start": 112.125, "end": 119.75},
        {"evidence_id": "v1", "start": 113.5, "end": 113.5}]
    assert len(by_field["Steps to prove"]["timestamps"]) == 3
    assert {card["field"] for card in cards if card["supplemented"]} == {"Step 2"}
    assert (by_field["Step 1"]["pdf_page"], by_field["Step 2"]["pdf_page"]) == (1, 2)
    markdown = (job / "anki/sources.md").read_text()
    assert markdown.count("\n## ") == 4
    assert "00:01:01.250–00:01:07.500" in markdown
    assert "#t=112.125,119.75" in markdown
    assert "#t=113.5)" in markdown
    assert "#nameddest=shad-block-b2" in markdown
    assert markdown.count("**Дополнено моделью.**") == 1
    assert "Проверить дополнение" in markdown
    outbound = json.dumps(client.complete.call_args.args, ensure_ascii=False)
    assert "Текст блока 1" in outbound
    assert "PRIVATE_" not in outbound
    assert "61.25" not in outbound
    assert str(job) not in outbound


@pytest.mark.parametrize("evidence_ids, expected", [
    (["v1_ill01"], [{"evidence_id": "v1_ill01", "start": 113.5, "end": 113.5}]),
    (["v1_ill01", "s1", "v1", "v1_ill02", "v1_ill01"], [
        {"evidence_id": "s1", "start": 61.25, "end": 67.5},
        {"evidence_id": "v1_ill01", "start": 113.5, "end": 113.5},
        {"evidence_id": "v1", "start": 113.5, "end": 113.5},
        {"evidence_id": "v1_ill02", "start": 113.5, "end": 113.5},
    ]),
])
def test_illustration_card_sources_keep_parent_frame_time_and_crop_identity(
        job, client, response, evidence_ids, expected):
    mapping = mapped_lecture(job, response)
    evidence = read_json(job / "evidence.json")
    evidence["visuals"][0].update(
        status="complete", illustrations=[{"id": "v1_ill01"}, {"id": "v1_ill02"}])
    write_json(job / "evidence.json", evidence)
    mapping["evidence_fingerprint"] = fingerprint(evidence)
    mapping["blocks"]["b1"]["evidence_ids"] = evidence_ids
    write_json(job / "notes/source-map.json", mapping)
    response["notes"] = [response["notes"][0]]
    client.complete.return_value = sourced_reply(response)

    package = anki.generate_anki(job, job / "auth", client=client)

    rows, _ = inspect_package(package)
    cards = read_json(job / "anki/report.json")["cards"]
    assert len(rows) == len(cards) == 1
    assert cards[0]["timestamps"] == expected
    markdown = (job / "anki/sources.md").read_text()
    assert "#t=113.5)" in markdown
    assert "`v1_ill01`" in markdown
    outbound = json.dumps(client.complete.call_args.args, ensure_ascii=False)
    assert "113.5" not in outbound
    assert "PRIVATE_" not in outbound


def test_unknown_illustration_source_does_not_borrow_its_frame_time(job, client, response):
    mapping = mapped_lecture(job, response)
    evidence = read_json(job / "evidence.json")
    evidence["visuals"][0].update(status="complete", illustrations=[{"id": "v1_ill01"}])
    write_json(job / "evidence.json", evidence)
    mapping["evidence_fingerprint"] = fingerprint(evidence)
    mapping["blocks"]["b1"]["evidence_ids"] = ["v1_ill99"]
    write_json(job / "notes/source-map.json", mapping)

    with pytest.raises(ValueError, match="Lecture block b1 has no valid timed evidence"):
        anki.generate_anki(job, job / "auth", client=client)

    assert not (job / "anki/deck.apkg").exists()
    assert not (job / "anki/sources.md").exists()
    assert not (job / "checkpoints/anki.json").exists()


@pytest.mark.parametrize("error", ["missing", "duplicate", "unknown", "page", "no_evidence"])
def test_invalid_card_sources_do_not_publish(job, client, response, error):
    mapping = mapped_lecture(job, response)
    reference = response["notes"][0]["card_sources"][0]
    if error == "missing":
        response["notes"][0]["card_sources"] = []
    elif error == "duplicate":
        response["notes"][0]["card_sources"].append(copy.deepcopy(reference))
    elif error == "unknown":
        reference["block_ids"] = ["invented"]
    elif error == "page":
        reference["page"] = 2
    else:
        mapping["blocks"]["b1"]["evidence_ids"] = ["invented"]
        write_json(job / "notes/source-map.json", mapping)
    with pytest.raises(ValueError):
        anki.generate_anki(job, job / "auth", client=client)
    assert not (job / "anki/deck.apkg").exists()
    assert not (job / "anki/sources.md").exists()
    assert not (job / "checkpoints/anki.json").exists()


@pytest.mark.parametrize("change", ["pdf", "evidence", "missing"])
def test_stale_lecture_map_fails_before_model(job, client, response, change):
    mapped_lecture(job, response)
    if change == "pdf":
        write_pdf(job / "notes/main.pdf", "Changed source")
    elif change == "evidence":
        evidence = read_json(job / "evidence.json")
        evidence["segments"][0]["start"] = 62
        write_json(job / "evidence.json", evidence)
    else:
        (job / "notes/source-map.json").unlink()
        write_json(job / "notes/project.json", {})
    with pytest.raises(ValueError, match="source map|evidence changed"):
        anki.generate_anki(job, job / "auth", client=client)
    client.complete.assert_not_called()


def test_cached_card_sources_are_validated_and_catalog_changes_invalidate_cache(
        job, client, response):
    mapping = mapped_lecture(job, response)
    anki.generate_anki(job, job / "auth", client=client)
    previous = (job / "anki/sources.md").read_bytes()
    cache_path = job / "checkpoints/anki.json"
    cache = read_json(cache_path)
    cache["result"]["notes"][0]["card_sources"] = []
    write_json(cache_path, cache)
    with pytest.raises(ValueError, match="Every generated answer"):
        anki.generate_anki(job, job / "auth", client=client)
    assert (job / "anki/sources.md").read_bytes() == previous
    assert client.complete.call_count == 1
    mapping["blocks"]["b1"]["content"] = "Уточнённая формулировка"
    write_json(job / "notes/source-map.json", mapping)
    anki.generate_anki(job, job / "auth", client=client)
    assert client.complete.call_count == 2


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
    assert read_json(job / "anki" / "identity.json") == identity
    assert anki.generate_anki(job, tmp_path / "auth", pdf=source,
                              client=client, rebuild=True) == package

    assert client.complete.call_count == 2
    assert read_json(job / "anki" / "identity.json") != identity
    assert {path.name for path in job.iterdir()} == {
        "anki", "checkpoints", "diagnostics", "run-diagnostics.json"}
    report = read_json(job / "run-diagnostics.json")
    assert report["command"] == "anki" and report["status"] == "completed"
    assert {step["name"] for step in report["steps"][0]["steps"]} == {
        "anki_generation", "anki_note_preparation", "anki_latex_rendering", "anki_packaging"}


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


def test_rebuild_deletes_checkpoint_before_generation_and_replaces_it(job, client, response):
    anki.generate_anki(job, job / "auth", client=client)
    cache_path = job / "checkpoints/anki.json"
    assert cache_path.exists()
    fresh = copy.deepcopy(response)
    fresh["notes"][0]["back"] = "Freshly generated answer"

    def generate(*args, **kwargs):
        assert not cache_path.exists()
        return fresh

    client.complete.reset_mock()
    client.complete.side_effect = generate
    anki.generate_anki(job, job / "auth", client=client, rebuild=True)
    assert read_json(cache_path)["result"]["notes"][0]["back"] == "Freshly generated answer"
    client.complete.assert_called_once()

    client.complete.reset_mock()
    anki.generate_anki(job, job / "auth", client=client)
    client.complete.assert_not_called()


@pytest.mark.parametrize("failure", ["missing_pdf", "invalid_pdf", "minimum_position", "collection"])
def test_rebuild_clears_all_anki_artifacts_before_checks(job, client, collection_reader, failure):
    for name in ("checkpoints/anki.json", "anki/identity.json", "anki/deck.apkg",
                 "anki/notes.json", "anki/report.json", "anki/sources.md", "anki/media/old.png",
                 "anki/figures/old.png", "anki/.build-stale/partial", "diagnostics/anki/old/status.json"):
        path = job / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"previous Anki artifact")
    preserved = {"evidence.json": (job / "evidence.json").read_bytes(),
                 "checkpoints/notes/ch001.json": b"notes checkpoint",
                 "diagnostics/vision/frame.json": b"vision diagnostic"}
    for name, contents in preserved.items():
        path = job / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    artifact_roots = [job / name for name in ("checkpoints/anki.json", "anki", "diagnostics/anki")]
    source = job / "notes/main.pdf"
    if failure == "missing_pdf":
        source.unlink()
    else:
        if failure == "invalid_pdf":
            source.write_bytes(b"not a PDF")
        preserved["notes/main.pdf"] = source.read_bytes()

    def unavailable_collection(_):
        assert all(not path.exists() for path in artifact_roots)
        raise ValueError("Collection unavailable")

    collection_reader.side_effect = unavailable_collection
    with pytest.raises(ValueError):
        anki.generate_anki(job, job / "auth", client=client, rebuild=True,
                           minimum_position=0 if failure == "minimum_position" else 1)

    assert all(not path.exists() for path in artifact_roots)
    assert {name: (job / name).read_bytes() for name in preserved} == preserved
    assert collection_reader.call_count == (failure == "collection")
    client.complete.assert_not_called()


def test_rebuild_stops_if_old_anki_checkpoint_cannot_be_deleted(
        job, client, monkeypatch, collection_reader):
    anki.generate_anki(job, job / "auth", client=client)
    cache_path = job / "checkpoints/anki.json"
    before = cache_path.read_bytes()
    unlink = Path.unlink

    def fail_checkpoint(path, *args, **kwargs):
        if path == cache_path:
            raise PermissionError("Cannot delete Anki checkpoint")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_checkpoint)
    client.complete.reset_mock()
    collection_reader.reset_mock()
    with pytest.raises(PermissionError, match="Cannot delete Anki checkpoint"):
        anki.generate_anki(job, job / "auth", client=client, rebuild=True)
    client.complete.assert_not_called()
    collection_reader.assert_not_called()
    assert cache_path.read_bytes() == before


@pytest.mark.parametrize("change", ["rebuild", "pdf", "model", "prompt"])
def test_generation_restarts_when_required_and_only_rebuild_resets_identity(
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
    assert (read_json(job / "anki" / "identity.json") == identity) is (change != "rebuild")
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
    "sources_error",
])
@pytest.mark.parametrize("rebuild", [False, True])
def test_failed_regeneration_preserves_previous_artifacts_only_without_rebuild(
        job, client, monkeypatch, no_retry_wait, collection_reader, failure, rebuild):
    anki.generate_anki(job, job / "auth", client=client)
    previous = {path.name: path.read_bytes() for path in (job / "anki").iterdir()
                if path.is_file()}
    client.complete.reset_mock()
    client.model = "another-available-model"
    if failure == "invalid_json":
        client.complete.return_value = {"title": "Invalid", "notes": []}
        expected_calls = 4
    elif failure == "response_error":
        client.complete.side_effect = ResponseError("Invalid response JSON")
        expected_calls = 4
    elif failure == "latex_error":
        monkeypatch.setattr(anki, "render_latex_media", Mock(side_effect=RuntimeError("TeX failed")))
        expected_calls = 1
    elif failure == "sources_error":
        monkeypatch.setattr(anki, "write_sources_report", Mock(side_effect=OSError("Disk full")))
        expected_calls = 1
    elif failure == "collection_refresh_error":
        collection_reader.side_effect = [collection_reader.return_value,
                                         RuntimeError("Collection is no longer available")]
        expected_calls = 1
    else:
        monkeypatch.setattr(anki, "write_package", Mock(side_effect=OSError("Disk unavailable")))
        expected_calls = 1

    with pytest.raises((ValueError, RuntimeError, OSError)):
        anki.generate_anki(job, job / "auth", client=client, rebuild=rebuild)

    assert client.complete.call_count == expected_calls
    assert (job / "checkpoints/anki.json").exists() is (
        not rebuild or failure not in {"invalid_json", "response_error"})
    if rebuild:
        assert all(not (job / "anki" / name).exists()
                   for name in ("notes.json", "report.json", "sources.md", "deck.apkg"))
        identity = job / "anki/identity.json"
        if identity.exists():
            assert identity.read_bytes() != previous["identity.json"]
    else:
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
    names = ("notes.json", "report.json", "sources.md", "deck.apkg")
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
    runs = list((job / "diagnostics/anki").iterdir())
    assert len(runs) == 1
    assert read_json(runs[0] / "attempt-01/response.json") == {"notes": []}
    assert read_json(runs[0] / "attempt-01/status.json")["status"] == "failed"
    assert read_json(runs[0] / "attempt-02/status.json")["status"] == "success"
    # Local validation errors and failed responses never become retry inputs.
    assert client.complete.call_args_list[0] == client.complete.call_args_list[1]


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
    assert "Строка &amp; смысл<br>" in back
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


def mapped_figure(job, response):
    mapping = mapped_lecture(job, response)
    evidence = read_json(job / "evidence.json")
    evidence["visuals"][0].update(status="complete", illustrations=[{"id": "v1_ill01"}])
    write_json(job / "evidence.json", evidence)
    mapping["evidence_fingerprint"] = fingerprint(evidence)
    relative = "figures/illustration-v1_ill01.png"
    path = job / "notes" / relative
    path.parent.mkdir()
    pixels = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 48, 24), False)
    pixels.clear_with(120)
    pixels.save(path)
    block = {"id": "fig1", "kind": "screenshot", "topic": "График", "content": "Точный график",
             "illustration_id": "v1_ill01", "evidence_ids": ["v1_ill01"],
             "anchor": "shad-block-fig1", "page_start": 1, "page_end": 1,
             "figure_asset": {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
    mapping["blocks"]["fig1"] = block
    write_json(job / "notes/source-map.json", mapping)
    write_json(job / "notes/project.json", {"version": 2, "tex_files": [],
               "image_files": [relative], "source_blocks": [block]})
    response["notes"][0]["figures"] = [{"field": "Back", "block_id": "fig1"}]
    return path


def test_original_figure_bytes_are_packaged_without_pdf_cropping_and_reused_offline(
        job, client, response, monkeypatch):
    original = mapped_figure(job, response).read_bytes()
    monkeypatch.setattr(pymupdf.Page, "get_pixmap", Mock(side_effect=AssertionError("PDF crop")))

    package = anki.generate_anki(job, job / "auth", client=client)

    rows, media = inspect_package(package)
    assert list(media.values()) == [original]
    report = read_json(job / "anki/report.json")
    figure = report["notes"][0]["figures"][0]
    assert figure["block_id"] == "fig1" and "box" not in figure
    assert figure["source_asset"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert f'<img src="{figure["file"]}">' in rows[0][1]
    outbound = json.dumps(client.complete.call_args.args, ensure_ascii=False)
    assert "figure_catalog" in outbound and "fig1" in outbound
    assert "PRIVATE_" not in outbound and "figures/" not in outbound
    assert client.complete.call_args.kwargs["schema"] == generation_schema(source_figures=True)
    before_guids = {row[0] for row in rows}
    anki.generate_anki(job, job / "auth", client=client)
    assert {row[0] for row in inspect_package(package)[0]} == before_guids
    assert client.complete.call_count == 1


@pytest.mark.parametrize("target", ["unknown", "b1", "pdf_crop"])
def test_lecture_figures_cannot_select_unknown_prose_or_pdf_crops(job, client, response, target):
    mapped_figure(job, response)
    response["notes"][0]["figures"] = (
        [{"field": "Back", "page": 1, "box": [0, 0, 1, 1]}] if target == "pdf_crop"
        else [{"field": "Back", "block_id": target}])
    with pytest.raises(ValueError, match="Figure must reference|Lecture figures require"):
        anki.generate_anki(job, job / "auth", client=client)
    assert not (job / "anki/deck.apkg").exists()
    assert not (job / "checkpoints/anki.json").exists()


@pytest.mark.parametrize("change", ["missing", "changed"])
def test_missing_or_changed_original_figure_fails_before_model(job, client, response, change):
    path = mapped_figure(job, response)
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(b"Changed source")
    with pytest.raises(ValueError, match="missing|changed"):
        anki.generate_anki(job, job / "auth", client=client)
    client.complete.assert_not_called()
    assert not (job / "anki/deck.apkg").exists()


def test_source_figure_reference_requires_lecture_assets(job, client, response):
    response["notes"][0]["figures"] = [{"field": "Back", "block_id": "fig1"}]
    with pytest.raises(ValueError, match="Figure must reference"):
        anki.generate_anki(job, job / "auth", client=client)
    assert not (job / "anki/deck.apkg").exists()


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
    response["notes"][0]["card_sources"] = [response["notes"][0]["card_sources"][0]]
    client.complete.return_value = sourced_reply(response)
    rows, _ = inspect_package(anki.generate_anki(job, job / "auth", client=client))
    assert len(rows) == 1
    assert rows[0][2:4] == (0, 1)
    assert "needs_review" in rows[0][6]
