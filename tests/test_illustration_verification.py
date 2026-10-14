import json
from copy import deepcopy
from pathlib import Path

import httpx
import pymupdf
import pytest

from shad_helper import illustration_verification as verification
from shad_helper.jobs import read_json, sha256_file, write_json
from shad_helper.openai_client import PlanClient, ResponseError
from shad_helper.rendering import RenderError


def png(path, shade=220):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 16, 12), False)
    pixels.clear_with(shade)
    pixels.save(path)
    return path


@pytest.fixture
def example(tmp_path, monkeypatch):
    job = tmp_path / "job"
    source = png(job / "illustrations" / "v01_ill01.png")
    observation = {"id": "v01_ill01", "source_kind": "drawn", "box": [0, 0, 1, 1],
                   "description": "Вершины A и B, стрелка A к B.",
                   "image_sha256": sha256_file(source)}
    evidence = {"visuals": [{"id": "v01", "timestamp": 2, "status": "complete",
                             "illustrations": [observation]}]}
    write_json(job / "illustrations" / "manifest.json", {"version": 1, "assets": {
        observation["id"]: {"path": "illustrations/v01_ill01.png",
                            "sha256": observation["image_sha256"]}}})
    block = {"id": "drawing", "kind": "diagram", "content": r"\draw[->] (0,0)--(1,1);",
             "illustration_id": observation["id"], "caption": "Связь вершин",
             "topic": "Граф", "evidence_ids": ["v01", observation["id"]],
             "language": "", "uncertain_fragments": []}
    notes = {"title": "Лекция", "language": "ru", "chapters": [
        {"id": "c01", "title": "Графы", "start": 0, "end": 10, "blocks": [block]}],
        "glossary": [], "issues": []}
    renders = []

    def preview(content, folder):
        renders.append(content)
        return png(folder / "preview.png")

    monkeypatch.setattr(verification.rendering, "render_diagram_preview", preview)
    monkeypatch.setattr(verification, "refine_crop", lambda *args: ("", False, ""))
    return job, evidence, notes, renders


class Client:
    model = "gpt-6.1-sol"

    def __init__(self, response=None):
        self.response = response if response is not None else {
            "verdict": "pass", "reason": "Все вершины, подписи и стрелки совпадают."}
        self.calls = []

    def complete(self, instructions, text, schema, *, images):
        self.calls.append((instructions, text, schema, images))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def block(notes):
    return notes["chapters"][0]["blocks"][0]


def finalize(example, client, name="attempt"):
    job, evidence, notes, _ = example
    return verification.finalize_illustrations(notes, evidence, job, job / name, client)


def test_verified_drawing_is_kept_and_archived_with_source_preview(example):
    job, _, notes, renders = example
    original = deepcopy(notes)
    client = Client()
    accepted, retry = finalize(example, client)
    assert accepted == original and notes == original
    assert retry is False
    assert renders == [block(notes)["content"]]
    instructions, _, schema, images = client.calls[0]
    assert "arrow direction" in instructions and "plotted relationship" in instructions
    assert schema == verification.VERIFICATION_SCHEMA
    assert [label for label, _ in images] == ["SOURCE", "RECONSTRUCTION"]
    assert all(Path(path).is_file() for _, path in images)
    archive = job / "attempt" / "illustration-verification" / "drawing"
    assert read_json(archive / "candidate.json") == block(notes)
    assert read_json(archive / "accepted.json") == block(accepted)
    assert read_json(archive / "verdict.json")["verdict"] == "pass"
    assert (archive / "source.png").is_file()
    assert (archive / "artifacts" / "preview.png").is_file()
    call = read_json(archive / "artifacts" / "call.json")
    assert call["model"] == client.model
    assert [image["file"] for image in call["images"]] == ["source.png", "preview.png"]
    assert read_json(archive / "artifacts" / "response.json") == client.response


@pytest.mark.parametrize("verdict,reason", [
    ("fail", "Направление стрелки изменено."),
    ("fail", "Подпись B заменена на C."),
    ("uncertain", "Подпись вершины не читается."),
])
def test_failed_or_uncertain_comparison_preserves_crop_and_review(example, verdict, reason):
    _, _, original, _ = example
    accepted, retry = finalize(example, Client({"verdict": verdict, "reason": reason}))
    expected = {**block(original), "kind": "screenshot",
                "content": "Связь вершин"}
    assert block(accepted) == expected
    assert block(original)["kind"] == "diagram"
    assert retry is False
    assert accepted["issues"][0]["evidence_ids"] == ["v01_ill01"]
    assert reason in accepted["issues"][0]["description"]


@pytest.mark.parametrize("response", [
    ResponseError("Network output incomplete"), OSError("Offline"),
    {"verdict": "pass", "reason": ""}, {"verdict": "yes", "reason": "Maybe"},
    {"verdict": "pass", "reason": "Faithful", "extra": "bad"}, ["pass"],
])
def test_unavailable_verification_falls_back_and_retries_original(example, response):
    client = Client(response)
    accepted, retry = finalize(example, client)
    assert block(accepted)["kind"] == "screenshot" and retry is True
    client.response = {"verdict": "pass", "reason": "Соответствует источнику."}
    accepted, retry = finalize(example, client, "second-attempt")
    assert block(accepted)["kind"] == "diagram" and retry is False
    assert len(client.calls) == 2
    assert accepted["issues"] == []


def test_missing_verification_client_falls_back(example):
    accepted, retry = finalize(example, None)
    assert block(accepted)["kind"] == "screenshot" and retry is True


def test_invalid_tikz_falls_back_before_comparison(example, monkeypatch):
    def broken(*args):
        raise RenderError("Unsupported TeX command")

    monkeypatch.setattr(verification.rendering, "render_diagram_preview", broken)
    client = Client()
    accepted, retry = finalize(example, client)
    assert block(accepted)["kind"] == "screenshot"
    assert "компиляцию" in accepted["issues"][0]["description"]
    assert not client.calls and retry is False


def test_fallback_never_promotes_unverified_local_description_into_notes(example):
    _, evidence, notes, _ = example
    description = "Неверное локальное предположение о максимуме функции."
    evidence["visuals"][0]["illustrations"][0]["description"] = description
    block(notes)["caption"] = ""
    accepted, _ = finalize(example, Client({"verdict": "fail", "reason": "Рисунок не совпадает."}))
    assert block(accepted)["content"] == block(accepted)["caption"] == "Иллюстрация из лекции"
    assert description not in json.dumps(accepted, ensure_ascii=False)


@pytest.mark.parametrize("verdict", ["pass", "fail", "uncertain"])
def test_completed_decisions_are_cached_and_copied_to_each_archive(example, verdict):
    job, _, _, renders = example
    client = Client({"verdict": verdict, "reason": "Результат проверки."})
    first, _ = finalize(example, client)
    second, retry = finalize(example, client, "second-attempt")
    assert second == first and retry is False
    assert len(client.calls) == len(renders) == 1
    assert (job / "second-attempt" / "illustration-verification" / "drawing"
            / "artifacts" / "preview.png").is_file()


@pytest.mark.parametrize("corruption", ["missing_preview", "changed_preview", "invalid_verdict"])
def test_incomplete_or_modified_verifier_cache_requires_fresh_comparison(example, corruption):
    job, _, _, renders = example
    client = Client()
    finalize(example, client)
    cache = next((job / "illustration-verification").iterdir())
    if corruption == "missing_preview":
        (cache / "preview.png").unlink()
    elif corruption == "changed_preview":
        png(cache / "preview.png", 100)
    else:
        (cache / "verdict.json").write_text("invalid JSON")
    finalize(example, client, "second-attempt")
    assert len(client.calls) == len(renders) == 2


def test_invalid_model_response_is_archived_before_fallback(example):
    job, _, _, _ = example
    client = Client({"verdict": "invalid", "reason": "Unknown"})
    finalize(example, client)
    archive = job / "attempt" / "illustration-verification" / "drawing"
    assert read_json(archive / "artifacts" / "response.json") == client.response
    assert read_json(archive / "error.json")["type"] == "ValueError"


def test_verification_records_exact_plan_request_and_restores_recorder(example):
    job, _, _, _ = example
    requests = []
    response = {"verdict": "pass", "reason": "Все стрелки и подписи сохранены."}

    def handle(request):
        requests.append(request)
        event = {"type": "response.completed", "response": {"status": "completed", "output": [
            {"content": [{"type": "output_text", "text": json.dumps(response)}]}]}}
        return httpx.Response(200, text="data: " + json.dumps(event) + "\n\n")

    def original_recorder(*args):
        pytest.fail("The verification must use its own archive")

    client = PlanClient(job, proxy="http://proxy.invalid:8080", transport=httpx.MockTransport(handle),
                        token_provider=lambda: "private-token", request_recorder=original_recorder)
    client._available_models = [client.model]
    accepted, retry = finalize(example, client)
    assert block(accepted)["kind"] == "diagram" and not retry
    assert client.request_recorder is original_recorder
    archive = job / "attempt" / "illustration-verification" / "drawing" / "artifacts"
    assert (archive / "http-001.body.json").read_bytes() == requests[0].content
    assert "private-token" not in (archive / "http-001.request.json").read_text()


@pytest.mark.parametrize("changed", ["tikz", "source", "model", "prompt", "render", "policy"])
def test_verification_cache_tracks_source_and_all_verification_inputs(example, monkeypatch, changed):
    job, evidence, notes, renders = example
    client = Client()
    finalize(example, client)
    if changed == "tikz":
        block(notes)["content"] = r"\draw[->] (0,0)--(2,1);"
    elif changed == "source":
        source = png(job / "illustrations" / "v01_ill01.png", shade=100)
        digest = sha256_file(source)
        evidence["visuals"][0]["illustrations"][0]["image_sha256"] = digest
        write_json(job / "illustrations" / "manifest.json", {"version": 1, "assets": {
            "v01_ill01": {"path": "illustrations/v01_ill01.png", "sha256": digest}}})
    elif changed == "model":
        client.model = "other-configured-model"
    elif changed == "prompt":
        monkeypatch.setattr(verification, "VERIFICATION_INSTRUCTIONS", "Changed policy")
    elif changed == "render":
        monkeypatch.setattr(verification.rendering, "diagram_preview_fingerprint", lambda: "new")
    elif changed == "policy":
        monkeypatch.setattr(verification, "VERIFICATION_POLICY_VERSION", 2)
    finalize(example, client, "second-attempt")
    assert len(client.calls) == len(renders) == 2


@pytest.mark.parametrize("source_kind,issue", [("printed", False), ("drawn", True), ("unknown", True)])
def test_direct_screenshots_skip_verification_and_flag_fallbacks(example, source_kind, issue):
    _, evidence, notes, renders = example
    evidence["visuals"][0]["illustrations"][0]["source_kind"] = source_kind
    block(notes)["kind"] = "screenshot"
    block(notes)["content"] = "Исходная иллюстрация."
    client = Client()
    accepted, retry = finalize(example, client)
    assert block(accepted) == block(notes)
    assert bool(accepted["issues"]) is issue
    assert not retry and not client.calls and not renders


def test_legacy_diagrams_without_source_are_left_unchanged(example):
    _, _, notes, renders = example
    block(notes).pop("illustration_id")
    client = Client()
    accepted, retry = finalize(example, client)
    assert accepted == notes
    assert not retry and not client.calls and not renders


def test_unknown_source_requires_review_even_for_verified_diagram(example):
    _, evidence, _, _ = example
    evidence["visuals"][0]["illustrations"][0]["source_kind"] = "unknown"
    accepted, retry = finalize(example, Client())
    assert block(accepted)["kind"] == "diagram"
    assert accepted["issues"][0]["evidence_ids"] == ["v01_ill01"]
    assert not retry


def add_aliased_observation(example, *, identical=False, timestamp=5):
    job, evidence, notes, _ = example
    block(notes).update(kind="screenshot", content="Учебный рисунок")
    evidence["visuals"][0]["illustrations"][0]["source_kind"] = "printed"
    path = png(job / "illustrations" / "v02_ill01.png", 220 if identical else 100)
    observation = {"id": "v02_ill01", "source_kind": "drawn", "box": [0, 0, 1, 1],
                   "description": "Новая содержательная стадия.", "image_sha256": sha256_file(path)}
    evidence["visuals"].append({"id": "v02", "timestamp": timestamp, "status": "complete",
                                 "illustrations": [observation]})
    manifest = read_json(job / "illustrations" / "manifest.json")
    manifest["assets"]["v02_ill01"] = {"path": "illustrations/v02_ill01.png",
                                       "sha256": observation["image_sha256"]}
    write_json(job / "illustrations" / "manifest.json", manifest)
    notes["chapters"][0]["illustration_coverage"] = [
        {"illustration_id": identifier, "block_id": block(notes)["id"]}
        for identifier in ("v01_ill01", "v02_ill01")]


def test_identical_observations_reuse_figure_without_semantic_request(example):
    add_aliased_observation(example, identical=True)
    client = Client()
    accepted, retry = finalize(example, client)
    assert len(accepted["chapters"][0]["blocks"]) == 1
    assert not client.calls and not retry and not accepted["issues"]


def test_verified_consolidation_preserves_mapping_and_caches_decision(example):
    add_aliased_observation(example)
    client = Client()
    accepted, retry = finalize(example, client)
    assert len(accepted["chapters"][0]["blocks"]) == 1
    assert not retry and not accepted["issues"]
    assert len(client.calls) == 1
    assert client.calls[0][0] == verification.CONSOLIDATION_INSTRUCTIONS
    finalize(example, client, "second-attempt")
    assert len(client.calls) == 1


@pytest.mark.parametrize("verdict", ["fail", "uncertain"])
def test_distinct_or_uncertain_observation_gets_separate_crop_and_mapping(example, verdict):
    job, _, notes, _ = example
    add_aliased_observation(example)
    client = Client({"verdict": verdict, "reason": "Отдельная содержательная стадия."})
    accepted, retry = finalize(example, client)
    blocks = accepted["chapters"][0]["blocks"]
    assert [item["illustration_id"] for item in blocks] == ["v01_ill01", "v02_ill01"]
    assert blocks[1]["kind"] == "screenshot"
    assert blocks[1]["caption"] == blocks[1]["content"] == "Иллюстрация к теме «Граф»"
    assert accepted["chapters"][0]["illustration_coverage"][1]["block_id"] == blocks[1]["id"]
    assert accepted["issues"][0]["evidence_ids"] == ["v02_ill01"]
    assert not retry and len(notes["chapters"][0]["blocks"]) == 1
    archive = job / "attempt" / "illustration-coverage" / "v02_ill01"
    assert read_json(archive / "accepted-block.json") == blocks[1]


def test_separate_intermediate_crop_precedes_completed_drawing(example):
    _, _, notes, _ = example
    add_aliased_observation(example, timestamp=1)
    block(notes)["evidence_ids"].append("v02_ill01")
    accepted, _ = finalize(example, Client({"verdict": "fail", "reason": "Отдельная стадия."}))
    assert [item["illustration_id"] for item in accepted["chapters"][0]["blocks"]] == [
        "v02_ill01", "v01_ill01"]


def test_consolidation_failure_retries_raw_mapping(example):
    add_aliased_observation(example)
    client = Client(OSError("Offline"))
    accepted, retry = finalize(example, client)
    assert len(accepted["chapters"][0]["blocks"]) == 2 and retry
    client.response = {"verdict": "pass", "reason": "Тот же рисунок, сдвинут кадр."}
    accepted, retry = finalize(example, client, "second-attempt")
    assert len(accepted["chapters"][0]["blocks"]) == 1 and not retry
    assert len(client.calls) == 2


def test_previous_chapter_consolidation_can_fall_back_in_current_chapter(example):
    job, evidence, notes, _ = example
    add_aliased_observation(example)
    previous = deepcopy(notes)
    previous["chapters"][0].update(end=3, illustration_coverage=[
        {"illustration_id": "v01_ill01", "block_id": "drawing"}])
    notes["chapters"][0].update(id="c02", start=3, blocks=[], illustration_coverage=[
        {"illustration_id": "v02_ill01", "block_id": "drawing"}])
    accepted, retry = verification.finalize_illustrations(
        notes, evidence, job, job / "attempt", Client({"verdict": "fail", "reason": "Новая стадия."}),
        previous=[previous])
    assert block(accepted)["illustration_id"] == "v02_ill01" and not retry
    assert accepted["chapters"][0]["illustration_coverage"][0]["block_id"] == block(accepted)["id"]
    assert block(previous)["illustration_id"] == "v01_ill01"


def test_rejected_group_reuses_clearer_crop_but_preserves_new_stage(example):
    job, evidence, notes, _ = example
    add_aliased_observation(example)
    chapter = notes["chapters"][0]
    manifest = read_json(job / "illustrations" / "manifest.json")
    # The original (220) is clipped; 100--104 show the complete figure, while
    # 50--51 show a different meaningful stage. One repeat is byte-identical.
    for number, shade in enumerate([101, 102, 103, 104, 100, 50, 51], 3):
        visual_id = f"v{number:02d}"
        identifier = visual_id + "_ill01"
        source = png(job / "illustrations" / f"{identifier}.png", shade)
        observation = {**evidence["visuals"][1]["illustrations"][0],
                       "id": identifier, "image_sha256": sha256_file(source)}
        evidence["visuals"].append({"id": visual_id, "timestamp": number + 3,
                                     "status": "complete", "illustrations": [observation]})
        manifest["assets"][identifier] = {"path": f"illustrations/{identifier}.png",
                                          "sha256": observation["image_sha256"]}
        chapter["illustration_coverage"].append({"illustration_id": identifier,
                                                 "block_id": "drawing"})
    write_json(job / "illustrations" / "manifest.json", manifest)

    class FigureClient(Client):
        def complete(self, instructions, text, schema, *, images):
            source, representative = [pymupdf.Pixmap(path).samples[0] for _, path in images]
            self.calls.append((source, representative))
            same_stage = (100 <= source < 200 and 100 <= representative < 200
                          or source < 100 and representative < 100)
            return {"verdict": "pass" if same_stage else "fail",
                    "reason": "Тот же рисунок." if same_stage else "Обрезан или другая стадия."}

    client = FigureClient()
    accepted, retry = finalize(example, client)
    blocks = accepted["chapters"][0]["blocks"]
    assert [item["illustration_id"] for item in blocks] == [
        "v01_ill01", "v02_ill01", "v08_ill01"]
    targets = [item["block_id"] for item in accepted["chapters"][0]["illustration_coverage"]]
    assert targets == ["drawing", *([blocks[1]["id"]] * 6), *([blocks[2]["id"]] * 2)]
    assert (101, 100) in client.calls and (51, 50) in client.calls
    assert (50, 100) in client.calls and (51, 100) in client.calls
    assert (100, 100) not in client.calls
    assert not retry and len(accepted["issues"]) == 2
    count = len(client.calls)
    repeated, retry = finalize(example, client, "second-attempt")
    assert repeated == accepted and not retry and len(client.calls) == count
    archive = job / "attempt" / "illustration-coverage" / "v03_ill01"
    assert read_json(archive / "accepted.json")["block_id"] == blocks[1]["id"]
    assert read_json(archive / "alternatives" / blocks[1]["id"] / "verdict.json")["verdict"] == "pass"


def test_excluded_observation_skips_consolidation(example):
    _, _, notes, _ = example
    add_aliased_observation(example)
    notes["chapters"][0]["illustration_coverage"][1].update(
        block_id="", exclusion_reason="decoration")
    client = Client()
    accepted, retry = finalize(example, client)
    assert accepted == notes and not retry and not client.calls


@pytest.mark.parametrize("verdict", ["pass", "fail"])
def test_reconstruction_and_its_fallback_use_the_refined_source(example, monkeypatch, verdict):
    job, _, _, _ = example
    cropped = png(job / "refined.png", 80)
    monkeypatch.setattr(verification, "refine_crop", lambda *args: ("crop_1", False, ""))

    def resolve(job, evidence, identifier, crop_id=""):
        assert identifier == "v01_ill01" and crop_id == "crop_1"
        return cropped

    monkeypatch.setattr(verification, "resolve_crop", resolve)
    client = Client({"verdict": verdict, "reason": "Результат сравнения."})
    accepted, retry = finalize(example, client)
    assert block(accepted)["illustration_crop_id"] == "crop_1"
    assert block(accepted)["kind"] == ("diagram" if verdict == "pass" else "screenshot")
    assert client.calls[0][3][0][1].read_bytes() == cropped.read_bytes()
    assert (job / "attempt/illustration-verification/drawing/source.png").read_bytes() == cropped.read_bytes()
    assert not retry


def test_direct_screenshot_gets_refined_crop_without_reconstruction_check(example, monkeypatch):
    job, evidence, notes, _ = example
    block(notes)["kind"] = "screenshot"
    evidence["visuals"][0]["illustrations"][0]["source_kind"] = "printed"
    cropped = png(job / "refined.png", 80)
    monkeypatch.setattr(verification, "refine_crop", lambda *args: ("crop_1", False, ""))
    monkeypatch.setattr(verification, "resolve_crop", lambda *args: cropped)
    client = Client()
    accepted, retry = finalize(example, client)
    assert block(accepted)["illustration_crop_id"] == "crop_1"
    assert not client.calls and not retry and not accepted["issues"]


def test_consolidation_checks_refined_pixels_even_when_original_hashes_match(example, monkeypatch):
    job, _, _, _ = example
    add_aliased_observation(example, identical=True)
    cropped = png(job / "refined.png", 80)
    monkeypatch.setattr(verification, "refine_crop", lambda *args: ("crop_1", False, ""))
    monkeypatch.setattr(verification, "resolve_crop", lambda *args: cropped)
    client = Client()
    accepted, retry = finalize(example, client)
    assert not retry and len(accepted["chapters"][0]["blocks"]) == 1
    assert len(client.calls) == 2
    for instructions, _, _, images in client.calls:
        assert instructions == verification.CONSOLIDATION_INSTRUCTIONS
        assert pymupdf.Pixmap(images[0][1]).samples[0] == 220
        assert images[1][1].read_bytes() == cropped.read_bytes()
    finalize(example, client, "second-attempt")
    assert len(client.calls) == 2
    png(cropped, 90)
    finalize(example, client, "third-attempt")
    assert len(client.calls) == 4


def test_failed_consolidation_refines_new_fallback_before_accepting_it(example, monkeypatch):
    job, _, _, _ = example
    add_aliased_observation(example)
    calls = []

    def refine(job, evidence, selected, client, folder):
        calls.append(selected["illustration_id"])
        return ("crop_2" if selected["illustration_id"] == "v02_ill01" else "", False, "")

    monkeypatch.setattr(verification, "refine_crop", refine)
    accepted, retry = finalize(example, Client({"verdict": "fail", "reason": "Новая стадия."}))
    fallback = accepted["chapters"][0]["blocks"][1]
    assert calls == ["v01_ill01", "v02_ill01"]
    assert fallback["illustration_crop_id"] == "crop_2" and not retry
    archived = read_json(job / "attempt/illustration-coverage/v02_ill01/accepted-block.json")
    assert archived["illustration_crop_id"] == "crop_2"


def test_crop_failure_is_reviewable_and_does_not_hide_reconstruction_failure(example, monkeypatch):
    reason = "Не удалось подобрать границы исходного рисунка."
    monkeypatch.setattr(verification, "refine_crop", lambda *args: ("", True, reason))
    accepted, retry = finalize(example, Client({"verdict": "fail", "reason": "Неверная стрелка."}))
    assert retry and block(accepted)["kind"] == "screenshot"
    assert reason in accepted["issues"][0]["description"]
    assert "Неверная стрелка." in accepted["issues"][0]["description"]
