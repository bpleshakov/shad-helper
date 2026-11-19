import json
from copy import deepcopy
from pathlib import Path

import httpx
import pymupdf
import pytest

from shad_helper import illustration_crops
from shad_helper import illustration_verification as verification
from shad_helper.crop_validation import CROP_VERIFICATION_SCHEMA
from shad_helper.jobs import read_json, sha256_file, write_json
from shad_helper.openai_client import PlanClient, ResponseError
from shad_helper.rendering import RenderError
from shad_helper.schemas import LectureNotes


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
    monkeypatch.setattr(verification, "refine_crop", lambda *args: ("", False, "", False))
    monkeypatch.setattr(verification.time, "sleep", lambda *_: None)
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
        if schema == verification.REPAIR_SCHEMA:
            return {"content": r"\draw[->] (0,0)--(1,1);"}
        if instructions.endswith(verification.BATCH_INSTRUCTIONS):
            return {label: self.response for label in schema["required"]}
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
    assert len(client.calls) == 1 and client.calls[0][2] == verification.REPAIR_SCHEMA
    assert retry is False


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
    assert len(client.calls) == (1 if verdict == "pass" else 3)
    assert len(renders) == (1 if verdict == "pass" else 2)
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
        monkeypatch.setattr(verification, "VERIFICATION_POLICY_VERSION",
                            verification.VERIFICATION_POLICY_VERSION + 1)
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


def add_equivalent_observations(example):
    import cv2
    import numpy as np

    job, evidence, notes, _ = example
    add_aliased_observation(example)
    image = np.full((240, 400, 3), 240, np.uint8)
    cv2.arrowedLine(image, (40, 190), (340, 190), (20,) * 3, 2)
    cv2.putText(image, "A -> B", (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,) * 3, 2)
    manifest = read_json(job / "illustrations" / "manifest.json")
    evidence["visuals"] = evidence["visuals"][:1]
    notes["chapters"][0]["illustration_coverage"] = [
        {"illustration_id": "v01_ill01", "block_id": "drawing"}]
    for number in range(2, 5):
        visual_id, identifier = f"v{number:02d}", f"v{number:02d}_ill01"
        source = job / "illustrations" / f"{identifier}.png"
        cv2.imwrite(str(source), image - number)
        observation = {"id": identifier, "source_kind": "drawn", "box": [0, 0, 1, 1],
                       "description": f"Граф, наблюдение {number}",
                       "image_sha256": sha256_file(source)}
        evidence["visuals"].append({"id": visual_id, "timestamp": number,
                                     "status": "complete", "illustrations": [observation]})
        manifest["assets"][identifier] = {"path": f"illustrations/{identifier}.png",
                                          "sha256": observation["image_sha256"]}
        notes["chapters"][0]["illustration_coverage"].append(
            {"illustration_id": identifier, "block_id": "drawing"})
    write_json(job / "illustrations" / "manifest.json", manifest)


def test_equivalent_sources_verify_once_and_keep_all_coverage_and_archives(example):
    job, _, notes, _ = example
    add_equivalent_observations(example)
    original = deepcopy(notes)
    client = Client()
    accepted, retry = finalize(example, client)
    assert accepted == original and notes == original and not retry
    assert len(client.calls) == 1
    state_folder = job / "attempt/illustration-source-states"
    states = read_json(state_folder / "groups.json")
    assert len(states["groups"]) == 1
    representative = next(iter(states["groups"]))
    for identifier in ("v02_ill01", "v03_ill01", "v04_ill01"):
        archive = job / "attempt/illustration-coverage" / identifier
        assert read_json(archive / "accepted.json") == {
            "illustration_id": identifier, "block_id": "drawing"}
        assert read_json(archive / "source-state.json") == {
            "observation_id": identifier, "representative_id": representative}
        assert (archive / "artifacts/source.png").read_bytes() == (
            job / "illustrations" / f"{representative}.png").read_bytes()
    repeated, retry = finalize(example, client, "second-attempt")
    assert repeated == accepted and not retry and len(client.calls) == 1


@pytest.mark.parametrize("native", [False, True])
def test_equivalent_rejections_share_check_and_reuse_fallback(example, monkeypatch, native):
    if native:
        add_native_observations(example)
    else:
        add_equivalent_observations(example)

    class RejectTarget(Client):
        def complete(self, instructions, text, schema, *, images):
            self.calls.append((instructions, text, schema, images))
            target = pymupdf.Pixmap(images[1][1])
            return {"verdict": "fail" if target.width == 16 else "pass", "reason": "Проверено."}

    client = RejectTarget()
    accepted, retry = finalize(example, client)
    coverage = accepted["chapters"][0]["illustration_coverage"]
    assert not retry
    assert len(accepted["chapters"][0]["blocks"]) == 2
    assert len({item["block_id"] for item in coverage[1:]}) == 1
    assert coverage[1]["block_id"] != "drawing"
    assert sum(pymupdf.Pixmap(call[3][1][1]).width == 16 for call in client.calls) == 1
    assert len(client.calls) <= 2
    monkeypatch.setattr(verification, "group_source_states",
                        lambda observations, sources, **kwargs: {key: [key] for key in sources})
    monkeypatch.setattr(verification, "_cached_verdict", lambda *args: None)
    monkeypatch.setitem(verification.BATCH_SETTINGS, "max_sources", 1)
    baseline_client = RejectTarget()
    baseline, retry = finalize(example, baseline_client, "ungrouped-attempt")
    assert baseline == accepted and not retry
    assert len(baseline_client.calls) == 5


@pytest.mark.parametrize("response", [
    {"verdict": "uncertain", "reason": "Подпись не читается."}, OSError("Offline"),
])
def test_uncertain_group_check_retains_individual_verification(example, response):
    add_equivalent_observations(example)
    client = Client(response)
    accepted, retry = finalize(example, client)
    target_calls = [call for call in client.calls if pymupdf.Pixmap(call[3][1][1]).width == 16]
    assert len({call[1] for call in target_calls}) == 3
    assert retry is isinstance(response, Exception)
    assert len(accepted["chapters"][0]["illustration_coverage"]) == 4
    assert len(accepted["chapters"][0]["blocks"]) == 2
    assert len(target_calls) == (12 if isinstance(response, Exception) else 3)


def test_group_verdicts_are_specific_to_selected_target(example):
    _, _, notes, _ = example
    add_equivalent_observations(example)
    chapter = notes["chapters"][0]
    chapter["blocks"].append({**deepcopy(block(notes)), "id": "other",
                               "caption": "Другая область рисунка"})
    chapter["illustration_coverage"][-1]["block_id"] = "other"
    client = Client()
    accepted, retry = finalize(example, client)
    assert accepted == notes and not retry and len(client.calls) == 2


def test_changed_member_leaves_group_and_gets_fresh_verification(example):
    job, evidence, _, _ = example
    add_equivalent_observations(example)
    client = Client()
    finalize(example, client)
    assert len(client.calls) == 1
    source = png(job / "illustrations/v04_ill01.png", 80)
    digest = sha256_file(source)
    evidence["visuals"][-1]["illustrations"][0]["image_sha256"] = digest
    manifest = read_json(job / "illustrations/manifest.json")
    manifest["assets"]["v04_ill01"]["sha256"] = digest
    write_json(job / "illustrations/manifest.json", manifest)
    _, retry = finalize(example, client)
    assert not retry and len(client.calls) == 2
    states = read_json(job / "attempt/illustration-source-states/groups.json")
    assert states["groups"]["v04_ill01"] == ["v04_ill01"]
    assert read_json(job / "attempt/illustration-coverage/v04_ill01/source-state.json")[
        "representative_id"] == "v04_ill01"


def test_grouping_uses_actual_refined_sources(example, monkeypatch):
    job, evidence, notes, _ = example
    add_equivalent_observations(example)
    cropped = png(job / "refined.png", 80)
    notes["chapters"][0]["blocks"].append({
        **deepcopy(block(notes)), "id": "refined", "illustration_id": "v04_ill01",
        "illustration_crop_id": "refined-source"})
    original_resolve = verification.resolve_crop

    def resolve(job, evidence, identifier, crop_id=""):
        if identifier == "v04_ill01":
            assert crop_id == "refined-source"
            return cropped
        return original_resolve(job, evidence, identifier, crop_id)

    monkeypatch.setattr(verification, "resolve_crop", resolve)
    client = Client()
    assert not verification._finalize_coverage(notes, evidence, job, job / "attempt", client, [], {})
    assert len(client.calls) == 1
    assert sha256_file(cropped) in [sha256_file(path) for _, path in client.calls[0][3]]
    states = read_json(job / "attempt/illustration-source-states/groups.json")
    assert states["groups"]["v04_ill01"] == ["v04_ill01"]
    assert states["observations"]["v04_ill01"]["source_sha256"] == sha256_file(cropped)


def add_batch_observations(example, count=16):
    job, evidence, notes, _ = example
    block(notes).update(kind="screenshot", content="Учебный рисунок")
    evidence["visuals"][0]["illustrations"][0]["source_kind"] = "printed"
    manifest = read_json(job / "illustrations/manifest.json")
    coverage = [{"illustration_id": "v01_ill01", "block_id": "drawing"}]
    for number in range(2, count + 2):
        visual_id, identifier = f"v{number:02d}", f"v{number:02d}_ill01"
        path = png(job / "illustrations" / f"{identifier}.png", number)
        observation = {"id": identifier, "source_kind": "printed", "box": [0, 0, 1, 1],
                       "description": f"Источник {number}", "image_sha256": sha256_file(path)}
        evidence["visuals"].append({"id": visual_id, "timestamp": number,
                                     "status": "complete", "illustrations": [observation]})
        manifest["assets"][identifier] = {"path": f"illustrations/{identifier}.png",
                                          "sha256": observation["image_sha256"]}
        coverage.append({"illustration_id": identifier, "block_id": "drawing"})
    write_json(job / "illustrations/manifest.json", manifest)
    notes["chapters"][0]["illustration_coverage"] = coverage


@pytest.mark.parametrize("count,calls", [(8, 1), (16, 2), (17, 3)])
def test_coverage_batches_preserve_notes_and_cache_independent_decisions(example, count, calls):
    job, _, notes, _ = example
    add_batch_observations(example, count)
    original = deepcopy(notes)
    client = Client()
    accepted, retry = finalize(example, client)
    assert accepted == original and notes == original and not retry
    assert len(client.calls) == calls
    assert all(len(call[3]) <= 9 for call in client.calls)
    for instructions, _, schema, images in client.calls:
        if instructions.endswith(verification.BATCH_INSTRUCTIONS):
            assert images[0][0] == "RECONSTRUCTION"
            assert schema["required"] == [label for label, _ in images[1:]]
            assert schema["additionalProperties"] is False
    for number in range(2, min(count, 16) + 2):
        archive = job / "attempt/illustration-coverage" / f"v{number:02d}_ill01"
        provenance = read_json(archive / "artifacts/batch.json")
        batch = job / "attempt/illustration-batches" / provenance["batch_id"]
        result = read_json(batch / "response.json")[provenance["source_label"]]
        assert result == client.response
        assert sha256_file(batch / f"{provenance['source_label']}.png") == sha256_file(
            job / "illustrations" / f"v{number:02d}_ill01.png")
        assert read_json(archive / "verdict.json")["verdict"] == result["verdict"]
        assert read_json(batch / "timing.json")["seconds"] >= 0
    repeated, retry = finalize(example, client, "repeated")
    assert repeated == accepted and not retry and len(client.calls) == calls
    assert len(list((job / "repeated/illustration-batches").iterdir())) == count // 8


def test_mixed_batch_results_preserve_sequential_fallbacks_and_prose_refs(example, monkeypatch):
    add_batch_observations(example)
    _, _, notes, _ = example
    notes["chapters"][0]["blocks"].append({
        "id": "explanation", "kind": "text", "content": "Сравним рисунки.",
        "evidence_ids": ["v01"], "figure_refs": ["drawing"]})

    class Mixed(Client):
        def complete(self, instructions, text, schema, *, images):
            self.calls.append((instructions, text, schema, images))
            target = next(path for label, path in images if label == "RECONSTRUCTION")
            target_shade = pymupdf.Pixmap(target).samples[0]

            def verdict(path):
                shade = pymupdf.Pixmap(path).samples[0]
                # Two distinct rejected sources can share the first new fallback.
                passed = shade not in (3, 4) or target_shade in (3, 4)
                return {"verdict": "pass" if passed else "fail", "reason": f"Источник {shade}."}

            if instructions.endswith(verification.BATCH_INSTRUCTIONS):
                # Deliberately reverse response order: association must use the keys.
                return {label: verdict(path) for label, path in reversed(images[1:])}
            return verdict(images[0][1])

    client = Mixed()
    accepted, retry = finalize(example, client)
    assert not retry and len(client.calls) == 3  # Two batches plus one conditional alternative.
    coverage = accepted["chapters"][0]["illustration_coverage"]
    assert coverage[2]["block_id"] == coverage[3]["block_id"] != "drawing"
    assert all(item["block_id"] == "drawing" for index, item in enumerate(coverage)
               if index not in (2, 3))
    monkeypatch.setitem(verification.BATCH_SETTINGS, "max_sources", 1)
    monkeypatch.setattr(verification, "_cached_verdict", lambda *args: None)
    baseline = Mixed()
    sequential, retry = finalize(example, baseline, "sequential")
    assert sequential == accepted and not retry
    assert len(baseline.calls) == 17


@pytest.mark.parametrize("problem", ["missing", "extra", "group_verdict", "invalid_item", "transport"])
def test_broken_batches_fall_back_without_caching_partial_passes(example, problem):
    job, _, notes, _ = example
    add_batch_observations(example, 8)

    class BrokenBatch(Client):
        def complete(self, instructions, text, schema, *, images):
            result = super().complete(instructions, text, schema, images=images)
            if not instructions.endswith(verification.BATCH_INSTRUCTIONS):
                return result
            if problem == "transport":
                raise ResponseError("Batch exceeds context", code="context_length_exceeded")
            if problem == "missing":
                result.pop("SOURCE_01")
            elif problem == "extra":
                result["UNKNOWN"] = self.response
            elif problem == "group_verdict":
                return self.response
            else:
                result["SOURCE_01"] = {"verdict": "pass", "reason": ""}
            return result

    client = BrokenBatch()
    accepted, retry = finalize(example, client)
    assert accepted == notes and not retry and len(client.calls) == 9
    errors = list((job / "attempt/illustration-batches").glob("*/error.json"))
    assert len(errors) == 1
    assert not list((job / "illustration-verification").glob("*/batch.json"))
    finalize(example, client, "retry")
    assert len(client.calls) == 9


def test_only_uncertain_source_is_rechecked_individually(example):
    job, _, notes, _ = example
    add_batch_observations(example, 8)

    class UncertainBatch(Client):
        def complete(self, instructions, text, schema, *, images):
            result = super().complete(instructions, text, schema, images=images)
            if instructions.endswith(verification.BATCH_INSTRUCTIONS):
                result["SOURCE_04"] = {"verdict": "uncertain", "reason": "Мелкая подпись."}
            return result

    client = UncertainBatch()
    accepted, retry = finalize(example, client)
    assert accepted == notes and not retry and len(client.calls) == 2
    assert sha256_file(client.calls[-1][3][0][1]) == sha256_file(job / "illustrations/v05_ill01.png")
    assert len(list((job / "illustration-verification").glob("*/batch.json"))) == 7


@pytest.mark.parametrize("change", ["caption", "content", "model", "prompt", "source", "target",
                                   "corrupt_source", "missing_response", "wrong_source_label"])
def test_batch_cache_tracks_pair_inputs_and_batch_provenance(example, monkeypatch, change):
    job, evidence, notes, _ = example
    add_batch_observations(example, 8)
    client = Client()
    finalize(example, client)
    if change in {"caption", "content"}:
        block(notes)[change] += " Новый контекст."
    elif change == "model":
        client.model = "other-model"
    elif change == "prompt":
        monkeypatch.setattr(verification, "BATCH_INSTRUCTIONS", verification.BATCH_INSTRUCTIONS + "Updated.")
    elif change in {"source", "target"}:
        index = 1 if change == "source" else 0
        observation = evidence["visuals"][index]["illustrations"][0]
        path = png(job / "illustrations" / f"{observation['id']}.png", 90)
        observation["image_sha256"] = sha256_file(path)
        manifest = read_json(job / "illustrations/manifest.json")
        manifest["assets"][observation["id"]]["sha256"] = sha256_file(path)
        write_json(job / "illustrations/manifest.json", manifest)
    elif change == "corrupt_source":
        path = next((job / "illustration-verification").glob("*/batch.json"))
        png(path.parent / "source.png", 90)
    elif change == "wrong_source_label":
        path = next((job / "illustration-verification").glob("*/batch.json"))
        record = read_json(path)
        record["source_label"] = "SOURCE_01" if record["source_label"] != "SOURCE_01" else "SOURCE_02"
        write_json(path, record)
    else:
        next((job / "illustration-verification/batches").glob("*/response.json")).unlink()
    accepted, retry = finalize(example, client, "changed")
    assert accepted == notes and not retry and len(client.calls) == 2
    assert len(client.calls[-1][3]) == (2 if change in {
        "source", "corrupt_source", "wrong_source_label"} else 9)


def test_batches_keep_target_captions_separate_even_for_identical_pixels(example):
    job, _, notes, _ = example
    add_batch_observations(example, 16)
    chapter = notes["chapters"][0]
    chapter["blocks"].append({**deepcopy(block(notes)), "id": "other", "caption": "Другая часть"})
    for coverage in chapter["illustration_coverage"][9:]:
        coverage["block_id"] = "other"
    client = Client()
    accepted, retry = finalize(example, client)
    assert accepted == notes and not retry and len(client.calls) == 2
    contexts = [json.loads(call[1].split("Контекст: ", 1)[1]) for call in client.calls]
    assert [context["selected_figure"] for context in contexts] == ["Связь вершин", "Другая часть"]
    assert all(len(context["sources"]) == 8 for context in contexts)
    assert len(list((job / "attempt/illustration-batches").iterdir())) == 2


def test_batch_replay_retains_individual_results_and_validates_source_hashes(example):
    from scripts.illustration_batching_benchmark import replay

    job, _, _, _ = example
    add_batch_observations(example, 8)
    # Build a historical single-comparison archive with both passes and failures.
    attempt = job / "handoff/verification/attempt-01/illustration-coverage"
    for number in range(2, 10):
        archive = attempt / f"v{number:02d}_ill01"
        artifacts = archive / "artifacts"
        artifacts.mkdir(parents=True)
        source = job / "illustrations" / f"v{number:02d}_ill01.png"
        target = job / "illustrations/v01_ill01.png"
        (artifacts / "source.png").write_bytes(source.read_bytes())
        (artifacts / "preview.png").write_bytes(target.read_bytes())
        verification._compare(Client(), artifacts / "source.png", artifacts / "preview.png",
                              json.dumps({"observation": str(number), "selected_figure": "Общий рисунок",
                                          "selected_content": "Сравнение"}), artifacts,
                              instructions=verification.CONSOLIDATION_INSTRUCTIONS)
        write_json(archive / "candidate.json", {"block_id": "drawing"})
        write_json(archive / "verdict.json", {"verdict": "fail" if number == 3 else "pass"})
    destination = job.parent / "replay"
    report = replay(job, destination)
    assert report["distinct_initial_pairs"] == report["source_images"] == 8
    assert report["packaged_requests"] == report["mixed_verdict_batches"] == 1
    assert report["request_reduction_fraction"] == 0.875
    assignments = read_json(destination / "pairs.json")
    assert sum(item["recorded_verdicts"] == ["fail"] for item in assignments) == 1
    png(attempt / "v03_ill01/artifacts/source.png", 100)
    with pytest.raises(ValueError, match="hash mismatch"):
        replay(job, destination)


def test_batching_skips_cached_identical_and_excluded_sources(example):
    job, evidence, notes, _ = example
    add_batch_observations(example, 8)
    client = Client()
    observation = verification.illustration_index(evidence)["v02_ill01"]
    verification._consolidation(observation, block(notes), job, evidence, job / "prior", client)
    target = job / "illustrations/v01_ill01.png"
    source = job / "illustrations/v03_ill01.png"
    source.write_bytes(target.read_bytes())
    evidence["visuals"][2]["illustrations"][0]["image_sha256"] = sha256_file(source)
    manifest = read_json(job / "illustrations/manifest.json")
    manifest["assets"]["v03_ill01"]["sha256"] = sha256_file(source)
    write_json(job / "illustrations/manifest.json", manifest)
    notes["chapters"][0]["illustration_coverage"][3].update(block_id="", exclusion_reason="unresolved")
    accepted, retry = finalize(example, client)
    assert accepted == notes and not retry and len(client.calls) == 2
    assert len(client.calls[-1][3]) == 6  # Five pending sources plus one target.


def test_batches_respect_encoded_byte_budget_without_resizing(example, monkeypatch):
    job, _, notes, _ = example
    add_batch_observations(example, 8)
    sizes = [4 * ((path.stat().st_size + 2) // 3) for path in (job / "illustrations").glob("*.png")]
    # Fit two sources and a target, but never three sources.
    budget = max(sizes) * 3
    assert min(sizes) * 4 > budget
    monkeypatch.setitem(verification.BATCH_SETTINGS, "max_encoded_image_bytes", budget)
    client = Client()
    accepted, retry = finalize(example, client)
    assert accepted == notes and not retry and len(client.calls) == 4
    for _, _, _, images in client.calls:
        assert len(images) == 3
        assert sum(4 * ((path.stat().st_size + 2) // 3) for _, path in images) <= budget
        assert all(pymupdf.Pixmap(path).width == 16 for _, path in images)


def test_batch_records_exact_request_and_restores_plan_recorder(example):
    job, _, notes, _ = example
    add_batch_observations(example, 8)
    requests = []

    def handle(request):
        requests.append(request)
        payload = json.loads(request.content)
        response = {key: {"verdict": "pass", "reason": key + " проверен."}
                    for key in payload["text"]["format"]["schema"]["required"]}
        event = {"type": "response.completed", "response": {"status": "completed", "output": [
            {"content": [{"type": "output_text", "text": json.dumps(response)}]}]}}
        return httpx.Response(200, text="data: " + json.dumps(event) + "\n\n")

    recorder = lambda *args: pytest.fail("Batch must use its own recorder")
    client = PlanClient(job, proxy="http://proxy.invalid:8080", transport=httpx.MockTransport(handle),
                        token_provider=lambda: "private",
                        request_recorder=recorder, api_key="")
    client._available_models = [client.model]
    accepted, retry = finalize(example, client)
    assert accepted == notes and not retry and len(requests) == 1
    assert client.request_recorder is recorder
    batch = next((job / "attempt/illustration-batches").iterdir())
    assert (batch / "http-001.body.json").read_bytes() == requests[0].content
    images = [item for item in json.loads(requests[0].content)["input"][0]["content"]
              if item["type"] == "input_image"]
    assert len(images) == 9 and all(item["detail"] == "high" for item in images)


def add_native_observations(example):
    import cv2

    job, evidence, _, _ = example
    add_equivalent_observations(example)
    evidence["media"] = {"sha256": "b" * 64}
    (job / "frames").mkdir()
    original = cv2.imread(str(job / "illustrations/v02_ill01.png"))
    manifest = read_json(job / "illustrations/manifest.json")
    for index, (visual, box) in enumerate(zip(evidence["visuals"][1:], [
        (0, 0, 380, 240), (20, 0, 400, 240), (40, 0, 400, 240),
    ])):
        observation = visual["illustrations"][0]
        image = original if index else cv2.addWeighted(
            original, 0.96, cv2.GaussianBlur(original, (3, 3), 0), 0.04, 0)
        frame = job / "frames" / f"{visual['id']}_{'b' * 12}.png"
        assert cv2.imwrite(str(frame), image)
        x0, y0, x1, y1 = box
        source = job / "illustrations" / f"{observation['id']}.png"
        assert cv2.imwrite(str(source), image[y0:y1, x0:x1])
        observation.update(box=[x0 / 400, y0 / 240, x1 / 400, y1 / 240],
                           image_sha256=sha256_file(source))
        manifest["assets"][observation["id"]]["sha256"] = observation["image_sha256"]
    write_json(job / "illustrations/manifest.json", manifest)


def test_group_verifier_receives_the_complete_native_state_and_its_provenance(example):
    job, _, notes, _ = example
    add_native_observations(example)
    client = Client()
    accepted, retry = finalize(example, client)
    assert accepted == notes and not retry and len(client.calls) == 1
    assert pymupdf.Pixmap(client.calls[0][3][0][1]).width == 400
    folder = job / "attempt/illustration-source-states"
    states = read_json(folder / "groups.json")
    representative = next(iter(states["groups"]))
    assert representative == "v03_ill01"
    assert states["contexts"][representative]["region"] == [0, 0, 400, 240]
    assert pymupdf.Pixmap(job / "illustrations" / f"{representative}.png").width == 380
    assert pymupdf.Pixmap(folder / representative / "source.png").width == 400
    repeated, retry = finalize(example, client, "second-attempt")
    assert repeated == accepted and not retry and len(client.calls) == 1


@pytest.mark.parametrize("offline", [False, True])
def test_uncertain_native_state_checks_every_original_crop_including_representative(example, offline):
    _, _, notes, _ = example
    add_native_observations(example)

    class NativeClient(Client):
        def complete(self, instructions, text, schema, *, images):
            self.calls.append((instructions, text, schema, images))
            if pymupdf.Pixmap(images[0][1]).width == 400:
                if offline:
                    raise OSError("Offline")
                return {"verdict": "uncertain", "reason": "Неясна область рисунка."}
            return {"verdict": "pass", "reason": "Исходный рисунок сохранён."}

    client = NativeClient()
    accepted, retry = finalize(example, client)
    assert accepted == notes and not retry
    assert len(client.calls) == (7 if offline else 4)
    assert sum(pymupdf.Pixmap(call[3][0][1]).width < 400 for call in client.calls) == 3


@pytest.mark.parametrize("verdict", ["fail", "uncertain"])
def test_distinct_or_uncertain_observation_gets_separate_crop_and_mapping(example, verdict):
    job, _, notes, _ = example
    add_aliased_observation(example)
    client = Client({"verdict": verdict, "reason": "Отдельная содержательная стадия."})
    accepted, retry = finalize(example, client)
    blocks = accepted["chapters"][0]["blocks"]
    assert [item["illustration_id"] for item in blocks] == ["v01_ill01", "v02_ill01"]
    assert blocks[1]["kind"] == "screenshot"
    assert blocks[1]["caption"] == blocks[1]["content"] == block(notes)["caption"]
    assert accepted["chapters"][0]["illustration_coverage"][1]["block_id"] == blocks[1]["id"]
    assert accepted["issues"][0]["evidence_ids"] == ["v02_ill01"]
    assert not retry and len(notes["chapters"][0]["blocks"]) == 1
    archive = job / "attempt" / "illustration-coverage" / "v02_ill01"
    assert read_json(archive / "accepted-block.json") == blocks[1]


@pytest.mark.parametrize("outcome", [
    "unusable", "uncertain", "occluded", "offline", "invalid", "correctable",
])
def test_real_crop_result_controls_coverage_fallback_without_losing_evidence(example, monkeypatch, outcome):
    job, evidence, notes, _ = example
    add_aliased_observation(example)
    notes["chapters"][0]["blocks"].append({
        "id": "explanation", "kind": "text", "content": "Сравним стадии.",
        "evidence_ids": ["v01"], "figure_refs": ["drawing"]})
    evidence["media"] = {"sha256": "a" * 64}
    png(job / "frames" / f"v01_{'a' * 12}.png", 220)
    png(job / "frames" / f"v02_{'a' * 12}.png", 220 if outcome == "correctable" else 100)
    original = {path: path.read_bytes() for directory in ("frames", "illustrations")
                for path in (job / directory).iterdir()}
    monkeypatch.setattr(verification, "refine_crop", illustration_crops.refine_crop)
    reason = "В кадре остались только заголовок и край графика."

    class CropClient(Client):
        def complete(self, instructions, text, schema, *, images):
            if schema == illustration_crops.SOURCE_SELECTION_SCHEMA:
                return {"ranked_frame_ids": [json.loads(text)["original_frame"]["visual_id"]],
                        "reason": "Сохраняем исходную стадию."}
            if schema == CROP_VERIFICATION_SCHEMA:
                return {"verdict": "pass", "checks": {
                    check: "pass" for check in
                    ("identity", "completeness", "readability", "cleanliness")},
                    "defects": [], "repair": "none", "reason": "Полный выбранный рисунок."}
            if instructions != illustration_crops.CROP_INSTRUCTIONS:
                return {"verdict": "uncertain", "reason": "По исходному фрагменту сравнить нельзя."}
            if json.loads(text)["original_frame"]["visual_id"] == "v01" or outcome == "correctable":
                return {"status": "cropped", "box": [0, 0, 16, 12], "reason": "Полный рисунок."}
            if outcome == "offline":
                raise OSError("Offline")
            if outcome == "invalid":
                return {"status": "unusable", "box": [False, 0, 0, 0], "reason": reason}
            return {"status": "unusable" if outcome == "unusable" else "uncertain",
                    "box": [0, 0, 0, 0],
                    "reason": "Лектор закрыл подпись, но стрелки читаются." if outcome == "occluded" else reason}

    accepted, retry = finalize(example, CropClient())
    chapter = accepted["chapters"][0]
    figures = [item for item in chapter["blocks"] if item.get("illustration_id")]
    coverage = chapter["illustration_coverage"][1]
    prose = next(item for item in chapter["blocks"] if item["id"] == "explanation")
    assert {item["illustration_id"] for item in chapter["illustration_coverage"]} == {"v01_ill01", "v02_ill01"}
    assert all(path.read_bytes() == data for path, data in original.items())
    assert retry is (outcome in {"offline", "invalid"})
    if outcome == "unusable":
        assert len(figures) == 1 and prose["figure_refs"] == ["drawing"]
        assert coverage == {"illustration_id": "v02_ill01", "block_id": "",
                            "exclusion_reason": "no_instructional_visual"}
        archive = job / "attempt/illustration-coverage/v02_ill01"
        assert read_json(archive / "accepted.json") == coverage
        assert not (archive / "accepted-block.json").exists()
    elif outcome == "correctable":
        assert len(figures) == 1 and coverage["block_id"] == "drawing"
        assert prose["figure_refs"] == ["drawing"]
    else:
        assert len(figures) == 1 and coverage["block_id"] == ""
        assert coverage["exclusion_reason"] == "unresolved"
        assert prose["figure_refs"] == ["drawing"]
        assert all(item["kind"] != "uncertain" for item in chapter["blocks"])
        assert any(issue["status"] == "open" and "v02_ill01" in issue["evidence_ids"]
                   for issue in accepted["issues"])
    LectureNotes.model_validate(accepted)


def test_unusable_selected_figure_is_archived_without_publishing_empty_pixels(example, monkeypatch):
    _, evidence, notes, _ = example
    block(notes)["kind"] = "screenshot"
    evidence["visuals"][0]["illustrations"][0]["source_kind"] = "printed"
    reason = "Видны только заголовок и край рисунка."
    monkeypatch.setattr(verification, "refine_crop", lambda *args: ("", False, reason, True))
    accepted, retry = finalize(example, Client())
    assert accepted["chapters"][0]["blocks"] == [] and not retry
    assert block(notes)["kind"] == "screenshot"
    assert reason in accepted["issues"][0]["description"]


def test_unusable_selected_sources_leave_coverage_and_review_records(example, monkeypatch):
    _, _, notes, _ = example
    add_aliased_observation(example)
    selected = {**deepcopy(block(notes)), "id": "selected",
                "illustration_id": "v02_ill01", "evidence_ids": ["v02_ill01"]}
    notes["chapters"][0]["blocks"].append(selected)
    monkeypatch.setattr(verification, "refine_crop", lambda *args: ("", False, "Фрагмент.", True))
    accepted, _ = finalize(example, Client({"verdict": "uncertain", "reason": "Сравнить нельзя."}))
    assert not any(item.get("illustration_id") for item in accepted["chapters"][0]["blocks"])
    assert accepted["chapters"][0]["illustration_coverage"][1]["exclusion_reason"]
    assert accepted["issues"]
    LectureNotes.model_validate(accepted)


def test_unpublishing_recovered_source_keeps_failed_alias_unresolved(example):
    _, _, notes, _ = example
    failed = block(notes)
    good = {**deepcopy(failed), "id": "good", "illustration_id": "v03_ill01",
            "evidence_ids": ["v03_ill01"]}
    chapter = notes["chapters"][0]
    chapter["blocks"].extend([good, {"id": "explanation", "kind": "text",
                                     "content": "См. рисунок.", "evidence_ids": ["v01_ill01"],
                                     "figure_refs": [failed["id"]]}])
    chapter["illustration_coverage"] = [
        {"illustration_id": "v01_ill01", "block_id": "good", "exclusion_reason": ""},
        {"illustration_id": "v02_ill01", "block_id": failed["id"], "exclusion_reason": ""}]
    verification._unpublish(notes, failed, "Второе наблюдение не удалось сопоставить.",
                            accepted_ids={"good"})
    assert chapter["illustration_coverage"] == [
        {"illustration_id": "v01_ill01", "block_id": "good", "exclusion_reason": ""},
        {"illustration_id": "v02_ill01", "block_id": "", "exclusion_reason": "unresolved"}]
    assert [item["id"] for item in chapter["blocks"]] == ["good", "explanation"]
    assert chapter["blocks"][1]["figure_refs"] == ["good"]
    assert any(issue["status"] == "open" and "v02_ill01" in issue["evidence_ids"]
               for issue in notes["issues"])
    assert all(item["kind"] != "uncertain" for item in chapter["blocks"])


@pytest.mark.parametrize("accepted_ids,retained,recovered", [
    (None, True, True), ({"good"}, True, True), (set(), True, False), (None, False, False),
])
def test_unpublishing_redirects_refs_only_to_retained_accepted_figures(
        example, accepted_ids, retained, recovered):
    _, _, notes, _ = example
    failed = block(notes)
    chapter = notes["chapters"][0]
    if retained:
        chapter["blocks"].append({**deepcopy(failed), "id": "good"})
    chapter["blocks"].append({"id": "explanation", "kind": "text", "content": "См. рисунок.",
                              "evidence_ids": ["v01_ill01"], "figure_refs": [failed["id"]]})
    chapter["illustration_coverage"] = [
        {"illustration_id": "v01_ill01", "block_id": "good", "exclusion_reason": ""}]
    verification._issue(notes, failed, "Рисунок не прошёл проверку.")
    verification._unpublish(notes, failed, "Источник не восстановлен.", accepted_ids=accepted_ids)
    assert all(item["id"] != failed["id"] for item in chapter["blocks"])
    prose = next(item for item in chapter["blocks"] if item["id"] == "explanation")
    assert prose["figure_refs"] == (["good"] if recovered else [])
    if recovered:
        assert notes["issues"] == []
    else:
        assert any(issue["status"] == "open" and "v01_ill01" in issue["evidence_ids"]
                   for issue in notes["issues"])


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
    assert len(client.calls) == 5


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
    assert (50, 100) in client.calls
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
    monkeypatch.setattr(verification, "refine_crop", lambda *args: ("crop_1", False, "", False))

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
    monkeypatch.setattr(verification, "refine_crop", lambda *args: ("crop_1", False, "", False))
    monkeypatch.setattr(verification, "resolve_crop", lambda *args: cropped)
    client = Client()
    accepted, retry = finalize(example, client)
    assert block(accepted)["illustration_crop_id"] == "crop_1"
    assert not client.calls and not retry and not accepted["issues"]


def test_consolidation_checks_refined_pixels_even_when_original_hashes_match(example, monkeypatch):
    job, _, _, _ = example
    add_aliased_observation(example, identical=True)
    cropped = png(job / "refined.png", 80)
    monkeypatch.setattr(verification, "refine_crop", lambda *args: ("crop_1", False, "", False))
    resolve = verification.resolve_crop
    monkeypatch.setattr(verification, "resolve_crop", lambda job, evidence, identifier, crop_id="":
                        cropped if crop_id else resolve(job, evidence, identifier))
    client = Client()
    accepted, retry = finalize(example, client)
    assert not retry and len(accepted["chapters"][0]["blocks"]) == 1
    assert len(client.calls) == 1
    for instructions, _, _, images in client.calls:
        assert instructions == verification.CONSOLIDATION_INSTRUCTIONS
        assert pymupdf.Pixmap(images[0][1]).samples[0] == 220
        assert images[1][1].read_bytes() == cropped.read_bytes()
    finalize(example, client, "second-attempt")
    assert len(client.calls) == 1
    png(cropped, 90)
    finalize(example, client, "third-attempt")
    assert len(client.calls) == 2


def test_failed_consolidation_refines_new_fallback_before_accepting_it(example, monkeypatch):
    job, _, _, _ = example
    add_aliased_observation(example)
    calls = []

    def refine(job, evidence, selected, client, folder):
        calls.append(selected["illustration_id"])
        return ("crop_2" if selected["illustration_id"] == "v02_ill01" else "", False, "", False)

    monkeypatch.setattr(verification, "refine_crop", refine)
    cropped = png(job / "refined.png", 80)
    resolve = verification.resolve_crop
    monkeypatch.setattr(verification, "resolve_crop", lambda job, evidence, identifier, crop_id="":
                        cropped if crop_id else resolve(job, evidence, identifier))
    accepted, retry = finalize(example, Client({"verdict": "fail", "reason": "Новая стадия."}))
    fallback = accepted["chapters"][0]["blocks"][1]
    assert calls == ["v01_ill01", "v02_ill01"]
    assert fallback["illustration_crop_id"] == "crop_2" and not retry
    archived = read_json(job / "attempt/illustration-coverage/v02_ill01/accepted-block.json")
    assert archived["illustration_crop_id"] == "crop_2"


@pytest.mark.parametrize("kind", ["diagram", "screenshot"])
@pytest.mark.parametrize("transient", [False, True])
def test_unverified_crop_preserves_source_for_review_without_reconstruction(
        example, monkeypatch, kind, transient):
    job, evidence, notes, renders = example
    block(notes)["kind"] = kind
    notes["chapters"][0]["illustration_coverage"] = [
        {"illustration_id": "v01_ill01", "block_id": "drawing"}]
    evidence["visuals"][0]["illustrations"][0]["source_kind"] = "printed"
    reason = "Выбран другой рисунок; исправить границы не удалось."
    monkeypatch.setattr(verification, "refine_crop", lambda *args: ("", transient, reason, False))
    client = Client()
    accepted, retry = finalize(example, client)
    assert retry is transient and accepted["chapters"][0]["blocks"] == []
    assert accepted["chapters"][0]["illustration_coverage"] == [
        {"illustration_id": "v01_ill01", "block_id": "", "exclusion_reason": "unresolved"}]
    assert reason in accepted["issues"][0]["description"]
    assert accepted["issues"][0]["evidence_ids"] == ["v01_ill01"]
    assert accepted["issues"][0]["status"] == "open"
    assert not renders and not client.calls
    archive = job / "attempt/illustration-verification/drawing"
    assert read_json(archive / "verdict.json") == {
        "verdict": "uncertain", "reason": reason, "retry_needed": transient}
    assert read_json(archive / "accepted.json") is None
    assert read_json(archive / "candidate.json") == block(notes)
    assert read_json(archive / "publication.json")["status"] == "unresolved"
    assert (archive / "source.png").read_bytes() == (
        job / "illustrations/v01_ill01.png").read_bytes()


def test_unsuccessful_refinement_removes_imported_crop_reference(example, monkeypatch):
    job, _, notes, renders = example
    block(notes)["illustration_crop_id"] = "f" * 64
    reason = "Исходный фрагмент не прошёл проверку."
    monkeypatch.setattr(verification, "refine_crop", lambda *args: ("", False, reason, False))
    accepted, retry = finalize(example, Client())
    assert not retry and not renders
    assert accepted["chapters"][0]["blocks"] == []
    assert block(notes)["illustration_crop_id"] == "f" * 64
    assert accepted["issues"][0]["evidence_ids"] == ["v01_ill01"]
    archive = job / "attempt/illustration-verification/drawing"
    assert read_json(archive / "candidate.json")["illustration_crop_id"] == "f" * 64
    assert (archive / "source.png").read_bytes() == (
        job / "illustrations/v01_ill01.png").read_bytes()


def test_own_source_does_not_create_duplicate_when_comparison_is_unavailable(example, monkeypatch):
    job, evidence, notes, _ = example
    block(notes)["kind"] = "screenshot"
    evidence["visuals"][0]["illustrations"][0]["source_kind"] = "printed"
    notes["chapters"][0]["illustration_coverage"] = [
        {"illustration_id": "v01_ill01", "block_id": "drawing"}]
    cropped = png(job / "refined.png", 80)
    monkeypatch.setattr(verification, "refine_crop", lambda *args: ("crop_1", False, "", False))
    monkeypatch.setattr(verification, "resolve_crop", lambda *args: cropped)
    client = Client(OSError("Connection interrupted"))
    accepted, retry = finalize(example, client)
    assert len(accepted["chapters"][0]["blocks"]) == 1
    assert not client.calls and not retry and not accepted["issues"]


@pytest.mark.parametrize("failure", [ResponseError("server_is_overloaded"), httpx.ReadError("Interrupted")])
def test_transient_comparison_retries_before_inserting_duplicate(example, failure):
    add_aliased_observation(example)

    class RecoveringClient(Client):
        def complete(self, *args, **kwargs):
            reply = super().complete(*args, **kwargs)
            if len(self.calls) == 1:
                raise failure
            return reply

    client = RecoveringClient()
    accepted, retry = finalize(example, client)
    assert len(client.calls) == 2 and not retry
    assert len(accepted["chapters"][0]["blocks"]) == 1 and not accepted["issues"]


def test_refined_source_is_recompared_before_adding_redundant_fallback(example, monkeypatch):
    job, _, _, _ = example
    add_aliased_observation(example)
    cropped = png(job / "refined.png", 220)
    monkeypatch.setattr(verification, "refine_crop", lambda job, evidence, block, *args:
                        ("crop_2" if block["illustration_id"] == "v02_ill01" else "", False, "", False))
    resolve = verification.resolve_crop
    monkeypatch.setattr(verification, "resolve_crop", lambda job, evidence, identifier, crop_id="":
                        cropped if crop_id else resolve(job, evidence, identifier))
    client = Client({"verdict": "fail", "reason": "Разные поля кадра."})
    accepted, retry = finalize(example, client)
    assert len(accepted["chapters"][0]["blocks"]) == 1 and not retry
    assert len(client.calls) == 1  # Refined source pixels exactly match the representative.


def test_refined_source_rechecks_other_matching_figures_before_publishing_duplicate(example, monkeypatch):
    job, evidence, notes, _ = example
    add_aliased_observation(example)
    source = png(job / "illustrations/v03_ill01.png", 180)
    observation = {**evidence["visuals"][1]["illustrations"][0], "id": "v03_ill01",
                   "image_sha256": sha256_file(source)}
    evidence["visuals"].append({"id": "v03", "timestamp": 8, "status": "complete",
                               "illustrations": [observation]})
    manifest = read_json(job / "illustrations/manifest.json")
    manifest["assets"]["v03_ill01"] = {"path": "illustrations/v03_ill01.png",
                                       "sha256": observation["image_sha256"]}
    write_json(job / "illustrations/manifest.json", manifest)
    chapter = notes["chapters"][0]
    chapter["blocks"].append({**deepcopy(block(notes)), "id": "other_figure",
                              "illustration_id": "v03_ill01", "evidence_ids": ["v03_ill01"]})
    chapter["illustration_coverage"].append({"illustration_id": "v03_ill01",
                                             "block_id": "other_figure"})
    monkeypatch.setattr(verification, "refine_crop", lambda job, evidence, candidate, *args:
                        ("crop_2" if candidate["illustration_id"] == "v02_ill01" else "",
                         False, "", False))
    resolve = verification.resolve_crop
    monkeypatch.setattr(verification, "resolve_crop", lambda job, evidence, identifier, crop_id="":
                        source if crop_id == "crop_2" else resolve(job, evidence, identifier))
    client = Client({"verdict": "fail", "reason": "Исходные фрагменты различаются."})
    accepted, retry = finalize(example, client)
    chapter = accepted["chapters"][0]
    assert not retry and [item["id"] for item in chapter["blocks"]] == ["drawing", "other_figure"]
    assert chapter["illustration_coverage"][1]["block_id"] == "other_figure"
    archive = job / "attempt/illustration-coverage/v02_ill01"
    assert read_json(archive / "alternatives/other_figure/verdict.json")["verdict"] == "fail"
    assert read_json(archive / "refined/other_figure/verdict.json")["verdict"] == "pass"
    assert not (archive / "accepted-block.json").exists()


def test_later_recovered_figure_rescues_earlier_failed_observations_and_prose_refs(example, monkeypatch):
    job, evidence, notes, _ = example
    add_aliased_observation(example)
    evidence["visuals"][1]["illustrations"][0]["source_kind"] = "printed"
    source = png(job / "illustrations/v03_ill01.png", 180)
    observation = {**evidence["visuals"][1]["illustrations"][0], "id": "v03_ill01",
                   "image_sha256": sha256_file(source)}
    evidence["visuals"].append({"id": "v03", "timestamp": 8, "status": "complete",
                               "illustrations": [observation]})
    manifest = read_json(job / "illustrations/manifest.json")
    manifest["assets"]["v03_ill01"] = {"path": "illustrations/v03_ill01.png",
                                       "sha256": observation["image_sha256"]}
    write_json(job / "illustrations/manifest.json", manifest)
    chapter = notes["chapters"][0]
    chapter["illustration_coverage"].append({"illustration_id": "v03_ill01", "block_id": "drawing"})
    chapter["blocks"].append({"id": "explanation", "kind": "text", "content": "Связи на рисунке.",
                              "evidence_ids": ["v01"], "figure_refs": ["drawing"]})
    monkeypatch.setattr(verification, "refine_crop", lambda job, evidence, candidate, *args:
                        ("", False, "" if candidate["illustration_id"] == "v03_ill01"
                         else "Не удаётся выделить полный рисунок.", False))
    accepted, retry = finalize(example, Client())
    chapter = accepted["chapters"][0]
    figures = [item for item in chapter["blocks"] if item.get("illustration_id")]
    assert not retry and len(figures) == 1 and figures[0]["illustration_id"] == "v03_ill01"
    assert all(item["block_id"] == figures[0]["id"] and not item.get("exclusion_reason")
               for item in chapter["illustration_coverage"])
    assert next(item for item in chapter["blocks"] if item["id"] == "explanation")["figure_refs"] == [
        figures[0]["id"]]
    assert all(item["kind"] != "uncertain" for item in chapter["blocks"])
    assert not accepted["issues"]
    for identifier in ("v01_ill01", "v02_ill01"):
        archive = job / "attempt/illustration-recovery" / identifier / figures[0]["id"]
        assert read_json(archive / "verdict.json")["verdict"] == "pass"


def test_matching_figures_prioritizes_topic_over_generic_description_similarity():
    generic = "Обзор кадра с возможными учебными иллюстрациями."
    observations = {"current": {"id": "current", "source_kind": "printed", "description": generic,
                                 "timestamp": 100, "image_sha256": "current"}}
    current = {"id": "current", "illustration_id": "current", "topic": "Распределения"}
    candidates = []
    for identifier in ("knn1", "knn2", "knn3", "knn4", "histograms"):
        observations[identifier] = {"id": identifier, "source_kind": "printed", "timestamp": 90,
                                    "image_sha256": identifier, "description": (
                                        "Три гистограммы дохода и числа покупок." if identifier == "histograms"
                                        else generic)}
        candidates.append({"id": identifier, "illustration_id": identifier,
                           "topic": "Распределения" if identifier == "histograms" else "Ближайшие соседи"})
    ranked = verification._matching_figures(current, candidates, observations)
    assert len(ranked) == 3 and ranked[0]["id"] == "histograms"


@pytest.mark.parametrize("has_match,accepted_match", [(False, False), (True, True), (True, False)])
def test_visual_shortlist_reduces_calls_without_changing_notes(
        example, monkeypatch, has_match, accepted_match):
    import cv2
    import numpy as np

    job, evidence, notes, _ = example
    add_aliased_observation(example)
    chapter = notes["chapters"][0]
    manifest = read_json(job / "illustrations/manifest.json")
    sources = []
    for seed in range(4):
        image = np.full((320, 480, 3), 245, np.uint8)
        rng = np.random.default_rng(seed)
        for number in range(45):
            point = tuple(int(value) for value in rng.integers([10, 20], [430, 290]))
            cv2.putText(image, f"{seed}:{number}", point, cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (20, 30, 40), 1, cv2.LINE_AA)
        if seed == 3 and has_match:
            image = cv2.resize(sources[0], None, fx=0.8, fy=0.8)
            if not accepted_match:
                cv2.putText(image, "CHANGED", (60, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 2)
        sources.append(image)
        identifier = f"v{seed + 2:02}_ill01"
        path = job / "illustrations" / f"{identifier}.png"
        assert cv2.imwrite(str(path), image)
        observation = {"id": identifier, "source_kind": "printed", "box": [0, 0, 1, 1],
                       "description": "Учебный график", "image_sha256": sha256_file(path)}
        manifest["assets"][identifier] = {"path": f"illustrations/{identifier}.png",
                                          "sha256": observation["image_sha256"]}
        if seed == 0:
            evidence["visuals"][1]["illustrations"] = [observation]
        else:
            evidence["visuals"].append({"id": f"v{seed + 2:02}", "timestamp": 5 + seed,
                                         "status": "complete", "illustrations": [observation]})
            chapter["blocks"].append({**deepcopy(block(notes)), "id": f"alternative{seed}",
                "illustration_id": identifier, "caption": "График", "content": "График",
                "topic": "Другое название" if seed == 3 else block(notes)["topic"]})
    write_json(job / "illustrations/manifest.json", manifest)
    match_hash = manifest["assets"]["v05_ill01"]["sha256"]

    class Compare(Client):
        def complete(self, instructions, text, schema, *, images):
            self.calls.append((instructions, text, schema, images))
            passed = accepted_match and sha256_file(images[1][1]) == match_hash
            return {"verdict": "pass" if passed else "fail", "reason": "Сравнение рисунков."}

    optimized = Compare()
    accepted, retry = finalize(example, optimized)
    assert not retry and len(optimized.calls) == (2 if has_match else 1)
    search = read_json(job / "attempt/illustration-coverage/v02_ill01/matching.json")
    assert search["selected"] == (["alternative3"] if has_match else [])
    matching = verification._matching_figures
    monkeypatch.setattr(verification, "_matching_figures", lambda current, retained, observations,
                        **kwargs: matching(current, retained, observations))
    monkeypatch.setattr(verification, "_cached_verdict", lambda *args: None)
    baseline = Compare()
    previous, retry = finalize(example, baseline, "baseline")
    assert not retry and previous == accepted
    assert len(baseline.calls) == 4
    if not accepted_match:
        assert any(item.get("illustration_id") == "v02_ill01"
                   for item in accepted["chapters"][0]["blocks"])


def test_new_distinct_fallback_extends_existing_prose_references(example):
    _, _, notes, _ = example
    add_aliased_observation(example)
    notes["chapters"][0]["blocks"].append({
        "id": "explanation", "kind": "text", "content": "Сравним стадии.",
        "evidence_ids": ["v01"], "figure_refs": ["drawing"]})
    accepted, _ = finalize(example, Client({"verdict": "fail", "reason": "Иная стадия."}))
    figures = [item for item in accepted["chapters"][0]["blocks"] if item.get("illustration_id")]
    prose = next(item for item in accepted["chapters"][0]["blocks"] if item["id"] == "explanation")
    assert prose["figure_refs"] == ["drawing", figures[1]["id"]]


def paired_figures(example, *, across_chapters=False):
    _, evidence, notes, _ = example
    add_aliased_observation(example)
    evidence["visuals"][1]["illustrations"][0]["source_kind"] = "printed"
    later = {**deepcopy(block(notes)), "id": "later", "illustration_id": "v02_ill01",
             "evidence_ids": ["v02_ill01"], "caption": "Завершённый рисунок"}
    chapter = notes["chapters"][0]
    chapter["blocks"].insert(0, {"id": "explanation", "kind": "text", "content": "Объяснение.",
                                 "evidence_ids": ["v01"], "figure_refs": ["drawing", "later"]})
    chapter["illustration_coverage"][1]["block_id"] = "later"
    if across_chapters:
        chapter["end"] = 3
        notes["chapters"].append({"id": "c02", "title": "Продолжение", "start": 3, "end": 10,
                                  "blocks": [later], "illustration_coverage": [
                                      chapter["illustration_coverage"].pop()]})
    else:
        chapter["blocks"].append(later)
    return notes


@pytest.mark.parametrize("relation,preferred", [("same", "left"), ("left_contains_right", "left"),
                                                ("right_contains_left", "right")])
def test_reconciliation_keeps_best_view_and_remaps_cross_chapter_references(example, relation, preferred):
    job, evidence, _, _ = example
    notes = paired_figures(example, across_chapters=True)
    original = deepcopy(notes)
    dropped = "drawing" if preferred == "right" else "later"
    kept = "later" if preferred == "right" else "drawing"
    figures = [item for chapter in notes["chapters"] for item in chapter["blocks"]
               if item.get("illustration_id")]
    verification._issue(notes, next(item for item in figures if item["id"] == dropped), "Старая обрезка.")
    client = Client({"relation": relation, "preferred": preferred, "reason": "Выбран полный вид."})
    accepted, retry = verification.reconcile_illustrations(notes, evidence, job, job / "reconcile", client)
    blocks = [item for chapter in accepted["chapters"] for item in chapter["blocks"]]
    assert [item["id"] for item in blocks] == ["explanation", kept]
    assert blocks[0]["figure_refs"] == [kept]
    assert all(item["block_id"] == kept for chapter in accepted["chapters"]
               for item in chapter["illustration_coverage"])
    assert accepted["chapters"][0 if preferred == "left" else 1]["blocks"][-1]["id"] == kept
    assert not accepted["issues"] and not retry and len(client.calls) == 1
    assert notes["chapters"] == original["chapters"]
    assert "Завершённый рисунок" in client.calls[0][1]
    again, retry = verification.reconcile_illustrations(notes, evidence, job, job / "again", client)
    assert again == accepted and not retry and len(client.calls) == 1


@pytest.mark.parametrize("relation", ["distinct", "uncertain"])
def test_reconciliation_preserves_independent_or_uncertain_figures(example, relation):
    job, evidence, _, _ = example
    notes = paired_figures(example)
    client = Client({"relation": relation, "preferred": "neither", "reason": "Самостоятельные стадии."})
    accepted, retry = verification.reconcile_illustrations(notes, evidence, job, job / "reconcile", client)
    assert accepted == notes and not retry and len(client.calls) == 1


def test_reconciliation_never_discards_figure_on_unavailable_verification(example):
    job, evidence, _, _ = example
    notes = paired_figures(example)
    client = Client(OSError("Offline"))
    accepted, retry = verification.reconcile_illustrations(notes, evidence, job, job / "reconcile", client)
    assert accepted == notes and retry and len(client.calls) == 4


def test_reconciliation_skips_different_source_kinds(example):
    job, evidence, _, _ = example
    notes = paired_figures(example)
    evidence["visuals"][1]["illustrations"][0]["source_kind"] = "drawn"
    client = Client()
    accepted, retry = verification.reconcile_illustrations(notes, evidence, job, job / "reconcile", client)
    assert accepted == notes and not retry and not client.calls


def test_reconciliation_same_source_preserves_prose_references_without_model_call(example):
    job, evidence, _, _ = example
    notes = paired_figures(example)
    chapter = notes["chapters"][0]
    chapter["blocks"][-1].update(illustration_id="v01_ill01", evidence_ids=["v01_ill01"],
                                 caption=chapter["blocks"][1]["caption"])
    chapter["blocks"].append({"id": "later_explanation", "kind": "text", "content": "Продолжение.",
                              "evidence_ids": ["v02"], "figure_refs": ["later"]})
    client = Client(OSError("Offline"))
    accepted, retry = verification.reconcile_illustrations(notes, evidence, job, job / "reconcile", client)
    assert [item["id"] for item in accepted["chapters"][0]["blocks"]] == [
        "explanation", "drawing", "later_explanation"]
    assert accepted["chapters"][0]["blocks"][0]["figure_refs"] == ["drawing"]
    assert accepted["chapters"][0]["blocks"][2]["figure_refs"] == ["drawing"]
    assert not client.calls and not retry


@pytest.mark.parametrize("identical_pixels", [False, True])
def test_reconciliation_compares_different_crops_of_same_source(example, monkeypatch, identical_pixels):
    job, evidence, _, _ = example
    notes = paired_figures(example)
    first, second = notes["chapters"][0]["blocks"][1:]
    first["illustration_crop_id"] = "a" * 64
    second.update(illustration_id=first["illustration_id"], illustration_crop_id="b" * 64,
                  caption=first["caption"],
                  evidence_ids=first["evidence_ids"])
    crops = {first["illustration_crop_id"]: png(job / "left.png", 220),
             second["illustration_crop_id"]: png(job / "right.png", 220 if identical_pixels else 50)}
    monkeypatch.setattr(verification, "resolve_crop",
                        lambda job, evidence, identifier, crop_id="": crops[crop_id])
    client = Client({"relation": "distinct", "preferred": "neither", "reason": "Разные рисунки."})
    accepted, retry = verification.reconcile_illustrations(notes, evidence, job, job / "reconcile", client)
    assert not retry
    if identical_pixels:
        assert not client.calls
        assert [item["id"] for item in accepted["chapters"][0]["blocks"]] == ["explanation", "drawing"]
    else:
        assert accepted == notes and len(client.calls) == 1
        assert [Path(path).read_bytes() for _, path in client.calls[0][3]] == [
            crops[first["illustration_crop_id"]].read_bytes(),
            crops[second["illustration_crop_id"]].read_bytes()]


def three_revisions(example):
    job, evidence, _, _ = example
    notes = paired_figures(example, across_chapters=True)
    later = notes["chapters"][1]["blocks"][0]
    later["content"] = "Полный рисунок с подписями."
    generic_caption = "Иллюстрация к теме «Граф»"
    final = {**deepcopy(later), "id": "best", "illustration_id": "v03_ill01",
             "evidence_ids": ["v03_ill01"], "caption": generic_caption, "content": generic_caption}
    source = png(job / "illustrations/v03_ill01.png", 50)
    observation = {**evidence["visuals"][1]["illustrations"][0], "id": "v03_ill01",
                   "image_sha256": sha256_file(source)}
    evidence["visuals"].append({"id": "v03", "timestamp": 6, "status": "complete",
                                 "illustrations": [observation]})
    manifest = read_json(job / "illustrations/manifest.json")
    manifest["assets"]["v03_ill01"] = {"path": "illustrations/v03_ill01.png",
                                       "sha256": observation["image_sha256"]}
    write_json(job / "illustrations/manifest.json", manifest)
    notes["chapters"][1]["blocks"].append(final)
    notes["chapters"][1]["illustration_coverage"].append({
        "illustration_id": "v03_ill01", "block_id": "best"})
    return job, evidence, notes, later


def test_reconciliation_remaps_chain_and_keeps_meaningful_caption_for_better_fallback(example):
    job, evidence, notes, later = three_revisions(example)
    client = Client({"relation": "right_contains_left", "preferred": "right", "reason": "Полнее."})
    accepted, retry = verification.reconcile_illustrations(notes, evidence, job, job / "reconcile", client)
    assert not retry and len(client.calls) == 3  # Includes the direct drawing-to-best comparison.
    [kept] = accepted["chapters"][1]["blocks"]
    assert kept["id"] == "best" and kept["illustration_id"] == "v03_ill01"
    assert kept["caption"] == later["caption"] and kept["content"] == later["content"]
    assert accepted["chapters"][0]["blocks"][0]["figure_refs"] == ["best"]
    assert all(item["block_id"] == "best" for chapter in accepted["chapters"]
               for item in chapter["illustration_coverage"])


@pytest.mark.parametrize("direct", ["distinct", "uncertain", "offline"])
def test_quicksort_tree_keeps_verified_board_when_lomuto_drops_it(example, direct):
    job, evidence, notes, later = three_revisions(example)
    tree = notes["chapters"][0]["blocks"][1]
    tree["caption"] = "Quicksort recursion tree"
    notes["chapters"][0]["blocks"][0]["figure_refs"] = [tree["id"]]
    later["caption"] = "Board with recursion tree and Lomuto partition"
    last = notes["chapters"][1]["blocks"].pop()
    last.update(kind="diagram", content=r"\draw (0,0)--(1,1);", caption="Lomuto partition")
    notes["chapters"][1]["end"] = 5.5
    coverage = notes["chapters"][1]["illustration_coverage"].pop()
    state = {}
    class Verifier(Client):
        def complete(self, instructions, text, schema, *, images):
            self.calls.append((text, images))
            pair = json.loads(text[text.index("{"):])
            if pair["left"]["caption"] == tree["caption"] and pair["right"]["caption"] == last["caption"]:
                if direct == "offline":
                    raise OSError("offline")
                return {"relation": direct, "preferred": "neither", "reason": "Missing recursion tree"}
            return {"relation": "right_contains_left", "preferred": "right", "reason": "More complete"}
    client = Verifier()
    archive = job / "reconciliation"
    first, _ = verification.reconcile_illustrations(notes, evidence, job, archive, client, state=state)
    assert first["chapters"][0]["blocks"][0]["figure_refs"] == ["later"]
    notes["chapters"].append({"id": "c03", "title": "Partition", "start": 5.5, "end": 10,
                              "blocks": [last], "illustration_coverage": [coverage]})
    accepted, retry = verification.reconcile_illustrations(notes, evidence, job, archive, client,
                                                           state=json.loads(json.dumps(state)))
    assert retry == (direct == "offline")
    assert {b["id"] for ch in accepted["chapters"] for b in ch["blocks"]
            if b.get("illustration_id")} == {"later", "best"}
    assert accepted["chapters"][0]["blocks"][0]["figure_refs"] == ["later"]
    assert [c["illustration_coverage"][0]["block_id"] for c in accepted["chapters"]] == [
        "later", "best", "best"]
    assert read_json(archive / "replacements.json") == {"drawing": "later", "later": "best"}
    text, images = client.calls[-1]
    assert tree["caption"] in text and last["caption"] in text
    assert images[1][1].read_bytes() == png(job / "expected-preview.png").read_bytes()
    if direct != "offline":
        calls = len(client.calls)
        replay, _ = verification.reconcile_illustrations(notes, evidence, job, archive, client)
        assert replay == accepted and len(client.calls) == calls


def test_identical_pixels_with_distinct_intended_figures_need_a_verdict(example):
    job, evidence, _, _ = example
    notes = paired_figures(example)
    first, second = notes["chapters"][0]["blocks"][1:]
    second.update(illustration_id=first["illustration_id"], caption="Different panel")
    client = Client({"relation": "distinct", "preferred": "neither", "reason": "Independent panels"})
    accepted, retry = verification.reconcile_illustrations(notes, evidence, job, job / "pairs", client)
    assert accepted == notes and len(client.calls) == 1 and not retry


def test_later_chapter_recovers_unpublished_source_prose_and_review(example, monkeypatch):
    job, evidence, notes, _ = example
    source_notes = paired_figures(example, across_chapters=True)
    first, second = [dict(source_notes, chapters=[ch]) for ch in source_notes["chapters"]]
    first["chapters"][0]["blocks"][0]["figure_refs"] = ["drawing"]
    monkeypatch.setattr(verification, "refine_crop", lambda *args: ("", False, "occluded", False))
    pending = {}
    old, _ = verification.finalize_illustrations(first, evidence, job, job / "old", Client(),
                                                 pending=pending)
    assert old["issues"] and old["chapters"][0]["illustration_coverage"][0]["exclusion_reason"] == "unresolved"
    matcher = verification.figure_matcher(job, evidence)
    client = Client()
    verification.recover_later([old], second, pending, evidence, job, job / "later", client, matcher)
    assert not pending and not old["issues"] and len(client.calls) == 1
    assert old["chapters"][0]["blocks"][0]["figure_refs"] == ["later"]
    assert old["chapters"][0]["illustration_coverage"][0]["block_id"] == "later"
    verification.recover_later([old], second, pending, evidence, job, job / "again", client, matcher)
    assert len(client.calls) == 1


def test_one_tikz_correction_is_rendered_verified_and_reused(example):
    job, _, original, renders = example

    class RepairClient(Client):
        def complete(self, instructions, text, schema, *, images):
            self.calls.append((instructions, text, schema, images))
            if schema == verification.REPAIR_SCHEMA:
                assert "Неверная стрелка" in text
                return {"content": r"\draw[<-] (0,0)--(1,1);"}
            return {"verdict": "fail" if len(self.calls) == 1 else "pass",
                    "reason": "Неверная стрелка" if len(self.calls) == 1 else "Исправлено."}

    client = RepairClient()
    accepted, retry = finalize(example, client)
    assert block(accepted)["kind"] == "diagram" and not retry
    assert block(accepted)["content"] == renders[-1] == r"\draw[<-] (0,0)--(1,1);"
    assert len(client.calls) == 3 and len(renders) == 2
    assert block(original)["content"] != block(accepted)["content"]
    repeated, retry = finalize(example, client, "second")
    assert repeated == accepted and not retry and len(client.calls) == 3
    repair = job / "attempt/illustration-verification/drawing/artifacts/repair"
    assert (repair / "candidate.tex").read_text() == block(accepted)["content"]


def test_reconstruction_only_source_cannot_be_published_as_screenshot(example, monkeypatch):
    job, _, notes, _ = example
    source = png(job / "cluttered.png")
    purposes = []

    def refine(job, evidence, candidate, client, folder):
        purposes.append(candidate["kind"])
        if candidate["kind"] == "diagram":
            return "reconstruction_crop", False, "", False
        return "", False, "Рука рядом с рисунком.", False

    monkeypatch.setattr(verification, "refine_crop", refine)
    monkeypatch.setattr(verification, "resolve_crop", lambda *args: source)
    notes["chapters"][0]["illustration_coverage"] = [
        {"illustration_id": "v01_ill01", "block_id": "drawing"}]
    accepted, retry = finalize(example, Client({"verdict": "fail", "reason": "Неверная стрелка."}))
    assert purposes == ["diagram", "screenshot"] and not retry
    assert accepted["chapters"][0]["blocks"] == []
    assert any(issue["status"] == "open" and issue["evidence_ids"] == ["v01_ill01"]
               and "Рука рядом с рисунком." in issue["description"] for issue in accepted["issues"])
    assert accepted["chapters"][0]["illustration_coverage"][0] == {
        "illustration_id": "v01_ill01", "block_id": "", "exclusion_reason": "unresolved"}
    LectureNotes.model_validate(accepted)


def test_nonconsecutive_figures_with_different_topics_are_compared_visually(example):
    job, evidence, _, _ = example
    notes = paired_figures(example)
    first, later = notes["chapters"][0]["blocks"][1:]
    later["topic"] = "Повторение свойства"
    evidence["visuals"][1]["illustrations"][0]["description"] = (
        evidence["visuals"][0]["illustrations"][0]["description"])
    unrelated = deepcopy(first)
    unrelated.update(id="other", topic="Другая тема")
    # A separate intervening figure must not make the earlier representative unreachable.
    notes["chapters"][0]["blocks"].insert(2, unrelated)
    client = Client({"relation": "same", "preferred": "left", "reason": "Тот же рисунок."})
    accepted, retry = verification.reconcile_illustrations(
        notes, evidence, job, job / "reconcile", client)
    assert not retry
    assert [item["id"] for item in accepted["chapters"][0]["blocks"]] == ["explanation", "drawing"]
    assert len(client.calls) == 1  # Same-source intervening observation merges without a call.


@pytest.mark.parametrize("different_observation", [False, True])
def test_failed_selected_screenshot_reuses_verified_diagram_and_preserves_refs(
    example, monkeypatch, different_observation,
):
    job, evidence, notes, _ = example
    failed = block(notes)
    failed.update(kind="screenshot", content=failed["caption"])
    retained = deepcopy(failed)
    retained.update(id="good_diagram", kind="diagram", content=r"\draw[->] (0,0)--(1,1);")
    if different_observation:
        source = png(job / "illustrations/v02_ill01.png")
        observation = {**evidence["visuals"][0]["illustrations"][0],
                       "id": "v02_ill01", "image_sha256": sha256_file(source)}
        evidence["visuals"].append({"id": "v02", "timestamp": 3, "status": "complete",
                                   "illustrations": [observation]})
        manifest = read_json(job / "illustrations/manifest.json")
        manifest["assets"][observation["id"]] = {
            "path": "illustrations/v02_ill01.png", "sha256": observation["image_sha256"]}
        write_json(job / "illustrations/manifest.json", manifest)
        retained.update(illustration_id="v02_ill01", evidence_ids=["v02", "v02_ill01"])
    chapter = notes["chapters"][0]
    chapter["blocks"] = [{"id": "explanation", "kind": "text", "content": "Связь вершин.",
                           "evidence_ids": ["v01_ill01"], "figure_refs": [failed["id"]]},
                          failed, retained]
    chapter["illustration_coverage"] = [{"illustration_id": "v01_ill01",
                                         "block_id": failed["id"], "exclusion_reason": ""}]
    if different_observation:
        chapter["illustration_coverage"].append({"illustration_id": "v02_ill01",
                                                 "block_id": retained["id"], "exclusion_reason": ""})

    def refine(job, evidence, candidate, client, folder):
        return ("", False, "Рука скрывает рисунок.", False) if candidate["id"] == failed["id"] else (
            "", False, "", False)

    monkeypatch.setattr(verification, "refine_crop", refine)
    accepted, retry = finalize(example, Client())
    chapter = accepted["chapters"][0]
    assert not retry
    assert [(item["id"], item["kind"]) for item in chapter["blocks"]] == [
        ("explanation", "text"), ("good_diagram", "diagram")]
    assert chapter["blocks"][0]["figure_refs"] == ["good_diagram"]
    assert all(item["block_id"] == "good_diagram" and not item["exclusion_reason"]
               for item in chapter["illustration_coverage"])
    assert not accepted["issues"]
    LectureNotes.model_validate(accepted)
