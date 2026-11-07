import json
import shutil

import pytest
from PIL import Image

from shad_helper import notes, rendering
from shad_helper.illustrations import materialize_illustrations
from shad_helper.jobs import read_json, sha256_file, write_json


@pytest.fixture(autouse=True)
def no_crop_refinement(monkeypatch):
    # These orchestration fixtures have synthetic crops but no original video
    # frames. Native-frame refinement is covered separately.
    monkeypatch.setattr("shad_helper.illustration_verification.refine_crop",
                        lambda *args, **kwargs: ("", False, "", False))


def prepare(job, *, times=(10,), source_kind="printed", duration=120, identical=False):
    frames = []
    for number, timestamp in enumerate(times):
        image = job / f"frame{number}.png"
        Image.new("RGB", (80, 50), (255, 200 if identical else 40 + number % 200, 0)).save(image)
        frames.append({"id": f"f{number}", "timestamp": timestamp, "image": str(image),
                       "content": {"illustrations": [{"source_kind": source_kind,
                           "description": "Учебный рисунок", "box": [0, 0, 1, 1]}]}})
    crops = materialize_illustrations(job, frames)
    evidence = {"schema_version": 5, "title": "Лекция", "media": {"duration": duration},
                "segments": [], "visuals": [
                    {"id": frame["id"], "timestamp": frame["timestamp"], "status": "complete",
                     "kind": "slides" if source_kind == "printed" else "whiteboard",
                     "text": "", "formulas": [], "diagrams": [], "unreadable": [],
                     "illustrations": crops[frame["id"]]} for frame in frames]}
    write_json(job / "evidence.json", evidence)
    return evidence


def chapter_reply(context, *, drawn=False, reuse=False):
    chapter = context["chapter"]
    required = context["required_illustration_ids"]
    blocks = []
    coverage = []
    for identifier in required:
        if reuse and context.get("prior_illustrations"):
            target = context["prior_illustrations"][0]["block_id"]
        else:
            target = f"{chapter['id']}_{identifier}"
            blocks.append({"id": target, "kind": "diagram" if drawn else "screenshot",
                           "content": r"\draw (0,0)--(1,1);" if drawn else "Учебный рисунок",
                           "topic": "Пример", "caption": "Иллюстрация",
                           "illustration_id": identifier, "evidence_ids": [identifier]})
        coverage.append({"illustration_id": identifier, "block_id": target})
    return {"title": "Лекция", "chapters": [{**chapter, "blocks": blocks,
             "illustration_coverage": coverage}], "issues": [], "glossary": []}


class Client:
    model = "fake"

    def __init__(self, *, drawn=False, reuse=False, fail_once=False):
        self.drawn, self.reuse, self.fail_once = drawn, reuse, fail_once
        self.calls, self.verifications = [], 0

    def complete(self, instructions, text, schema=None, *, images=None):
        if "verdict" in schema.get("properties", {}):
            self.verifications += 1
            if self.fail_once and self.verifications == 1:
                raise RuntimeError("Temporary verifier outage")
            return {"verdict": "pass", "reason": "Связи и подписи совпадают."}
        context = json.loads(text)
        self.calls.append((context, [(label, path.read_bytes()) for label, path in images or []]))
        return chapter_reply(context, drawn=self.drawn, reuse=self.reuse)


def test_printed_illustrations_reach_notes_and_portable_assets(tmp_path):
    prepare(tmp_path)
    client = Client()
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client).exists()
    block = read_json(tmp_path / "notes.json")["chapters"][0]["blocks"][0]
    assert block["kind"] == "screenshot" and block["illustration_id"] == "f0_ill01"
    manifest = read_json(tmp_path / "notes/project.json")
    assert manifest["image_files"] == ["figures/illustration-f0_ill01.png"]
    source = tmp_path / "illustrations/f0_ill01.png"
    assert (tmp_path / "notes" / manifest["image_files"][0]).read_bytes() == source.read_bytes()
    assert client.verifications == 0
    assert read_json(tmp_path / "review.json")["ready"]
    context, images = client.calls[0]
    assert context["required_illustration_ids"] == ["f0_ill01"]
    assert images == [("f0_ill01", source.read_bytes())]
    root = tmp_path / "handoff"
    assert (root / "inputs/ch001/images/f0_ill01.png").read_bytes() == source.read_bytes()
    assert read_json(root / "inputs/ch001/context.json") == context
    assert not (root / "runs").exists()


def test_identical_crops_share_attachment_without_losing_coverage(tmp_path):
    prepare(tmp_path, times=(10, 20), identical=True)
    client = Client()
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    context, images = client.calls[0]
    assert len(images) == 1 and images[0][0] == "f0_ill01 / f1_ill01"
    assert context["required_illustration_ids"] == ["f0_ill01", "f1_ill01"]
    assert len(list((tmp_path / "handoff/inputs/ch001/images").glob("*.png"))) == 1


@pytest.mark.parametrize("failure", ["missing", "prose", "unknown", "duplicate"])
def test_coverage_rejects_silent_loss_and_invalid_targets(tmp_path, failure):
    evidence = prepare(tmp_path)
    context = notes.chapter_context(evidence, notes.default_outline(evidence)["chapters"][0])
    reply = chapter_reply(context)
    chapter = reply["chapters"][0]
    if failure == "missing":
        chapter["illustration_coverage"] = []
    elif failure == "prose":
        chapter["blocks"][0].update(kind="text", illustration_id="")
    elif failure == "unknown":
        chapter["illustration_coverage"][0]["block_id"] = "nonexistent"
    else:
        chapter["illustration_coverage"] *= 2
    with pytest.raises(ValueError, match="[Ii]llustration"):
        notes.validate_notes(reply, evidence)


@pytest.mark.parametrize("target", ["missing", "prose"])
def test_figure_reference_requires_an_existing_figure(tmp_path, target):
    evidence = prepare(tmp_path)
    context = notes.chapter_context(evidence, notes.default_outline(evidence)["chapters"][0])
    reply = chapter_reply(context)
    reply["chapters"][0]["blocks"].append({
        "id": "prose", "kind": "text", "content": "Объяснение рисунка.",
        "evidence_ids": ["f0_ill01"], "figure_refs": [target],
    })
    with pytest.raises(ValueError, match="figure references"):
        notes.validate_notes(reply, evidence)


def test_final_reconciliation_can_reference_a_later_complete_figure(tmp_path):
    evidence = prepare(tmp_path, times=(10, 130), duration=240)
    parts = []
    for number, start in enumerate((0, 120)):
        chapter = {"id": f"ch{number}", "title": "Рисунок", "start": start, "end": start + 120}
        parts.append(chapter_reply(notes.chapter_context(evidence, chapter, overlap=0)))
    first, second = [part["chapters"][0] for part in parts]
    target = second["blocks"][0]["id"]
    first["blocks"] = [{"id": "explanation", "kind": "text", "content": "Объяснение рисунка.",
                         "evidence_ids": ["f0_ill01"], "figure_refs": [target]}]
    first["illustration_coverage"][0]["block_id"] = target
    reply = dict(parts[0], chapters=[first, second])
    with pytest.raises(ValueError, match="current or prior"):
        notes.validate_notes(reply, evidence)
    result = notes.validate_notes(reply, evidence, finalized=True)
    assert result["chapters"][0]["blocks"][0]["figure_refs"] == [target]
    # Reconciliation does not relax source provenance or permit missing targets.
    second["blocks"][0]["evidence_ids"] = ["f0_ill01"]
    with pytest.raises(ValueError, match="Unknown evidence|Invalid illustration source"):
        notes.validate_notes(reply, evidence, finalized=True)


@pytest.mark.parametrize("kind,refs", [("math", ["f"]), ("text", ["f", "f"])])
def test_figure_references_have_unambiguous_prose_placement(kind, refs):
    from shad_helper.schemas import NoteBlock

    with pytest.raises(ValueError, match="Figure references"):
        NoteBlock.model_validate({"id": "b", "kind": kind, "content": "x", "figure_refs": refs})


def test_import_accepts_final_coverage_and_references_to_a_later_figure(tmp_path, monkeypatch):
    evidence = prepare(tmp_path, times=(10, 130), duration=240)
    chapters = []
    for number, start in enumerate((0, 120)):
        interval = {"id": f"ch{number}", "title": "Рисунок", "start": start, "end": start + 120}
        chapters.append(chapter_reply(notes.chapter_context(evidence, interval))["chapters"][0])
    target = chapters[1]["blocks"][0]["id"]
    chapters[0]["blocks"] = [{"id": "explanation", "kind": "text", "content": "Объяснение.",
                               "evidence_ids": ["f0_ill01"], "figure_refs": [target]}]
    chapters[0]["illustration_coverage"][0]["block_id"] = target
    imported = tmp_path / "final.json"
    write_json(imported, {"title": "Лекция", "chapters": chapters, "issues": [], "glossary": []})
    clients = []

    class ImportClient(Client):
        def __init__(self, *args, **kwargs):
            super().__init__()
            clients.append(self)

    monkeypatch.setattr("shad_helper.openai_client.PlanClient", ImportClient)
    monkeypatch.setattr(rendering, "validate_chapter_latex", lambda *args, **kwargs: None)
    assert notes.generate_notes(tmp_path, tmp_path / "auth", imported=imported).exists()
    result = read_json(tmp_path / "notes.json")
    assert result["chapters"][0]["blocks"][0]["figure_refs"] == [target]
    assert result["chapters"][0]["illustration_coverage"][0]["block_id"] == target
    assert len(clients) == 1 and clients[0].verifications == 1
    assert sum(block["kind"] == "screenshot" for chapter in result["chapters"]
               for block in chapter["blocks"]) == 1


@pytest.mark.parametrize("reason", ["decoration", "lecturer_or_clothing", "code_or_interface",
                                    "text_or_formulas", "no_instructional_visual"])
def test_false_candidates_are_explicitly_excluded_without_losing_real_figure(tmp_path, reason):
    prepare(tmp_path, times=(10, 20))
    evidence_bytes = (tmp_path / "evidence.json").read_bytes()

    class FilteringClient(Client):
        def complete(self, instructions, text, schema=None, *, images=None):
            reply = super().complete(instructions, text, schema, images=images)
            chapter = reply["chapters"][0]
            chapter["blocks"].pop(0)
            chapter["illustration_coverage"][0].update(block_id="", exclusion_reason=reason)
            return reply

    client = FilteringClient()
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client) == tmp_path / "notes/main.tex"
    [chapter] = read_json(tmp_path / "notes.json")["chapters"]
    assert [block["illustration_id"] for block in chapter["blocks"]] == ["f1_ill01"]
    assert chapter["illustration_coverage"][0] == {
        "illustration_id": "f0_ill01", "block_id": "", "exclusion_reason": reason}
    assert read_json(tmp_path / "notes/project.json")["image_files"] == ["figures/illustration-f1_ill01.png"]
    assert read_json(tmp_path / "review.json")["ready"]
    assert client.verifications == 0
    assert (tmp_path / "evidence.json").read_bytes() == evidence_bytes


@pytest.mark.parametrize("failure", ["unexplained", "budget", "contradictory", "still_rendered"])
def test_coverage_rejects_invalid_exclusions(tmp_path, failure):
    evidence = prepare(tmp_path)
    context = notes.chapter_context(evidence, notes.default_outline(evidence)["chapters"][0])
    reply = chapter_reply(context)
    chapter = reply["chapters"][0]
    decision = chapter["illustration_coverage"][0]
    decision.update(block_id="", exclusion_reason="text_or_formulas")
    if failure == "unexplained":
        decision["exclusion_reason"] = ""
    elif failure == "budget":
        decision["exclusion_reason"] = "too_many_images"
    elif failure == "contradictory":
        decision["block_id"] = chapter["blocks"][0]["id"]
    if failure != "still_rendered":
        chapter["blocks"] = []
    with pytest.raises(ValueError):
        notes.validate_notes(reply, evidence)


def test_repeated_observation_can_reuse_previous_chapter_figure(tmp_path):
    prepare(tmp_path, times=(890, 905), duration=1800, identical=True)
    client = Client(reuse=True)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    chapters = read_json(tmp_path / "notes.json")["chapters"]
    assert len(chapters[0]["blocks"]) == 1 and chapters[1]["blocks"] == []
    assert chapters[1]["illustration_coverage"][0]["block_id"] == chapters[0]["blocks"][0]["id"]
    assert client.calls[1][0]["prior_illustrations"][0]["block_id"] == chapters[0]["blocks"][0]["id"]
    cached = Client(reuse=True)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=cached)
    assert cached.calls == []


def test_distinct_stages_remain_separate_and_chronological(tmp_path):
    prepare(tmp_path, times=(890, 905), duration=1800)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=Client())
    chapters = read_json(tmp_path / "notes.json")["chapters"]
    assert [chapter["blocks"][0]["illustration_id"] for chapter in chapters] == ["f0_ill01", "f1_ill01"]


def test_identical_figure_recurring_after_another_frame_reuses_prior_block(tmp_path):
    evidence = prepare(tmp_path, times=(10, 150), duration=200, identical=True)
    evidence["visuals"].insert(1, {"id": "textonly", "timestamp": 50, "status": "complete",
                                 "kind": "slides", "text": "Объяснение", "formulas": [],
                                 "diagrams": [], "unreadable": [], "illustrations": []})
    write_json(tmp_path / "evidence.json", evidence)
    write_json(tmp_path / "outline.json", notes.default_outline(evidence, chunk_seconds=100))
    client = Client(reuse=True)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    chapters = read_json(tmp_path / "notes.json")["chapters"]
    assert chapters[1]["blocks"] == []
    assert client.calls[1][0]["prior_illustrations"][0]["source"]["id"] == "f0_ill01"


def test_changed_crop_can_reuse_distant_figure_without_reattaching_prior_catalog(tmp_path):
    evidence = prepare(tmp_path, times=(10, 150), duration=200)
    evidence["visuals"].insert(1, {"id": "textonly", "timestamp": 50, "status": "complete",
                                 "kind": "slides", "text": "Объяснение", "formulas": [],
                                 "diagrams": [], "unreadable": [], "illustrations": []})
    write_json(tmp_path / "evidence.json", evidence)
    write_json(tmp_path / "outline.json", notes.default_outline(evidence, chunk_seconds=100))
    client = Client(reuse=True)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    chapters = read_json(tmp_path / "notes.json")["chapters"]
    assert chapters[1]["blocks"] == []
    assert chapters[1]["illustration_coverage"][0]["block_id"] == chapters[0]["blocks"][0]["id"]
    context, images = client.calls[1]
    assert context["prior_illustrations"][0]["source"]["id"] == "f0_ill01"
    assert [label for label, _ in images] == ["f1_ill01"]
    assert client.verifications == 1
    assert len(list((tmp_path / "handoff/inputs/ch002/images").glob("*.png"))) == 1


@pytest.mark.parametrize("duration,count", [(30, 25), (900, 36)])
def test_more_than_24_unique_images_stay_in_one_chapter(tmp_path, duration, count):
    evidence = prepare(tmp_path, times=tuple(number * duration / count for number in range(count)),
                       duration=duration)
    outline = notes.default_outline(evidence)
    assert notes._bounded_outline(evidence, outline, job=tmp_path) == outline
    client = Client()
    assert notes.generate_notes(tmp_path, tmp_path / "auth", client=client) == tmp_path / "notes/main.tex"
    assert len(client.calls) == 1
    context, images = client.calls[0]
    expected = [f"f{number}_ill01" for number in range(count)]
    assert context["required_illustration_ids"] == expected
    assert [label for label, _ in images] == expected
    assert len({data for _, data in images}) == count
    [chapter] = read_json(tmp_path / "notes.json")["chapters"]
    assert [item["illustration_id"] for item in chapter["illustration_coverage"]] == expected
    assert read_json(tmp_path / "outline.json") == outline


@pytest.mark.parametrize("imported", [False, True])
def test_unavailable_verification_retries_automatically_without_regenerating_notes(
        tmp_path, monkeypatch, imported):
    evidence = prepare(tmp_path, source_kind="drawn")
    def preview(content, folder):
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "preview.png"
        shutil.copyfile(tmp_path / "illustrations/f0_ill01.png", target)
        return target
    monkeypatch.setattr(rendering, "render_diagram_preview", preview)
    client = Client(drawn=True, fail_once=True)
    kwargs = {}
    if imported:
        context = notes.chapter_context(evidence, notes.default_outline(evidence)["chapters"][0])
        write_json(tmp_path / "reply.json", chapter_reply(context, drawn=True))
        kwargs["imported"] = tmp_path / "reply.json"
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client, **kwargs)
    first = read_json(tmp_path / "notes.json")
    assert first["chapters"][0]["blocks"][0]["kind"] == "diagram"
    assert not first["issues"] and read_json(tmp_path / "review.json")["ready"]
    assert read_json(tmp_path / "handoff/verification/status.json") == {
        "attempts": 2, "retry_needed": False}
    assert len(client.calls) == (0 if imported else 1) and client.verifications == 2
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client, **kwargs)
    assert client.verifications == 2
    assert read_json(tmp_path / "handoff/verification/status.json")["attempts"] == 1


def test_verifier_outage_has_bounded_automatic_retry(tmp_path, monkeypatch):
    prepare(tmp_path, source_kind="drawn")
    monkeypatch.setattr(rendering, "render_diagram_preview",
                        lambda content, folder: tmp_path / "illustrations/f0_ill01.png")

    class Unavailable(Client):
        def complete(self, instructions, text, schema=None, *, images=None):
            if "verdict" in schema.get("properties", {}):
                self.verifications += 1
                raise RuntimeError("Verifier unavailable")
            return super().complete(instructions, text, schema, images=images)

    client = Unavailable(drawn=True)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert len(client.calls) == 1 and client.verifications == 2
    assert read_json(tmp_path / "handoff/verification/status.json") == {
        "attempts": 2, "retry_needed": True}


def test_automatic_finalization_can_reuse_a_complete_figure_from_a_later_chapter(
        tmp_path, monkeypatch):
    evidence = prepare(tmp_path, times=(10, 130), duration=240)
    write_json(tmp_path / "outline.json", notes.default_outline(evidence, chunk_seconds=120))
    monkeypatch.setattr("shad_helper.illustration_verification.refine_crop",
                        lambda job, evidence, block, *args: (
                            "", False, "Рисунок обрезан." if block["illustration_id"] == "f0_ill01"
                            else "", False))

    class Referenced(Client):
        def complete(self, instructions, text, schema=None, *, images=None):
            reply = super().complete(instructions, text, schema, images=images)
            if "chapters" in reply:
                chapter = reply["chapters"][0]
                figure = chapter["blocks"][0]
                chapter["blocks"].append({"id": chapter["id"] + "_explanation", "kind": "text",
                                           "content": "Связи показаны на рисунке.",
                                           "evidence_ids": [figure["illustration_id"]],
                                           "figure_refs": [figure["id"]]})
            return reply

    notes.generate_notes(tmp_path, tmp_path / "auth", client=Referenced())
    result = notes.validate_notes(read_json(tmp_path / "notes.json"), evidence, finalized=True)
    first, second = result["chapters"]
    target = next(block["id"] for block in second["blocks"] if block.get("illustration_id"))
    assert all(not block.get("illustration_id") for block in first["blocks"])
    assert first["illustration_coverage"][0]["block_id"] == target
    assert first["blocks"][0]["figure_refs"] == [target]
    assert all(block["kind"] != "uncertain" for chapter in result["chapters"]
               for block in chapter["blocks"])


def test_reconciliation_retries_original_figures_once(tmp_path, monkeypatch):
    from copy import deepcopy

    prepare(tmp_path, times=(10, 20))
    seen = []

    def reconcile(candidate, *args):
        result = deepcopy(candidate)
        chapter = result["chapters"][0]
        seen.append(len(chapter["blocks"]))
        if len(seen) == 1:
            chapter["blocks"].pop()
            chapter["illustration_coverage"][1]["block_id"] = chapter["blocks"][0]["id"]
        return result, True

    monkeypatch.setattr("shad_helper.illustration_verification.reconcile_illustrations", reconcile)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=Client())
    assert seen == [2, 2]
    assert len(read_json(tmp_path / "notes.json")["chapters"][0]["blocks"]) == 2
    assert read_json(tmp_path / "handoff/figure-reconciliation/status.json") == {
        "attempts": 2, "retry_needed": True}


def test_rebuild_refreshes_visual_caches_and_preserves_stage_inputs(tmp_path, monkeypatch):
    evidence = prepare(tmp_path, source_kind="drawn")
    outline = notes.default_outline(evidence)
    outline["chapters"][0]["title"] = "Custom chapter"
    write_json(tmp_path / "outline.json", outline)
    write_json(tmp_path / "review-resolutions.json", {})
    write_json(tmp_path / "checkpoints/anki.json", {"result": "previous deck"})
    write_json(tmp_path / "checkpoints/extraction.json", {"result": "source"})
    monkeypatch.setattr(rendering, "render_diagram_preview",
                        lambda content, folder: tmp_path / "illustrations/f0_ill01.png")
    client = Client(drawn=True)
    legacy_run = tmp_path / "handoff/runs/legacy"
    write_json(legacy_run / "run.json", {"mode": "automatic"})
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    preserved = [tmp_path / name for name in (
        "evidence.json", "review-resolutions.json",
        "illustrations/f0_ill01.png", "frame0.png", "checkpoints/anki.json",
        "checkpoints/extraction.json")]
    before = {path: path.read_bytes() for path in preserved}
    stale = [tmp_path / folder / "stale.json" for folder in (
        "checkpoints/notes", "checkpoints/notes_candidates", "checkpoints/notes_final",
        "illustration-crops", "illustration-verification", "illustration-stitches")]
    for path in stale:
        write_json(path, {"stale": True})

    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert len(client.calls) == client.verifications == 1
    assert all(path.exists() for path in stale)
    assert read_json(tmp_path / "outline.json") == outline
    assert legacy_run.exists()
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client, rebuild=True)
    assert len(client.calls) == client.verifications == 2
    assert all(not path.exists() for path in stale)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not (tmp_path / "handoff/runs").exists()
    assert read_json(tmp_path / "outline.json") == notes.default_outline(evidence)

    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert len(client.calls) == client.verifications == 2


def test_import_runs_drawing_verification_and_missing_assets_preserve_previous_notes(tmp_path, monkeypatch):
    evidence = prepare(tmp_path, source_kind="drawn")
    context = notes.chapter_context(evidence, notes.default_outline(evidence)["chapters"][0])
    reply = chapter_reply(context, drawn=True)
    imported = tmp_path / "reply.json"
    write_json(imported, reply)
    monkeypatch.setattr(rendering, "render_diagram_preview",
                        lambda content, folder: tmp_path / "illustrations/f0_ill01.png")
    client = Client(drawn=True)
    notes.generate_notes(tmp_path, tmp_path / "auth", imported=imported, client=client)
    assert client.verifications == 1 and client.calls == []
    original = (tmp_path / "notes.json").read_bytes()
    (tmp_path / "illustrations/f0_ill01.png").unlink()
    with pytest.raises(ValueError, match="[Ii]llustration"):
        notes.generate_notes(tmp_path, tmp_path / "auth", manual=True)
    assert (tmp_path / "notes.json").read_bytes() == original


def test_unreadable_visual_candidate_still_produces_reviewable_crop(tmp_path):
    evidence = prepare(tmp_path, source_kind="unknown")
    evidence["visuals"][0]["status"] = "incomplete"
    write_json(tmp_path / "evidence.json", evidence)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=Client())
    assert read_json(tmp_path / "notes.json")["chapters"][0]["blocks"][0]["kind"] == "screenshot"
    assert not read_json(tmp_path / "review.json")["ready"]


def test_crop_content_change_invalidates_candidate_cache(tmp_path):
    prepare(tmp_path)
    client = Client()
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    prepare(tmp_path, identical=True)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert len(client.calls) == 2


def add_native_frames(job, evidence):
    evidence["media"]["sha256"] = "a" * 64
    folder = job / "frames"
    folder.mkdir(exist_ok=True)
    paths = []
    for number, visual in enumerate(evidence["visuals"]):
        path = folder / f"{visual['id']}_{'a' * 12}.png"
        Image.new("RGB", (160, 100), (20 + number, 80, 140)).save(path)
        paths.append(path)
    write_json(job / "evidence.json", evidence)
    return paths


def test_native_context_is_bounded_without_dropping_candidate_images(tmp_path):
    evidence = prepare(tmp_path, times=(10, 20, 30, 40, 50, 60))
    add_native_frames(tmp_path, evidence)
    context = notes.chapter_context(evidence, notes.default_outline(evidence)["chapters"][0])
    images = notes._chapter_images(tmp_path, evidence, context)
    assert [label for label, _ in images[:6]] == [f"f{i}_ill01" for i in range(6)]
    assert [label for label, _ in images[6:]] == [
        f"SOURCE_FRAME f{i} (context only)" for i in range(4)]


@pytest.mark.parametrize("kind,box,expected", [
    ("drawn", [0.2, 0.2, 0.6, 0.6], True),
    ("unknown", [0.2, 0.2, 0.6, 0.6], True),
    ("printed", [0, 0, 1, 1], True),
    ("printed", [0.1, 0.3, 0.6, 0.4], True),
    ("printed", [0.2, 0.2, 0.6, 0.6], False),
])
def test_native_context_only_for_ambiguous_candidates(tmp_path, kind, box, expected):
    evidence = prepare(tmp_path, source_kind=kind)
    evidence["visuals"][0]["illustrations"][0]["box"] = box
    add_native_frames(tmp_path, evidence)
    context = notes.chapter_context(evidence, notes.default_outline(evidence)["chapters"][0])
    assert len(notes._chapter_images(tmp_path, evidence, context)) == 1 + expected


def test_native_context_attachments_match_manual_export_and_bind_actual_pixels(tmp_path):
    evidence = prepare(tmp_path)
    [frame] = add_native_frames(tmp_path, evidence)
    client = Client()
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    [(_, images)] = client.calls
    records = read_json(tmp_path / "handoff/inputs/ch001/images.json")
    assert records[1] == {
        "illustration_ids": [], "visual_id": "f0", "role": "source_frame",
        "label": "SOURCE_FRAME f0 (context only)", "file": "images/source-frame-f0.png",
        "sha256": sha256_file(frame)}
    assert images[1] == (records[1]["label"], frame.read_bytes())
    notes.generate_notes(tmp_path, tmp_path / "auth", manual=True)
    manual_records = read_json(tmp_path / "handoff/inputs/ch001/images.json")
    assert manual_records == records
    exported = tmp_path / "handoff/inputs/ch001" / records[1]["file"]
    assert exported.read_bytes() == frame.read_bytes()


def test_native_frame_pixel_change_invalidates_notes_candidate_cache(tmp_path):
    evidence = prepare(tmp_path)
    [frame] = add_native_frames(tmp_path, evidence)
    client = Client()
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert len(client.calls) == 1
    evidence_bytes = (tmp_path / "evidence.json").read_bytes()
    Image.new("RGB", (160, 100), "green").save(frame)
    notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert len(client.calls) == 2
    assert (tmp_path / "evidence.json").read_bytes() == evidence_bytes
    assert client.calls[1][1][1][1] == frame.read_bytes()


def test_missing_native_frames_keep_portable_crop_attachments(tmp_path):
    evidence = prepare(tmp_path, source_kind="drawn")
    [frame] = add_native_frames(tmp_path, evidence)
    frame.unlink()
    context = notes.chapter_context(evidence, notes.default_outline(evidence)["chapters"][0])
    assert [label for label, _ in notes._chapter_images(tmp_path, evidence, context)] == ["f0_ill01"]


def test_native_context_counts_toward_the_existing_image_budget(tmp_path, monkeypatch):
    from shad_helper import openai_client

    evidence = prepare(tmp_path)
    [frame] = add_native_frames(tmp_path, evidence)
    outline = notes.default_outline(evidence)
    context = notes.chapter_context(evidence, outline["chapters"][0])
    crop = tmp_path / "illustrations/f0_ill01.png"
    limit = 4 * ((crop.stat().st_size + 2) // 3)
    monkeypatch.setattr(openai_client, "MAX_IMAGE_ENCODED_BYTES", limit)
    assert "image attachments" in notes._chapter_budget_error(
        context, outline, [], job=tmp_path, evidence=evidence)
    frame.unlink()
    assert notes._chapter_budget_error(context, outline, [], job=tmp_path, evidence=evidence) is None


def test_notes_entry_rejects_audio_only_evidence_for_video_job(tmp_path):
    evidence = prepare(tmp_path)
    evidence.pop("visuals")
    evidence["schema_version"] = 3
    evidence["segments"] = [{"id": "speech", "start": 0, "end": 120, "text": "Определение"}]
    write_json(tmp_path / "evidence.json", evidence)
    write_json(tmp_path / "extraction-config.json", {
        "schema_version": 5, "pipeline": "audio_video", "vision_model": "35b"})
    client = Client()
    with pytest.raises(ValueError, match="Visual extraction is missing"):
        notes.generate_notes(tmp_path, tmp_path / "auth", client=client)
    assert client.calls == []
    assert not (tmp_path / "notes.json").exists()


@pytest.mark.parametrize("missing", ["issue", "open_issue", "matching_id", None])
def test_unresolved_coverage_retains_internal_issue_without_publishing_recovery_text(tmp_path, missing):
    evidence = prepare(tmp_path, times=(10, 20))
    context = notes.chapter_context(evidence, notes.default_outline(evidence)["chapters"][0])
    reply = chapter_reply(context)
    chapter = reply["chapters"][0]
    chapter["illustration_coverage"][0].update(block_id="", exclusion_reason="unresolved")
    chapter["illustration_coverage"][1]["exclusion_reason"] = ""
    chapter["blocks"].pop(0)
    reference = "f1_ill01" if missing == "matching_id" else "f0_ill01"
    if missing != "issue":
        reply["issues"] = [{"id": "pending_issue", "description": "Рисунок не восстановлен.",
                            "evidence_ids": [reference],
                            "status": "resolved" if missing == "open_issue" else "open",
                            "resolution": "Проверено" if missing == "open_issue" else ""}]
    if missing is None:
        notes._validate_illustration_coverage(reply, evidence, [])
        validated = notes.validate_notes(reply, evidence)
        assert validated["chapters"][0]["illustration_coverage"][0]["exclusion_reason"] == "unresolved"
        assert all(block["kind"] != "uncertain" for block in validated["chapters"][0]["blocks"])
    else:
        with pytest.raises(ValueError, match="requires an open review issue"):
            notes._validate_illustration_coverage(reply, evidence, [])


@pytest.mark.parametrize("complete_import,forward", [(True, True), (True, False), (False, False)])
def test_import_unpublishes_shared_figure_without_dangling_chapter_references(
        tmp_path, monkeypatch, complete_import, forward):
    evidence = prepare(tmp_path, times=(10, 130), duration=240)
    outline = notes.default_outline(evidence, chunk_seconds=120)
    write_json(tmp_path / "outline.json", outline)
    replies = [chapter_reply(notes.chapter_context(evidence, chapter))
               for chapter in outline["chapters"]]
    target_index, alias_index = (1, 0) if forward else (0, 1)
    target = replies[target_index]["chapters"][0]["blocks"][0]["id"]
    alias = replies[alias_index]["chapters"][0]
    alias_id = alias["illustration_coverage"][0]["illustration_id"]
    alias["blocks"] = [{"id": "explanation", "kind": "text", "content": "См. рисунок.",
                         "evidence_ids": [alias_id], "figure_refs": [target]}]
    alias["illustration_coverage"][0]["block_id"] = target
    imported = tmp_path / ("final.json" if complete_import else "replies")
    if complete_import:
        write_json(imported, {**replies[0], "chapters": [
            reply["chapters"][0] for reply in replies]})
    else:
        for reply in replies:
            write_json(imported / (reply["chapters"][0]["id"] + ".json"), reply)
    original = {path: path.read_bytes() for path in (tmp_path / "illustrations").glob("*.png")}
    monkeypatch.setattr("shad_helper.illustration_verification.refine_crop",
                        lambda *args, **kwargs: ("", False, "Рисунок закрыт рукой.", False))
    compiled = []
    monkeypatch.setattr(rendering, "validate_chapter_latex",
                        lambda part, *args, **kwargs: compiled.append(part))
    assert notes.generate_notes(tmp_path, tmp_path / "auth", imported=imported,
                                client=Client()).exists()
    result = notes.validate_notes(read_json(tmp_path / "notes.json"), evidence, finalized=True)
    assert len(compiled) == 2
    assert all(not block["figure_refs"] for chapter in result["chapters"]
               for block in chapter["blocks"])
    assert all(coverage["exclusion_reason"] == "unresolved" for chapter in result["chapters"]
               for coverage in chapter["illustration_coverage"])
    assert all(block["kind"] != "uncertain" for chapter in result["chapters"]
               for block in chapter["blocks"])
    assert result["issues"] and all(issue["status"] == "open" for issue in result["issues"])
    assert not read_json(tmp_path / "review.json")["ready"]
    assert all(path.read_bytes() == data for path, data in original.items())


def test_imported_reply_directory_still_rejects_forward_figure_references(tmp_path):
    evidence = prepare(tmp_path, times=(10, 130), duration=240)
    outline = notes.default_outline(evidence, chunk_seconds=120)
    write_json(tmp_path / "outline.json", outline)
    replies = [chapter_reply(notes.chapter_context(evidence, chapter))
               for chapter in outline["chapters"]]
    first = replies[0]["chapters"][0]
    first["blocks"] = []
    first["illustration_coverage"][0]["block_id"] = replies[1]["chapters"][0]["blocks"][0]["id"]
    for reply in replies:
        write_json(tmp_path / "replies" / (reply["chapters"][0]["id"] + ".json"), reply)
    with pytest.raises(ValueError, match="current or prior"):
        notes.generate_notes(tmp_path, tmp_path / "auth", imported=tmp_path / "replies",
                             client=Client())
