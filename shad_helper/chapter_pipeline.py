"""Release ready video evidence to a single chronological notes consumer."""

from bisect import bisect_left
from contextvars import copy_context
from copy import deepcopy
from itertools import accumulate
from pathlib import Path
from queue import SimpleQueue
from threading import Thread

from . import notes
from .auth import PlanAccessError
from .chapter_progress import (
    ChapterCancelled,
    ChapterProgress,
    NotesCancellation,
    notes_display,
    notes_phase,
)
from .illustration_verification import (
    figure_matcher,
    finalize_illustrations,
    reconcile_illustrations,
    recover_later,
)
from .illustrations import chapter_assets, illustration_index
from .jobs import (
    VIDEO_EVIDENCE_VERSION,
    fingerprint,
    read_json,
    sha256_file,
    verification_files,
    write_json,
)
from .openai_client import DEFAULT_REASONING_EFFORT, PlanClient, ResponseError
from .rendering import RenderError, validate_chapter_latex
from .run_diagnostics import timed_step


def _base_outline(saved, existing):
    """Reuse base numbering only when the current outline contains its actual split leaves."""
    if not existing or any(saved[k] != existing[k] for k in ("title", "glossary")):
        return saved
    leaves = iter(saved["chapters"])
    leaf = next(leaves, None)

    def matches(chapter):
        nonlocal leaf
        if leaf == chapter:
            leaf = next(leaves, None)
            return True
        if not leaf or (leaf["start"] != chapter["start"] or leaf["end"] >= chapter["end"]):
            return False
        children = notes._split_chapter(chapter)
        return children is not None and all(matches(child) for child in children)

    return existing if all(matches(ch) for ch in existing["chapters"]) and leaf is None else saved


def chapter_plan(evidence, base, frames, batch_ends):
    positions = {frame["id"]: index for index, frame in enumerate(frames, 1)}
    skeleton = dict(evidence, visuals=frames)
    plan = []
    stop = 0
    for chapter in base["chapters"]:
        context = notes.chapter_context(skeleton, chapter)
        ids = [item["id"] for item in context["visuals"]]
        if ids:
            stop = max(stop, max(positions[identifier] for identifier in ids))
        plan.append({"chapter": chapter, "stop": stop,
                     "batch_end": batch_ends[bisect_left(batch_ends, stop)] if stop else 0,
                     "frame_ids": ids, "segment_ids": [s["id"] for s in context["segments"]]})
    return plan


def _policy():
    # Include changes to prompts, verification, retrieval and LaTeX checks on resume.
    modules = ("chapter_pipeline", "notes", "illustrations", "illustration_verification",
               "illustration_crops", "crop_validation", "illustration_stitching",
               "illustration_states", "illustration_matching", "rendering", "schemas")
    return fingerprint([sha256_file(Path(__file__).with_name(name + ".py")) for name in modules])


class ChapterPipeline:
    def __init__(self, job, state_dir, *, proxy=None, rebuild=False):
        self.job, self.state_dir = Path(job).resolve(), state_dir
        self.proxy, self.rebuild = proxy, rebuild
        self.root = self.job / "checkpoints" / "chapter-pipeline"
        self.handoff = self.job / "handoff"
        self.progress = ChapterProgress()
        self.ready = SimpleQueue()
        self.cancel = NotesCancellation()
        self.context = copy_context()
        self.worker = None
        self.result = self.error = None

    def __enter__(self):
        return self

    def start(self, evidence, request):
        if self.rebuild:
            notes.reset_notes(self.job)
        saved = notes.load_outline(self.job, evidence)
        path = self.root / "plan.json"
        base = _base_outline(saved, read_json(path)["base"] if path.exists() else None)
        self.outline = deepcopy(saved)
        write_json(self.job / "outline.json", saved)
        notes.clear_handoff(self.handoff)
        notes.prepare_handoff(self.job, self.outline)
        self.transcript = evidence
        self.frames = [{"id": item["id"], "timestamp": item["timestamp"],
                        "sha256": sha256_file(Path(item["image"]))} for item in request["items"]]
        if request["mode"] == "video":
            from .video_vision import group_frames

            batches = group_frames(request["items"])
            counts = [len(batch) for batch in batches]
            # Inference must consume the exact global grouping used by the scheduler.
            request["batch_sizes"] = counts
        else:
            counts = [1] * len(self.frames)
        self.plan = chapter_plan(evidence, base, self.frames, list(accumulate(counts)))
        binding = {"base": base, "input": fingerprint(evidence), "frames": self.frames,
                   "settings": {k: v for k, v in request.items() if k != "items"},
                   "chapters": self.plan}
        self.binding = fingerprint(binding)
        write_json(path, dict(binding, fingerprint=self.binding))
        self.visuals = []
        self.next_ready = 0
        self.notes_info = {"mode": "automatic", "status": "running", "rebuild": self.rebuild}
        write_json(self.handoff / "run.json", self.notes_info)
        self.progress.start()
        self.progress.update("Notes", self.label(0), "waiting for vision")
        self.worker = Thread(target=self.context.run, args=(self._consume,), name="chapter-notes")
        self.worker.start()
        self.accept([], {})

    def label(self, index):
        return f"{self.plan[index]['chapter']['id']}  {index + 1}/{len(self.plan)}"

    def vision_status(self, phase, frame):
        if phase == "retry":
            self.progress.update("Vision", self.label(self.next_ready), f"retrying {frame}")
        elif phase == "cached" and self.progress.rows:
            self.progress.update("Vision", self.label(self.next_ready), f"cached {frame}")

    def accept(self, batch, assets):
        first = len(self.visuals)
        expected = self.frames[first:first + len(batch)]
        if [(f["id"], f["timestamp"]) for f in batch] != [(f["id"], f["timestamp"]) for f in expected]:
            raise ValueError("Unexpected vision frame or timestamp")
        self.visuals.extend(batch)
        while self.next_ready < len(self.plan):
            index = self.next_ready
            entry = self.plan[index]
            previous = self.plan[index - 1] if index else {"stop": 0, "batch_end": 0}
            total = entry["batch_end"] - previous["batch_end"]
            done = min(total, len(self.visuals) - previous["batch_end"])
            self.progress.update("Vision", self.label(index), "inspecting", done=done,
                                 total=total, unit="frames")
            if len(self.visuals) < entry["batch_end"]:
                break
            delta = self.visuals[previous["stop"]:entry["stop"]]
            manifest = {obs["id"]: assets[obs["id"]] for frame in delta for obs in frame["illustrations"]}
            record = {"plan": self.binding, "index": index, "visuals": delta, "assets": manifest}
            path = self.root / "ready" / f"{entry['chapter']['id']}.json"
            write_json(path, dict(record, fingerprint=fingerprint(record)))
            self.ready.put(path)
            self.next_ready += 1
        if self.next_ready == len(self.plan):
            self.progress.update("Vision", self.label(self.next_ready - 1), "complete")

    def __exit__(self, kind, value, traceback):
        if self.worker is None:
            return
        try:
            if kind is not None:
                self.cancel.set()
                self.progress.update("Vision", self.label(min(self.next_ready, len(self.plan) - 1)),
                                     "interrupted" if kind is KeyboardInterrupt else "failed")
            self.ready.put(None)
            try:
                self.worker.join()
            except KeyboardInterrupt:
                self.cancel.set()
                self.worker.join()
                raise
        finally:
            self.progress.close()
        if kind is None and self.error is not None:
            raise self.error

    def _consume(self):
        try:
            with timed_step("notes") as timing:
                self.result = self._run_notes()
                if self.result == self.handoff:
                    timing["status"] = "handoff"
                    self.notes_info["status"] = "handoff"
                else:
                    self.notes_info["status"] = "completed"
        except ChapterCancelled:
            self.notes_info["status"] = "interrupted"
        except BaseException as error:  # noqa: BLE001 - transport the worker exception to its owner
            self.error = error
            self.notes_info["status"] = "failed"
        finally:
            write_json(self.handoff / "run.json", self.notes_info)

    def _run_notes(self):
        evidence = dict(self.transcript, schema_version=VIDEO_EVIDENCE_VERSION, visuals=[])
        manifest = {"version": 1, "assets": {}}
        contexts = {}
        state = None
        failed = None
        count = 0
        while (path := self.ready.get()) is not None:
            if self.cancel.is_set():
                raise ChapterCancelled
            record = read_json(path)
            digest = record.pop("fingerprint")
            if record["plan"] != self.binding or record["index"] != count or fingerprint(record) != digest:
                raise ValueError("Chapter readiness record does not match the current run")
            index = count
            count += 1
            if failed:
                continue
            evidence["visuals"].extend(record["visuals"])
            manifest["assets"].update(record["assets"])
            base = self.plan[index]["chapter"]
            label = self.label(index)
            chapter = base
            try:
                with chapter_assets(self.job, manifest, self.frames):
                    if state is None:
                        client = PlanClient(self.state_dir, proxy=self.proxy)
                        state = ChapterNotes(self.job, self.outline, client, contexts)
                    position = next(i for i, ch in enumerate(self.outline["chapters"])
                                    if ch["start"] == base["start"])
                    while position < len(self.outline["chapters"]):
                        chapter = self.outline["chapters"][position]
                        if chapter["start"] >= base["end"]:
                            break
                        label = self.label(index)
                        if chapter["id"] != base["id"]:
                            leaves = [ch for ch in self.outline["chapters"]
                                      if base["start"] <= ch["start"] < base["end"]]
                            label += f"  part {leaves.index(chapter) + 1}/{len(leaves)} ({chapter['id']})"
                        with notes_display(self.progress, label, self.cancel):
                            notes_phase("drafting")
                            with timed_step("chapter/" + chapter["id"]):
                                if not state.process(evidence, chapter, self.binding):
                                    continue
                            notes_phase("complete")
                        position += 1
                if index + 1 < len(self.plan):
                    self.progress.update("Notes", self.label(index + 1), "waiting for vision")
            except (PlanAccessError, ResponseError, ValueError, RenderError, OSError) as error:
                if self.cancel.is_set():
                    raise ChapterCancelled from error
                failed = {"status": "manual_required", "chapter": chapter["id"], "reason": str(error)}
                write_json(self.handoff / "status.json", failed)
                self.progress.update("Notes", label, "manual handoff")
                self.progress.log(f"Notes {chapter['id']}: {error}")
        if self.cancel.is_set():
            raise ChapterCancelled
        if count != len(self.plan):
            raise ValueError("Extraction ended before all chapters were ready")
        full = read_json(self.job / "evidence.json")
        from .extraction import assert_extraction_complete

        assert_extraction_complete(self.job, full)
        if failed:
            return notes.export_handoff(self.job, full, self.outline, contexts=contexts)
        if evidence != full:
            raise ValueError("Published evidence differs from the ready chapters")
        notes.require_content(full)
        if [ch["id"] for ch in state.result["chapters"]] != [ch["id"] for ch in self.outline["chapters"]]:
            raise ValueError("Validated notes do not cover the complete outline")
        resolutions = self.job / "review-resolutions.json"
        result = notes.validate_notes(state.result, full, read_json(resolutions) if resolutions.exists()
                                      else {}, finalized=True)
        return notes.publish_notes(self.job, self.outline, result)


class ChapterNotes:
    """The notes thread owns drafts, unresolved sources and ordered figure decisions."""

    def __init__(self, job, outline, client, contexts):
        self.job, self.client = job, client
        self.root = job / "checkpoints" / "chapter-pipeline"
        self.drafts = notes.ChapterDrafts(job, outline, client, {}, contexts)
        self.parts, self.pending, self.matches = [], {}, {}
        self.result = None
        self.signature = fingerprint((_policy(), client.model, DEFAULT_REASONING_EFFORT))

    def process(self, evidence, chapter, binding):
        previous = [self.result] if self.result else []
        draft = self.drafts.generate(evidence, chapter, previous=previous)
        if draft is None:
            return False
        path = self.root / "notes" / (chapter["id"] + ".json")
        key = fingerprint((self.signature, binding, evidence, draft))
        try:
            saved = read_json(path)
        except (FileNotFoundError, ValueError):
            saved = {}
        digest = saved.pop("fingerprint", None)
        if (saved.get("inputs") == key and digest == fingerprint(saved) and not saved["retry"] and
                all((self.job / file).is_file() and sha256_file(self.job / file) == digest
                    for file, digest in saved["files"].items())):
            self.parts, self.pending, self.matches, self.result = (
                saved[name] for name in ("parts", "pending", "matches", "notes"))
            notes_phase("cached")
        else:
            with verification_files(self.job) as used:
                retry = self._verify(draft, evidence, previous)
            files = {str(file): sha256_file(self.job / file) for file in sorted(used)
                     if (self.job / file).is_file()}
            saved = {"inputs": key, "parts": self.parts, "pending": self.pending,
                     "matches": self.matches, "notes": self.result, "files": files, "retry": retry}
            digest = fingerprint(saved)
            write_json(path, dict(saved, fingerprint=digest))
        self.signature = digest
        write_json(self.root / "notes-state.json", {"chapter": chapter["id"],
                                                   "fingerprint": self.signature, "plan": binding})
        write_json(self.job / "handoff" / "figure-reconciliation" / "replacements.json",
                   self.matches.get("replacements", {}))
        return True

    def _verify(self, draft, evidence, previous):
        identifier = draft["chapters"][0]["id"]
        handoff = self.job / "handoff"
        folder = handoff / "verification" / identifier
        prior, pending, matches = self.parts, self.pending, self.matches
        for attempt in range(2):
            self.parts = deepcopy(prior)
            self.pending = deepcopy(pending)
            self.matches = deepcopy(matches)
            matcher = figure_matcher(self.job, evidence, self.matches.get("examples", []))
            part = draft
            retry = False
            if illustration_index(notes.chapter_context(evidence, draft["chapters"][0])):
                notes_phase("verifying illustrations")
                part, retry = finalize_illustrations(draft, evidence, self.job,
                    folder / f"attempt-{attempt + 1:02d}", self.client, previous=previous,
                    pending=self.pending, matcher=matcher)
            notes_phase("recovering earlier observations")
            retry |= recover_later(self.parts, part, self.pending, evidence, self.job,
                                   folder / "earlier-recovery", self.client, matcher)
            self.parts.append(part)
            combined = notes._merge_chapters(self.parts, self.drafts.outline, evidence)
            notes_phase("reconciling figures")
            result, transient = reconcile_illustrations(combined, evidence, self.job,
                handoff / "figure-reconciliation", self.client, state=self.matches, matcher=matcher, publish=False)
            retry |= transient
            self.result = notes.validate_notes(result, evidence, finalized=True)
            if not retry:
                break
        write_json(folder / "status.json", {"attempts": attempt + 1, "retry_needed": retry})
        notes_phase("checking LaTeX")
        # Unillustrated drafts already passed their isolated compile during generation.
        before = {ch["id"]: ch for ch in previous[0]["chapters"]} if previous else {}
        for chapter in self.result["chapters"]:
            if (chapter != before.get(chapter["id"]) and
                    illustration_index(notes.chapter_context(evidence, chapter))):
                validate_chapter_latex(dict(self.result, chapters=[chapter]), evidence,
                                       folder / "latex-check" / chapter["id"], asset_job=self.job)
        return retry
