"""Resumable frame selection within the single loaded vision worker."""

import time
from collections import OrderedDict
from pathlib import Path

from PIL import Image

from .extraction import _artifact_checkpoint, _scan_video
from .jobs import checkpoint, fingerprint, read_json, sha256_file
from .video_frames import iter_previews, read_frame
from .vision_mode import PAGE_SELECTION_VERSION


def export_selections(job, video, video_hash, selections, prefix="visual"):
    items = []
    for position, selection in enumerate(selections):
        identifier = f"{prefix}_{position:05d}"
        path = job / "frames" / f"{identifier}.png"

        def export(path=path, identifier=identifier, selection=selection):
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(read_frame(video, selection["timestamp"])).save(path)
            return {"id": identifier, "image": str(path.resolve()),
                    "sha256": sha256_file(path), "timestamp": selection["timestamp"],
                    **{key: selection[key] for key in
                       ("page_id", "page_start", "page_end", "selection_issues") if key in selection}}

        items.append(_artifact_checkpoint(
            job, f"page-frames/{identifier}", {"video": video_hash, "selection": selection},
            export, lambda result: [result["image"]],
        ))
    return items


def prepare_page_images(request, analyze, segmenter, prompt, generation):
    from .whiteboard_pages import POLICY, select_pages

    job, video = Path(request["job"]), Path(request["video"])
    start, end = request["start"], request["end"]
    video_hash = request["video_sha256"]
    inputs = {"video": video_hash, "start": start, "end": end,
              "metadata": request["metadata"], "model": request["model"], "prompt": prompt,
              "generation": generation, "page_selection_version": PAGE_SELECTION_VERSION,
              "policy": POLICY}
    bootstraps = []
    recent = OrderedDict()

    def full_frame(timestamp):
        if timestamp not in recent:
            recent[timestamp] = read_frame(video, timestamp)
            if len(recent) > 4:
                recent.popitem(last=False)
        return recent[timestamp]

    def classify(timestamp):
        identifier = f"bootstrap_{len(bootstraps):05d}"
        path = job / "selection" / f"{identifier}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(full_frame(timestamp)).save(path)
        item = {"id": identifier, "image": str(path.resolve()), "timestamp": timestamp,
                "sha256": sha256_file(path)}
        analysis = analyze(item)
        bootstraps.append({**item, "kind": analysis["kind"], "issues": analysis["issues"]})
        return analysis

    def select():
        started = time.monotonic()
        selections, audit = select_pages(
            iter_previews(video, request["metadata"], start, end),
            start=start, end=end, classify=classify, segmenter=segmenter, read_frame=full_frame,
        )
        return {"selections": selections, "audit": {**audit,
                "selection_seconds": time.monotonic() - started}, "bootstraps": bootstraps}

    saved = job / "checkpoints/page-selection.json"
    reused = saved.exists() and read_json(saved).get("fingerprint") == fingerprint(inputs)
    selected = checkpoint(job, "page-selection", inputs, select)
    # Bootstrap images are audit artifacts; restore them without rescanning or inference.
    for bootstrap in selected["bootstraps"]:
        path = Path(bootstrap["image"])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(full_frame(bootstrap["timestamp"])).save(path)
    started = time.monotonic()
    items = export_selections(job, video, video_hash, selected["selections"])
    audit = {**selected["audit"], "page_selection_version": PAGE_SELECTION_VERSION,
             "policy": POLICY, "bootstraps": selected["bootstraps"],
             "selection_reused": reused, "selections": selected["selections"],
             "selected_video_frames": len(items),
             "export_seconds": time.monotonic() - started}
    return items, audit


def fallback_page_items(request, item, analysis):
    """An unexpected final media type cannot authorize dropping earlier evidence."""
    job, video = Path(request["job"]), Path(request["video"])
    inputs = {"video": request["video_sha256"], "item": item, "analysis": analysis,
              "page_selection_version": PAGE_SELECTION_VERSION}
    selections = checkpoint(
        job, f"page-fallback/{item['page_id']}", inputs,
        lambda: _scan_video(video, request["metadata"], item["page_start"],
                           item["page_end"] - item["page_start"]),
    )
    reason = "Тип финального изображения доски не подтвержден; сохранены промежуточные состояния."
    selections = [{**selection, "selection_issues": [reason]} for selection in selections]
    return export_selections(job, video, request["video_sha256"], selections,
                             prefix=f"{item['id']}_fallback")
