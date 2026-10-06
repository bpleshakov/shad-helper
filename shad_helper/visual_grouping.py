"""Conservative reuse of real visual analyses while preserving every occurrence."""

import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .jobs import sha256_file
from .vision_mode import GROUPING_VERSION

NOISE_TOLERANCE = 3
PERSON_PROBABILITY = 0.1
PERSON_CONFIDENCE = 0.9
MASK_PADDING_PIXELS = 3
SURFACE_PADDING_PIXELS = 8


def _box(value):
    if (not isinstance(value, list) or len(value) != 4
            or any(isinstance(number, bool) or not isinstance(number, (int, float))
                   or not 0 <= number <= 1 or not math.isfinite(number) for number in value)
            or value[0] >= value[2] or value[1] >= value[3]):
        return None
    return value


def _rectangle(box, shape, padding=0):
    height, width = shape[:2]
    left, top, right, bottom = box
    return (max(0, math.floor(left * width) - padding),
            max(0, math.floor(top * height) - padding),
            min(width, math.ceil(right * width) + padding),
            min(height, math.ceil(bottom * height) + padding))


def _surface(analysis, shape):
    box = _box(analysis.get("lecture_surface_bbox"))
    # A current text-only box cannot stand in for the entire lecture surface.
    if box is not None and any(
            region["bbox"][0] < box[0] or region["bbox"][1] < box[1]
            or region["bbox"][2] > box[2] or region["bbox"][3] > box[3]
            for region in analysis["regions"]):
        box = None
    rectangle = _rectangle(box or [0, 0, 1, 1], shape, SURFACE_PADDING_PIXELS)
    mask = np.zeros(shape[:2], dtype=bool)
    left, top, right, bottom = rectangle
    mask[top:bottom, left:right] = True
    return mask, list(rectangle)


def _foreground(rgb, analysis, segmenter):
    empty = np.zeros(rgb.shape[:2], dtype=bool)
    lecturer = _box(analysis.get("lecturer_bbox"))
    # The prompt only supplies lecturer_bbox for live physical-whiteboard footage.
    if analysis["kind"] != "whiteboard" or lecturer is None:
        return empty, empty, None
    if segmenter is None:
        return empty, empty, "segmentation_unavailable"
    probabilities = segmenter(rgb)
    if probabilities is None:
        return empty, empty, "segmentation_unavailable"
    probabilities = np.asarray(probabilities)
    if (probabilities.shape != rgb.shape[:2]
            or not np.issubdtype(probabilities.dtype, np.number)
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))):
        return empty, empty, "segmentation_uncertain"
    possible = probabilities >= PERSON_PROBABILITY
    confident = probabilities >= PERSON_CONFIDENCE
    left, top, right, bottom = _rectangle(lecturer, rgb.shape)
    bounds = empty.copy()
    bounds[top:bottom, left:right] = True
    base = empty.copy()
    if possible.any():
        seeds = confident & bounds
        if not seeds.any():
            return empty, empty, "segmentation_uncertain"
        _, labels = cv2.connectedComponents(possible.astype(np.uint8))
        live_labels = np.unique(labels[seeds])
        base = np.isin(labels, live_labels) & bounds
    kernel = np.ones((2 * MASK_PADDING_PIXELS + 1,) * 2, dtype=np.uint8)
    padded = cv2.dilate(base.astype(np.uint8), kernel).astype(bool) & bounds
    return base, padded, None


def _changed(reference, current, visible):
    difference = np.max(np.abs(reference.astype(np.int16) - current.astype(np.int16)), axis=2)
    # Never divide by image area: one faint colored pixel can be a decimal point.
    return bool(np.any((difference > NOISE_TOLERANCE) & visible))


def _pixel_key(rgb):
    return rgb.shape, hashlib.sha256(rgb.tobytes()).hexdigest()


def _eligible(analysis):
    return (analysis.get("issues") == [] and isinstance(analysis.get("summary"), str)
            and isinstance(analysis.get("regions"), list)
            and "diagram" in analysis
            and analysis.get("kind") in ("slide", "notebook", "whiteboard", "other"))


def _quality(rgb, surface, mask):
    visible = surface & ~mask
    coverage = float(np.count_nonzero(visible) / np.count_nonzero(surface))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # Do not score the artificial edge of an occlusion as clearer writing.
    interior = cv2.erode(visible.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    sharpness = float(np.var(laplacian[interior])) if interior.any() else 0.0
    return coverage, sharpness


def group_visuals(items, analyze, segmenter, max_pending_seconds=30):
    """Return one record per occurrence; analyze owns inference and checkpoints.

    Unknown occlusions remain pending until observed against the same fixed
    anchor. A change or timeout analyzes those original frames independently.
    """
    outputs = [None] * len(items)
    decisions = [None] * len(items)
    scores = [None] * len(items)
    pixel_keys = [None] * len(items)
    exact = {}
    pending = []
    anchor = None
    calls = 0

    def emit(position, state, reason):
        item = items[position]
        original = str(item["image"])
        record = {**state["analysis"], "id": item["id"], "timestamp": item.get("timestamp"),
                  "image": state["image"], "sha256": state["sha256"],
                  "original_image": original, "original_sha256": sha256_file(Path(original)),
                  "analysis_id": state["id"]}
        record["issues"] = list(dict.fromkeys([
            *record.get("issues", []), *item.get("selection_issues", [])]))
        if "source_id" in item:
            record["source_id"] = item["source_id"]
        else:
            record.pop("source_id", None)
        for field in ("page_id", "page_start", "page_end"):
            if field in item:
                record[field] = item[field]
            else:
                record.pop(field, None)
        outputs[position] = record
        previous = decisions[position]
        decisions[position] = {"id": item["id"], "timestamp": item.get("timestamp"),
                               "original_image": original,
                               "original_sha256": record["original_sha256"],
                               "analysis_id": state["id"], "reason": reason,
                               "comparison_reason": (previous["comparison_reason"]
                                                     if previous else reason)}

    def full(position, reason, rgb=None, *, track_foreground=True):
        nonlocal calls
        item = items[position]
        if rgb is None:
            with Image.open(item["image"]) as image:
                rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        calls += 1
        analysis = analyze(item)
        state = {"id": item["id"], "image": str(item["image"]),
                 "sha256": sha256_file(Path(item["image"])), "analysis": analysis, "rgb": rgb}
        state["surface"], state["rectangle"] = _surface(analysis, rgb.shape)
        if track_foreground and item.get("timestamp") is not None and _eligible(analysis):
            state["base"], state["mask"], state["uncertain"] = _foreground(rgb, analysis, segmenter)
        else:
            state["base"] = state["mask"] = np.zeros(rgb.shape[:2], dtype=bool)
            state["uncertain"] = None
        if track_foreground or scores[position] is None:
            scores[position] = _quality(rgb, state["surface"], state["mask"])
        pixel_keys[position] = _pixel_key(rgb)
        emit(position, state, reason)
        if item.get("timestamp") is not None and _eligible(analysis):
            exact[_pixel_key(rgb)] = {field: state[field]
                                     for field in ("id", "image", "sha256", "analysis")}
        return state

    def flush(reason):
        for candidate in pending:
            full(candidate["position"], reason, track_foreground=False)
        pending.clear()

    def resolve_pending(mask, now):
        retained = []
        for candidate in pending:
            revealed = candidate["unknown"] & ~mask & ~anchor["mask"]
            candidate["unknown"][revealed] = False
            if not candidate["unknown"].any():
                emit(candidate["position"], anchor, "occlusion_resolved")
            elif now - items[candidate["position"]]["timestamp"] >= max_pending_seconds:
                full(candidate["position"], "pending_timeout", track_foreground=False)
            else:
                retained.append(candidate)
        pending[:] = retained

    for position, item in enumerate(items):
        with Image.open(item["image"]) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        if item.get("timestamp") is None:
            flush("source_boundary")
            full(position, "source", rgb)
            anchor = None
            continue
        key = _pixel_key(rgb)
        pixel_keys[position] = key
        if key in exact:
            if anchor is not None and key == _pixel_key(anchor["rgb"]):
                resolve_pending(anchor["mask"], item["timestamp"])
            else:
                flush("exact_state_changed")
                anchor = {**exact[key], "rgb": rgb}
                anchor["surface"], anchor["rectangle"] = _surface(anchor["analysis"], rgb.shape)
                anchor["base"], anchor["mask"], anchor["uncertain"] = _foreground(
                    rgb, anchor["analysis"], segmenter)
            emit(position, exact[key], "exact_pixels")
            scores[position] = _quality(rgb, anchor["surface"], anchor["mask"])
            continue
        if anchor is None:
            anchor = full(position, "first", rgb)
            continue
        if not _eligible(anchor["analysis"]):
            flush("ineligible_anchor")
            anchor = full(position, "ineligible_anchor", rgb)
            continue
        if rgb.shape != anchor["rgb"].shape:
            flush("frame_geometry_changed")
            anchor = full(position, "frame_geometry_changed", rgb)
            continue
        base, mask, uncertainty = _foreground(rgb, anchor["analysis"], segmenter)
        if uncertainty or anchor["uncertain"]:
            reason = uncertainty or anchor["uncertain"]
            flush(reason)
            anchor = full(position, reason, rgb)
            continue
        surface = anchor["surface"]
        visible = ~(anchor["mask"] | mask)
        # Padding is not permission to discard visible ink beside a hand.
        boundary = (anchor["mask"] | mask) & ~(anchor["base"] | base)
        if _changed(anchor["rgb"], rgb, visible | boundary):
            flush("changed_before_occlusion_resolved")
            reason = ("outside_surface_changed" if _changed(anchor["rgb"], rgb,
                      (visible | boundary) & ~surface) else "visible_content_changed")
            anchor = full(position, reason, rgb)
            continue
        newly_visible = anchor["base"] & ~base
        if newly_visible.any():
            flush("unknown_anchor_baseline")
            anchor = full(position, "unknown_anchor_baseline", rgb)
            continue
        resolve_pending(mask, item["timestamp"])
        scores[position] = _quality(rgb, surface, mask)
        unknown = anchor["mask"] | mask
        if unknown.any():
            pending.append({"position": position, "unknown": unknown.copy()})
        else:
            emit(position, anchor, "near_duplicate")
    flush("clip_end_unresolved")
    members = {}
    for position, output in enumerate(outputs):
        members.setdefault(output["analysis_id"], []).append(position)
    for positions in members.values():
        if len(positions) < 2:
            continue
        canonical = next(position for position in positions
                         if items[position]["id"] == outputs[position]["analysis_id"])
        best = max(positions, key=lambda position: scores[position])
        if scores[best] <= scores[canonical] or pixel_keys[best] == pixel_keys[canonical]:
            continue
        representative = full(best, "better_representative", track_foreground=False)
        if _eligible(representative["analysis"]):
            for position in positions:
                emit(position, representative, "clearest_real_representative")
        else:
            decisions[best]["reason"] = "representative_analysis_issues"
    groups = {}
    for output in outputs:
        group = groups.setdefault(output["analysis_id"], {
            "analysis_id": output["analysis_id"], "image": output["image"],
            "sha256": output["sha256"], "occurrence_ids": []})
        group["occurrence_ids"].append(output["id"])
    audit = {"version": GROUPING_VERSION, "input_count": len(items),
             "analysis_calls": calls,
             "reused_count": sum(output["id"] != output["analysis_id"] for output in outputs),
             "policy": {"noise_tolerance_per_channel": NOISE_TOLERANCE,
                        "person_probability": PERSON_PROBABILITY,
                        "person_confidence": PERSON_CONFIDENCE,
                        "mask_padding_pixels": MASK_PADDING_PIXELS,
                        "surface_padding_pixels": SURFACE_PADDING_PIXELS,
                        "max_pending_seconds": max_pending_seconds,
                        "foreground": "confirmed_live_whiteboard_only",
                        "mask_extent": "seeded_components_clipped_to_lecturer_bbox",
                        "outside_surface": "same_per_pixel_change_guard",
                        "representative": "visible_surface_coverage_then_laplacian_sharpness",
                        "unknown": "retain_until_same_anchor_revealed_or_analyze"},
             "decisions": decisions, "groups": list(groups.values())}
    return outputs, audit
