"""Select actual final whiteboard views before exporting full-size images."""

import math

import cv2
import numpy as np

from .extraction import select_keyframes

POLICY = {
    "version": 1,
    "decision_seconds": 2.0,
    "reset_probe_seconds": 0.5,
    "reset_loss": 0.7,
    "reference_visibility": 0.9,
    "confirmation_seconds": 1.0,
    "ambiguity_seconds": 30.0,
    "layout_change": 0.35,
    "layout_settle_seconds": 0.7,
    "ink_contrast": 12,
    "full_resolution_ink_contrast": 3,
    "known_ink_contrast": 3,
    "ink_match_padding": 1,
    "ink_step_noise": 0.01,
    "preview_noise": 3,
    "person_probability": 0.1,
    "person_confidence": 0.9,
    "person_padding": 3,
    "person_tracking_margin": 0.12,
    "surface_inset_pixels": 3,
    "early_decisions": "co_visible_rgb_change_or_reset_candidate",
}


def _valid_rgb(rgb):
    return (isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8
            and rgb.ndim == 3 and rgb.shape[2] == 3 and min(rgb.shape[:2]) > 0)


def _box(value):
    if (not isinstance(value, list) or len(value) != 4
            or any(isinstance(number, bool) or not isinstance(number, (int, float))
                   or not 0 <= number <= 1 or not math.isfinite(number) for number in value)
            or value[0] >= value[2] or value[1] >= value[3]):
        return None
    return value


def _rectangle_mask(box, shape, inset=0):
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    left, top, right, bottom = box
    y0, y1 = math.floor(top * height) + inset, math.ceil(bottom * height) - inset
    x0, x1 = math.floor(left * width) + inset, math.ceil(right * width) - inset
    if y0 < y1 and x0 < x1:
        mask[y0:y1, x0:x1] = True
    return mask


def _ink(rgb, roi, contrast):
    # Per-channel local contrast also detects colored strokes of similar luminance.
    background = cv2.medianBlur(rgb, 15)
    difference = np.max(np.abs(rgb.astype(np.int16) - background.astype(np.int16)), axis=2)
    return (difference > contrast) & roi


def _expand(mask, pixels):
    kernel = np.ones((2 * pixels + 1, 2 * pixels + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _difference(old, new, visible):
    denominator = max(1, np.count_nonzero(old))
    lost = old & ~_expand(new, POLICY["ink_match_padding"]) & visible
    added = new & ~_expand(old, POLICY["ink_match_padding"]) & visible
    return np.count_nonzero(lost) / denominator, np.count_nonzero(added) / denominator


def select_pages(previews, *, start, end, classify, segmenter, read_frame):
    """Select whole-page endings; ambiguous scenes keep the existing keyframes.

    Callbacks receive actual timestamps. Only a few RGB/mask states are retained;
    native-frame scalar metadata remains available for conservative fallback.
    """
    selections, scenes = [], []
    counts = {"preview_count": 0, "classification_calls": 0,
              "segmentation_calls": 0, "full_resolution_checks": 0}
    scene, pending, last_native = None, [], None
    layout_start, last_layout_change = None, None
    page_number = 0

    def fallback(current, reason):
        current["mode"], current["reason"] = "states", reason
        current["review"] = True

    def new_scene(sample, beginning, metadata):
        counts["classification_calls"] += 1
        try:
            analysis = classify(sample["timestamp"])
        except (OSError, RuntimeError, TypeError, ValueError):
            analysis = {"kind": "other", "issues": ["classification_failed"]}
        box = _box(analysis.get("lecture_surface_bbox"))
        valid = (analysis.get("kind") == "whiteboard" and analysis.get("issues") == []
                 and box is not None and all(
                     box[0] <= region["bbox"][0] < region["bbox"][2] <= box[2]
                     and box[1] <= region["bbox"][1] < region["bbox"][3] <= box[3]
                     for region in analysis.get("regions", [])))
        current = {"start": beginning, "metadata": metadata, "analysis": analysis,
                   "box": box, "mode": "pages" if valid else "states",
                   "reason": "whole_whiteboard" if valid else "other_surface",
                   "review": bool(analysis.get("issues")) or
                   (analysis.get("kind") == "whiteboard" and not valid),
                   "pages": [], "page_start": beginning, "latest": None,
                   "reference": None, "current": None, "erase": None, "cleanup": None,
                   "known_ink": np.zeros(sample["rgb"].shape[:2], bool),
                   "last_decision": -math.inf, "last_probe": -math.inf,
                   "person_seed": None, "unknown_since": None}
        if valid:
            current["roi"] = _rectangle_mask(box, sample["rgb"].shape, POLICY["surface_inset_pixels"])
            if not current["roi"].any():
                fallback(current, "surface_too_small")
            lecturer = _box(analysis.get("lecturer_bbox"))
            if lecturer is not None:
                current["person_seed"] = _rectangle_mask(lecturer, sample["rgb"].shape)
        return current

    def foreground(current, rgb):
        counts["segmentation_calls"] += 1
        probabilities = segmenter(rgb) if segmenter is not None else None
        if probabilities is None:
            raise ValueError("segmentation_unavailable")
        probabilities = np.asarray(probabilities)
        if (probabilities.shape != rgb.shape[:2]
                or not np.issubdtype(probabilities.dtype, np.number)
                or not np.isfinite(probabilities).all()
                or np.any((probabilities < 0) | (probabilities > 1))):
            raise ValueError("segmentation_uncertain")
        possible = probabilities >= POLICY["person_probability"]
        empty = np.zeros(rgb.shape[:2], bool)
        if not possible.any():
            return empty
        confident = probabilities >= POLICY["person_confidence"]
        seed = current["person_seed"]
        if seed is None or not confident.any():
            raise ValueError("untracked_person")
        margin = max(1, round(max(rgb.shape[:2]) * POLICY["person_tracking_margin"]))
        _, labels = cv2.connectedComponents(possible.astype(np.uint8))
        tracked = np.unique(labels[confident & _expand(seed, margin)])
        if not len(tracked):
            raise ValueError("untracked_person")
        if len(tracked) != 1:
            raise ValueError("ambiguous_people")
        live = np.isin(labels, tracked)
        if np.any(confident & ~live):
            raise ValueError("ambiguous_people")
        current["person_seed"] = live
        # Track the complete live component, rather than clipping to its old box.
        return _expand(live, POLICY["person_padding"])

    def emit_page(current, state, boundary, *, exclude_ink=None):
        nonlocal page_number
        if state is None:
            return
        known = state["known_ink"]
        if exclude_ink is not None:
            known = known & ~_expand(exclude_ink, POLICY["ink_match_padding"])
        if np.any(known & state["mask"]):
            fallback(current, "final_content_occluded")
            return
        counts["full_resolution_checks"] += 1
        try:
            rgb = read_frame(state["timestamp"])
            if not _valid_rgb(rgb):
                raise ValueError("Invalid full-resolution frame")
            inset = math.ceil(POLICY["surface_inset_pixels"] * rgb.shape[1] / state["rgb"].shape[1])
            roi = _rectangle_mask(current["box"], rgb.shape, inset)
            mask = cv2.resize(state["mask"].astype(np.uint8),
                              (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            ink = _ink(rgb, roi, POLICY["full_resolution_ink_contrast"]) & ~mask
            if exclude_ink is not None:
                old = cv2.resize(_expand(exclude_ink, POLICY["ink_match_padding"]).astype(np.uint8),
                                 (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                ink &= ~old
        except (OSError, RuntimeError, TypeError, ValueError, cv2.error):
            fallback(current, "full_resolution_verification_failed")
            return
        if not ink.any():
            return
        current["pages"].append({"timestamp": state["timestamp"],
                                 "page_id": f"page_{page_number:05d}",
                                 "page_start": current["page_start"], "page_end": boundary,
                                 "selection_issues": []})
        page_number += 1

    def observe(current, sample, *, force=False):
        if current["mode"] != "pages":
            return
        now, rgb = sample["timestamp"], sample["rgb"]
        if now == current["last_decision"]:
            return
        reference = (current["erase"] or current["cleanup"] or {}).get("reference", current["reference"])
        previous = current["current"]
        content_changed = False
        if previous is not None:
            delta = np.max(np.abs(rgb.astype(np.int16) - previous["rgb"].astype(np.int16)), axis=2)
            content_changed = bool(np.any((delta > POLICY["preview_noise"]) & ~previous["mask"]))
        probe = now - current["last_probe"] >= POLICY["reset_probe_seconds"]
        ink = None
        if probe or force:
            ink = _ink(rgb, current["roi"], POLICY["ink_contrast"])
            current["last_probe"] = now
        possible_reset = (current["cleanup"] is None and reference is not None and ink is not None
                          and _difference(reference["ink"], ink, current["roi"])[0]
                          >= POLICY["reset_loss"])
        if (not force and not possible_reset and not content_changed
                and now - current["last_decision"] < POLICY["decision_seconds"]):
            latest = current["latest"]
            if latest is not None and np.max(np.abs(
                    rgb.astype(np.int16) - latest["rgb"].astype(np.int16))) <= POLICY["preview_noise"]:
                current["latest"] = {**latest, "timestamp": now, "rgb": rgb.copy()}
            return
        try:
            mask = foreground(current, rgb)
        except (OSError, RuntimeError, TypeError, ValueError, cv2.error) as error:
            fallback(current, str(error) if isinstance(error, ValueError) else "segmentation_failed")
            return
        current["last_decision"] = now
        if ink is None:
            ink = _ink(rgb, current["roi"], POLICY["ink_contrast"])
        ink &= ~mask
        visible = current["roi"] & ~mask
        if reference is not None:
            outside = ~_rectangle_mask(current["box"], rgb.shape) & ~(reference["mask"] | mask)
            delta = np.max(np.abs(rgb.astype(np.int16) - reference["rgb"].astype(np.int16)), axis=2)
            if np.any((delta > POLICY["preview_noise"]) & outside):
                fallback(current, "outside_surface_changed")
                return
        coverage = (1.0 if reference is None or not reference["ink"].any() else
                    np.count_nonzero(reference["ink"] & visible) / np.count_nonzero(reference["ink"]))
        if coverage < POLICY["reference_visibility"]:
            current["unknown_since"] = current["unknown_since"] if current["unknown_since"] is not None else now
            if now - current["unknown_since"] >= POLICY["ambiguity_seconds"]:
                fallback(current, "occlusion_unresolved")
            return
        current["unknown_since"] = None
        known = _ink(rgb, current["roi"], POLICY["known_ink_contrast"]) & ~mask
        current["known_ink"] = (current["known_ink"] & mask) | known
        state = {"timestamp": now, "rgb": rgb.copy(), "ink": ink, "mask": mask,
                 "known_ink": current["known_ink"]}
        cleanup = current["cleanup"]
        if cleanup is not None:
            added = ink & ~_expand(cleanup["remaining"], POLICY["ink_match_padding"])
            new_writing = np.count_nonzero(added) / max(1, np.count_nonzero(cleanup["reference"]["ink"]))
            cleanup["remaining"] &= known | ~visible
            current["current"] = state
            if new_writing <= POLICY["ink_step_noise"]:
                return
            current["cleanup"], current["reference"] = None, state
            current["latest"] = state
            return
        loss, _ = (0.0, 0.0) if reference is None else _difference(reference["ink"], ink, visible)
        if current["erase"] is None and loss > POLICY["ink_step_noise"]:
            current["erase"] = {"reference": reference, "before": current["latest"],
                                "start": now, "progress": now, "loss": loss,
                                "confirm": None, "writing": 0}
        erase = current["erase"]
        if erase is not None:
            if loss >= POLICY["reset_loss"]:
                if erase["confirm"] is None:
                    erase["confirm"] = now
                elif now - erase["confirm"] >= POLICY["confirmation_seconds"]:
                    emit_page(current, erase["before"], erase["start"])
                    current["page_start"] = erase["start"]
                    _, added = _difference(erase["reference"]["ink"], ink, visible)
                    if added <= POLICY["ink_step_noise"]:
                        current["cleanup"] = {"reference": erase["reference"], "remaining": known.copy()}
                    current["erase"] = None
                    current["reference"] = state
                    current["known_ink"] = known.copy()
                    state["known_ink"] = current["known_ink"]
            else:
                erase["confirm"] = None
                removed, added = (0.0, 0.0) if previous is None else _difference(previous["ink"], ink, visible)
                if removed > POLICY["ink_step_noise"] and added > POLICY["ink_step_noise"]:
                    fallback(current, "mixed_writing_and_erasing")
                    return
                if abs(loss - erase["loss"]) > POLICY["ink_step_noise"] or added > POLICY["ink_step_noise"]:
                    erase["progress"], erase["loss"] = now, loss
                _, replacement = _difference(erase["reference"]["ink"], ink, visible)
                erase["writing"] = erase["writing"] + 1 if (
                    replacement > POLICY["ink_step_noise"] and removed <= POLICY["ink_step_noise"]) else 0
                if erase["writing"] >= 2:
                    current["erase"], current["reference"] = None, state
                elif now - erase["progress"] >= POLICY["ambiguity_seconds"]:
                    fallback(current, "erasure_unresolved")
                    return
        else:
            current["reference"] = state
        current["latest"] = None if current["cleanup"] is not None else state
        current["current"] = state

    def finish(current, last, boundary):
        if current is None:
            return
        observe(current, last, force=True)
        if current["mode"] == "pages" and current["unknown_since"] is not None:
            fallback(current, "final_visibility_unresolved")
        if current["mode"] == "pages":
            if current["cleanup"] is not None:
                emit_page(current, current["current"], boundary, exclude_ink=current["cleanup"]["remaining"])
            else:
                emit_page(current, current["latest"], boundary)
        if current["mode"] == "pages":
            selected = current["pages"]
        else:
            issues = ([("Нельзя уверенно определить окончательное состояние доски; "
                        "сохранены промежуточные кадры для проверки.")] if current["review"] else [])
            selected = [{**sample, "selection_issues": list(issues)}
                        for sample in select_keyframes(current["metadata"])]
        selections.extend(selected)
        scenes.append({"start": current["start"], "end": boundary, "mode": current["mode"],
                       "reason": current["reason"], "selected_count": len(selected)})

    for sample in previews:
        timestamp, rgb = sample["timestamp"], sample["rgb"]
        if not _valid_rgb(rgb) or not math.isfinite(timestamp) or not start <= timestamp <= end:
            raise ValueError("Preview must contain RGB pixels and an actual timestamp within the clip")
        if last_native is not None and timestamp < last_native["timestamp"]:
            raise ValueError("Preview timestamps must be chronological")
        counts["preview_count"] += 1
        metadata = {key: sample[key] for key in ("timestamp", "change", "sharpness")}
        cut = last_native is not None and (rgb.shape != last_native["rgb"].shape
                                          or sample["change"] > POLICY["layout_change"])
        if cut and scene is not None:
            finish(scene, last_native, timestamp)
            scene, layout_start, last_layout_change = None, timestamp, timestamp
        if scene is None and counts["preview_count"] == 1:
            scene = new_scene(sample, start, [])
        if scene is None:
            pending.append(metadata)
            if cut:
                last_layout_change = timestamp
            if timestamp - last_layout_change >= POLICY["layout_settle_seconds"]:
                transition = [value for value in pending if value["timestamp"] < last_layout_change]
                if transition:
                    selected = [{**value, "selection_issues": []} for value in select_keyframes(transition)]
                    selections.extend(selected)
                    scenes.append({"start": layout_start, "end": last_layout_change,
                                   "mode": "states", "reason": "layout_transition",
                                   "selected_count": len(selected)})
                stable = [value for value in pending if value["timestamp"] >= last_layout_change]
                scene = new_scene(sample, last_layout_change, stable)
                pending = []
                observe(scene, sample, force=True)
        else:
            scene["metadata"].append(metadata)
            observe(scene, sample)
        last_native = {**sample, "rgb": rgb.copy()}
    if scene is not None:
        finish(scene, last_native, end)
    elif pending:
        selected = [{**sample, "selection_issues": ["Смена изображения не завершилась до конца видео."]}
                    for sample in select_keyframes(pending)]
        selections.extend(selected)
        scenes.append({"start": layout_start, "end": end, "mode": "states",
                       "reason": "layout_unsettled", "selected_count": len(selected)})
    selections.sort(key=lambda selection: selection["timestamp"])
    return selections, {"policy": dict(POLICY), **counts, "scenes": scenes,
                        "selected_count": len(selections)}
