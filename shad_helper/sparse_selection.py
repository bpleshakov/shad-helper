"""Bounded real-frame evidence for completed lecture pages."""

import math
from collections import deque

import cv2
import numpy as np

POLICY = {
    "version": 5,
    "reset_loss": 0.7,
    "confirmation_seconds": 0.5,
    "layout_fraction": 0.18,
    "layout_contrast": 24,
    "board_edge_fraction": 0.6,
    "board_changed_edges": 3,
    "surface_inset_pixels": 3,
    "ink_contrast": 12,
    "full_resolution_contrast": 3,
    "ink_padding": 1,
    "ink_change_fraction": 0.015,
    "person_probability": 0.1,
    "route_attempts": 3,
    "route_retry_seconds": 5.0,
    "max_views": 3,
    "max_erase_probes": 2,
    "erase_probe_seconds": 2.0,
}


def _box(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or not 0 <= x <= 1 for x in value)
            or value[0] >= value[2] or value[1] >= value[3]):
        return None
    return list(value)


def _rectangle(box, shape, padding=0):
    height, width = shape[:2]
    mask = np.zeros((height, width), bool)
    x0, y0, x1, y1 = box
    left, top = max(0, int(x0 * width) - padding), max(0, int(y0 * height) - padding)
    right = min(width, math.ceil(x1 * width) + padding)
    bottom = min(height, math.ceil(y1 * height) + padding)
    mask[top:bottom, left:right] = True
    return mask


def _ink(rgb, roi, contrast=None):
    rows, columns = np.where(roi)
    result = np.zeros(roi.shape, bool)
    if not len(rows):
        return result
    top, bottom, left, right = rows.min(), rows.max() + 1, columns.min(), columns.max() + 1
    crop = rgb[top:bottom, left:right]
    background = cv2.medianBlur(crop, 15)
    difference = np.max(np.abs(crop.astype(np.int16) - background.astype(np.int16)), axis=2)
    result[top:bottom, left:right] = (
        difference > (POLICY["ink_contrast"] if contrast is None else contrast))
    return result & roi


def _expand(ink):
    return cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def _difference(old, new, visible):
    denominator = max(1, np.count_nonzero(old & visible))
    removed = old & ~_expand(new) & visible
    added = new & ~_expand(old) & visible
    return np.count_nonzero(removed) / denominator, np.count_nonzero(added) / denominator


def _content(rgb):
    return bool(_ink(rgb, np.ones(rgb.shape[:2], bool)).any())


def select_sparse(previews, *, start, end, route, segmenter, read_frame):
    """Select at most three actual views per final page, never dense fallback.

    Routing errors that signal a stopped model worker propagate to its caller.
    Unknown routes and mask failures retain bounded evidence with review issues.
    Only a bounded set of preview pixels is retained throughout the scan.
    """
    selections, pages = [], []
    counts = {"preview_count": 0, "classification_calls": 0,
              "segmentation_calls": 0, "full_resolution_checks": 0}
    scene, previous = None, None
    pending_layout = None
    page_number = 0

    def routing(sample):
        counts["classification_calls"] += 1
        result = route(sample["timestamp"])
        if not isinstance(result, dict):
            result = {}
        kind = result.get("kind", "other")
        if kind not in {"whiteboard", "slide", "notebook", "other"}:
            kind = "other"
        box = _box(result.get("lecture_surface_bbox"))
        person = _box(result.get("lecturer_bbox")) if kind == "whiteboard" else None
        return kind, box or [0, 0, 1, 1], person, list(result.get("issues") or [])

    def new_scene(sample, beginning):
        kind, box, person, issues = routing(sample)
        inset = POLICY["surface_inset_pixels"] if kind == "whiteboard" else 0
        roi = _rectangle(box, sample["rgb"].shape, -inset)
        if kind == "other":
            issues.append("Visual layout is unknown; sparse representative evidence requires review.")
        return {"kind": kind, "box": box, "person": person, "roi": roi,
                "issues": issues, "start": beginning, "anchor": sample["rgb"].copy(),
                "attempts": 1, "last_route": sample["timestamp"], "latest": None,
                "anchor_state": None, "probe": None,
                "last_readable": None,
                "foreground_detected": False, "loss_uncertain": False, "uncertain_before": None,
                "replacement_start": None,
                "reference": None, "erase": None, "cleanup": None,
                "views": deque(maxlen=POLICY["max_views"]), "last_mask": -math.inf,
                "native_disabled": segmenter is None, "erase_probes": 0,
                "last_erase_probe": -math.inf,
                "mask": _rectangle(person, sample["rgb"].shape) if person else
                np.zeros(sample["rgb"].shape[:2], bool)}

    def checked_mask(current, state, *, detect_foreground=False):
        if state.get("checked") and not (detect_foreground and not state.get("native_checked")):
            return state["mask"]
        state["checked"] = True
        if current["person"] is None and not current["foreground_detected"] and not detect_foreground:
            state["mask"] = np.zeros(state["rgb"].shape[:2], bool)
            return state["mask"]
        if current["native_disabled"]:
            state["mask_uncertain"] = True
            return state["mask"]
        # Native calls are chronological; a cached/unknown older view cannot
        # rewind a stateful Apple Vision request after a later erasure probe.
        if state["timestamp"] - current["last_mask"] < 0.5:
            state["mask_uncertain"] = True
            return state["mask"]
        counts["segmentation_calls"] += 1
        state["native_checked"] = True
        current["last_mask"] = state["timestamp"]
        try:
            probabilities = segmenter(state["rgb"]) if segmenter else None
        except (OSError, RuntimeError, TypeError, ValueError, cv2.error):
            probabilities = None
        if (probabilities is None or np.shape(probabilities) != state["rgb"].shape[:2]
                or not np.issubdtype(np.asarray(probabilities).dtype, np.number)
                or not np.isfinite(probabilities).all()
                or np.any(np.asarray(probabilities) < 0) or np.any(np.asarray(probabilities) > 1)):
            state["mask_uncertain"] = True
            current["native_disabled"] = True
        else:
            state["mask"] = np.asarray(probabilities) >= POLICY["person_probability"]
            if detect_foreground and state["mask"].any():
                current["foreground_detected"] = True
        return state["mask"]

    def emit(current, boundary, *, views=None, reason="final_state", exclude=None):
        nonlocal page_number
        candidates = list(current["views"] if views is None else views)
        if not candidates:
            return
        for state in candidates:
            checked_mask(current, state)
        final = candidates[-1]
        if exclude is not None:
            candidates = [{**state, "ink": state["ink"] & ~_expand(exclude)} for state in candidates]
            final = candidates[-1]
        # Complementary views may expose hidden ink, but never resurrect a
        # visibly corrected equation from a previous content state.
        compatible = []
        for state in candidates:
            visible = ~(state["mask"] | final["mask"])
            changed = ((state["ink"] & ~_expand(final["ink"]))
                       | (final["ink"] & ~_expand(state["ink"]))) & visible
            if not changed.any():
                compatible.append(state)
        if not compatible:
            compatible = [final]
        preview_known = np.zeros(final["ink"].shape, bool)
        for state in compatible:
            preview_known |= state["ink"]
        if not np.any(preview_known & final["mask"]):
            compatible = [final]
        full = {}

        def full_state(state):
            timestamp = state["timestamp"]
            if timestamp not in full:
                counts["full_resolution_checks"] += 1
                rgb = read_frame(timestamp)
                inset = (math.ceil(POLICY["surface_inset_pixels"] * rgb.shape[1]
                                   / state["rgb"].shape[1])
                         if current["kind"] == "whiteboard" else 0)
                roi = _rectangle(current["box"], rgb.shape, -inset)
                mask = cv2.resize(state["mask"].astype(np.uint8),
                                  (rgb.shape[1], rgb.shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
                ink = _ink(rgb, roi, POLICY["full_resolution_contrast"])
                if exclude is not None:
                    ink &= ~cv2.resize(_expand(exclude).astype(np.uint8),
                                       (rgb.shape[1], rgb.shape[0]),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                full[timestamp] = rgb, roi, mask, ink
            return full[timestamp]

        final_rgb, final_roi, final_mask, final_ink = full_state(final)
        if (final.get("native_checked") and not final.get("mask_uncertain")
                and not np.any(final_mask & final_roi) and final_ink.any()):
            current["loss_uncertain"], current["uncertain_before"] = False, None
            current["issues"] = [issue for issue in current["issues"] if issue != (
                "A suspected erasure may instead be covered by foreground; "
                "page boundaries and terminal content remain uncertain.")]
        checked = [final]
        for state in compatible:
            if state is final:
                continue
            rgb, roi, mask, ink = full_state(state)
            if rgb.shape != final_rgb.shape:
                continue
            visible = roi & final_roi & ~(mask | final_mask)
            writing = ink | final_ink
            difference = np.max(np.abs(rgb.astype(np.int16) - final_rgb.astype(np.int16)), axis=2)
            if not np.any((difference > POLICY["full_resolution_contrast"]) & visible & writing):
                checked.append(state)
        compatible = checked
        known = np.zeros(final["ink"].shape, bool)
        for state in compatible:
            known |= state["ink"] & ~state["mask"]
        # Latest real pixels are primary; complementary views expose covered ink.
        ranked = [final] + sorted(compatible[1:], key=lambda state: (
            np.count_nonzero(known & ~state["mask"]), state["timestamp"],
            state.get("sharpness", 0)), reverse=True)
        chosen, uncovered = [], known.copy()
        for state in ranked:
            if chosen and not np.any(uncovered & ~state["mask"]):
                continue
            chosen.append(state)
            uncovered &= state["mask"]
            if not uncovered.any() or len(chosen) == POLICY["max_views"]:
                break
        issues = list(current["issues"])
        if any(state.get("mask_uncertain") for state in compatible):
            issues.append("Foreground coverage is uncertain; selected real frames require review.")
        if uncovered.any():
            issues.append("Some final-page writing remains occluded in the selected views.")
        if len(chosen) > 1:
            issues.append("Complementary views agree on visible writing; covered corrections remain uncertain.")
        readable = []
        for state in chosen:
            _, _, mask, ink = full_state(state)
            if ink.any() and (np.any(ink & ~mask) or state.get("mask_uncertain")):
                readable.append(state)
        if not readable:
            prior = current["uncertain_before"] or current["last_readable"]
            if (prior is None or prior["timestamp"] >= final["timestamp"]
                    or not np.any(prior["ink"] & final["mask"])
                    and not final.get("mask_uncertain") and not current["loss_uncertain"]):
                return
            _, _, _, prior_ink = full_state(prior)
            if not prior_ink.any():
                return
            readable = [prior]
            issues.append("The terminal writing state is unobservable; an earlier real frame is retained "
                          "only as uncertain evidence and may precede later corrections.")
            reason = "terminal_occlusion_uncertain"
        prior = current["uncertain_before"]
        if (prior is not None and all(state["timestamp"] != prior["timestamp"] for state in readable)
                and full_state(prior)[3].any()):
            readable = readable[:POLICY["max_views"] - 1] + [prior]
        if current["loss_uncertain"]:
            reason = "foreground_boundary_uncertain"
        page_id = f"page_{page_number:05d}"
        page_number += 1
        for position, state in enumerate(readable):
            uncertain = (position and current["uncertain_before"] is not None
                         and state["timestamp"] == current["uncertain_before"]["timestamp"])
            selection = {"timestamp": state["timestamp"], "page_id": page_id,
                         "page_start": current["start"], "page_end": boundary,
                         "selection_role": ("uncertain" if uncertain else
                                            "primary" if position == 0 else "complementary"),
                         "selection_issues": list(dict.fromkeys(issues))}
            if position and not uncertain:
                selection["complement_of"] = page_id
            selections.append(selection)
        pages.append({"page_id": page_id, "start": current["start"], "end": boundary,
                      "kind": current["kind"], "reason": reason,
                      "selected_count": len(readable), "issues": list(dict.fromkeys(issues))})

    def broad_change(current, sample):
        if sample["rgb"].shape != current["anchor"].shape:
            return True
        delta = np.max(np.abs(cv2.GaussianBlur(sample["rgb"], (5, 5), 0).astype(np.int16)
                              - cv2.GaussianBlur(current["anchor"], (5, 5), 0)
                              .astype(np.int16)), axis=2)
        area = np.ones(delta.shape, bool)
        if current["kind"] == "whiteboard":
            rows, columns = np.where(current["roi"])
            if not len(rows):
                return True
            top, bottom, left, right = rows.min(), rows.max() + 1, columns.min(), columns.max() + 1
            thickness = max(1, round(min(bottom - top, right - left) * 0.03))
            # Actor motion outside the initial lecturer box is not a camera
            # change. A physical board exit must alter most of three edges.
            edges = (delta[top:top + thickness, left:right],
                     delta[bottom - thickness:bottom, left:right],
                     delta[top:bottom, left:left + thickness],
                     delta[top:bottom, right - thickness:right])
            changed = sum(np.count_nonzero(edge > POLICY["layout_contrast"]) / edge.size
                          >= POLICY["board_edge_fraction"] for edge in edges)
            if changed >= POLICY["board_changed_edges"] and (current["person"] or current["foreground_detected"]):
                anchor = current["anchor_state"]
                if anchor is not None:
                    checked_mask(current, anchor)
                probe = snapshot(current, sample)
                checked_mask(current, probe)
                current["probe"], current["mask"] = probe, probe["mask"]
                visible = ~(probe["mask"] | (anchor["mask"] if anchor is not None else
                                             current["mask"]))
                visible_edges = (visible[top:top + thickness, left:right],
                                 visible[bottom - thickness:bottom, left:right],
                                 visible[top:bottom, left:left + thickness],
                                 visible[top:bottom, right - thickness:right])
                changed = sum(bool(mask.any()) and np.count_nonzero(
                    (edge > POLICY["layout_contrast"]) & mask) / np.count_nonzero(mask)
                    >= POLICY["board_edge_fraction"] for edge, mask in zip(edges, visible_edges))
            return changed >= POLICY["board_changed_edges"]
        return (np.count_nonzero((delta > POLICY["layout_contrast"]) & area)
                / np.count_nonzero(area) >= POLICY["layout_fraction"])

    def snapshot(current, sample):
        if current["probe"] is not None and current["probe"]["timestamp"] == sample["timestamp"]:
            return current["probe"]
        return {**sample, "rgb": sample["rgb"].copy(), "mask": current["mask"].copy(),
                "ink": _ink(sample["rgb"], current["roi"]), "checked": False}

    def append_view(current, state):
        views = current["views"]
        if views and state["timestamp"] - views[-1]["timestamp"] < 0.5:
            views[-1] = state
        else:
            views.append(state)

    def track_mask(current, sample):
        latest = current["latest"]
        if (latest is None or not (current["person"] or current["foreground_detected"])
                or not current["mask"].any()):
            return
        old = cv2.cvtColor(latest["rgb"], cv2.COLOR_RGB2GRAY)
        new = cv2.cvtColor(sample["rgb"], cv2.COLOR_RGB2GRAY)
        points = cv2.goodFeaturesToTrack(old, 40, 0.1, 3,
                                        mask=current["mask"].astype(np.uint8) * 255)
        if points is None:
            return
        moved, status, error = cv2.calcOpticalFlowPyrLK(old, new, points, None,
                                                      winSize=(15, 15), maxLevel=2)
        if moved is None:
            return
        good = status.reshape(-1).astype(bool) & (error.reshape(-1) < 30)
        if np.count_nonzero(good) < 2:
            return
        shift = np.median((moved - points).reshape(-1, 2)[good], axis=0)
        height, width = old.shape
        if abs(shift[0]) <= width * 0.15 and abs(shift[1]) <= height * 0.15:
            current["mask"] = cv2.warpAffine(
                current["mask"].astype(np.uint8),
                np.array([[1, 0, shift[0]], [0, 1, shift[1]]], np.float32),
                (width, height), flags=cv2.INTER_NEAREST).astype(bool)

    def observe(current, sample):
        track_mask(current, sample)
        state = snapshot(current, sample)
        latest = current["latest"]
        if latest is None:
            if current["person"]:
                checked_mask(current, state)
                current["mask"] = state["mask"]
            current["latest"] = current["reference"] = state
            current["anchor_state"] = state
            if state["ink"].any():
                current["last_readable"] = state
            append_view(current, state)
            return
        visible = ~(latest["mask"] | state["mask"])
        lost, added = _difference(latest["ink"], state["ink"], visible)
        reference = (current["erase"] or {}).get("reference", current["reference"])
        cumulative, _ = _difference(reference["ink"], state["ink"],
                                     ~(reference["mask"] | state["mask"]))
        raw_cumulative, _ = _difference(reference["ink"], state["ink"], current["roi"])
        detect_entry = (current["kind"] == "whiteboard" and current["person"] is None
                        and not current["foreground_detected"] and raw_cumulative >= POLICY["reset_loss"])
        changed = max(lost, added) > POLICY["ink_change_fraction"]
        if current["cleanup"] is not None:
            cleanup = current["cleanup"]
            new_ink = state["ink"] & ~_expand(cleanup)
            if not new_ink.any():
                current["latest"] = state
                return
            current["cleanup"] = None
            current["reference"] = state
            current["views"].clear()
        if ((max(lost, cumulative) >= 0.2 or detect_entry
             or current["native_disabled"] and raw_cumulative >= POLICY["reset_loss"])
                and (current["person"] is not None or current["foreground_detected"] or detect_entry)
                and current["erase_probes"] < POLICY["max_erase_probes"]
                and (detect_entry or state["timestamp"] - current["last_erase_probe"]
                     >= POLICY["erase_probe_seconds"])):
            current["erase_probes"] += 1
            current["last_erase_probe"] = state["timestamp"]
            # Check frozen terminal views before advancing the native request.
            for candidate in current["views"]:
                checked_mask(current, candidate)
            if state["timestamp"] - current["last_mask"] >= 0.5:
                checked_mask(current, state, detect_foreground=detect_entry)
                current["mask"] = state["mask"]
            if (current["kind"] == "whiteboard"
                    and (state.get("mask_uncertain") or detect_entry and state["mask"].any())):
                current["loss_uncertain"] = True
                if current["uncertain_before"] is None:
                    current["uncertain_before"] = current["last_readable"] or latest
                current["issues"].append("A suspected erasure may instead be covered by foreground; "
                                         "page boundaries and terminal content remain uncertain.")
            visible = ~(latest["mask"] | state["mask"])
            lost, added = _difference(latest["ink"], state["ink"], visible)
            changed = max(lost, added) > POLICY["ink_change_fraction"]
        if current["kind"] == "slide" and lost >= 0.5 and state["ink"].any():
            emit(current, sample["timestamp"], reason="slide_replacement")
            current["start"] = sample["timestamp"]
            current["erase"] = current["cleanup"] = None
            current["views"].clear()
            append_view(current, state)
            current["latest"] = current["reference"] = state
            return
        if current["loss_uncertain"] and current["native_disabled"]:
            reference = current["uncertain_before"]
            removal, replacement = _difference(reference["ink"], state["ink"], current["roi"])
            old_sites = reference["ink"] & ~_expand(state["ink"])
            background = cv2.medianBlur(reference["rgb"], 15)
            matches = np.max(np.abs(state["rgb"].astype(np.int16)
                                    - background.astype(np.int16)), axis=2) <= POLICY["layout_contrast"]
            cleared = (old_sites.any() and np.count_nonzero(old_sites & matches)
                       / np.count_nonzero(old_sites) >= 0.9)
            if (removal >= POLICY["reset_loss"] and replacement >= POLICY["reset_loss"]
                    and cleared and np.any(state["ink"] & ~state["mask"])):
                if current["replacement_start"] is None or changed:
                    current["replacement_start"] = state["timestamp"]
                elif state["timestamp"] - current["replacement_start"] >= POLICY["confirmation_seconds"]:
                    beginning = current["replacement_start"]
                    emit(current, beginning, views=[reference], reason="foreground_boundary_uncertain")
                    current.update(start=beginning, erase=None, cleanup=None,
                                   loss_uncertain=False, uncertain_before=None, replacement_start=None,
                                   latest=state, reference=state, last_readable=state, erase_probes=0,
                                   native_disabled=segmenter is None)
                    current["views"].clear()
                    append_view(current, state)
                    current["issues"] = [issue for issue in current["issues"] if issue != (
                        "A suspected erasure may instead be covered by foreground; "
                        "page boundaries and terminal content remain uncertain.")]
                    current["issues"].append("New page inferred after an uncertain boundary; "
                                             "foreground visibility requires review.")
                    return
            else:
                current["replacement_start"] = None
        erase = current["erase"]
        if erase is None and lost > POLICY["ink_change_fraction"]:
            erase = {"start": sample["timestamp"], "reference": current["reference"],
                     "views": list(current["views"]), "confirm": None}
            current["erase"] = erase
        if erase is not None:
            visible = ~(erase["reference"]["mask"] | state["mask"])
            loss, replacement = _difference(erase["reference"]["ink"], state["ink"], visible)
            threshold = POLICY["reset_loss"] if current["kind"] == "whiteboard" else 0.5
            if loss >= threshold and not (current["loss_uncertain"] and current["native_disabled"]):
                if current["kind"] == "notebook" and changed:
                    erase["confirm"] = None
                if erase["confirm"] is None:
                    erase["confirm"] = sample["timestamp"]
                if sample["timestamp"] - erase["confirm"] >= POLICY["confirmation_seconds"]:
                    emit(current, erase["start"], views=erase["views"], reason="major_reset")
                    current["start"] = erase["start"]
                    current["erase"] = None
                    current["last_readable"] = None
                    current["loss_uncertain"] = False
                    current["uncertain_before"] = None
                    current["replacement_start"] = None
                    current["erase_probes"] = 0
                    current["cleanup"] = (state["ink"].copy() if
                                          replacement <= POLICY["ink_change_fraction"] else None)
                    current["views"].clear()
                    if current["cleanup"] is None:
                        append_view(current, state)
                    current["reference"] = state
                    current["latest"] = state
                    return
            elif added > POLICY["ink_change_fraction"] and lost <= POLICY["ink_change_fraction"]:
                current["erase"] = None
                current["reference"] = state
        # A genuine visible content change starts a fresh final-state pool.
        # Occluded changes remain compatible after final native mask checks.
        if changed and current["erase"] is None:
            current["views"].clear()
        append_view(current, state)
        current["latest"] = state
        if (state["ink"].any() and (np.any(state["ink"] & ~state["mask"])
                                   or current["native_disabled"])):
            current["last_readable"] = state
        if current["erase"] is None and not current["loss_uncertain"]:
            current["reference"] = state

    for sample in previews:
        timestamp, rgb = sample["timestamp"], sample["rgb"]
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
                or rgb.shape[2] != 3 or not min(rgb.shape[:2]) or not math.isfinite(timestamp)
                or not start <= timestamp <= end
                or previous is not None and timestamp < previous):
            raise ValueError("Sparse previews require chronological RGB frames within the clip.")
        counts["preview_count"] += 1
        previous = timestamp
        if scene is None:
            if not _content(rgb):
                continue
            scene = new_scene(sample, start)
        elif broad_change(scene, sample):
            if pending_layout is None:
                pending_layout = sample
                continue
            if timestamp - pending_layout["timestamp"] >= POLICY["confirmation_seconds"]:
                emit(scene, pending_layout["timestamp"], reason="layout_change")
                beginning = pending_layout["timestamp"]
                scene = new_scene(sample, beginning) if _content(rgb) else None
                pending_layout = None
                if scene is None:
                    continue
        else:
            # A brief layout need not last a full second to be useful evidence.
            if pending_layout is not None:
                emit(scene, pending_layout["timestamp"], reason="brief_layout")
                if _content(pending_layout["rgb"]):
                    brief = new_scene(pending_layout, pending_layout["timestamp"])
                    observe(brief, pending_layout)
                    emit(brief, timestamp, reason="brief_layout")
                scene = new_scene(sample, timestamp)
                pending_layout = None
            if (scene["kind"] == "other" and scene["attempts"] < POLICY["route_attempts"]
                    and timestamp - scene["last_route"] >= POLICY["route_retry_seconds"]):
                kind, box, person, issues = routing(sample)
                scene.update(kind=kind, box=box, person=person,
                             roi=_rectangle(box, rgb.shape, -POLICY["surface_inset_pixels"]
                                            if kind == "whiteboard" else 0), issues=issues,
                             attempts=scene["attempts"] + 1, last_route=timestamp)
                if kind == "other":
                    scene["issues"].append("Visual layout remains unknown; review sparse evidence.")
        if scene is not None:
            observe(scene, sample)
    if scene is not None:
        if pending_layout is not None:
            emit(scene, pending_layout["timestamp"], reason="layout_exit")
            if _content(pending_layout["rgb"]):
                brief = new_scene(pending_layout, pending_layout["timestamp"])
                observe(brief, pending_layout)
                emit(brief, end, reason="clip_end")
        elif scene["cleanup"] is None:
            emit(scene, end, reason="clip_end")
        else:
            emit(scene, end, reason="clip_end", exclude=scene["cleanup"])
    selections.sort(key=lambda item: (item["timestamp"], item["page_id"]))
    return selections, {"policy": dict(POLICY), **counts, "pages": pages,
                        "selected_count": len(selections)}
