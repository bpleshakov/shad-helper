"""Grouping must preserve tiny changes, occlusions, chronology and real images."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from shad_helper.jobs import sha256_file
from shad_helper.visual_grouping import NOISE_TOLERANCE, group_visuals


class VisualGroupingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.job = Path(self.directory.name)
        self.calls = []

    def item(self, number, rgb=None, timestamp=None, source=False):
        if rgb is None:
            rgb = np.full((64, 64, 3), 200, dtype=np.uint8)
        path = self.job / f"visual_{number}.png"
        Image.fromarray(rgb).save(path)
        result = {"id": f"visual_{number}", "image": str(path),
                  "timestamp": None if source else number if timestamp is None else timestamp,
                  "sha256": sha256_file(path)}
        if source:
            result["source_id"] = f"source_{number}"
        return result

    def analyze(self, item, *, kind="whiteboard", issues=None, surface=None, lecturer=None):
        self.calls.append(item["id"])
        return {"kind": kind, "summary": "Реально видимое содержимое доски.",
                "regions": [], "diagram": None, "issues": issues or [],
                "lecture_surface_bbox": surface, "lecturer_bbox": lecturer}

    def live_analyze(self, item):
        return self.analyze(item, lecturer=[0, 0, 1, 1])

    @staticmethod
    def person(rgb):
        mask = np.zeros(rgb.shape[:2], dtype=np.float32)
        if rgb[20, 20, 0] < 180:
            mask[16:32, 16:32] = 1
        return mask

    @staticmethod
    def occluded():
        rgb = np.full((64, 64, 3), 200, dtype=np.uint8)
        rgb[16:32, 16:32] = 30
        return rgb

    def test_exact_pixels_keep_occurrence_and_actual_analysis_provenance(self):
        first, second = self.item(0), self.item(1)
        outputs, audit = group_visuals([first, second], self.analyze, None)
        self.assertEqual(self.calls, ["visual_0"])
        self.assertEqual(outputs[1]["id"], "visual_1")
        self.assertEqual(outputs[1]["timestamp"], 1)
        self.assertEqual(outputs[1]["analysis_id"], "visual_0")
        self.assertEqual(outputs[1]["image"], first["image"])
        self.assertEqual(outputs[1]["sha256"], first["sha256"])
        self.assertEqual(outputs[1]["original_image"], second["image"])
        self.assertEqual(outputs[1]["original_sha256"], second["sha256"])
        self.assertEqual(audit["analysis_calls"], 1)
        self.assertEqual(audit["decisions"][1]["reason"], "exact_pixels")
        json.dumps(audit, allow_nan=False)

    def test_tolerated_noise_is_compared_to_a_fixed_anchor(self):
        first = self.item(0)
        noise = np.full((64, 64, 3), 200 + NOISE_TOLERANCE, dtype=np.uint8)
        outputs, _ = group_visuals([first, self.item(1, noise)], self.analyze, None)
        self.assertEqual(self.calls, ["visual_0"])
        self.assertEqual(outputs[1]["image"], first["image"])
        self.assertNotEqual(outputs[1]["original_sha256"], outputs[1]["sha256"])
        self.calls.clear()
        small = np.full((64, 64, 3), 200, dtype=np.uint8)
        small[40, 40, 1] += 2
        accumulated = small.copy()
        accumulated[40, 40, 1] += 2
        outputs, _ = group_visuals([first, self.item(2, small), self.item(3, accumulated)],
                                  self.analyze, None)
        self.assertEqual(self.calls, ["visual_0", "visual_3", "visual_2"])
        self.assertEqual([output["analysis_id"] for output in outputs],
                         ["visual_2", "visual_2", "visual_3"])

    def test_one_faint_colored_pixel_addition_or_removal_forces_analysis(self):
        for delta in (4, 8):
            for removed in (False, True):
                with self.subTest(delta=delta, removed=removed):
                    self.calls.clear()
                    blank = np.full((64, 64, 3), 200, dtype=np.uint8)
                    marked = blank.copy()
                    marked[40, 40, 2] += delta
                    pair = [marked, blank] if removed else [blank, marked]
                    outputs, audit = group_visuals(
                        [self.item(0, pair[0]), self.item(1, pair[1])], self.analyze, None)
                    self.assertEqual(self.calls, ["visual_0", "visual_1"])
                    self.assertEqual(outputs[1]["analysis_id"], "visual_1")
                    self.assertEqual(audit["decisions"][1]["reason"], "visible_content_changed")

    def test_occluded_occurrence_waits_for_a_real_unchanged_reveal(self):
        for exact_reveal in (False, True):
            with self.subTest(exact_reveal=exact_reveal):
                self.calls.clear()
                reveal = np.full((64, 64, 3), 200 if exact_reveal else 201, dtype=np.uint8)
                items = [self.item(0), self.item(1, self.occluded()), self.item(2, reveal)]
                outputs, audit = group_visuals(items, self.live_analyze, self.person)
                self.assertEqual(self.calls, ["visual_0"])
                self.assertEqual([output["analysis_id"] for output in outputs], ["visual_0"] * 3)
                self.assertEqual(audit["decisions"][1]["reason"], "occlusion_resolved")
                self.assertEqual(outputs[1]["original_image"], items[1]["image"])

    def test_stateful_segmentation_does_not_revisit_flushed_older_frames(self):
        changed = np.full((64, 64, 3), 200, dtype=np.uint8)
        changed[50, 50, 0] += 4
        items = [self.item(0), self.item(1, self.occluded()), self.item(2, changed)]
        sequence = []

        def segment(rgb):
            position = 1 if rgb[20, 20, 0] < 180 else 2 if rgb[50, 50, 0] > 200 else 0
            sequence.append(position)
            return self.person(rgb)

        group_visuals(items, self.live_analyze, segment)
        self.assertEqual(self.calls, ["visual_0", "visual_1", "visual_2"])
        self.assertEqual(sequence, [0, 1, 2, 2])

    def test_new_ink_after_occlusion_is_not_assigned_to_earlier_occurrence(self):
        reveal = np.full((64, 64, 3), 200, dtype=np.uint8)
        reveal[20, 20, 2] += 4
        items = [self.item(0), self.item(1, self.occluded()), self.item(2, reveal)]
        outputs, audit = group_visuals(items, self.live_analyze, self.person)
        self.assertEqual(self.calls, ["visual_0", "visual_1", "visual_2"])
        self.assertEqual([output["analysis_id"] for output in outputs],
                         ["visual_0", "visual_1", "visual_2"])
        self.assertEqual(outputs[1]["image"], items[1]["image"])
        self.assertEqual(audit["decisions"][1]["reason"], "changed_before_occlusion_resolved")

    def test_visible_faint_strokes_in_mask_padding_override_similarity(self):
        for removed in (False, True):
            with self.subTest(removed=removed):
                self.calls.clear()
                first = np.full((64, 64, 3), 200, dtype=np.uint8)
                later = self.occluded()
                (first if removed else later)[20, 32, 1] += 4
                _, audit = group_visuals([self.item(0, first), self.item(1, later)],
                                        self.live_analyze, self.person)
                self.assertEqual(self.calls, ["visual_0", "visual_1"])
                self.assertEqual(audit["decisions"][1]["reason"], "visible_content_changed")

    def test_pre_erasure_and_brief_changed_states_are_retained(self):
        marked = np.full((64, 64, 3), 200, dtype=np.uint8)
        marked[40, 40, 0] += 8
        items = [self.item(0), self.item(1, marked, timestamp=0.04), self.item(2, timestamp=0.08)]
        outputs, _ = group_visuals(items, self.analyze, None)
        self.assertEqual(self.calls, ["visual_0", "visual_1"])
        self.assertEqual(outputs[1]["analysis_id"], "visual_1")
        self.assertEqual(outputs[1]["timestamp"], 0.04)

    def test_occlusion_at_clip_end_or_timeout_is_fully_analyzed(self):
        _, audit = group_visuals([self.item(0), self.item(1, self.occluded())],
                                 self.live_analyze, self.person)
        self.assertEqual(self.calls, ["visual_0", "visual_1"])
        self.assertEqual(audit["decisions"][1]["reason"], "clip_end_unresolved")
        self.calls.clear()
        later = self.occluded() + 1
        _, audit = group_visuals([self.item(0), self.item(1, self.occluded()),
                                 self.item(2, later, timestamp=32)], self.live_analyze, self.person)
        self.assertEqual(self.calls, ["visual_0", "visual_1", "visual_2"])
        self.assertEqual(audit["decisions"][1]["reason"], "pending_timeout")

    def test_newly_revealed_pixels_without_anchor_baseline_are_analyzed(self):
        outputs, audit = group_visuals([self.item(0, self.occluded()), self.item(1)],
                                      self.live_analyze, self.person)
        self.assertEqual(self.calls, ["visual_0", "visual_1"])
        self.assertEqual(outputs[1]["analysis_id"], "visual_1")
        self.assertEqual(audit["decisions"][1]["reason"], "unknown_anchor_baseline")

    def test_sources_and_issue_bearing_analyses_never_group(self):
        for source, issues in ((True, []), (False, ["Неясная запись."])):
            with self.subTest(source=source):
                self.calls.clear()
                items = [self.item(0, source=source), self.item(1, source=source)]
                outputs, _ = group_visuals(
                    items, lambda item, issues=issues: self.analyze(item, issues=issues), None)
                self.assertEqual(self.calls, ["visual_0", "visual_1"])
                self.assertEqual([output["analysis_id"] for output in outputs],
                                 ["visual_0", "visual_1"])

    def test_people_in_slides_or_notebooks_are_not_masked(self):
        def forbidden_segmentation(_):
            self.fail("Slide/notebook depictions must not be removed.")

        for kind in ("slide", "notebook", "other"):
            with self.subTest(kind=kind):
                self.calls.clear()
                changed = self.occluded()
                group_visuals([self.item(0), self.item(1, changed)],
                              lambda item, kind=kind: self.analyze(item, kind=kind),
                              forbidden_segmentation)
                self.assertEqual(self.calls, ["visual_0", "visual_1"])

    def test_uncertain_segmentation_does_not_enable_reuse(self):
        for mask in (np.full((64, 64), np.nan), np.full((64, 64), 0.5), np.zeros((1, 1))):
            with self.subTest(shape=mask.shape):
                self.calls.clear()
                noise = np.full((64, 64, 3), 201, dtype=np.uint8)
                _, audit = group_visuals([self.item(0), self.item(1, noise)], self.live_analyze,
                                         lambda _, mask=mask: mask)
                self.assertEqual(self.calls, ["visual_0", "visual_1"])
                self.assertEqual(audit["decisions"][1]["reason"], "segmentation_uncertain")

    def test_missing_masks_retain_frames_with_an_explicit_reason(self):
        for segmenter in (None, lambda _: None):
            with self.subTest(segmenter=segmenter):
                self.calls.clear()
                noise = np.full((64, 64, 3), 201, dtype=np.uint8)
                _, audit = group_visuals([self.item(0), self.item(1, noise)],
                                         self.live_analyze, segmenter)
                self.assertEqual(self.calls, ["visual_0", "visual_1"])
                self.assertEqual(audit["decisions"][1]["reason"], "segmentation_unavailable")

    def test_invalid_surface_proposal_falls_back_to_full_frame(self):
        changed = np.full((64, 64, 3), 200, dtype=np.uint8)
        changed[0, 0, 0] += 4

        def analyze(item):
            result = self.analyze(item, surface=[0.3, 0.3, 0.7, 0.7])
            result["regions"] = [{"kind": "text", "bbox": [0, 0, 1, 1]}]
            return result

        group_visuals([self.item(0), self.item(1, changed)], analyze, None)
        self.assertEqual(self.calls, ["visual_0", "visual_1"])

    def test_new_writing_second_board_or_layout_change_outside_roi_is_retained(self):
        for changed_area in ("single_pixel", "second_board", "camera_layout"):
            with self.subTest(changed_area=changed_area):
                self.calls.clear()
                later = np.full((60, 100, 3), 200, dtype=np.uint8)
                if changed_area == "single_pixel":
                    later[30, 80, 2] += 4
                elif changed_area == "second_board":
                    later[20:22, 70:90] = 20
                else:
                    later[:, 80:90] = 80

                def analyze(item):
                    result = self.analyze(item, surface=[0, 0, 0.5, 1])
                    result["regions"] = [{"kind": "text", "bbox": [0.1, 0.1, 0.2, 0.2]}]
                    return result

                first = np.full((60, 100, 3), 200, dtype=np.uint8)
                outputs, audit = group_visuals(
                    [self.item(0, first), self.item(1, later)], analyze, None)
                self.assertEqual(self.calls, ["visual_0", "visual_1"])
                self.assertEqual(outputs[1]["analysis_id"], "visual_1")
                self.assertEqual(audit["decisions"][1]["reason"], "outside_surface_changed")

    def test_separate_depicted_person_on_board_is_not_masked_with_live_lecturer(self):
        first = np.full((60, 100, 3), 200, dtype=np.uint8)
        first[15:45, 2:8] = 0
        depicted = first.copy()
        depicted[20:40, 50:56] = 0

        def analyze(item):
            return self.analyze(item, surface=[0.2, 0, 1, 1], lecturer=[0, 0.2, 0.1, 0.8])

        def people(rgb):
            return np.where(np.all(rgb == 0, axis=2), 0.99, 0.0)

        outputs, audit = group_visuals(
            [self.item(0, first), self.item(1, depicted), self.item(2, first)], analyze, people)
        self.assertEqual(self.calls, ["visual_0", "visual_1"])
        self.assertEqual([output["analysis_id"] for output in outputs],
                         ["visual_0", "visual_1", "visual_0"])
        self.assertEqual(audit["decisions"][1]["reason"], "visible_content_changed")

    def test_live_component_cannot_mask_changes_outside_lecturer_box(self):
        first = np.full((64, 64, 3), 200, dtype=np.uint8)
        first[16:32, 16:32] = 30
        later = first.copy()
        later[20, 40, 1] += 4

        def analyze(item):
            return self.analyze(item, lecturer=[0.2, 0.2, 0.55, 0.55])

        def overbroad_mask(_):
            return np.ones((64, 64), dtype=np.float32)

        _, audit = group_visuals([self.item(0, first), self.item(1, later)],
                                 analyze, overbroad_mask)
        self.assertEqual(self.calls, ["visual_0", "visual_1"])
        self.assertEqual(audit["decisions"][1]["reason"], "visible_content_changed")

    def test_better_actual_representative_is_analyzed_before_rebinding_provenance(self):
        later = np.full((64, 64, 3), 200, dtype=np.uint8)
        later[20:40:2, 20:40:2] += 2
        items = [self.item(0), self.item(1, later)]

        def analyze(item):
            result = self.analyze(item)
            result["regions"] = [{"kind": "text", "bbox": [0.2, 0.2, 0.8, 0.8],
                                  "text": f"Parsed from {item['id']}"}]
            return result

        outputs, audit = group_visuals(items, analyze, None)
        self.assertEqual(self.calls, ["visual_0", "visual_1"])
        self.assertEqual(audit["analysis_calls"], 2)
        self.assertEqual(audit["reused_count"], 1)
        for position, output in enumerate(outputs):
            self.assertEqual(output["analysis_id"], "visual_1")
            self.assertEqual(output["image"], items[1]["image"])
            self.assertEqual(output["sha256"], items[1]["sha256"])
            self.assertEqual(output["original_image"], items[position]["image"])
            self.assertEqual(output["regions"][0]["text"], "Parsed from visual_1")

    def test_uncertain_better_representative_does_not_replace_valid_group_analysis(self):
        later = np.full((64, 64, 3), 200, dtype=np.uint8)
        later[20:40:2, 20:40:2] += 2
        items = [self.item(0), self.item(1, later), self.item(2)]

        def analyze(item):
            return self.analyze(item, issues=["Unreadable."] if item["id"] == "visual_1" else [])

        outputs, audit = group_visuals(items, analyze, None)
        self.assertEqual(self.calls, ["visual_0", "visual_1"])
        self.assertEqual([output["analysis_id"] for output in outputs],
                         ["visual_0", "visual_1", "visual_0"])
        self.assertEqual(outputs[0]["image"], items[0]["image"])
        self.assertEqual(outputs[1]["image"], items[1]["image"])
        self.assertEqual(outputs[1]["issues"], ["Unreadable."])
        self.assertEqual(audit["decisions"][1]["reason"], "representative_analysis_issues")
        self.assertEqual(audit["analysis_calls"], 2)
        self.assertEqual(audit["reused_count"], 1)

    def test_exact_pixels_can_reuse_even_if_person_masks_are_unavailable(self):
        outputs, audit = group_visuals([self.item(0), self.item(1)], self.live_analyze, None)
        self.assertEqual(self.calls, ["visual_0"])
        self.assertEqual(outputs[1]["analysis_id"], "visual_0")
        self.assertEqual(audit["decisions"][1]["reason"], "exact_pixels")

    def test_fullframe_geometry_changes_force_analysis(self):
        rgb = np.full((32, 64, 3), 200, dtype=np.uint8)
        _, audit = group_visuals([self.item(0), self.item(1, rgb)], self.analyze, None)
        self.assertEqual(self.calls, ["visual_0", "visual_1"])
        self.assertEqual(audit["decisions"][1]["reason"], "frame_geometry_changed")

    def test_exif_orientation_is_applied_before_comparing_pixels(self):
        rotated = np.full((40, 60, 3), 200, dtype=np.uint8)
        rotated[0:10, 0:20] = 30
        path = self.job / "rotated.png"
        exif = Image.Exif()
        exif[274] = 6
        Image.fromarray(rotated).save(path, exif=exif)
        first = {"id": "rotated", "image": str(path), "timestamp": 0}
        second = self.item(1, np.rot90(rotated, k=3))
        outputs, audit = group_visuals([first, second], self.analyze, None)
        self.assertEqual(self.calls, ["rotated"])
        self.assertEqual(outputs[1]["analysis_id"], "rotated")
        self.assertEqual(audit["decisions"][1]["reason"], "exact_pixels")


if __name__ == "__main__":
    unittest.main()
