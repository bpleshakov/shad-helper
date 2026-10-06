"""Whole-page selection must preserve final real pixels and review ambiguity."""

import json
import unittest

import numpy as np

from shad_helper.extraction import select_keyframes
from shad_helper.whiteboard_pages import POLICY, select_pages


class WhiteboardPageTests(unittest.TestCase):
    def setUp(self):
        self.classified, self.segmented, self.checked = [], [], []

    @staticmethod
    def board(columns=(15, 25, 35, 45, 55)):
        rgb = np.full((80, 100, 3), 240, dtype=np.uint8)
        for column in columns:
            rgb[20:50, column:column + 2] = [30, 50, 100]
        return rgb

    def classify(self, timestamp):
        self.classified.append(timestamp)
        return {"kind": "whiteboard", "issues": [], "regions": [],
                "lecture_surface_bbox": [0, 0, 1, 1], "lecturer_bbox": None}

    def segmenter(self, rgb):
        self.segmented.append(rgb.copy())
        return np.zeros(rgb.shape[:2], np.float32)

    def run_pages(self, states, *, classify=None, segmenter=None, full_frames=None, end=None):
        previews, frames, previous = [], {}, None
        for timestamp, rgb in states:
            change = 1.0 if previous is None or previous.shape != rgb.shape else np.count_nonzero(
                np.max(np.abs(rgb.astype(np.int16) - previous.astype(np.int16)), axis=2) > 12) / rgb.shape[0] / rgb.shape[1]
            previews.append({"timestamp": timestamp, "rgb": rgb, "change": change, "sharpness": 10})
            frames[timestamp] = rgb if full_frames is None else full_frames.get(timestamp, rgb)
            previous = rgb

        def read_frame(timestamp):
            self.checked.append(timestamp)
            return frames[timestamp]

        return select_pages(iter(previews), start=states[0][0], end=states[-1][0] if end is None else end,
                            classify=classify or self.classify,
                            segmenter=segmenter or self.segmenter, read_frame=read_frame), previews

    def test_additions_are_one_final_page_with_real_bounds(self):
        states = [(0, self.board((15,))), (2, self.board((15, 25))),
                  (4, self.board((15, 25, 35))), (5, self.board((15, 25, 35)))]
        (selections, audit), _ = self.run_pages(states, end=5.5)
        self.assertEqual([selection["timestamp"] for selection in selections], [5])
        self.assertEqual(selections[0]["page_start"], 0)
        self.assertEqual(selections[0]["page_end"], 5.5)
        self.assertEqual(self.classified, [0])
        self.assertEqual(self.checked, [5])
        self.assertEqual(audit["scenes"][0]["mode"], "pages")
        json.dumps(audit, allow_nan=False)

    def test_seventy_percent_major_reset_keeps_final_pre_wipe_view(self):
        full, remaining = self.board(), self.board((15,))
        states = [(0, full), (0.5, full), (0.9, full), (1, remaining),
                  (1.5, remaining), (2, remaining), (4, self.board((15, 65))),
                  (6, self.board((15, 65, 75)))]
        (selections, _), _ = self.run_pages(states)
        self.assertEqual([selection["timestamp"] for selection in selections], [0.9, 6])
        self.assertEqual(selections[0]["page_end"], 1)
        self.assertEqual(selections[1]["page_start"], 1)
        self.assertEqual(self.checked, [0.9, 6])

    def test_local_correction_discards_earlier_erased_material(self):
        old = self.board()
        corrected = self.board((15, 25, 35, 45))
        states = [(0, old), (2, corrected), (4, self.board((15, 25, 35, 45, 65))),
                  (6, self.board((15, 25, 35, 45, 65))), (8, self.board((15, 25, 35, 45, 65, 75)))]
        (selections, audit), _ = self.run_pages(states)
        self.assertEqual([selection["timestamp"] for selection in selections], [8])
        self.assertEqual(audit["scenes"][0]["mode"], "pages")

    def test_partial_erasure_at_clip_end_emits_current_state_only(self):
        states = [(0, self.board()), (2, self.board((15, 25, 35, 45))),
                  (4, self.board((15, 25, 35, 45)))]
        (selections, _), _ = self.run_pages(states)
        self.assertEqual([selection["timestamp"] for selection in selections], [4])
        self.assertEqual(self.checked, [4])

    def test_wipe_reference_survives_short_pauses(self):
        full, half, nearly_erased = self.board(), self.board((15, 25, 35)), self.board((15,))
        states = [(0, full), (1, full), (2, half), (10, half),
                  (15, nearly_erased), (16, nearly_erased), (18, self.board(()))]
        (selections, audit), _ = self.run_pages(states)
        self.assertEqual([selection["timestamp"] for selection in selections], [1])
        self.assertEqual(audit["scenes"][0]["mode"], "pages")

    def test_no_progress_for_thirty_seconds_falls_back_with_review_issue(self):
        states = [(0, self.board()), (2, self.board((15, 25, 35, 45))),
                  (20, self.board((15, 25, 35, 45))), (32, self.board((15, 25, 35, 45)))]
        (selections, audit), previews = self.run_pages(states)
        expected = select_keyframes([{key: sample[key] for key in ("timestamp", "change", "sharpness")}
                                     for sample in previews])
        self.assertEqual([selection["timestamp"] for selection in selections],
                         [selection["timestamp"] for selection in expected])
        self.assertTrue(all(selection["selection_issues"] for selection in selections))
        self.assertEqual(audit["scenes"][0]["reason"], "erasure_unresolved")

    def test_mixed_writing_and_erasing_is_reviewable_fallback(self):
        states = [(0, self.board()), (2, self.board((15, 25, 35, 45, 65))),
                  (4, self.board((15, 25, 35, 45, 65)))]
        (selections, audit), _ = self.run_pages(states)
        self.assertEqual(audit["scenes"][0]["reason"], "mixed_writing_and_erasing")
        self.assertTrue(all(selection["selection_issues"] for selection in selections))

    def test_faint_late_full_resolution_ink_uses_latest_actual_frame(self):
        preview = self.board((15,))
        final = np.repeat(np.repeat(preview, 2, axis=0), 2, axis=1)
        final[100, 130, 2] += 4
        states = [(0, preview), (0.5, preview), (0.9, preview)]
        (selections, _), _ = self.run_pages(states, full_frames={0.9: final})
        self.assertEqual([selection["timestamp"] for selection in selections], [0.9])
        self.assertEqual(self.checked, [0.9])

    def test_faint_late_ink_before_reset_does_not_choose_earlier_sharp_frame(self):
        preview = self.board()
        final = preview.copy()
        final[60, 70, 0] += 4
        states = [(0, preview), (0.5, preview), (0.9, preview),
                  (1, self.board(())), (2, self.board(()))]
        (selections, _), _ = self.run_pages(states, full_frames={0.9: final})
        self.assertEqual([selection["timestamp"] for selection in selections], [0.9])
        self.assertEqual(self.checked[0], 0.9)

    def test_faint_preview_pixel_before_fast_wipe_is_observed_before_throttle(self):
        base = self.board()
        faint = base.copy()
        faint[60, 70, 0] += 4
        states = [(0, base), (0.5, faint), (0.9, faint),
                  (1, self.board(())), (2, self.board(()))]
        (selections, _), _ = self.run_pages(states)
        self.assertEqual([selection["timestamp"] for selection in selections], [0.9])
        self.assertEqual(self.checked[0], 0.9)

    def test_short_complete_page_written_and_erased_inside_two_seconds_is_kept(self):
        blank, written = self.board(()), self.board((15,))
        states = [(0, blank), (0.2, written), (0.6, written), (0.9, written),
                  (1, blank), (2, blank)]
        (selections, audit), _ = self.run_pages(states)
        self.assertEqual([selection["timestamp"] for selection in selections], [0.9])
        self.assertEqual(audit["scenes"][0]["mode"], "pages")

    def test_blank_final_board_is_omitted(self):
        (selections, audit), _ = self.run_pages([(0, self.board(())), (2, self.board(()))])
        self.assertEqual(selections, [])
        self.assertEqual(audit["selected_count"], 0)

    def test_progressive_wipe_cleanup_does_not_emit_remaining_old_ink(self):
        full = self.board(tuple(range(10, 90, 8)))
        remaining = self.board((10, 18, 26))
        almost_empty = self.board((10,))
        for trailing_writing in (False, True):
            with self.subTest(trailing_writing=trailing_writing):
                states = [(0, full), (1, full), (2, remaining), (3, remaining),
                          (4, almost_empty), (6, almost_empty)]
                if trailing_writing:
                    states += [(8, self.board(())), (10, self.board((65,))),
                               (12, self.board((65, 75)))]
                (selections, audit), _ = self.run_pages(states)
                expected = [1, 12] if trailing_writing else [1]
                self.assertEqual([selection["timestamp"] for selection in selections], expected)
                self.assertEqual(audit["scenes"][0]["mode"], "pages")
                if trailing_writing:
                    self.assertEqual(selections[0]["page_end"], selections[1]["page_start"])

    def test_static_board_frame_does_not_prevent_reset_or_make_blank_page(self):
        def framed(rgb):
            rgb = rgb.copy()
            rgb[:2] = rgb[-2:] = 0
            rgb[:, :2] = rgb[:, -2:] = 0
            return rgb

        full, blank = framed(self.board()), framed(self.board(()))
        final_full_size = np.repeat(np.repeat(blank, 5, axis=0), 5, axis=1)
        (selections, audit), _ = self.run_pages(
            [(0, full), (1, full), (2, blank), (3, blank), (4, blank)],
            full_frames={4: final_full_size})
        self.assertEqual([selection["timestamp"] for selection in selections], [1])
        self.assertEqual(audit["scenes"][0]["mode"], "pages")

    def test_tiny_surface_cannot_be_eroded_into_a_confirmed_board(self):
        def classify(_):
            return {"kind": "whiteboard", "issues": [], "regions": [],
                    "lecture_surface_bbox": [0.49, 0.49, 0.51, 0.51]}

        (selections, audit), _ = self.run_pages([(0, self.board()), (2, self.board())], classify=classify)
        self.assertEqual(audit["scenes"][0]["reason"], "surface_too_small")
        self.assertTrue(all(selection["selection_issues"] for selection in selections))

    def test_layout_exit_closes_board_and_classifies_new_surface(self):
        slide = self.board((15,)) - 80

        def classify(timestamp):
            self.classified.append(timestamp)
            result = {"kind": "whiteboard" if timestamp < 2 else "slide", "issues": [],
                      "regions": [], "lecture_surface_bbox": [0, 0, 1, 1]}
            return result

        states = [(0, self.board()), (1, self.board()), (2, slide), (3, slide), (4, slide)]
        (selections, audit), _ = self.run_pages(states, classify=classify)
        self.assertEqual(self.classified, [0, 3])
        self.assertEqual([scene["mode"] for scene in audit["scenes"]], ["pages", "states"])
        self.assertEqual(selections[0]["timestamp"], 1)
        self.assertEqual(selections[0]["page_end"], 2)
        self.assertTrue(all("page_id" not in selection for selection in selections[1:]))

    def test_brief_slide_before_settled_board_keeps_transition_evidence(self):
        slide = np.full((80, 100, 3), 100, np.uint8)
        new_board = self.board((75,))
        states = [(0, self.board()), (0.5, self.board()), (1, slide), (1.2, slide),
                  (1.3, new_board), (2.1, new_board), (3, new_board)]
        (selections, audit), _ = self.run_pages(states)
        retained_slide = [selection for selection in selections if 1 <= selection["timestamp"] < 1.3]
        self.assertTrue(retained_slide)
        self.assertEqual(retained_slide[0]["timestamp"], 1)
        self.assertTrue(all("page_id" not in selection for selection in retained_slide))
        self.assertEqual([scene["mode"] for scene in audit["scenes"]], ["pages", "states", "pages"])
        self.assertEqual(selections[-1]["timestamp"], 3)

    def test_person_can_enter_after_bootstrap_with_no_lecturer(self):
        def people(rgb):
            mask = np.zeros(rgb.shape[:2], np.float32)
            if rgb[65, 10, 0] == 150:
                mask[60:75, 5:15] = 0.99
            return mask

        with_person = self.board()
        with_person[60:75, 5:15] = 150
        (selections, audit), _ = self.run_pages([(0, self.board()), (2, with_person)], segmenter=people)
        self.assertEqual(audit["scenes"][0]["reason"], "untracked_person")
        self.assertTrue(all(selection["selection_issues"] for selection in selections))

    def test_live_component_moves_beyond_initial_lecturer_box(self):
        def person_at(left):
            rgb = self.board()
            rgb[60:75, left:left + 8] = [150, 70, 40]
            return rgb

        def classify(timestamp):
            result = self.classify(timestamp)
            result["lecturer_bbox"] = [0.02, 0.65, 0.15, 0.98]
            return result

        def people(rgb):
            return np.where(np.all(rgb == [150, 70, 40], axis=2), 0.99, 0.0)

        (selections, audit), _ = self.run_pages([(0, person_at(4)), (2, person_at(16)),
                                                (4, person_at(24))], classify=classify, segmenter=people)
        self.assertEqual([selection["timestamp"] for selection in selections], [4])
        self.assertEqual(audit["scenes"][0]["mode"], "pages")

    def test_person_occlusion_never_confirms_false_reset(self):
        clear = self.board()
        covered = clear.copy()
        covered[10:60, 10:60] = 150

        def classify(timestamp):
            result = self.classify(timestamp)
            result["lecturer_bbox"] = [0.1, 0.1, 0.6, 0.8]
            return result

        def people(rgb):
            return np.where(np.all(rgb == 150, axis=2), 0.99, 0.0)

        (selections, audit), _ = self.run_pages([(0, clear), (2, covered), (4, covered), (6, clear)],
                                               classify=classify, segmenter=people)
        self.assertEqual([selection["timestamp"] for selection in selections], [6])
        self.assertEqual(audit["scenes"][0]["mode"], "pages")

    def test_occlusion_unresolved_for_thirty_seconds_falls_back(self):
        clear = self.board()
        covered = clear.copy()
        covered[10:60, 10:60] = 150

        def classify(timestamp):
            result = self.classify(timestamp)
            result["lecturer_bbox"] = [0.1, 0.1, 0.6, 0.8]
            return result

        def people(rgb):
            return np.where(np.all(rgb == 150, axis=2), 0.99, 0.0)

        (selections, audit), _ = self.run_pages([(0, clear), (2, covered), (32, covered)],
                                               classify=classify, segmenter=people)
        self.assertEqual(audit["scenes"][0]["reason"], "occlusion_unresolved")
        self.assertTrue(all(selection["selection_issues"] for selection in selections))

    def test_short_occlusion_at_clip_end_never_silently_uses_older_final_state(self):
        clear = self.board()
        covered = clear.copy()
        covered[10:60, 10:60] = 150

        def classify(timestamp):
            result = self.classify(timestamp)
            result["lecturer_bbox"] = [0.1, 0.1, 0.6, 0.8]
            return result

        def people(rgb):
            return np.where(np.all(rgb == 150, axis=2), 0.99, 0.0)

        (selections, audit), _ = self.run_pages([(0, clear), (1, covered)],
                                               classify=classify, segmenter=people)
        self.assertEqual(audit["scenes"][0]["reason"], "final_visibility_unresolved")
        self.assertTrue(all(selection["selection_issues"] for selection in selections))
        self.assertTrue(all("page_id" not in selection for selection in selections))

    def test_one_known_symbol_covered_at_end_cannot_be_hidden_by_ninety_percent_rule(self):
        clear = self.board()
        clear[60, 65] = [30, 50, 100]
        covered = clear.copy()
        covered[58:63, 63:68] = 150

        def classify(timestamp):
            result = self.classify(timestamp)
            result["lecturer_bbox"] = [0.55, 0.65, 0.75, 0.85]
            return result

        def people(rgb):
            return np.where(np.all(rgb == 150, axis=2), 0.99, 0.0)

        (selections, audit), _ = self.run_pages([(0, clear), (1, covered)],
                                               classify=classify, segmenter=people)
        self.assertEqual(audit["scenes"][0]["reason"], "final_content_occluded")
        self.assertTrue(all(selection["selection_issues"] for selection in selections))
        self.assertTrue(all("page_id" not in selection for selection in selections))

    def test_faint_known_point_covered_at_end_is_also_reviewable(self):
        clear = self.board()
        clear[60, 65, 2] += 4
        covered = clear.copy()
        covered[58:63, 63:68] = 150

        def classify(timestamp):
            result = self.classify(timestamp)
            result["lecturer_bbox"] = [0.55, 0.65, 0.75, 0.85]
            return result

        def people(rgb):
            return np.where(np.all(rgb == 150, axis=2), 0.99, 0.0)

        (selections, audit), _ = self.run_pages([(0, clear), (1, covered)],
                                               classify=classify, segmenter=people)
        self.assertEqual(audit["scenes"][0]["reason"], "final_content_occluded")
        self.assertTrue(all(selection["selection_issues"] for selection in selections))

    def test_unconfirmed_extra_person_components_are_ambiguous(self):
        rgb = self.board()
        rgb[60:75, 5:15] = 150
        rgb[60:75, 50:60] = 150

        def classify(timestamp):
            result = self.classify(timestamp)
            result["lecturer_bbox"] = [0.02, 0.65, 0.15, 0.98]
            return result

        def people(frame):
            return np.where(np.all(frame == 150, axis=2), 0.99, 0.0)

        (selections, audit), _ = self.run_pages([(0, rgb), (2, rgb)], classify=classify, segmenter=people)
        self.assertEqual(audit["scenes"][0]["reason"], "ambiguous_people")
        self.assertTrue(all(selection["selection_issues"] for selection in selections))

    def test_pre_wipe_full_resolution_check_does_not_segment_backwards(self):
        seen = []
        states = []
        for timestamp, rgb in [(0, self.board()), (1, self.board()),
                               (2, self.board(())), (3, self.board(())), (4, self.board((75,)))]:
            rgb[0, 0, 0] = int(timestamp)
            states.append((timestamp, rgb))

        def segmenter(rgb):
            seen.append(int(rgb[0, 0, 0]))
            return np.zeros(rgb.shape[:2], np.float32)

        self.run_pages(states, segmenter=segmenter)
        self.assertEqual(seen, sorted(seen))

    def test_new_visible_ink_outside_tight_surface_causes_fallback(self):
        later = self.board((15,))
        later[60, 80, 2] += 4

        def classify(timestamp):
            result = self.classify(timestamp)
            result["lecture_surface_bbox"] = [0, 0, 0.5, 1]
            return result

        (selections, audit), _ = self.run_pages([(0, self.board((15,))), (2, later)], classify=classify)
        self.assertEqual(audit["scenes"][0]["reason"], "outside_surface_changed")
        self.assertTrue(all(selection["selection_issues"] for selection in selections))

    def test_slide_and_notebook_selection_matches_existing_selector(self):
        for kind in ("slide", "notebook"):
            with self.subTest(kind=kind):
                def classify(_, kind=kind):
                    return {"kind": kind, "issues": [], "regions": [], "lecture_surface_bbox": [0, 0, 1, 1]}

                states = [(0, self.board()), (1, self.board()), (2, self.board((15,))),
                          (3, self.board((15,))), (4, self.board((15,)))]
                (selections, audit), previews = self.run_pages(states, classify=classify)
                expected = select_keyframes([{key: sample[key] for key in ("timestamp", "change", "sharpness")}
                                             for sample in previews])
                self.assertEqual([selection["timestamp"] for selection in selections],
                                 [selection["timestamp"] for selection in expected])
                self.assertTrue(all(not selection["selection_issues"] for selection in selections))
                self.assertEqual(audit["segmentation_calls"], 0)

    def test_invalid_or_text_only_roi_never_uses_page_selection(self):
        for box in (None, [0.4, 0.4, 0.6, 0.6]):
            with self.subTest(box=box):
                def classify(_, box=box):
                    return {"kind": "whiteboard", "issues": [], "lecture_surface_bbox": box,
                            "regions": [{"bbox": [0.1, 0.1, 0.9, 0.9]}]}

                (selections, audit), _ = self.run_pages([(0, self.board()), (2, self.board())],
                                                      classify=classify)
                self.assertEqual(audit["scenes"][0]["mode"], "states")
                self.assertTrue(all(selection["selection_issues"] for selection in selections))

    def test_unavailable_segmentation_is_reviewable_fallback(self):
        (selections, audit), _ = self.run_pages([(0, self.board()), (2, self.board())],
                                               segmenter=lambda _: None)
        self.assertEqual(audit["scenes"][0]["reason"], "segmentation_unavailable")
        self.assertTrue(all(selection["selection_issues"] for selection in selections))

    def test_policy_is_explicit_and_audit_serializable(self):
        (selections, audit), _ = self.run_pages([(0, self.board()), (2, self.board())])
        self.assertEqual(audit["policy"], POLICY)
        self.assertEqual(POLICY["reset_loss"], 0.7)
        self.assertEqual(POLICY["reference_visibility"], 0.9)
        self.assertEqual(POLICY["ambiguity_seconds"], 30)
        self.assertEqual(len(selections), 1)
        json.dumps(audit, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
