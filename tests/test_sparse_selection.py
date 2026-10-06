"""Sparse selection preserves completed evidence without dense fallback."""

import json
import unittest

import numpy as np

from shad_helper.sparse_selection import select_sparse


class SparseSelectionTests(unittest.TestCase):
    @staticmethod
    def board(columns=(15, 25, 35, 45, 55), *, background=240):
        rgb = np.full((80, 100, 3), background, np.uint8)
        for column in columns:
            rgb[20:50, column:column + 2] = [30, 50, 100]
        return rgb

    def run_sparse(self, states, *, route=None, segmenter=None, end=None, full_frames=None,
                   native_available=True):
        frames = dict(states)
        if full_frames:
            frames.update(full_frames)
        routed, masks, checked = [], [], []

        def routing(timestamp):
            routed.append(timestamp)
            return (route(timestamp) if route else
                    {"kind": "whiteboard", "lecture_surface_bbox": [0, 0, 1, 1],
                     "lecturer_bbox": None})

        def mask(rgb):
            masks.append(rgb.copy())
            return segmenter(rgb) if segmenter else np.zeros(rgb.shape[:2], np.float32)

        def read(timestamp):
            checked.append(timestamp)
            return frames[timestamp]

        previews = [{"timestamp": timestamp, "rgb": rgb, "change": 0.0, "sharpness": 10}
                    for timestamp, rgb in states]
        selected, audit = select_sparse(iter(previews), start=states[0][0],
                                        end=states[-1][0] + 0.5 if end is None else end,
                                        route=routing, segmenter=mask if native_available else None,
                                        read_frame=read)
        json.dumps(audit, allow_nan=False)
        return selected, audit, routed, masks, checked

    def test_hundreds_of_writing_pauses_emit_one_latest_real_view(self):
        states = [(i * 0.5, self.board(tuple(range(15, 15 + min(i // 30 + 1, 7) * 10, 10))))
                  for i in range(400)]
        selected, audit, routed, masks, checked = self.run_sparse(states)
        self.assertEqual([x["timestamp"] for x in selected], [199.5])
        self.assertEqual(selected[0]["selection_role"], "primary")
        self.assertEqual(routed, [0])
        self.assertEqual(masks, [])
        self.assertEqual(checked, [199.5])
        self.assertEqual(audit["preview_count"], 400)

    def test_blank_intro_is_not_routed_or_exported(self):
        states = [(0, self.board(())), (1, self.board(())), (2, self.board()),
                  (3, self.board())]
        selected, _, routed, _, _ = self.run_sparse(states)
        self.assertEqual(routed, [2])
        self.assertEqual([x["timestamp"] for x in selected], [3])
        self.assertEqual(selected[0]["page_start"], 0)

    def test_confirmed_blank_video_has_no_calls_or_frames(self):
        selected, audit, routed, masks, checked = self.run_sparse(
            [(0, self.board(())), (5, self.board(()))])
        self.assertEqual((selected, routed, masks, checked), ([], [], [], []))
        self.assertEqual(audit["selected_count"], 0)

    def test_title_to_board_reclassifies_against_scene_anchor(self):
        title = self.board((15, 25), background=30)
        route = lambda timestamp: {"kind": "slide" if timestamp < 2 else "whiteboard",
                                   "lecture_surface_bbox": [0, 0, 1, 1], "lecturer_bbox": None}
        selected, _, routed, _, _ = self.run_sparse(
            [(0, title), (0.5, title), (2, self.board()), (2.5, self.board()),
             (4, self.board((15, 25, 35, 45, 55, 65)))], route=route)
        self.assertEqual(routed, [0, 2.5])
        self.assertEqual([x["timestamp"] for x in selected], [0.5, 4])
        self.assertEqual(selected[0]["page_end"], 2)

    def test_unknown_layout_retries_at_most_three_and_stays_sparse(self):
        states = [(i, self.board()) for i in range(61)]
        selected, _, routed, _, _ = self.run_sparse(states, route=lambda _: None)
        self.assertEqual(routed, [0, 5, 10])
        self.assertEqual([x["timestamp"] for x in selected], [60])
        self.assertTrue(selected[0]["selection_issues"])

    def test_route_budget_error_is_not_swallowed(self):
        class DeadlineError(RuntimeError):
            pass

        def route(_):
            raise DeadlineError("stop model worker")

        with self.assertRaises(DeadlineError):
            self.run_sparse([(0, self.board())], route=route)

    def test_unknown_retry_can_activate_board_without_layout_change(self):
        def route(timestamp):
            return {"kind": "other" if timestamp < 5 else "whiteboard",
                    "lecture_surface_bbox": [0, 0, 1, 1], "lecturer_bbox": None}

        selected, _, routed, _, _ = self.run_sparse([(i, self.board()) for i in range(11)],
                                                   route=route)
        self.assertEqual(routed, [0, 5])
        self.assertEqual([x["timestamp"] for x in selected], [10])
        self.assertEqual(selected[0]["selection_issues"], [])

    def test_board_frame_is_not_ink_and_does_not_hide_major_wipe(self):
        def framed(columns):
            rgb = self.board(columns)
            rgb[:2] = rgb[-2:] = 10
            rgb[:, :2] = rgb[:, -2:] = 10
            return rgb

        selected, _, _, _, _ = self.run_sparse(
            [(0, framed((15, 25))), (1, framed((15, 25))), (2, framed(())),
             (3, framed(())), (4, framed((65, 75))), (5, framed((65, 75)))])
        self.assertEqual([x["timestamp"] for x in selected], [1, 5])
        selected, _, _, _, _ = self.run_sparse([(0, framed(())), (1, framed(()))])
        self.assertEqual(selected, [])

    def test_partial_correction_does_not_export_obsolete_equation(self):
        old, corrected = self.board(), self.board((15, 25, 35, 45, 65))
        selected, _, _, _, _ = self.run_sparse(
            [(0, old), (1, old), (2, self.board((15, 25, 35, 45))),
             (3, corrected), (4, corrected)])
        self.assertEqual([x["timestamp"] for x in selected], [4])

    def test_partial_erasure_at_clip_end_keeps_current_state(self):
        selected, _, _, _, _ = self.run_sparse(
            [(0, self.board()), (2, self.board((15, 25, 35, 45))),
             (3, self.board((15, 25, 35, 45)))])
        self.assertEqual([x["timestamp"] for x in selected], [3])

    def test_slow_wipe_freezes_complete_pre_erasure_state(self):
        selected, audit, _, _, _ = self.run_sparse(
            [(0, self.board()), (1, self.board()), (2, self.board((15, 25, 35))),
             (10, self.board((15, 25, 35))), (15, self.board((15,))),
             (16, self.board((15,))), (20, self.board(()))])
        self.assertEqual([x["timestamp"] for x in selected], [1])
        self.assertEqual(selected[0]["page_end"], 2)
        self.assertEqual(audit["pages"][0]["reason"], "major_reset")

    def test_reset_cleanup_does_not_create_blank_or_fragment_page(self):
        selected, _, _, _, _ = self.run_sparse(
            [(0, self.board()), (1, self.board()), (2, self.board((15,))),
             (3, self.board((15,))), (4, self.board(())),
             (5, self.board((65,))), (6, self.board((65, 75))),
             (7, self.board((65, 75)))])
        self.assertEqual([x["timestamp"] for x in selected], [1, 7])
        self.assertEqual(selected[1]["page_start"], 2)

    def test_compression_noise_does_not_emit_duplicate_states(self):
        rng = np.random.default_rng(9)
        base = self.board()
        states = [(i * 0.5, np.clip(base.astype(np.int16)
                                   + rng.integers(-3, 4, base.shape), 0, 255).astype(np.uint8))
                  for i in range(80)]
        selected, _, routed, _, _ = self.run_sparse(states)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["timestamp"], 39.5)
        self.assertEqual(len(routed), 1)

    def test_brief_distinct_slide_is_not_lost_between_layouts(self):
        slide, brief = self.board((15, 25)), self.board((65, 75), background=30)
        route = lambda _: {"kind": "slide", "lecture_surface_bbox": [0, 0, 1, 1]}
        selected, audit, _, _, _ = self.run_sparse(
            [(0, slide), (0.5, slide), (1, brief), (1.5, slide), (2, slide)], route=route)
        self.assertIn(1, [x["timestamp"] for x in selected])
        self.assertTrue(any(page["reason"] == "brief_layout" for page in audit["pages"]))

    def test_brief_text_slide_on_same_white_background_is_retained(self):
        slide, brief = self.board((15, 25)), self.board((65, 75))
        route = lambda _: {"kind": "slide", "lecture_surface_bbox": [0, 0, 1, 1]}
        selected, _, _, _, _ = self.run_sparse(
            [(0, slide), (0.5, slide), (1, brief), (1.5, slide), (2, slide)], route=route)
        self.assertEqual([x["timestamp"] for x in selected], [0.5, 1, 2])

    def test_notebook_scroll_emits_settled_state_instead_of_every_update(self):
        route = lambda _: {"kind": "notebook", "lecture_surface_bbox": [0, 0, 1, 1]}
        states = [(0, self.board((15,))), (1, self.board((15,)))]
        states.extend((i * 0.5, self.board((25 + i * 2,))) for i in range(4, 20))
        states.extend([(10, self.board((63,))), (10.5, self.board((63,))),
                       (11, self.board((63,)))])
        selected, _, _, _, _ = self.run_sparse(states, route=route)
        self.assertEqual([x["timestamp"] for x in selected], [1, 11])

    def test_unknown_interval_never_reintroduces_sealed_pages(self):
        selected, _, _, _, _ = self.run_sparse(
            [(0, self.board()), (1, self.board()), (2, self.board(())),
             (3, self.board(())), *[(i, self.board((65, 75))) for i in range(4, 100)]],
            route=lambda _: {"kind": "other", "lecture_surface_bbox": [0, 0, 1, 1]})
        self.assertEqual([x["timestamp"] for x in selected], [1, 99])
        self.assertTrue(all(x["selection_issues"] for x in selected))

    def test_complementary_real_views_expose_same_final_ink(self):
        base = self.board((15, 75))
        left, right = base.copy(), base.copy()
        left[:, :40] = [120, 10, 10]
        right[:, 60:] = [120, 10, 10]

        def segmenter(rgb):
            return ((rgb[:, :, 0] == 120) & (rgb[:, :, 1] == 10)).astype(np.float32)

        route = lambda _: {"kind": "whiteboard", "lecture_surface_bbox": [0, 0, 1, 1],
                           "lecturer_bbox": [0, 0, 1, 1]}
        selected, audit, _, masks, _ = self.run_sparse(
            [(0, left), (1, right)], route=route, segmenter=segmenter)
        self.assertLessEqual(len(selected), 3)
        self.assertEqual(len(selected), 2)
        self.assertEqual({x["timestamp"] for x in selected}, {0, 1})
        self.assertEqual([x["selection_role"] for x in selected].count("primary"), 1)
        self.assertEqual([x["selection_role"] for x in selected].count("complementary"), 1)
        self.assertEqual(len(masks), 2)
        self.assertEqual(audit["full_resolution_checks"], 2)

    def test_failed_person_mask_keeps_bounded_review_evidence(self):
        route = lambda _: {"kind": "whiteboard", "lecture_surface_bbox": [0, 0, 1, 1],
                           "lecturer_bbox": [0.1, 0.1, 0.3, 0.8]}
        selected, _, _, masks, _ = self.run_sparse([(i, self.board()) for i in range(30)],
                                                 route=route, segmenter=lambda _: None)
        self.assertEqual(len(selected), 1)
        self.assertLessEqual(len(masks), 3)
        self.assertTrue(selected[0]["selection_issues"])
        self.assertEqual(len(masks), 1)

    def test_native_preview_cadence_does_not_call_masks_per_frame(self):
        route = lambda _: {"kind": "whiteboard", "lecture_surface_bbox": [0, 0, 1, 1],
                           "lecturer_bbox": [0.1, 0.1, 0.3, 0.8]}
        selected, _, _, masks, _ = self.run_sparse([(i / 25, self.board()) for i in range(100)],
                                                 route=route)
        self.assertEqual(len(selected), 1)
        self.assertLessEqual(len(masks), 4)

    def test_actor_crossing_far_from_initial_box_does_not_split_layout(self):
        states = []
        for index in range(24):
            rgb = self.board()
            column = 5 + (index % 10) * 8
            rgb[58:80, column:column + 15] = [120, 10, 10]
            states.append((index * 0.5, rgb))

        def mask(rgb):
            return ((rgb[:, :, 0] == 120) & (rgb[:, :, 1] == 10)).astype(np.float32)

        route = lambda _: {"kind": "whiteboard", "lecture_surface_bbox": [0, 0, 1, 0.7],
                           "lecturer_bbox": [0.05, 0.7, 0.2, 1]}
        selected, _, routed, _, _ = self.run_sparse(states, route=route, segmenter=mask)
        self.assertEqual(len(routed), 1)
        self.assertEqual(len(selected), 1)

    def test_physical_board_exit_changes_three_edges_and_reclassifies(self):
        board, slide = self.board(), self.board((65, 75), background=30)
        route = lambda timestamp: {"kind": "whiteboard" if timestamp < 2 else "slide",
                                   "lecture_surface_bbox": [0, 0, 1, 1], "lecturer_bbox": None}
        selected, _, routed, _, _ = self.run_sparse(
            [(0, board), (1, board), (2, slide), (2.5, slide), (3, slide)], route=route)
        self.assertEqual(routed, [0, 2.5])
        self.assertEqual([x["timestamp"] for x in selected], [1, 3])

    def test_raised_mask_error_is_a_local_review_issue(self):
        def failure(_):
            raise RuntimeError("native model failed")

        route = lambda _: {"kind": "whiteboard", "lecture_surface_bbox": [0, 0, 1, 1],
                           "lecturer_bbox": [0.1, 0.1, 0.3, 0.8]}
        selected, _, _, _, _ = self.run_sparse([(0, self.board()), (1, self.board())],
                                              route=route, segmenter=failure)
        self.assertEqual(len(selected), 1)
        self.assertTrue(selected[0]["selection_issues"])

    def test_full_resolution_latest_capture_preserves_faint_late_symbol(self):
        base = self.board()
        final = np.repeat(np.repeat(base, 2, axis=0), 2, axis=1)
        final[100, 150, 0] += 4
        selected, _, _, _, checked = self.run_sparse([(0, base), (1, base), (2, base)],
                                                    full_frames={2: final})
        self.assertEqual([x["timestamp"] for x in selected], [2])
        self.assertEqual(checked, [2])

    def test_tiny_visible_correction_rejects_older_complement(self):
        old, final = self.board((15, 75)), self.board((15, 75))
        old[55, 50, 0] -= 4
        final[:, :40] = [120, 10, 10]

        def mask(rgb):
            return ((rgb[:, :, 0] == 120) & (rgb[:, :, 1] == 10)).astype(np.float32)

        route = lambda _: {"kind": "whiteboard", "lecture_surface_bbox": [0, 0, 1, 1],
                           "lecturer_bbox": [0, 0, 0.4, 1]}
        selected, _, _, _, _ = self.run_sparse([(0, old), (1, final)], route=route,
                                              segmenter=mask)
        self.assertEqual([x["timestamp"] for x in selected], [1])

    def test_long_body_motion_has_bounded_native_calls(self):
        states = []
        for i in range(1200):
            rgb = self.board((15, 25, 35, 45, 55))
            column = 5 + (i % 20) * 3
            rgb[58:80, column:column + 20] = [120, 10, 10]
            states.append((i * 0.5, rgb))

        def mask(rgb):
            return ((rgb[:, :, 0] == 120) & (rgb[:, :, 1] == 10)).astype(np.float32)

        route = lambda _: {"kind": "whiteboard", "lecture_surface_bbox": [0, 0, 1, 1],
                           "lecturer_bbox": [0.05, 0.7, 0.25, 1]}
        selected, _, _, masks, _ = self.run_sparse(states, route=route, segmenter=mask)
        self.assertEqual(len(selected), 1)
        self.assertLessEqual(len(masks), 12)

    def test_remaining_old_ink_after_reset_is_not_a_new_fragment_page(self):
        selected, _, _, _, _ = self.run_sparse(
            [(0, self.board()), (1, self.board()), (2, self.board((15,))),
             (3, self.board((15,))), (4, self.board((15,)))])
        self.assertEqual([x["timestamp"] for x in selected], [1])

    def test_prolonged_terminal_occlusion_keeps_earlier_real_uncertain_evidence(self):
        base = self.board((15, 25))
        covered = base.copy()
        covered[:, :50] = [120, 10, 10]
        route = lambda _: {"kind": "whiteboard", "lecture_surface_bbox": [0.1, 0.1, 0.9, 0.9],
                           "lecturer_bbox": [0, 0, 0.5, 1]}

        def mask(rgb):
            return ((rgb[:, :, 0] == 120) & (rgb[:, :, 1] == 10)).astype(np.float32)

        selected, audit, _, _, _ = self.run_sparse(
            [(0, base), (1, base), *[(i, covered) for i in range(2, 7)]],
            route=route, segmenter=mask)
        self.assertEqual([x["timestamp"] for x in selected], [1])
        self.assertTrue(any("terminal writing state is unobservable" in issue
                            for issue in selected[0]["selection_issues"]))
        self.assertEqual(audit["pages"][0]["reason"], "terminal_occlusion_uncertain")

    def test_native_failure_retains_latest_corrected_frame_after_occlusion(self):
        old, corrected = self.board((15, 25)), self.board((15, 35))
        covered = corrected.copy()
        covered[:, :50] = [120, 10, 10]
        route = lambda _: {"kind": "whiteboard", "lecture_surface_bbox": [0.1, 0.1, 0.9, 0.9],
                           "lecturer_bbox": [0, 0, 0.5, 1]}
        selected, _, _, masks, _ = self.run_sparse(
            [(0, old), (1, corrected), *[(i, covered) for i in range(2, 7)]],
            route=route, segmenter=lambda _: None)
        self.assertEqual([x["timestamp"] for x in selected], [1])
        self.assertEqual(len(masks), 1)
        self.assertTrue(selected[0]["selection_issues"])

    def test_absent_native_mask_keeps_corrected_evidence_without_fake_calls(self):
        old, corrected = self.board((15, 25)), self.board((15, 35))
        covered = corrected.copy()
        covered[:, :50] = [120, 10, 10]
        route = lambda _: {"kind": "whiteboard", "lecture_surface_bbox": [0.1, 0.1, 0.9, 0.9],
                           "lecturer_bbox": [0, 0, 0.5, 1]}
        selected, audit, _, masks, _ = self.run_sparse(
            [(0, old), (1, corrected), *[(i, covered) for i in range(2, 7)]],
            route=route, native_available=False)
        self.assertEqual([x["timestamp"] for x in selected], [1])
        self.assertEqual((masks, audit["segmentation_calls"]), ([], 0))
        self.assertTrue(selected[0]["selection_issues"])

    def test_visible_correction_precedes_prolonged_masked_occlusion(self):
        old, corrected = self.board((15, 25)), self.board((15, 35))
        covered = corrected.copy()
        covered[:, :50] = [120, 10, 10]
        route = lambda _: {"kind": "whiteboard", "lecture_surface_bbox": [0.1, 0.1, 0.9, 0.9],
                           "lecturer_bbox": [0, 0, 0.5, 1]}

        def mask(rgb):
            return ((rgb[:, :, 0] == 120) & (rgb[:, :, 1] == 10)).astype(np.float32)

        selected, _, _, _, _ = self.run_sparse(
            [(0, old), (1, corrected), *[(i, covered) for i in range(2, 7)]],
            route=route, segmenter=mask)
        self.assertEqual([x["timestamp"] for x in selected], [1])

    def test_confirmed_blank_reset_clears_older_uncertainty_candidate(self):
        board, blank = self.board((15, 25)), self.board(())
        route = lambda _: {"kind": "whiteboard", "lecture_surface_bbox": [0.1, 0.1, 0.9, 0.9],
                           "lecturer_bbox": [0, 0, 0.5, 1]}
        selected, audit, _, _, _ = self.run_sparse(
            [(0, board), (1, board), (2, blank), (3, blank), (4, blank), (5, blank)], route=route)
        self.assertEqual([x["timestamp"] for x in selected], [1])
        self.assertEqual(audit["pages"][0]["reason"], "major_reset")

    def test_late_entering_lecturer_does_not_become_a_false_blank_reset(self):
        old, corrected = self.board((15, 25)), self.board((15, 35))
        covered = corrected.copy()
        covered[:, :50] = [120, 10, 10]
        route = lambda _: {"kind": "whiteboard", "lecture_surface_bbox": [0.1, 0.1, 0.9, 0.9],
                           "lecturer_bbox": None}

        def mask(rgb):
            return ((rgb[:, :, 0] == 120) & (rgb[:, :, 1] == 10)).astype(np.float32)

        selected, audit, _, masks, _ = self.run_sparse(
            [(0, old), (0.5, old), (1, corrected), (1.5, corrected),
             *[(i * 0.5, covered) for i in range(4, 13)]], route=route, segmenter=mask)
        self.assertEqual([x["timestamp"] for x in selected], [1.5])
        self.assertEqual(selected[0]["page_end"], 6.5)
        self.assertEqual(audit["pages"][0]["reason"], "foreground_boundary_uncertain")
        self.assertTrue(selected[0]["selection_issues"])
        self.assertGreater(len(masks), 0)

    def test_person_free_unchanged_board_uses_no_native_masks(self):
        selected, audit, _, masks, _ = self.run_sparse([(i * 0.5, self.board()) for i in range(100)])
        self.assertEqual(len(selected), 1)
        self.assertEqual((masks, audit["segmentation_calls"]), ([], 0))

    def test_unavailable_late_entry_masks_preserve_pre_loss_to_clip_end(self):
        base = self.board((15, 25))
        covered = base.copy()
        covered[:, :50] = [120, 10, 10]
        route = lambda _: {"kind": "whiteboard", "lecture_surface_bbox": [0.1, 0.1, 0.9, 0.9],
                           "lecturer_bbox": None}
        states = [(0, base), (1, base), *[(i * 0.5, covered) for i in range(4, 13)]]
        for native_available in (True, False):
            with self.subTest(native_available=native_available):
                selected, audit, _, masks, _ = self.run_sparse(
                    states, route=route, segmenter=lambda _: None, native_available=native_available)
                self.assertEqual([x["timestamp"] for x in selected], [1])
                self.assertEqual(selected[0]["page_end"], 6.5)
                self.assertEqual(len(masks), int(native_available))
                self.assertTrue(selected[0]["selection_issues"])
                self.assertEqual(audit["pages"][0]["reason"], "foreground_boundary_uncertain")

    def test_true_person_free_major_erasure_is_confirmed_by_empty_native_mask(self):
        base, blank = self.board((15, 25)), self.board(())
        selected, audit, _, masks, _ = self.run_sparse(
            [(0, base), (1, base), (2, blank), (2.5, blank), (3, blank)])
        self.assertEqual([x["timestamp"] for x in selected], [1])
        self.assertEqual(audit["pages"][0]["reason"], "major_reset")
        self.assertTrue(masks)

    def test_native_failed_boundary_older_evidence_is_uncertain_not_complementary(self):
        old, new, blank = self.board((15, 25)), self.board((65, 75)), self.board(())
        new[:, :50] = [120, 10, 10]
        selected, audit, _, _, _ = self.run_sparse(
            [(0, old), (1, old), (2, blank), (2.5, blank), (3, new), (4, new)],
            segmenter=lambda _: None)
        self.assertEqual([x["timestamp"] for x in selected], [1, 4])
        self.assertEqual([x["selection_role"] for x in selected], ["uncertain", "primary"])
        self.assertNotIn("complement_of", selected[0])
        self.assertEqual(audit["pages"][0]["end"], 4.5)
        self.assertTrue(all(x["selection_issues"] for x in selected))

    def test_native_failure_keeps_later_distinct_pages_after_local_uncertainty(self):
        first, second, third = self.board((15, 25)), self.board((65, 75)), self.board((35, 45))
        blank = self.board(())
        selected, audit, _, _, _ = self.run_sparse(
            [(0, first), (1, first), (2, blank), (2.5, blank),
             (3, second), (3.5, second), (4, second), (5, blank), (5.5, blank),
             (6, third), (6.5, third), (7, third)], segmenter=lambda _: None)
        self.assertEqual([x["timestamp"] for x in selected], [1, 4, 7])
        self.assertEqual([page["end"] for page in audit["pages"]], [3, 6, 7.5])
        self.assertTrue(all(x["selection_issues"] for x in selected))
        self.assertEqual([x["selection_role"] for x in selected], ["primary"] * 3)

    def test_late_foreground_exit_with_readable_correction_discards_obsolete_frame(self):
        old, corrected = self.board((15, 25)), self.board((15, 35))
        covered = old.copy()
        covered[:, :50] = [120, 10, 10]

        def mask(rgb):
            return ((rgb[:, :, 0] == 120) & (rgb[:, :, 1] == 10)).astype(np.float32)

        selected, audit, _, _, _ = self.run_sparse(
            [(0, old), (1, old), (2, covered), (2.5, covered), (3, covered),
             (4, corrected), (4.5, corrected), (5, corrected)], segmenter=mask)
        self.assertEqual([x["timestamp"] for x in selected], [5])
        self.assertEqual(selected[0]["selection_role"], "primary")
        self.assertEqual(selected[0]["selection_issues"], [])
        self.assertEqual(audit["pages"][0]["reason"], "clip_end")

    def test_nonchronological_previews_are_rejected(self):
        with self.assertRaises(ValueError):
            self.run_sparse([(1, self.board()), (0, self.board())], end=2)
