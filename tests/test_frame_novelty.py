from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from shad_helper import frame_novelty
from shad_helper.frame_novelty import FrameNoveltyFilter, _PersonSegmenter, filter_frames


def board():
    image = np.full((240, 400, 3), 240, np.uint8)
    for row in (35, 80, 125, 170):
        cv2.putText(image, "x + y = 2", (25, row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,) * 3, 2)
    return image


def fake_person(image):
    return np.all(image == 80, axis=2)


def occlude(image, left, right):
    image = image.copy()
    image[50:, left:right] = 80
    return image


def test_moving_lecturer_over_previously_seen_board_is_redundant():
    filterer = FrameNoveltyFilter(fake_person)
    clear = board()
    assert filterer.consider(clear, "initial")[0]
    for left in (40, 140, 220, 80):
        keep, decision = filterer.consider(occlude(clear, left, left + 60), "before_erasure")
        assert not keep
        assert decision["reason"] == "unchanged_content"


def test_newly_revealed_writing_is_kept_and_then_reused():
    filterer = FrameNoveltyFilter(fake_person)
    clear = board()
    assert filterer.consider(occlude(clear, 20, 180), "initial")[0]
    keep, decision = filterer.consider(occlude(clear, 230, 290), "before_erasure")
    assert keep
    assert decision["newly_visible_fraction"] > 0
    assert not filterer.consider(occlude(clear, 240, 300), "change_time_guard")[0]


def test_one_clear_representative_is_kept_after_partial_views():
    filterer = FrameNoveltyFilter(fake_person)
    clear = board()
    assert filterer.consider(occlude(clear, 20, 110), "initial")[0]
    assert filterer.consider(occlude(clear, 240, 330), "before_erasure")[0]
    keep, decision = filterer.consider(clear, "final_change")
    assert keep
    assert decision["reason"] == "clearer_view"
    assert not filterer.consider(clear, "final_change")[0]


def test_short_lived_label_its_erasure_and_revisit_are_kept():
    filterer = FrameNoveltyFilter(fake_person)
    blank = board()
    written = blank.copy()
    written[195:198, 250:280] = 0
    for image, reason in ((blank, "initial"), (written, "before_erasure"),
                          (blank, "final_change"), (written, "before_transition")):
        assert filterer.consider(image, reason)[0]


def test_small_local_annotation_survives_even_on_a_large_board(monkeypatch):
    monkeypatch.setattr(cv2, "findTransformECC", Mock(return_value=(1.0, np.eye(2, 3, dtype=np.float32))))
    image = np.full((405, 720, 3), 240, np.uint8)
    cv2.putText(image, "Median of fives", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,) * 3, 2)
    for x in range(340, 470, 10):
        cv2.line(image, (x, 80), (x, 120), (100,) * 3)
    for y in range(80, 121, 10):
        cv2.line(image, (340, y), (460, y), (100,) * 3)
    annotated = image.copy()
    cv2.ellipse(annotated, (390, 100), (6, 3), 0, 0, 360, (130, 40, 40), 1)
    filterer = FrameNoveltyFilter(fake_person)
    assert filterer.consider(image, "initial")[0]
    keep, decision = filterer.consider(annotated, "before_erasure")
    assert decision["changed_fraction"] < frame_novelty.NOVELTY_SETTINGS["minimum_change_fraction"]
    assert keep
    assert filterer.consider(image, "final_change")[0]


@pytest.mark.parametrize("box", [(195, 198, 250, 280), (185, 225, 240, 340)])
def test_low_contrast_writing_and_filled_panels_are_not_discarded(box):
    filterer = FrameNoveltyFilter(fake_person)
    image = board()
    assert filterer.consider(image, "initial")[0]
    top, bottom, left, right = box
    changed = image.copy()
    changed[top:bottom, left:right] = 210
    assert filterer.consider(changed, "before_erasure")[0]
    assert filterer.consider(image, "final_change")[0]


def test_unknown_blank_area_does_not_count_as_new_writing():
    filterer = FrameNoveltyFilter(fake_person)
    clear = board()
    assert filterer.consider(occlude(clear, 280, 340), "initial")[0]
    keep, decision = filterer.consider(occlude(clear, 310, 370), "before_erasure")
    assert not keep
    assert decision["newly_visible_fraction"] == 0


def test_scrolling_and_uniform_slide_transitions_are_kept():
    filterer = FrameNoveltyFilter(fake_person)
    for shade in (255, 210, 100, 255):
        assert filterer.consider(np.full((240, 400, 3), shade, np.uint8), "before_transition")[0]
    assert filterer.consider(np.full((240, 400, 3), 255, np.uint8), "scroll_overlap")[0]


def test_small_camera_shift_is_redundant_but_new_label_is_not():
    filterer = FrameNoveltyFilter(fake_person)
    image = board()
    assert filterer.consider(image, "initial")[0]
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 2], [0, 1, 1]]), (400, 240),
                            borderValue=(240,) * 3)
    assert not filterer.consider(shifted, "change_time_guard")[0]
    shifted[200:205, 270:320] = 0
    assert filterer.consider(shifted, "before_erasure")[0]


def test_soft_lighting_change_does_not_replace_visible_content():
    filterer = FrameNoveltyFilter(fake_person)
    image = board()
    assert filterer.consider(image, "initial")[0]
    shadow = np.linspace(0, 20, 400)[None, :, None]
    darker = np.clip(image.astype(float) - shadow, 0, 255).astype(np.uint8)
    assert not filterer.consider(darker, "change_time_guard")[0]


def test_unreliable_large_mask_and_failed_alignment_preserve_changes(monkeypatch):
    filterer = FrameNoveltyFilter(lambda image: np.ones(image.shape[:2], bool))
    assert filterer.consider(board(), "initial")[1]["reason"] == "uncertain_mask"
    filterer.segmenter = fake_person
    assert filterer.consider(board(), "before_erasure")[0]
    monkeypatch.setattr(cv2, "findTransformECC", Mock(side_effect=cv2.error("no alignment")))
    changed = board()
    changed[190:210, 200:300] = 0
    assert filterer.consider(changed, "final_change")[0]


@pytest.mark.parametrize("bottom", [False, True])
def test_segmenter_masks_foreground_but_not_an_interior_portrait(bottom):
    output = np.zeros((1, 2, 192, 192), np.float32)
    output[0, 1, 80:192 if bottom else 130, 60:120] = 1
    segmenter = _PersonSegmenter.__new__(_PersonSegmenter)
    segmenter.net = Mock()
    segmenter.net.forward.return_value = output
    image = np.full((240, 240, 3), (20, 100, 200), np.uint8)
    mask = segmenter(image)
    assert mask.shape == image.shape[:2]
    assert mask.dtype == bool
    assert bool(mask.any()) == bottom
    blob = segmenter.net.setInput.call_args.args[0]
    assert blob.shape == (1, 3, 192, 192)
    np.testing.assert_allclose(blob[0, :, 0, 0], np.array([200, 100, 20]) / 127.5 - 1,
                               atol=1e-6)


def test_segmenter_covers_both_edges_of_a_wide_recording():
    segmenter = _PersonSegmenter.__new__(_PersonSegmenter)
    segmenter.net = Mock()
    output = np.zeros((1, 2, 192, 192), np.float32)
    output[0, 1, 90:, :] = 1
    segmenter.net.forward.return_value = output
    mask = segmenter(np.full((240, 720, 3), 200, np.uint8))
    assert mask[-1].all()
    assert segmenter.net.forward.call_count == 5


@pytest.fixture
def segmenter():
    instance = _PersonSegmenter.__new__(_PersonSegmenter)
    instance.net = Mock()
    return instance


def test_person_geometry_includes_weak_edges_but_requires_a_confident_core(segmenter):
    score = np.zeros((1, 2, 192, 192), np.float32)
    score[0, 1, 80:, 60:120] = 0.3
    score[0, 1, 100:120, 80:100] = 0.9
    segmenter.net.forward.return_value = score
    image = np.full((240, 240, 3), 180, np.uint8)
    assert segmenter(image)[220, 100]
    score[0, 1, 100:120, 80:100] = 0.3
    assert not segmenter(image).any()


def test_wide_board_diagram_near_bottom_is_not_a_standing_person(segmenter):
    score = np.zeros((1, 2, 192, 192), np.float32)
    score[0, 1, 60:180, 25:165] = 0.9
    segmenter.net.forward.return_value = score
    image = np.full((240, 240, 3), 240, np.uint8)
    cv2.ellipse(image, (120, 150), (75, 45), 0, 0, 360, (180, 60, 60), 2)
    assert not segmenter(image).any()


@pytest.mark.parametrize("left,expected", [(150, False), (360, True)])
def test_internal_pane_edges_support_cropped_people_but_not_portrait_outlines(
        segmenter, left, expected):
    image = np.full((240, 720, 3), 240, np.uint8)
    image[:, 360:] = 160
    image[70:170, left:left + 30] = (20, 100, 200)

    def predict():
        blob = segmenter.net.setInput.call_args.args[0]
        person = (blob[0, 0] > 0.3) & (blob[0, 1] < 0) & (blob[0, 2] < 0)
        score = np.zeros((1, 2, 192, 192), np.float32)
        score[0, 1] = person * 0.9
        return score

    segmenter.net.forward.side_effect = predict
    assert bool(segmenter(image)[120, left + 15]) == expected


def test_confidence_drop_does_not_hide_brief_writing_revealed_under_a_pale_shirt(segmenter):
    score = np.zeros((1, 2, 192, 192), np.float32)
    score[0, 1, 80:, 60:120] = 0.9
    segmenter.net.forward.return_value = score
    image = np.full((240, 240, 3), 240, np.uint8)
    cv2.putText(image, "A + B", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,) * 3, 2)
    filterer = FrameNoveltyFilter(segmenter)
    assert filterer.consider(image, "initial")[0]
    score[0, 1, 80:, 60:120] = 0.1
    written = image.copy()
    written[180:184, 100:130] = 20
    assert filterer.consider(written, "before_erasure")[0]
    assert filterer.consider(image, "final_change")[0]


def test_small_alignment_does_not_create_unknown_border_bands(monkeypatch):
    monkeypatch.setattr(cv2, "findTransformECC", Mock(return_value=(
        1.0, np.float32([[1, 0, 2], [0, 1, 1]]),
    )))
    image = np.full((240, 400), 240, np.uint8)
    known = np.ones(image.shape, bool)
    aligned, coverage, _ = frame_novelty._align(image, known, image, known)
    assert coverage.all()
    assert np.all(aligned == 240)
    filterer = FrameNoveltyFilter(fake_person)
    assert filterer.consider(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), "initial")[0]
    image[100:130, -5:] = 20
    assert filterer.consider(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), "final_change")[0]


def test_later_additions_cover_earlier_writing_but_erasure_does_not():
    earlier = board()
    later = earlier.copy()
    cv2.putText(later, "A + B", (240, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,) * 3, 2)
    visible = np.ones(earlier.shape[:2], bool)
    assert frame_novelty._covers_content((earlier, visible), (later, visible))
    assert not frame_novelty._covers_content((later, visible), (earlier, visible))


def test_combining_preserves_marks_that_move_out_of_view(monkeypatch):
    monkeypatch.setattr(cv2, "findTransformECC", Mock(return_value=(
        1.0, np.float32([[1, 0, -4], [0, 1, 0]]),
    )))
    earlier = board()
    earlier[195:215, 1:3] = 20
    later = cv2.warpAffine(earlier, np.float32([[1, 0, -4], [0, 1, 0]]), (400, 240),
                           borderMode=cv2.BORDER_REPLICATE)
    visible = np.ones(earlier.shape[:2], bool)
    assert not frame_novelty._covers_content((earlier, visible), (later, visible))


@pytest.mark.parametrize("loss", ["erased", "hidden", "recolored", "blurred", "scene"])
def test_combining_preserves_small_marks_color_visibility_and_readability(loss):
    earlier = board()
    cv2.ellipse(earlier, (300, 200), (6, 3), 0, 0, 360, (160, 40, 40), 1)
    visible = np.ones(earlier.shape[:2], bool)
    later = earlier.copy()
    later_visible = visible.copy()
    if loss == "erased":
        later[190:210, 290:310] = 240
    elif loss == "hidden":
        later[190:210, 290:310] = 80
        later_visible[190:210, 290:310] = False
    elif loss == "recolored":
        cv2.ellipse(later, (300, 200), (6, 3), 0, 0, 360, (40, 40, 160), 1)
    elif loss == "blurred":
        later = cv2.GaussianBlur(later, (9, 9), 2)
    else:
        later[:] = 100
    assert not frame_novelty._covers_content((earlier, visible), (later, later_visible))


def test_additive_groups_preserve_initial_view_bound_time_and_resolve_representatives(
        tmp_path, monkeypatch):
    monkeypatch.setattr(frame_novelty, "_PersonSegmenter", lambda: fake_person)
    image = board()
    items = []
    # The final frame is within five seconds of its predecessor, but outside
    # the first group's window. It must not extend that group indefinitely.
    for index, timestamp in enumerate((0, 1, 3, 6, 7)):
        if index:
            cv2.putText(image, str(index), (210 + index * 30, 215),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,) * 3, 2)
        path = tmp_path / f"{index}.png"
        cv2.imwrite(str(path), image)
        items.append({"id": f"f{index}", "timestamp": timestamp, "image": str(path),
                      "reason": "initial" if index == 0 else "before_erasure"})
    selected, diagnostics = filter_frames(tmp_path, items)
    assert selected == [items[0], items[3], items[4]]
    assert diagnostics["merged_frames"] == 2
    decisions = frame_novelty.read_json(tmp_path / "checkpoints/video/novelty.json")["result"]["decisions"]
    assert [d["represented_by"] for d in decisions if not d["retained"]] == ["f3", "f3"]
    assert all(d["novelty_retained"] for d in decisions)
    assert filter_frames(tmp_path, items)[0] == selected
    assert filter_frames(tmp_path, items)[1]["cache_hit"]


@pytest.mark.parametrize("reason", ["scroll_overlap", "final_change"])
def test_combining_keeps_scroll_boundaries_and_short_lived_pre_erasure_content(
        tmp_path, monkeypatch, reason):
    monkeypatch.setattr(frame_novelty, "_PersonSegmenter", lambda: fake_person)
    initial = board()
    written = initial.copy()
    written[195:199, 250:280] = 20
    final = written.copy() if reason == "scroll_overlap" else initial.copy()
    items = []
    for index, (image, selected_reason) in enumerate(((initial, "initial"),
                                                     (written, "before_erasure"),
                                                     (final, reason))):
        path = tmp_path / f"{index}.png"
        cv2.imwrite(str(path), image)
        items.append({"id": f"f{index}", "timestamp": index, "image": str(path),
                      "reason": selected_reason})
    assert filter_frames(tmp_path, items)[0] == items


def test_model_checksum_is_checked_before_loading(tmp_path, monkeypatch):
    import huggingface_hub

    path = tmp_path / "incorrect.onnx"
    path.write_bytes(b"wrong model")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: str(path))
    read_net = Mock()
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", read_net)
    with pytest.raises(ValueError, match="checksum"):
        _PersonSegmenter()
    read_net.assert_not_called()


def test_filter_cache_tracks_images_model_and_policy_without_loading_on_resume(tmp_path, monkeypatch):
    factory = Mock(return_value=fake_person)
    monkeypatch.setattr(frame_novelty, "_PersonSegmenter", factory)
    images = [board(), occlude(board(), 50, 110), board().copy()]
    images[2][195:200, 260:310] = 0
    items = []
    for index, image in enumerate(images):
        path = tmp_path / f"{index}.png"
        cv2.imwrite(str(path), image)
        items.append({"id": f"f{index}", "timestamp": index / 2, "image": str(path),
                      "reason": "initial" if index == 0 else "before_erasure"})
    selected, diagnostics = filter_frames(tmp_path, items)
    assert [item["id"] for item in selected] == ["f0", "f2"]
    assert diagnostics["suppressed_frames"] == 1
    assert not diagnostics["cache_hit"]
    assert filter_frames(tmp_path, items)[1]["cache_hit"]
    factory.assert_called_once()
    changed = images[1].copy()
    changed[210:220, 250:330] = 0
    cv2.imwrite(items[1]["image"], changed)
    assert not filter_frames(tmp_path, items)[1]["cache_hit"]
    monkeypatch.setitem(frame_novelty.NOVELTY_SETTINGS, "version", 999)
    assert not filter_frames(tmp_path, items)[1]["cache_hit"]
    monkeypatch.setitem(frame_novelty.PERSON_MODEL, "revision", "different revision")
    assert not filter_frames(tmp_path, items)[1]["cache_hit"]
    assert factory.call_count == 4
