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
