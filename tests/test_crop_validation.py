from copy import deepcopy
from types import SimpleNamespace

import pymupdf
import pytest

from shad_helper.crop_validation import crop_warnings, mark_crop, valid_crop_verdict


def verdict(**changes):
    return {"verdict": "pass", "checks": dict.fromkeys(
        ("identity", "completeness", "readability", "cleanliness"), "pass"),
        "defects": [], "repair": "none", "reason": "Рисунок полностью соответствует источнику.",
        **changes}


def pixels(width=80, height=64, shade=255):
    image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    image.clear_with(shade)
    return image


def test_accepts_complete_consistent_pass_without_mutation():
    value = verdict()
    original = deepcopy(value)
    assert valid_crop_verdict(value)
    assert value == original


@pytest.mark.parametrize("status,defect,repair", [
    ("fail", "wrong_figure", "recrop"),
    ("fail", "wrong_stage", "none"),
    ("uncertain", "occluded", "nearby_frame"),
])
def test_nonpassing_verdict_requires_consistent_checks(status, defect, repair):
    value = verdict(verdict=status, defects=[defect], repair=repair)
    assert not valid_crop_verdict(value)
    value["checks"]["identity"] = status
    assert valid_crop_verdict(value)
    value["verdict"] = "pass"
    assert not valid_crop_verdict(value)


def test_failure_takes_precedence_over_uncertainty():
    value = verdict(verdict="fail", defects=["clipped_content"], repair="recrop")
    value["checks"].update(completeness="fail", readability="uncertain")
    assert valid_crop_verdict(value)
    value["verdict"] = "uncertain"
    assert not valid_crop_verdict(value)


@pytest.mark.parametrize("changes", [
    {"extra": True}, {"verdict": "yes"}, {"verdict": ["pass"]},
    {"checks": None}, {"checks": {}}, {"checks": {"identity": "pass"}},
    {"defects": "wrong_figure"}, {"defects": ["wrong_figure"]},
    {"defects": ["unknown_defect"]}, {"defects": [{}]},
    {"repair": "recrop"}, {"repair": "invent"}, {"reason": " \n"}, {"reason": 1},
])
def test_rejects_malformed_or_contradictory_pass(changes):
    assert not valid_crop_verdict(verdict(**changes))


@pytest.mark.parametrize("value", [None, [], "pass", {"verdict": "pass"}])
def test_rejects_incomplete_judgments(value):
    assert not valid_crop_verdict(value)


def test_rejects_missing_extra_and_invalid_criteria():
    value = verdict()
    value["checks"]["identity"] = "probably"
    assert not valid_crop_verdict(value)
    value["checks"]["identity"] = "pass"
    value["checks"]["extra"] = "pass"
    assert not valid_crop_verdict(value)
    value["checks"].pop("extra")
    value["checks"].pop("cleanliness")
    assert not valid_crop_verdict(value)


def test_nonpassing_judgment_needs_a_concrete_defect():
    value = verdict(verdict="uncertain")
    value["checks"]["readability"] = "uncertain"
    assert not valid_crop_verdict(value)


@pytest.mark.parametrize("shade", [0, 125, 255])
def test_uniform_crops_warn_for_dark_and_light_backgrounds(shade):
    assert crop_warnings(pixels(shade=shade), [10, 10, 90, 74], [100, 84]) == ["nearly_blank"]


def test_distinguishes_internal_content_from_boundary_content():
    image = pixels()
    image.set_rect(pymupdf.IRect(20, 20, 60, 44), (0, 0, 0))
    before = bytes(image.samples)
    assert crop_warnings(image, [10, 10, 90, 74], [100, 84]) == []
    assert bytes(image.samples) == before
    image.set_rect(pymupdf.IRect(20, 0, 60, 24), (0, 0, 0))
    assert crop_warnings(image, [10, 10, 90, 74], [100, 84]) == ["content_at_boundary"]


@pytest.mark.parametrize("box", [[0, 10, 80, 74], [10, 0, 90, 64],
                                [20, 10, 100, 74], [10, 20, 90, 84]])
def test_warns_when_any_crop_side_touches_native_frame(box):
    assert "frame_edge" in crop_warnings(pixels(), box, [100, 84])


@pytest.mark.parametrize("size", [(200, 20), (20, 200), (80, 12)])
def test_warns_for_thin_fragments(size):
    image = pixels(*size)
    assert "thin_fragment" in crop_warnings(
        image, [10, 10, 10 + size[0], 10 + size[1]], [300, 300])


def test_sampling_cost_is_bounded_for_large_frames():
    coordinates = []

    def pixel(x, y):
        coordinates.append((x, y))
        return (255, 255, 255)

    image = SimpleNamespace(width=100_000, height=200_000, n=3, alpha=0, pixel=pixel)
    assert crop_warnings(image, [1, 1, 100_001, 200_001], [200_000, 300_000]) == ["nearly_blank"]
    assert len(coordinates) <= 64 * 64 + 4 * 64
    assert all(0 <= x < image.width and 0 <= y < image.height for x, y in coordinates)


@pytest.mark.parametrize("colorspace,alpha", [(pymupdf.csRGB, False), (pymupdf.csGRAY, False),
                                           (pymupdf.csRGB, True)])
def test_marks_exact_bounds_and_preserves_source(tmp_path, colorspace, alpha):
    frame, output = tmp_path / "frame.png", tmp_path / "marked.png"
    image = pymupdf.Pixmap(colorspace, pymupdf.IRect(0, 0, 80, 64), alpha)
    image.clear_with(255)
    image.save(frame)
    original = frame.read_bytes()
    mark_crop(frame, [10, 12, 70, 54], output)
    result = pymupdf.Pixmap(output)
    assert (result.width, result.height) == (80, 64)
    for point in [(10, 12), (69, 12), (10, 53), (69, 53), (40, 12), (10, 30)]:
        assert result.pixel(*point)[:3] == (255, 0, 0)
    for point in [(0, 0), (9, 12), (70, 12), (40, 30)]:
        assert result.pixel(*point)[:3] == (255, 255, 255)
    assert frame.read_bytes() == original


def test_overlay_refuses_to_replace_native_source(tmp_path):
    frame = tmp_path / "frame.png"
    pixels().save(frame)
    original = frame.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        mark_crop(frame, [10, 12, 70, 54], frame)
    assert frame.read_bytes() == original
