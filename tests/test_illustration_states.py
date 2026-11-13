import cv2
import numpy as np
import pytest

from shad_helper.illustration_states import (
    SOURCE_STATE_SETTINGS,
    _equivalent,
    _prepare,
    group_source_states,
    save_state_source,
    source_contexts,
)


def figure():
    image = np.full((240, 400, 3), 240, np.uint8)
    cv2.arrowedLine(image, (45, 190), (340, 190), (20,) * 3, 2)
    cv2.arrowedLine(image, (45, 190), (45, 45), (20,) * 3, 2)
    cv2.putText(image, "A", (65, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,) * 3, 2)
    cv2.line(image, (80, 150), (250, 80), (50, 70, 160), 2)
    return image


def group(tmp_path, images, *, timestamps=None, descriptions=None):
    observations, sources = {}, {}
    for index, image in enumerate(images):
        identifier = f"v{index}_ill01"
        path = tmp_path / f"{identifier}.png"
        assert cv2.imwrite(str(path), image)
        sources[identifier] = path
        observations[identifier] = {
            "id": identifier, "visual_id": f"v{index}", "source_kind": "drawn",
            "timestamp": timestamps[index] if timestamps else index * 2, "status": "complete",
            "description": descriptions[index] if descriptions else "The same graph",
        }
    return group_source_states(observations, sources)


def test_minor_pixel_noise_and_empty_margins_share_one_source_state(tmp_path):
    original = figure()
    noisy = np.clip(original.astype(np.int16) + np.random.default_rng(4).integers(
        -2, 3, original.shape), 0, 255).astype(np.uint8)
    padded = cv2.copyMakeBorder(original, 7, 11, 9, 13, cv2.BORDER_CONSTANT, value=(240,) * 3)
    groups = group(tmp_path, [original, noisy, padded],
                   descriptions=["Axes A and i", "A graph with axes", "The same drawing"])
    assert len(groups) == 1
    assert set(next(iter(groups.values()))) == {"v0_ill01", "v1_ill01", "v2_ill01"}


@pytest.mark.parametrize("change", [
    "label", "arrow", "panel", "geometry", "intermediate", "occlusion", "clipped", "color",
    "faint_mark", "small_mark", "single_pixel", "scroll",
])
def test_same_description_does_not_merge_different_content(tmp_path, change):
    original = figure()
    changed = original.copy()
    if change == "label":
        changed[40:75, 60:95] = 240
        cv2.putText(changed, "B", (65, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,) * 3, 2)
    elif change == "arrow":
        changed[175:205] = 240
        cv2.arrowedLine(changed, (340, 190), (45, 190), (20,) * 3, 2)
    elif change == "panel":
        cv2.rectangle(changed, (280, 60), (330, 140), (20,) * 3, 2)
    elif change == "geometry":
        cv2.line(changed, (80, 150), (250, 110), (20,) * 3, 2)
    elif change == "intermediate":
        changed[80:110, 170:250] = 240
    elif change == "occlusion":
        changed[:, 120:150] = 80
    elif change == "clipped":
        changed = changed[:, 80:]
    elif change == "color":
        changed[np.all(changed == (50, 70, 160), axis=2)] = (160, 70, 50)
    elif change == "faint_mark":
        changed[120:123, 200:210] = 215
    elif change == "small_mark":
        changed[120:122, 200:202] = 0
    elif change == "single_pixel":
        changed[120, 200] = 0
    elif change == "scroll":
        changed = np.roll(changed, 100, axis=0)
    assert len(group(tmp_path, [original, changed])) == 2


def test_distant_observations_and_blank_crops_stay_independent(tmp_path):
    gap = SOURCE_STATE_SETTINGS["nearby_seconds"] + 1
    assert len(group(tmp_path, [figure(), figure()], timestamps=[0, gap])) == 2
    blank = np.full((240, 400, 3), 220, np.uint8)
    assert len(group(tmp_path, [blank, blank])) == 2


def test_tiny_local_change_is_not_hidden_by_large_unchanged_area(tmp_path):
    original = np.full((1080, 1920, 3), 240, np.uint8)
    cv2.rectangle(original, (30, 30), (1890, 1050), (20,) * 3, 2)
    changed = original.copy()
    changed[500:502, 500:502] = 0
    assert len(group(tmp_path, [original, changed])) == 2


@pytest.mark.parametrize("field,value", [
    ("status", "partial"), ("source_kind", "unknown"), ("source_kind", "printed"),
    ("visual_id", "v0"),
])
def test_uncertain_or_separate_source_observations_are_not_grouped(tmp_path, field, value):
    path = tmp_path / "figure.png"
    assert cv2.imwrite(str(path), figure())
    first = {"status": "complete", "source_kind": "drawn", "visual_id": "v0", "timestamp": 0}
    second = {**first, "visual_id": "v1", "timestamp": 2, field: value}
    groups = group_source_states({"first": first, "second": second},
                                  {"first": path, "second": path})
    assert len(groups) == 2


def test_representative_is_the_sharper_source_and_does_not_change(tmp_path):
    sharp = figure()
    soft = cv2.addWeighted(sharp, 0.96, cv2.GaussianBlur(sharp, (3, 3), 0), 0.04, 0)
    groups = group(tmp_path, [soft, sharp])
    assert groups == {"v1_ill01": ["v0_ill01", "v1_ill01"]}


def test_two_figures_in_one_frame_cannot_join_through_a_third_observation(tmp_path):
    path = tmp_path / "figure.png"
    assert cv2.imwrite(str(path), figure())
    first = {"status": "complete", "source_kind": "drawn", "visual_id": "v0", "timestamp": 0}
    second = {**first, "visual_id": "v1", "timestamp": 2}
    groups = group_source_states({"a": first, "b": second, "c": second},
                                  {"a": path, "b": path, "c": path})
    assert groups == {"a": ["a", "b"], "c": ["c"]}


def test_every_member_must_match_fixed_representative_without_transitive_drift(tmp_path):
    original = figure()
    images = []
    for offset in (0, 6, 12):
        image = original.copy()
        image[120:124, 200:204] -= offset
        images.append(image)
    assert len(group(tmp_path, images)) == 2


@pytest.mark.parametrize("change", ["lighting", "compression", "crop", "camera", "zoom"])
def test_same_content_survives_capture_and_crop_changes(tmp_path, change):
    original = figure()
    gradient = np.linspace(-12, 12, original.shape[1])[None, :, None]
    original = np.clip(original.astype(np.float32) + gradient, 0, 255).astype(np.uint8)
    if change == "lighting":
        changed = np.clip(original.astype(np.float32) + gradient + 12, 0, 255).astype(np.uint8)
    elif change == "compression":
        encoded = cv2.imencode(".jpg", original, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
        changed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    elif change == "crop":
        changed = original[10:220, 20:380]
    elif change == "camera":
        changed = cv2.warpAffine(original, np.float32([[1, 0, 3], [0, 1, 2]]),
                                  (400, 240), borderMode=cv2.BORDER_REPLICATE)
    else:
        changed = cv2.warpAffine(original, np.float32([[1.01, 0, -2], [0, 1.01, -1]]),
                                  (400, 240), borderMode=cv2.BORDER_REPLICATE)
    assert len(group(tmp_path, [original, changed])) == 1


def test_continuous_unchanged_state_is_not_split_at_representative_time_limit(tmp_path):
    gap = SOURCE_STATE_SETTINGS["nearby_seconds"] - 1
    images = [figure() for _ in range(4)]
    assert len(group(tmp_path, images, timestamps=[0, gap, gap * 2, gap * 3])) == 1


def test_registration_cannot_resample_away_a_new_single_pixel_mark(tmp_path):
    original = figure()
    gradient = np.linspace(-12, 12, 400)[None, :, None]
    original = np.clip(original.astype(np.float32) + gradient, 0, 255).astype(np.uint8)
    changed = cv2.warpAffine(original, np.float32([[1, 0, 0.5], [0, 1, 0.5]]),
                              (400, 240), borderMode=cv2.BORDER_REPLICATE)
    changed[120, 200] = 0
    assert not _equivalent(_prepare(original), _prepare(changed),
                           np.float32([[1, 0, -0.5], [0, 1, -0.5]]))
    assert len(group(tmp_path, [original, changed])) == 2


def native_sources(tmp_path, images, boxes):
    observations, sources = {}, {}
    evidence = {"media": {"sha256": "a" * 64}}
    (tmp_path / "frames").mkdir()
    (tmp_path / "illustrations").mkdir()
    for index, (image, box) in enumerate(zip(images, boxes)):
        identifier = f"v{index}_ill01"
        x0, y0, x1, y1 = box
        frame = tmp_path / "frames" / f"v{index}_{'a' * 12}.png"
        source = tmp_path / "illustrations" / f"{identifier}.png"
        assert cv2.imwrite(str(frame), image)
        assert cv2.imwrite(str(source), image[y0:y1, x0:x1])
        sources[identifier] = source
        height, width = image.shape[:2]
        observations[identifier] = {
            "id": identifier, "visual_id": f"v{index}", "timestamp": index * 10,
            "source_kind": "drawn", "status": "complete",
            "box": [x0 / width, y0 / height, x1 / width, y1 / height],
        }
    contexts = source_contexts(observations, sources, tmp_path, evidence)
    return observations, sources, contexts, evidence


def test_native_region_recovers_crop_clipping_without_losing_any_source_content(tmp_path):
    original = figure()
    observations, sources, contexts, _ = native_sources(
        tmp_path, [original, original], [[20, 20, 380, 220], [40, 20, 380, 220]])
    assert len(contexts) == 2
    assert all(item["region"] == [20, 20, 380, 220] for item in contexts.values())
    assert len(group_source_states(observations, sources)) == 2
    groups = group_source_states(observations, sources, contexts=contexts)
    assert len(groups) == 1
    representative = next(iter(groups))
    path = tmp_path / "state/source.png"
    save_state_source(representative, sources, contexts, path)
    assert np.array_equal(cv2.imread(str(path)), original[20:220, 20:380])


def test_native_region_separates_a_changed_state_outside_the_original_crop(tmp_path):
    original = figure()
    changed = original.copy()
    changed[120:123, 30:35] = 0
    observations, sources, contexts, _ = native_sources(
        tmp_path, [original, changed], [[20, 20, 380, 220], [40, 20, 380, 220]])
    assert len(group_source_states(observations, sources, contexts=contexts)) == 2


@pytest.mark.parametrize("problem", ["missing", "changed", "refined"])
def test_unbound_native_context_is_not_used(tmp_path, problem):
    observations, sources, _, evidence = native_sources(
        tmp_path, [figure()], [[20, 20, 380, 220]])
    frame = next((tmp_path / "frames").iterdir())
    if problem == "missing":
        frame.unlink()
    elif problem == "changed":
        assert cv2.imwrite(str(frame), np.zeros((240, 400, 3), np.uint8))
    else:
        refined = tmp_path / "refined.png"
        refined.write_bytes(sources["v0_ill01"].read_bytes())
        sources["v0_ill01"] = refined
    assert source_contexts(observations, sources, tmp_path, evidence) == {}


def test_changed_native_frame_cannot_be_saved_under_earlier_group_provenance(tmp_path):
    _, sources, contexts, _ = native_sources(
        tmp_path, [figure()], [[20, 20, 380, 220]])
    frame = next((tmp_path / "frames").iterdir())
    assert cv2.imwrite(str(frame), np.zeros((240, 400, 3), np.uint8))
    with pytest.raises(ValueError, match="frame changed"):
        save_state_source("v0_ill01", sources, contexts, tmp_path / "state.png")


def test_optional_local_dependencies_are_not_required(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "cv2", None)
    assert group_source_states({}, {"one": tmp_path / "unused", "two": tmp_path / "unused"}) == {
        "one": ["one"], "two": ["two"]}
