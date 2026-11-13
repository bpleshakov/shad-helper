import sys

import pytest

from shad_helper.illustration_matching import VisualMatcher, visually_plausible
from shad_helper.illustration_verification import _matching_figures
from shad_helper.jobs import read_json, sha256_file, write_json


@pytest.fixture
def scenes(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    paths, observations, blocks = {}, {}, {}

    def add(identifier, seed=0, *, image=None, topic="Common topic", status="complete"):
        if image is None:
            image = np.full((320, 480, 3), 245, np.uint8)
            rng = np.random.default_rng(seed)
            for index in range(45):
                point = tuple(int(value) for value in rng.integers([10, 20], [430, 290]))
                cv2.putText(image, f"{seed}:{index}", point, cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (20, 30, 40), 1, cv2.LINE_AA)
        path = tmp_path / f"{identifier}.png"
        assert cv2.imwrite(str(path), image)
        paths[identifier, ""] = path
        observations[identifier] = {"id": identifier, "source_kind": "printed",
                                    "status": status, "description": "Lecture figure",
                                    "timestamp": len(observations), "image_sha256": identifier}
        blocks[identifier] = {"id": identifier, "illustration_id": identifier, "topic": topic}
        return image

    matcher = VisualMatcher(lambda identifier, crop_id: paths[identifier, crop_id])
    return add, matcher, observations, blocks, paths


def test_visual_match_beats_three_same_topic_distractors_and_different_caption(scenes, tmp_path):
    import cv2

    add, matcher, observations, blocks, _ = scenes
    source = add("source")
    for seed in (1, 2, 3):
        add(f"distractor{seed}", seed)
    add("match", image=cv2.resize(source, None, fx=0.8, fy=0.8), topic="Another name")
    observations["match"]["description"] = "Completely different extracted description"
    selected = _matching_figures(blocks["source"], list(blocks.values()), observations,
                                 matcher=matcher, archive=tmp_path / "search")
    assert [block["id"] for block in selected] == ["match"]
    record = read_json(tmp_path / "search/matching.json")
    assert record["selected"] == ["match"]
    assert sum(not item["eligible"] for item in record["candidates"]) == 3


def test_partial_view_can_match_a_complete_multi_panel_figure(scenes):
    import numpy as np

    add, matcher, _, blocks, _ = scenes
    source = add("partial")
    extra = add("extra", 4)
    add("complete", image=np.concatenate([source, extra], axis=1))
    assert visually_plausible(matcher.score(blocks["partial"], blocks["complete"]))
    assert visually_plausible(matcher.score(blocks["complete"], blocks["partial"]))


@pytest.mark.parametrize("uncertainty", ["sparse", "incomplete", "unknown", "dependencies"])
def test_uncertain_visual_evidence_keeps_text_candidates(scenes, monkeypatch, uncertainty):
    import numpy as np

    add, matcher, observations, blocks, paths = scenes
    add("source", image=np.full((120, 160, 3), 240, np.uint8) if uncertainty == "sparse" else None)
    for seed in (1, 2, 3):
        add(str(seed), seed)
    if uncertainty == "incomplete":
        observations["source"]["status"] = "incomplete"
    if uncertainty == "unknown":
        observations["source"]["source_kind"] = "unknown"
    if uncertainty == "dependencies":
        monkeypatch.setitem(sys.modules, "cv2", None)
        matcher = VisualMatcher(lambda identifier, crop_id: paths[identifier, crop_id])
    assert len(_matching_figures(blocks["source"], list(blocks.values()), observations,
                                matcher=matcher)) == 3


def test_recovery_keeps_visually_rejected_candidates(scenes):
    add, matcher, observations, blocks, _ = scenes
    add("source")
    add("target", 2)
    assert not _matching_figures(blocks["source"], [blocks["target"]], observations, matcher=matcher)
    assert _matching_figures(blocks["source"], [blocks["target"]], observations,
                             matcher=matcher, broad=True) == [blocks["target"]]


def test_verified_track_promotes_only_its_exact_target_crop(scenes):
    import cv2

    add, matcher, observations, blocks, paths = scenes
    source = add("source")
    add("target", 2)
    add("previous", image=cv2.resize(source, None, fx=0.85, fy=0.85))
    assert not _matching_figures(blocks["source"], [blocks["target"]], observations, matcher=matcher)
    matcher.remember(blocks["previous"], blocks["target"])
    assert _matching_figures(blocks["source"], [blocks["target"]], observations,
                             matcher=matcher) == [blocks["target"]]
    paths["target", "new-crop"] = paths["target", ""]
    changed = {**blocks["target"], "illustration_crop_id": "new-crop"}
    assert not _matching_figures(blocks["source"], [changed], observations, matcher=matcher)


def test_refined_source_replaces_original_pixels_and_reuses_cached_features(scenes):
    add, matcher, _, blocks, paths = scenes
    add("source")
    add("target", 2)
    assert not visually_plausible(matcher.score(blocks["source"], blocks["target"]))
    paths["source", "refined"] = paths["target", ""]
    refined = {**blocks["source"], "illustration_crop_id": "refined"}
    assert matcher.score(refined, blocks["target"])["identical"]
    misses = matcher.features.cache_info().misses
    assert matcher.score(refined, blocks["target"])["identical"]
    assert matcher.features.cache_info().misses == misses


def test_featureless_narrow_crop_does_not_crash_or_reject(scenes):
    import numpy as np

    add, matcher, _, blocks, _ = scenes
    add("source", image=np.full((1, 2000, 3), 240, np.uint8))
    add("target", 9)
    assert matcher.score(blocks["source"], blocks["target"]) is None


def test_wide_and_tall_crops_do_not_resize_a_template_to_zero(scenes):
    import numpy as np

    add, matcher, _, blocks, _ = scenes
    image = np.random.default_rng(123).integers(0, 256, (64, 800, 3), dtype=np.uint8)
    add("wide", image=image)
    add("tall", image=image.transpose(1, 0, 2).copy())
    score = matcher.score(blocks["wide"], blocks["tall"])
    assert score is not None and 0 <= score["inlier_fraction"] <= 1


def test_strong_leader_gets_one_check_but_recovery_keeps_other_plausible_views(scenes):
    import cv2

    add, matcher, observations, blocks, _ = scenes
    source = add("source")
    add("best", image=source)
    other = add("other", 4)
    add("partial", image=cv2.addWeighted(source, 0.5, other, 0.5, 0))
    assert visually_plausible(matcher.score(blocks["source"], blocks["partial"]))
    candidates = [blocks["best"], blocks["partial"]]
    assert _matching_figures(blocks["source"], candidates, observations,
                             matcher=matcher) == [blocks["best"]]
    assert len(_matching_figures(blocks["source"], candidates, observations,
                                matcher=matcher, broad=True)) == 2
    observations["source"]["status"] = "incomplete"
    assert len(_matching_figures(blocks["source"], candidates, observations,
                                matcher=matcher)) == 2


def test_replay_checks_archived_pixels_and_leaves_input_unchanged(scenes, tmp_path):
    from scripts.illustration_matching_benchmark import replay

    add, _, observations, blocks, paths = scenes
    add("source")
    add("target", 2)
    job = tmp_path / "job"
    attempt = job / "handoff/verification/attempt-01"
    write_json(job / "evidence.json", {"visuals": [
        {"id": identifier, "status": "complete", "timestamp": 1, "illustrations": [item]}
        for identifier, item in observations.items()]})
    write_json(attempt / "illustration-verification/target/accepted.json", blocks["target"])
    coverage = attempt / "illustration-coverage/source"
    write_json(coverage / "candidate.json", {"illustration_id": "source", "block_id": "original"})
    alternative = coverage / "alternatives/target"
    write_json(alternative / "verdict.json", {"verdict": "fail"})
    archive = alternative / "artifacts"
    archive.mkdir()
    images = []
    for identifier, filename in (("source", "source.png"), ("target", "preview.png")):
        (archive / filename).write_bytes(paths[identifier, ""].read_bytes())
        images.append({"file": filename, "sha256": sha256_file(archive / filename)})
    write_json(archive / "call.json", {"images": images})
    before = {path: path.read_bytes() for path in job.rglob("*") if path.is_file()}
    report = replay(job, tmp_path / "report")
    assert report["archived_comparisons"] == 1 and report["retained_comparisons"] == 0
    assert before == {path: path.read_bytes() for path in job.rglob("*") if path.is_file()}
    (archive / "source.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        replay(job, tmp_path / "invalid")
