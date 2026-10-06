import json

import pytest

from shad_helper import vision_mode


@pytest.mark.parametrize("requested", [None, "content", "baseline"])
def test_new_job_persists_default_or_explicit_mode(tmp_path, requested):
    out = tmp_path / "new-job"
    assert not out.exists()
    expected = {"vision_mode": requested or "content", "grouping_version": 1}
    assert vision_mode.resolve_vision_mode(out, requested) == expected
    assert json.loads((out / "extraction-config.json").read_text()) == expected


def test_empty_existing_directory_tree_is_a_new_content_job(tmp_path):
    out = tmp_path / "empty-job"
    (out / "checkpoints" / "vision").mkdir(parents=True)
    expected = {"vision_mode": "content", "grouping_version": 1}
    assert vision_mode.resolve_vision_mode(out, None) == expected
    assert json.loads((out / "extraction-config.json").read_text()) == expected


@pytest.mark.parametrize("requested", [None, "baseline"])
def test_legacy_artifacts_keep_baseline_without_any_writes(tmp_path, requested):
    out = tmp_path / "legacy-job"
    artifact = out / "checkpoints" / "vision" / "visual_00000.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"result": "existing work"}\n')
    before = {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    assert vision_mode.resolve_vision_mode(out, requested) == {
        "vision_mode": "baseline", "grouping_version": 1,
    }
    assert {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()} == before


def test_legacy_job_cannot_switch_to_content(tmp_path):
    out = tmp_path / "legacy-job"
    out.mkdir()
    artifact = out / "transcript.srt"
    artifact.write_text("existing subtitles")
    with pytest.raises(ValueError, match="new output directory"):
        vision_mode.resolve_vision_mode(out, "content")
    assert artifact.read_text() == "existing subtitles"
    assert not (out / "extraction-config.json").exists()


@pytest.mark.parametrize("mode", ["content", "baseline"])
def test_saved_mode_is_sticky_and_unchanged(tmp_path, mode):
    config_path = tmp_path / "extraction-config.json"
    config = {"vision_mode": mode, "grouping_version": 1}
    original = json.dumps(config, indent=4) + "\n"
    config_path.write_text(original)
    assert vision_mode.resolve_vision_mode(tmp_path, None) == config
    assert vision_mode.resolve_vision_mode(tmp_path, mode) == config
    with pytest.raises(ValueError, match="new output directory"):
        vision_mode.resolve_vision_mode(tmp_path, "baseline" if mode == "content" else "content")
    assert config_path.read_text() == original


@pytest.mark.parametrize("config", [
    {}, {"vision_mode": "unknown", "grouping_version": 1},
    {"vision_mode": [], "grouping_version": 1},
    {"vision_mode": "content", "grouping_version": 2},
    {"vision_mode": "content", "grouping_version": 0},
    {"vision_mode": "content", "grouping_version": True},
    {"vision_mode": "content", "grouping_version": 1.0},
    ["content", 1],
])
def test_invalid_or_unsupported_configuration_fails_without_writes(tmp_path, config):
    config_path = tmp_path / "extraction-config.json"
    original = json.dumps(config)
    config_path.write_text(original)
    with pytest.raises(ValueError, match="new output directory"):
        vision_mode.resolve_vision_mode(tmp_path, None)
    assert config_path.read_text() == original


def test_future_grouping_version_cannot_regroup_an_existing_job(tmp_path, monkeypatch):
    config_path = tmp_path / "extraction-config.json"
    original = '{"vision_mode":"content","grouping_version":1}'
    config_path.write_text(original)
    monkeypatch.setattr(vision_mode, "GROUPING_VERSION", 2)
    with pytest.raises(ValueError, match="grouping version"):
        vision_mode.resolve_vision_mode(tmp_path, None)
    assert config_path.read_text() == original


def test_corrupt_configuration_does_not_fall_back_to_a_new_mode(tmp_path):
    config_path = tmp_path / "extraction-config.json"
    config_path.write_text("{broken JSON")
    with pytest.raises(ValueError, match="Invalid extraction-config"):
        vision_mode.resolve_vision_mode(tmp_path, None)
    assert config_path.read_text() == "{broken JSON"


def test_invalid_requested_mode_is_rejected_before_creating_the_job(tmp_path):
    out = tmp_path / "new-job"
    with pytest.raises(ValueError, match="Vision mode"):
        vision_mode.resolve_vision_mode(out, "unknown")
    assert not out.exists()


def test_saved_configuration_cannot_override_worker_inputs(tmp_path):
    config_path = tmp_path / "extraction-config.json"
    config = {"vision_mode": "content", "grouping_version": 1,
              "job": "existing-lecture", "items": [], "model": "different-model"}
    original = json.dumps(config)
    config_path.write_text(original)
    assert vision_mode.resolve_vision_mode(tmp_path) == {
        "vision_mode": "content", "grouping_version": 1}
    assert config_path.read_text() == original
