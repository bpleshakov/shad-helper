import shutil
import stat
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from shad_helper import local_worker
from shad_helper.jobs import read_json, write_json


@pytest.fixture
def conversion(tmp_path, monkeypatch):
    source = tmp_path / "cached-snapshot"
    source.mkdir()
    for name, content in (("tokenizer.json", "original tokenizer"),
                          ("config.json", "{}"), ("model.safetensors", "original weights")):
        path = source / name
        path.write_text(content)
        path.chmod(0o444)
    destination = tmp_path / "converted"
    request = {"job": str(tmp_path), "model": "PaddlePaddle/PaddleOCR-VL-1.6",
               "destination": str(destination)}
    stages = []

    def convert(**kwargs):
        stage = Path(kwargs["hf_path"])
        stages.append(stage)
        assert stage != source
        assert (stage / "tokenizer.json").stat().st_mode & stat.S_IWUSR
        destination.mkdir(exist_ok=True)
        # Upstream copies sidecars, then processor.save_pretrained overwrites them.
        for name in ("tokenizer.json", "config.json"):
            shutil.copy(stage / name, destination / name)
        assert (destination / "tokenizer.json").stat().st_mode & stat.S_IWUSR
        (destination / "tokenizer.json").write_text("saved processor tokenizer")
        (destination / "model.safetensors").write_text("converted weights")

    module, utils = ModuleType("mlx_vlm"), ModuleType("mlx_vlm.utils")
    module.convert = Mock(side_effect=convert)
    utils.get_model_path = Mock(return_value=source)
    monkeypatch.setitem(sys.modules, "mlx_vlm", module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils)
    return SimpleNamespace(source=source, destination=destination, request=request,
                           stages=stages, convert=module.convert, resolve=utils.get_model_path)


def test_first_conversion_can_overwrite_readonly_snapshot_sidecars(conversion):
    result = local_worker.run(conversion.request, "convert")
    assert result == {"model": str(conversion.destination.resolve())}
    assert (conversion.destination / "tokenizer.json").read_text() == "saved processor tokenizer"
    assert read_json(conversion.destination / "shad_conversion.json") == {
        "source": conversion.request["model"], "dtype": "bfloat16"}
    conversion.resolve.assert_called_once_with(conversion.request["model"])
    assert not conversion.stages[0].exists()


def test_retry_repairs_readonly_destination_before_conversion(conversion):
    conversion.destination.mkdir()
    for name in ("tokenizer.json", "model.safetensors"):
        path = conversion.destination / name
        path.write_text("incomplete conversion")
        path.chmod(0o444)
    original = conversion.convert.side_effect

    def retry(**kwargs):
        for name in ("tokenizer.json", "model.safetensors"):
            assert (conversion.destination / name).stat().st_mode & stat.S_IWUSR
        original(**kwargs)

    conversion.convert.side_effect = retry
    local_worker.run(conversion.request, "convert")
    assert (conversion.destination / "model.safetensors").read_text() == "converted weights"
    assert (conversion.destination / "shad_conversion.json").exists()


def test_staged_weight_symlinks_resolve_without_mutating_cached_files(conversion):
    weights = conversion.source / "model.safetensors"
    blob = conversion.source.parent / "cached-weight-blob"
    weights.rename(blob)
    weights.symlink_to(blob)
    original = conversion.convert.side_effect

    def inspect(**kwargs):
        staged_weights = Path(kwargs["hf_path"]) / "model.safetensors"
        assert staged_weights.is_symlink()
        assert staged_weights.resolve() == blob.resolve()
        assert staged_weights.read_text() == "original weights"
        original(**kwargs)

    conversion.convert.side_effect = inspect
    local_worker.run(conversion.request, "convert")
    assert (conversion.source / "tokenizer.json").read_text() == "original tokenizer"
    assert stat.S_IMODE((conversion.source / "tokenizer.json").stat().st_mode) == 0o444
    assert blob.read_text() == "original weights"
    assert stat.S_IMODE(blob.stat().st_mode) == 0o444
    assert weights.is_symlink()


def test_completed_conversion_avoids_snapshot_resolution_and_conversion(conversion):
    conversion.destination.mkdir()
    (conversion.destination / "config.json").write_text("{}")
    (conversion.destination / "model.safetensors").write_text("converted")
    write_json(conversion.destination / "shad_conversion.json",
               {"source": conversion.request["model"], "dtype": "bfloat16"})
    assert local_worker.run(conversion.request, "convert") == {
        "model": str(conversion.destination.resolve())}
    conversion.resolve.assert_not_called()
    conversion.convert.assert_not_called()


def test_failed_conversion_has_no_success_marker_and_cleans_staging(conversion):
    stages = []

    def fail(**kwargs):
        stage = Path(kwargs["hf_path"])
        stages.append(stage)
        assert stage.exists()
        conversion.destination.mkdir(exist_ok=True)
        raise PermissionError("processor save failed")

    conversion.convert.side_effect = fail
    with pytest.raises(PermissionError, match="processor save failed"):
        local_worker.run(conversion.request, "convert")
    assert not (conversion.destination / "shad_conversion.json").exists()
    assert not stages[0].exists()
    assert (conversion.source / "tokenizer.json").read_text() == "original tokenizer"
    assert stat.S_IMODE((conversion.source / "tokenizer.json").stat().st_mode) == 0o444
