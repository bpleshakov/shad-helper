"""Proxy configuration is passed unchanged through the command-line entry point."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from shad_helper import cli
from shad_helper.cli import main


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("SHAD_HELPER_OPENAI_PROXY", "http://environment-proxy:8080")


@pytest.mark.parametrize(("proxy", "position"), [
    (None, None), ("http://configured-proxy:8080", "global"),
    ("http://user:secret@configured-proxy:8080", "local")])
def test_login_forwards_proxy_flag_or_none(monkeypatch, tmp_path, capsys, proxy, position):
    login = Mock(return_value={"email": "lecture@example.com"})
    monkeypatch.setitem(sys.modules, "shad_helper.auth", SimpleNamespace(login=login))
    state_dir = tmp_path / "auth"
    arguments = ["--state-dir", str(state_dir)]
    if position == "global":
        arguments.extend(["--openai-proxy", proxy])
    arguments.append("login")
    if position == "local":
        arguments.extend(["--openai-proxy", proxy])

    assert main(arguments) == 0

    login.assert_called_once_with(state_dir, proxy=proxy)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("vision_mode", [None, "content", "baseline"])
def test_extract_forwards_vision_mode_without_resolving_or_writing(monkeypatch, tmp_path, vision_mode):
    extract = Mock(return_value=tmp_path / "evidence.json")
    monkeypatch.setitem(sys.modules, "shad_helper.extraction", SimpleNamespace(extract=extract))
    video, out = tmp_path / "lecture.mp4", tmp_path / "job"
    arguments = ["extract", str(video), "--out", str(out)]
    if vision_mode is not None:
        arguments.extend(["--vision-mode", vision_mode])
    assert main(arguments) == 0
    extract.assert_called_once_with(video, [], out, language="ru", start=0, duration=None,
                                    vision_mode=vision_mode)
    assert not out.exists()


def test_extract_rejects_unknown_vision_mode_before_any_work(monkeypatch, tmp_path):
    extract = Mock()
    monkeypatch.setitem(sys.modules, "shad_helper.extraction", SimpleNamespace(extract=extract))
    with pytest.raises(SystemExit):
        main(["extract", "lecture.mp4", "--out", str(tmp_path / "job"),
              "--vision-mode", "unknown"])
    extract.assert_not_called()
    assert not (tmp_path / "job").exists()


@pytest.mark.parametrize(("platform_name", "missing", "healthy"), [
    ("darwin", "Vision", False), ("darwin", "Quartz", False),
    ("darwin", None, True), ("linux", "Vision", True),
])
def test_doctor_checks_apple_vision_dependencies_only_on_macos(monkeypatch, capsys,
                                                            platform_name, missing, healthy):
    find_spec = Mock(side_effect=lambda name: None if name == missing else SimpleNamespace())
    monkeypatch.setattr(cli.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(cli.sys, "platform", platform_name)
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert cli.doctor() is healthy
    packages = json.loads(capsys.readouterr().out)["packages"]
    if platform_name == "darwin":
        assert packages["Vision"] is (missing != "Vision")
        assert packages["Quartz"] is (missing != "Quartz")
    else:
        assert "Vision" not in packages and "Quartz" not in packages
        assert not any(call.args[0] in {"Vision", "Quartz"} for call in find_spec.call_args_list)


@pytest.mark.parametrize(("proxy", "position"), [
    (None, None), ("http://configured-proxy:8080", "global"),
    ("http://user:secret@configured-proxy:8080", "local")])
def test_notes_forwards_proxy_flag_or_none(monkeypatch, tmp_path, capsys, proxy, position):
    generate_notes = Mock(return_value=tmp_path / "notes.json")
    monkeypatch.setitem(sys.modules, "shad_helper.notes",
                        SimpleNamespace(generate_notes=generate_notes))
    state_dir = tmp_path / "auth"
    job = tmp_path / "lecture"
    arguments = ["--state-dir", str(state_dir)]
    if position == "global":
        arguments.extend(["--openai-proxy", proxy])
    arguments.extend(["notes", str(job)])
    if position == "local":
        arguments.extend(["--openai-proxy", proxy])

    assert main(arguments) == 0

    generate_notes.assert_called_once_with(job, state_dir, imported=None,
                                          manual=False, rebuild=False, proxy=proxy)
    assert capsys.readouterr().err == ""
