"""Proxy configuration is passed unchanged through the command-line entry point."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

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
