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
    monkeypatch.chdir(tmp_path)
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


@pytest.fixture
def doctor_catalog(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: SimpleNamespace())
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    state_dir = tmp_path / "auth"
    state_dir.mkdir()
    (state_dir / "credentials.json").write_text("{}")
    models = Mock(return_value=["gpt-6-astra", "another-model"])
    client = SimpleNamespace(model="gpt-6-astra", models=models)
    factory = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, "shad_helper.openai_client", SimpleNamespace(PlanClient=factory))
    return state_dir, factory, models


@pytest.mark.parametrize("available", [
    ["gpt-6-astra", "another-model"], ["gpt-6.1-sol"], [],
])
def test_doctor_reports_account_models_and_default_availability(doctor_catalog, capsys, available):
    state_dir, factory, models = doctor_catalog
    models.return_value = available
    assert main(["--state-dir", str(state_dir), "doctor"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["openai"]["status"] == "available"
    assert report["openai"]["models"] == available
    assert report["openai"]["default_model"] == "gpt-6-astra"
    assert report["openai"]["default_model_available"] is ("gpt-6-astra" in available)
    assert all(report["packages"].values())
    factory.assert_called_once_with(state_dir, proxy=None)
    models.assert_called_once_with()


@pytest.mark.parametrize("missing", ["credentials", "proxy", "explicit_blank", "dependency"])
def test_doctor_skips_catalog_without_required_setup(doctor_catalog, monkeypatch, capsys, missing):
    state_dir, factory, models = doctor_catalog
    proxy = None
    if missing == "credentials":
        (state_dir / "credentials.json").unlink()
    elif missing == "proxy":
        monkeypatch.delenv("SHAD_HELPER_OPENAI_PROXY")
        monkeypatch.setenv("HTTPS_PROXY", "http://unrelated-proxy:8080")
        monkeypatch.setenv("NO_PROXY", "*")
    elif missing == "explicit_blank":
        proxy = ""
    else:
        monkeypatch.setattr(cli.importlib.util, "find_spec",
                            lambda name: None if name == "httpx" else SimpleNamespace())
    assert cli.doctor(state_dir, proxy=proxy) is (missing != "dependency")
    report = json.loads(capsys.readouterr().out)
    assert report["openai"]["status"] == "not_checked"
    assert report["openai"]["models"] == []
    assert report["openai"]["default_model"] == "gpt-6-astra"
    assert report["openai"]["default_model_available"] is None
    expected = {"credentials": "login", "proxy": "--openai-proxy",
                "explicit_blank": "--openai-proxy", "dependency": "uv sync"}
    assert expected[missing] in report["openai"]["message"]
    factory.assert_not_called()
    models.assert_not_called()
    assert not (state_dir / ".lock").exists()


@pytest.mark.parametrize("failure", ["proxy", "account", "network", "catalog"])
def test_doctor_catalog_errors_are_safe_and_do_not_fail_local_checks(doctor_catalog, capsys, failure):
    state_dir, factory, models = doctor_catalog
    secret = "PRIVATE_TOKEN_http://user:password@proxy.example"
    if failure == "proxy":
        factory.side_effect = ValueError(secret)
    else:
        models.side_effect = OSError(secret) if failure == "network" else RuntimeError(secret)
    assert main(["--state-dir", str(state_dir), "doctor"]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["openai"]["status"] == "error"
    assert report["openai"]["models"] == []
    assert report["openai"]["default_model_available"] is None
    assert "proxy" in report["openai"]["message"]
    assert secret not in captured.out + captured.err
    assert "password" not in captured.out + captured.err


def test_doctor_catalog_error_preserves_local_dependency_failure(doctor_catalog, monkeypatch, capsys):
    state_dir, _, models = doctor_catalog
    models.side_effect = RuntimeError("unavailable")
    monkeypatch.setattr(cli.importlib.util, "find_spec",
                        lambda name: None if name == "mlx_whisper" else SimpleNamespace())
    assert main(["--state-dir", str(state_dir), "doctor"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["packages"]["mlx_whisper"] is False
    assert report["openai"]["status"] == "error"


@pytest.mark.parametrize(("proxy", "position"), [
    (None, None), ("http://configured-proxy:8080", "global"),
    ("http://user:secret@configured-proxy:8080", "local"),
])
def test_doctor_forwards_state_directory_and_proxy(monkeypatch, tmp_path, proxy, position):
    check = Mock(return_value=True)
    monkeypatch.setattr(cli, "doctor", check)
    state_dir = tmp_path / "auth"
    arguments = ["--state-dir", str(state_dir)]
    if position == "global":
        arguments.extend(["--openai-proxy", proxy])
    arguments.append("doctor")
    if position == "local":
        arguments.extend(["--openai-proxy", proxy])
    assert main(arguments) == 0
    check.assert_called_once_with(state_dir, proxy=proxy)


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

@pytest.mark.parametrize("media", ["lecture.mp4", "lecture.m4a", "lecture.wav"])
def test_extract_forwards_audio_options_without_writing(monkeypatch, tmp_path, media):
    extract = Mock(return_value=tmp_path / "evidence.json")
    monkeypatch.setitem(sys.modules, "shad_helper.extraction", SimpleNamespace(extract=extract))
    out = tmp_path / "job"
    assert main(["extract", media, "--out", str(out), "--start", "30",
                 "--duration", "60", "--language", "ru"]) == 0
    extract.assert_called_once_with(
        cli.Path(media), out, language="ru", start=30, duration=60, vision_model="35b")
    assert not out.exists()


@pytest.mark.parametrize("command,flag", [
    ("extract", "--source"), ("extract", "--vision-mode"),
    ("extract", "--vision-budget-minutes"), ("render", "--images"),
])
def test_removed_options_are_rejected_before_work(tmp_path, command, flag):
    arguments = [command, "lecture.mp4" if command == "extract" else str(tmp_path)]
    if command == "extract":
        arguments.extend(["--out", str(tmp_path / "job")])
    with pytest.raises(SystemExit) as caught:
        main([*arguments, flag, "unused"])
    assert caught.value.code == 2
    assert not (tmp_path / "job").exists()


def test_doctor_checks_audio_vision_notes_and_render_dependencies(monkeypatch, capsys):
    expected = {"httpx", "jwt", "pydantic", "pymupdf", "mlx", "mlx_whisper",
                "mlx_vlm", "numpy", "cv2", "PIL"}
    monkeypatch.setattr(cli.importlib.util, "find_spec",
                        lambda name: SimpleNamespace() if name in expected else None)
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert cli.doctor()
    report = json.loads(capsys.readouterr().out)
    assert set(report["packages"]) == expected
    assert set(report["tools"]) == {"ffmpeg", "ffprobe", "latexmk", "lualatex"}


def test_doctor_smoke_runs_only_transcription(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "doctor", Mock(return_value=True))
    extract = Mock()
    monkeypatch.setitem(sys.modules, "shad_helper.extraction", SimpleNamespace(extract=extract))
    assert main(["doctor", "--smoke", "lecture.wav"]) == 0
    extract.assert_called_once_with(cli.Path("lecture.wav"), cli.Path(".shad-helper/smoke"),
                                    duration=30, vision_model="off")


@pytest.mark.parametrize("model", ["4b", "9b", "35b", "off"])
def test_extract_forwards_vision_model(monkeypatch, tmp_path, model):
    extract = Mock(return_value=tmp_path / "evidence.json")
    monkeypatch.setitem(sys.modules, "shad_helper.extraction", SimpleNamespace(extract=extract))
    assert main(["extract", "lecture.mp4", "--out", str(tmp_path),
                 "--vision-model", model]) == 0
    assert extract.call_args.kwargs["vision_model"] == model


def test_unknown_vision_model_is_rejected_before_work(monkeypatch, tmp_path):
    extract = Mock()
    monkeypatch.setitem(sys.modules, "shad_helper.extraction", SimpleNamespace(extract=extract))
    with pytest.raises(SystemExit) as caught:
        main(["extract", "lecture.mp4", "--out", str(tmp_path / "job"),
              "--vision-model", "unknown"])
    assert caught.value.code == 2
    extract.assert_not_called()
    assert not (tmp_path / "job").exists()
