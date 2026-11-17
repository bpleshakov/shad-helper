"""Proxy configuration is passed unchanged through the command-line entry point."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from shad_helper import cli
from shad_helper.cli import main
from shad_helper.jobs import read_json
from shad_helper.run_diagnostics import record_run


def zero_spending(command):
    return (f"OpenAI spending [{command}]: estimated API cost $0.000000; "
            "API requests: 0; ChatGPT-plan requests: 0.\n")


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("SHAD_HELPER_OPENAI_PROXY", "http://environment-proxy:8080")


@pytest.mark.parametrize("position", ["global", "local"])
def test_run_forwards_multiple_files_and_stage_options(monkeypatch, tmp_path, position):
    run = Mock(return_value=True)
    monkeypatch.setitem(sys.modules, "shad_helper.pipeline", SimpleNamespace(run_pipeline=run))
    state, out = tmp_path / "auth", tmp_path / "jobs"
    files = [tmp_path / "Лекция 1.mp4", tmp_path / "Лекция 2.mkv"]
    collection = tmp_path / "profile" / "collection.anki2"
    proxy = "http://configured-proxy:8080"
    arguments = ["--state-dir", str(state)]
    if position == "global":
        arguments.extend(["--openai-proxy", proxy])
    arguments.extend(["run", *(str(file) for file in files), "--out", str(out),
                      "--start", "30", "--duration", "60", "--language", "en",
                      "--vision-model", "4b", "--vision-mode", "video",
                      "--rebuild", "--anki-collection", str(collection)])
    if position == "local":
        arguments.extend(["--openai-proxy", proxy])

    assert main(arguments) == 0
    run.assert_called_once_with(files, out, state, language="en", start=30, duration=60,
                                vision_model="4b", vision_mode="video", rebuild=True, proxy=proxy,
                                anki_collection=collection)


@pytest.mark.parametrize("completed", [True, False])
def test_run_defaults_and_batch_exit_status(monkeypatch, tmp_path, completed):
    run = Mock(return_value=completed)
    monkeypatch.setitem(sys.modules, "shad_helper.pipeline", SimpleNamespace(run_pipeline=run))
    assert main(["run", "lecture.mp4", "--out", "jobs"]) == (0 if completed else 1)
    run.assert_called_once_with([cli.Path("lecture.mp4")], cli.Path("jobs"),
                                cli.Path(".shad-helper/auth"), language="ru", start=0,
                                duration=None, vision_model="35b", vision_mode="frames", rebuild=False,
                                proxy=None, anki_collection=None)


@pytest.mark.parametrize("arguments", [
    ["run", "--out", "jobs"],
    ["run", "lecture.mp4"],
    ["run", "lecture.mp4", "--out", "jobs", "--vision-model", "unknown"],
    ["run", "lecture.mp4", "--out", "jobs", "--vision-mode", "unknown"],
    ["extract", "lecture.mp4", "--out", "jobs", "--vision-mode", "unknown"],
])
def test_run_rejects_missing_inputs_output_or_unknown_model(monkeypatch, arguments):
    run = Mock()
    monkeypatch.setitem(sys.modules, "shad_helper.pipeline", SimpleNamespace(run_pipeline=run))
    with pytest.raises(SystemExit) as caught:
        main(arguments)
    assert caught.value.code == 2
    run.assert_not_called()


@pytest.mark.parametrize("position", ["global", "local"])
def test_anki_forwards_pdf_rebuild_state_and_proxy(monkeypatch, tmp_path, position):
    generate = Mock(return_value=tmp_path / "deck.apkg")
    monkeypatch.setitem(sys.modules, "shad_helper.anki", SimpleNamespace(generate_anki=generate))
    state, job, pdf = tmp_path / "auth", tmp_path / "lecture", tmp_path / "notes.pdf"
    collection = tmp_path / "anki-profile" / "collection.anki2"
    arguments = ["--state-dir", str(state)]
    proxy = "http://configured-proxy:8080"
    if position == "global":
        arguments.extend(["--openai-proxy", proxy])
    arguments.extend(["anki", str(job), "--pdf", str(pdf), "--rebuild",
                      "--anki-collection", str(collection)])
    if position == "local":
        arguments.extend(["--openai-proxy", proxy])
    assert main(arguments) == 0
    generate.assert_called_once_with(job, state, pdf=pdf, rebuild=True, proxy=proxy,
                                     anki_collection=collection)


def test_anki_defaults_and_failure_exit_code(monkeypatch, tmp_path, capsys):
    generate = Mock(side_effect=ValueError("Anki requires a finished PDF"))
    monkeypatch.setitem(sys.modules, "shad_helper.anki", SimpleNamespace(generate_anki=generate))
    job = tmp_path / "lecture"
    assert main(["anki", str(job)]) == 1
    generate.assert_called_once_with(job, cli.Path(".shad-helper/auth"),
                                     pdf=None, rebuild=False, proxy=None, anki_collection=None)
    assert "finished PDF" in capsys.readouterr().err


@pytest.fixture
def anki_batch(monkeypatch, tmp_path):
    jobs = [tmp_path / f"lecture-{number}" for number in range(3)]
    for job in jobs:
        job.mkdir()

    def export(job, state, *, minimum_position, **kwargs):
        folder = job / "anki"
        folder.mkdir(exist_ok=True)
        (folder / "report.json").write_text(json.dumps({
            "end_position": max(100, minimum_position) + 2,
        }))
        return folder / "deck.apkg"

    generate = Mock(side_effect=export)
    monkeypatch.setitem(sys.modules, "shad_helper.anki", SimpleNamespace(generate_anki=generate))
    return jobs, generate


@pytest.mark.parametrize("position", ["global", "local"])
def test_anki_batch_runs_in_order_and_forwards_options(anki_batch, tmp_path, position, capsys):
    jobs, generate = anki_batch
    jobs = jobs[::-1]
    state, collection = tmp_path / "auth", tmp_path / "collection.anki2"
    proxy = "http://configured-proxy:8080"
    arguments = ["--state-dir", str(state)]
    if position == "global":
        arguments.extend(["--openai-proxy", proxy])
    arguments.extend(["anki", *(str(job) for job in jobs), "--rebuild",
                      "--anki-collection", str(collection)])
    if position == "local":
        arguments.extend(["--openai-proxy", proxy])

    assert main(arguments) == 0
    assert generate.call_args_list == [
        call(job, state, pdf=None, rebuild=True, proxy=proxy, anki_collection=collection,
             minimum_position=minimum)
        for job, minimum in zip(jobs, (1, 103, 106), strict=True)
    ]
    output = capsys.readouterr()
    assert "3 completed, 0 failed" in output.out
    assert output.err == zero_spending("anki")


def test_anki_batch_continues_after_failure_without_using_stale_report(anki_batch, capsys):
    jobs, generate = anki_batch
    stale_folder = jobs[1] / "anki"
    stale_folder.mkdir()
    (stale_folder / "report.json").write_text(json.dumps({"end_position": 9999}))
    export = generate.side_effect

    def fail_middle(job, *args, **kwargs):
        if job == jobs[1]:
            raise RuntimeError("Generation failed")
        return export(job, *args, **kwargs)

    generate.side_effect = fail_middle
    assert main(["anki", *(str(job) for job in jobs)]) == 1
    assert generate.call_args_list == [
        call(job, cli.Path(".shad-helper/auth"), pdf=None, rebuild=False, proxy=None,
             anki_collection=None, minimum_position=minimum)
        for job, minimum in zip(jobs, (1, 103, 103), strict=True)
    ]
    output = capsys.readouterr()
    assert "2 completed, 1 failed" in output.out
    assert str(jobs[1]) in output.err
    assert "Generation failed" in output.err


@pytest.mark.parametrize("duplicate", ["same", "relative", "symlink"])
def test_anki_batch_rejects_duplicate_destinations_before_work(anki_batch, tmp_path, duplicate):
    jobs, generate = anki_batch
    repeated = jobs[0]
    if duplicate == "relative":
        repeated = cli.Path(jobs[0].name)
    elif duplicate == "symlink":
        repeated = tmp_path / "alias"
        repeated.symlink_to(jobs[0], target_is_directory=True)

    assert main(["anki", str(jobs[0]), str(jobs[1]), str(repeated)]) == 1
    generate.assert_not_called()


def test_anki_batch_rejects_shared_pdf_before_work(anki_batch, tmp_path, capsys):
    jobs, generate = anki_batch
    assert main(["anki", *(str(job) for job in jobs), "--pdf", str(tmp_path / "notes.pdf")]) == 1
    generate.assert_not_called()
    assert "--pdf" in capsys.readouterr().err


def test_anki_requires_at_least_one_job(anki_batch):
    _, generate = anki_batch
    with pytest.raises(SystemExit) as caught:
        main(["anki"])
    assert caught.value.code == 2
    generate.assert_not_called()


def test_anki_keyboard_interrupt_stops_batch_immediately(anki_batch):
    jobs, generate = anki_batch
    generate.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        main(["anki", *(str(job) for job in jobs)])
    assert generate.call_count == 1


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
    assert capsys.readouterr().err == zero_spending("login")


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
    login = report["openai"]["chatgpt_login"]
    assert login["status"] == "available"
    assert login["models"] == available
    assert login["default_model"] == "gpt-6-astra"
    assert login["default_model_available"] is ("gpt-6-astra" in available)
    assert report["openai"]["api_key"]["status"] == "not_checked"
    assert all(report["packages"].values())
    factory.assert_called_once_with(state_dir, proxy=None, api_key="")
    models.assert_called_once_with(allow_fallback=False)


@pytest.mark.parametrize("source", ["environment", "file"])
def test_doctor_checks_api_key_without_saved_login(
        doctor_catalog, monkeypatch, tmp_path, capsys, source):
    state_dir, factory, models = doctor_catalog
    (state_dir / "credentials.json").unlink()
    if source == "environment":
        monkeypatch.setenv("OPENAI_API_KEY", "private-key")
    else:
        (tmp_path / "config").mkdir()
        (tmp_path / "config/openai.env").write_text("OPENAI_API_KEY=private-key\n")
    assert main(["--state-dir", str(state_dir), "doctor"]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["openai"]["api_key"]["status"] == "available"
    assert report["openai"]["chatgpt_login"]["status"] == "not_checked"
    factory.assert_called_once_with(state_dir, proxy=None, api_key="private-key")
    models.assert_called_once_with(allow_fallback=False)
    assert "private-key" not in captured.out + captured.err
    assert not (state_dir / ".lock").exists()


@pytest.mark.parametrize(("contents", "status"), [
    ("# No key\nUNRELATED=value\n", "not_checked"),
    ("OPENAI_API_KEY=\n", "not_checked"),
    ('OPENAI_API_KEY="private-key\n', "error"),
])
def test_doctor_missing_or_invalid_key_does_not_block_login(
        doctor_catalog, tmp_path, capsys, contents, status):
    state_dir, factory, models = doctor_catalog
    (tmp_path / "config").mkdir()
    (tmp_path / "config/openai.env").write_text(contents)

    assert main(["--state-dir", str(state_dir), "doctor"]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)["openai"]
    assert report["api_key"]["status"] == status
    assert report["api_key"]["models"] == []
    assert report["chatgpt_login"]["status"] == "available"
    factory.assert_called_once_with(state_dir, proxy=None, api_key="")
    models.assert_called_once_with(allow_fallback=False)
    assert "private-key" not in captured.out + captured.err


@pytest.mark.parametrize("key", ["", "  "])
def test_doctor_explicit_empty_key_disables_file_key(
        doctor_catalog, monkeypatch, tmp_path, capsys, key):
    state_dir, factory, _ = doctor_catalog
    (tmp_path / "config").mkdir()
    (tmp_path / "config/openai.env").write_text("OPENAI_API_KEY=private-key\n")
    monkeypatch.setenv("OPENAI_API_KEY", key)

    assert main(["--state-dir", str(state_dir), "doctor"]) == 0
    report = json.loads(capsys.readouterr().out)["openai"]
    assert report["api_key"]["status"] == "not_checked"
    assert report["chatgpt_login"]["status"] == "available"
    factory.assert_called_once_with(state_dir, proxy=None, api_key="")


@pytest.mark.parametrize("missing", ["credentials", "proxy", "explicit_blank", "dependency"])
def test_doctor_skips_catalog_without_required_setup(doctor_catalog, monkeypatch, capsys, missing):
    state_dir, factory, models = doctor_catalog
    proxy = None
    if missing != "credentials":
        monkeypatch.setenv("OPENAI_API_KEY", "private-key")
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
    expected = {"credentials": "login", "proxy": "--openai-proxy",
                "explicit_blank": "--openai-proxy", "dependency": "uv sync"}
    for source, result in report["openai"].items():
        assert result["status"] == "not_checked"
        assert result["models"] == []
        assert result["default_model"] is None
        assert result["default_model_available"] is None
        hint = "OPENAI_API_KEY" if missing == "credentials" and source == "api_key" else expected[missing]
        assert hint in result["message"]
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
    report = json.loads(captured.out)["openai"]["chatgpt_login"]
    assert report["status"] == "error"
    assert report["models"] == []
    assert report["default_model_available"] is None
    assert "proxy" in report["message"]
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
    assert report["openai"]["chatgpt_login"]["status"] == "error"


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
    assert capsys.readouterr().err == zero_spending("notes")


@pytest.fixture
def notes_batch(monkeypatch, tmp_path):
    jobs = [tmp_path / f"lecture-{number}" for number in range(3)]
    for job in jobs:
        job.mkdir()
    events = []

    def generate(job, state, *, manual, **kwargs):
        events.append(("notes", job))
        return job / "handoff" if manual else job / "notes" / "main.tex"

    def render(job, **kwargs):
        events.append(("render", job))
        return job / "notes" / "main.pdf"

    generate_mock, render_mock = Mock(side_effect=generate), Mock(side_effect=render)
    monkeypatch.setitem(sys.modules, "shad_helper.notes",
                        SimpleNamespace(generate_notes=generate_mock))
    monkeypatch.setitem(sys.modules, "shad_helper.rendering", SimpleNamespace(render=render_mock))
    return jobs, generate_mock, render_mock, events


@pytest.mark.parametrize("rebuild", [False, True])
@pytest.mark.parametrize("position", ["global", "local"])
def test_notes_batch_runs_in_order_with_options(notes_batch, tmp_path, rebuild, position, capsys):
    jobs, generate, render, events = notes_batch
    jobs = jobs[::-1]
    state, proxy = tmp_path / "auth", "http://configured-proxy:8080"
    arguments = ["--state-dir", str(state)]
    if position == "global":
        arguments.extend(["--openai-proxy", proxy])
    arguments.extend(["notes", *(str(job) for job in jobs)])
    if rebuild:
        arguments.append("--rebuild")
    if position == "local":
        arguments.extend(["--openai-proxy", proxy])

    assert main(arguments) == 0
    assert generate.call_args_list == [
        call(job, state, imported=None, manual=False, rebuild=rebuild, proxy=proxy) for job in jobs
    ]
    stages = ("notes", "render") if rebuild else ("notes",)
    assert events == [(stage, job) for job in jobs for stage in stages]
    assert render.call_args_list == ([call(job, rebuild=True) for job in jobs] if rebuild else [])
    output = capsys.readouterr()
    assert "3 completed, 0 failed" in output.out
    assert output.err == zero_spending("notes")
    if rebuild:
        assert all(str(job / "notes" / "main.pdf") in output.out for job in jobs)


@pytest.mark.parametrize("imported", [False, True])
def test_notes_single_rebuild_renders_completed_notes(notes_batch, tmp_path, imported, capsys):
    jobs, generate, render, events = notes_batch
    job = jobs[0]
    source = tmp_path / "response.json" if imported else None
    arguments = ["notes", str(job), "--rebuild"]
    if source is not None:
        arguments.extend(["--import", str(source)])

    assert main(arguments) == 0
    generate.assert_called_once_with(job, cli.Path(".shad-helper/auth"), imported=source,
                                     manual=False, rebuild=True, proxy=None)
    render.assert_called_once_with(job, rebuild=True)
    assert events == [("notes", job), ("render", job)]
    assert str(job / "notes" / "main.pdf") in capsys.readouterr().out


@pytest.mark.parametrize("fail_render", [False, True])
def test_notes_rebuild_keeps_generation_and_render_timings_in_one_report(
        notes_batch, monkeypatch, fail_render):
    jobs, generate, render, _ = notes_batch

    @record_run("notes")
    def tracked_notes(job, state, **kwargs):
        return generate(job, state, **kwargs)

    @record_run("render")
    def tracked_render(job, **kwargs):
        return render(job, **kwargs)

    monkeypatch.setattr(sys.modules["shad_helper.notes"], "generate_notes", tracked_notes)
    monkeypatch.setattr(sys.modules["shad_helper.rendering"], "render", tracked_render)
    if fail_render:
        render.side_effect = RuntimeError("Compilation failed")
    assert main(["notes", str(jobs[0]), "--rebuild"]) == int(fail_render)
    report = read_json(jobs[0] / "run-diagnostics.json")
    assert report["command"] == "notes"
    assert report["status"] == ("failed" if fail_render else "completed")
    assert [step["name"] for step in report["steps"]] == ["notes", "render"]
    assert all(step["elapsed_seconds"] >= 0 for step in report["steps"])


def test_notes_single_rebuild_does_not_render_automatic_handoff(notes_batch, capsys):
    jobs, generate, render, _ = notes_batch
    generate.side_effect = None
    generate.return_value = jobs[0] / "handoff"

    assert main(["notes", str(jobs[0]), "--rebuild"]) == 1
    render.assert_not_called()
    assert str(jobs[0] / "handoff") in capsys.readouterr().err


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("rebuild", [False, True])
def test_notes_manual_exports_each_handoff_without_rendering(notes_batch, count, rebuild, capsys):
    jobs, generate, render, events = notes_batch
    jobs = jobs[:count]
    arguments = ["notes", *(str(job) for job in jobs), "--manual"]
    if rebuild:
        arguments.append("--rebuild")

    assert main(arguments) == 0
    assert generate.call_args_list == [
        call(job, cli.Path(".shad-helper/auth"), imported=None, manual=True,
             rebuild=rebuild, proxy=None) for job in jobs
    ]
    assert events == [("notes", job) for job in jobs]
    render.assert_not_called()
    output = capsys.readouterr()
    assert all(str(job / "handoff") in output.out for job in jobs)
    assert output.err == zero_spending("notes")


@pytest.mark.parametrize(("failure", "rebuild"), [
    ("generation", False), ("generation", True), ("handoff", False), ("handoff", True),
    ("render", True),
])
def test_notes_batch_continues_after_incomplete_project(notes_batch, failure, rebuild, capsys):
    jobs, generate, render, events = notes_batch
    original_generate, original_render = generate.side_effect, render.side_effect

    def generate_notes(job, *args, **kwargs):
        result = original_generate(job, *args, **kwargs)
        if job == jobs[1]:
            if failure == "generation":
                raise ValueError("Invalid notes")
            if failure == "handoff":
                return job / "handoff"
        return result

    def render_notes(job, **kwargs):
        result = original_render(job, **kwargs)
        if job == jobs[1] and failure == "render":
            raise RuntimeError("PDF compilation failed")
        return result

    generate.side_effect, render.side_effect = generate_notes, render_notes
    arguments = ["notes", *(str(job) for job in jobs)]
    if rebuild:
        arguments.append("--rebuild")
    assert main(arguments) == 1
    expected = [("notes", jobs[0])]
    if rebuild:
        expected.append(("render", jobs[0]))
    expected.append(("notes", jobs[1]))
    if failure == "render":
        expected.append(("render", jobs[1]))
    expected.append(("notes", jobs[2]))
    if rebuild:
        expected.append(("render", jobs[2]))
    assert events == expected
    output = capsys.readouterr()
    assert "2 completed, 1 failed" in output.out
    assert str(jobs[1]) in output.err


@pytest.mark.parametrize("duplicate", ["same", "relative", "symlink"])
def test_notes_batch_rejects_duplicate_destinations_before_work(notes_batch, tmp_path, duplicate):
    jobs, generate, render, events = notes_batch
    repeated = jobs[0]
    if duplicate == "relative":
        repeated = cli.Path(jobs[0].name)
    elif duplicate == "symlink":
        repeated = tmp_path / "alias"
        repeated.symlink_to(jobs[0], target_is_directory=True)

    assert main(["notes", str(jobs[0]), str(jobs[1]), str(repeated), "--rebuild"]) == 1
    generate.assert_not_called()
    render.assert_not_called()
    assert events == []


def test_notes_batch_rejects_shared_import_before_work(notes_batch, tmp_path, capsys):
    jobs, generate, render, _ = notes_batch
    assert main(["notes", *(str(job) for job in jobs),
                 "--import", str(tmp_path / "response.json"), "--rebuild"]) == 1
    generate.assert_not_called()
    render.assert_not_called()
    assert "--import" in capsys.readouterr().err


def test_notes_requires_at_least_one_job(notes_batch):
    _, generate, render, _ = notes_batch
    with pytest.raises(SystemExit) as caught:
        main(["notes"])
    assert caught.value.code == 2
    generate.assert_not_called()
    render.assert_not_called()


@pytest.mark.parametrize("stage", ["notes", "render"])
def test_notes_keyboard_interrupt_stops_batch_immediately(notes_batch, stage):
    jobs, generate, render, _ = notes_batch
    (generate if stage == "notes" else render).side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        main(["notes", *(str(job) for job in jobs), "--rebuild"])
    assert generate.call_count == 1
    assert render.call_count == (stage == "render")


@pytest.mark.parametrize("media", ["lecture.mp4", "lecture.m4a", "lecture.wav"])
def test_extract_forwards_audio_options_without_writing(monkeypatch, tmp_path, media):
    extract = Mock(return_value=tmp_path / "evidence.json")
    monkeypatch.setitem(sys.modules, "shad_helper.extraction", SimpleNamespace(extract=extract))
    out = tmp_path / "job"
    assert main(["extract", media, "--out", str(out), "--start", "30",
                 "--duration", "60", "--language", "ru"]) == 0
    extract.assert_called_once_with(
        cli.Path(media), out, language="ru", start=30, duration=60, vision_model="35b",
        vision_mode="frames")
    assert not out.exists()


@pytest.mark.parametrize("command,flag", [
    ("extract", "--source"),
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
@pytest.mark.parametrize("mode", ["frames", "video"])
def test_extract_forwards_vision_options(monkeypatch, tmp_path, model, mode):
    extract = Mock(return_value=tmp_path / "evidence.json")
    monkeypatch.setitem(sys.modules, "shad_helper.extraction", SimpleNamespace(extract=extract))
    assert main(["extract", "lecture.mp4", "--out", str(tmp_path),
                 "--vision-model", model, "--vision-mode", mode]) == 0
    assert extract.call_args.kwargs["vision_model"] == model
    assert extract.call_args.kwargs["vision_mode"] == mode


def test_unknown_vision_model_is_rejected_before_work(monkeypatch, tmp_path):
    extract = Mock()
    monkeypatch.setitem(sys.modules, "shad_helper.extraction", SimpleNamespace(extract=extract))
    with pytest.raises(SystemExit) as caught:
        main(["extract", "lecture.mp4", "--out", str(tmp_path / "job"),
              "--vision-model", "unknown"])
    assert caught.value.code == 2
    extract.assert_not_called()
    assert not (tmp_path / "job").exists()
