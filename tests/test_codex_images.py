import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest
from PIL import Image

from shad_helper.auth import PlanAccessError
from shad_helper.chapter_progress import (
    ChapterCancelled,
    ChapterProgress,
    NotesCancellation,
    notes_display,
)
from shad_helper.openai_client import PlanClient, ResponseError
from shad_helper.openai_usage import command_usage


@pytest.fixture
def image_run(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "unused-api-key")
    monkeypatch.setenv("CODEX_API_KEY", "unused-codex-key")
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("https_proxy", "http://wrong.example")
    source, view, generated = (tmp_path / name for name in ("board.png", "view.png", "figure.png"))
    for path, color in ((source, "white"), (view, "black"), (generated, "blue")):
        Image.new("RGB", (12, 8), color).save(path)
    client = PlanClient(tmp_path, model="test-model", proxy="http://user:pass@proxy.example:8080",
                        transport=httpx.MockTransport(lambda r: pytest.fail("Unexpected HTTP request")),
                        token_provider=lambda: pytest.fail("Unexpected app login"))
    session = str(uuid4())
    folder = tmp_path / ".codex" / "generated_images" / session
    folder.mkdir(parents=True)
    target = folder / "exec-result.png"
    target.write_bytes(generated.read_bytes())
    events = [{"type": "thread.started", "thread_id": session},
              {"type": "item.completed", "item": {"type": "agent_message", "text": str(source)}},
              {"type": "turn.completed"}]
    child = MagicMock()
    child.returncode = 0
    child.poll.return_value = 0
    spawn = MagicMock(return_value=child)
    monkeypatch.setattr(subprocess, "Popen", spawn)

    def communicate(prompt, timeout):
        args = spawn.call_args.args[0]
        attached = [Path(args[i + 1]) for i, flag in enumerate(args) if flag == "--image"]
        assert [p.read_bytes() for p in attached] == [source.read_bytes(), view.read_bytes()]
        assert all(p.parent == Path(spawn.call_args.kwargs["cwd"]) for p in attached)
        assert timeout > 0
        return b"\n".join(json.dumps(e).encode() for e in events), None

    child.communicate.side_effect = communicate

    def run():
        return client.generate_image("Illustrate this construction.", "Меняем корень и потомка.",
                                     images=[("PRIMARY board", source), ("VIEW later", view)])

    return run, spawn, child, target, events


def test_drawing_uses_proxied_codex_login_and_its_own_generated_file(image_run):
    run, spawn, child, target, _ = image_run
    with command_usage("notes") as usage:
        assert run() == target.read_bytes()
    assert (usage.plan_requests, usage.api_requests) == (1, 0)
    args = spawn.call_args.args[0]
    assert args[:2] == ["codex", "exec"] and args[-1] == "-"
    assert args[args.index("--model") + 1] == "test-model"
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert 'forced_login_method="chatgpt"' in args and "--ephemeral" in args
    env = spawn.call_args.kwargs["env"]
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "WSS_PROXY", "WS_PROXY"):
        assert env[key] == env[key.lower()] == "http://user:pass@proxy.example:8080"
    assert env["NO_PROXY"] == env["no_proxy"] == ""
    assert "OPENAI_API_KEY" not in env and "CODEX_API_KEY" not in env
    prompt = child.communicate.call_args.args[0].decode()
    assert "Illustrate this construction." in prompt and "Меняем корень и потомка." in prompt
    assert prompt.index("PRIMARY board") < prompt.index("VIEW later")
    assert "user:pass" not in prompt and "потомка" not in str(args)


@pytest.mark.parametrize("failure", ["exit", "missing", "incomplete", "wrong_session",
                                      "symlink", "multiple", "invalid", "truncated", "timeout", "uninstalled"])
def test_failed_drawing_never_uses_an_unverified_file(image_run, failure):
    run, spawn, child, target, events = image_run
    if failure == "exit":
        child.returncode = 1
    elif failure == "missing":
        target.unlink()
    elif failure == "incomplete":
        events.pop()
    elif failure == "wrong_session":
        events[0]["thread_id"] = "../"
    elif failure == "symlink":
        outside = target.rename(target.parent.parent / "unrelated.png")
        target.symlink_to(outside)
    elif failure == "multiple":
        target.with_name("second.png").write_bytes(target.read_bytes())
    elif failure == "invalid":
        target.write_bytes(b"incomplete PNG")
    elif failure == "truncated":
        target.write_bytes(target.read_bytes()[:-12])  # Missing the final IEND chunk.
    elif failure == "timeout":
        child.communicate.side_effect = subprocess.TimeoutExpired("codex", 600)
        child.poll.return_value = None
    elif failure == "uninstalled":
        spawn.side_effect = FileNotFoundError
    with pytest.raises((PlanAccessError, ResponseError), match="Codex|codex"):
        run()
    if failure == "timeout":
        child.kill.assert_called_once()


def test_extraction_cancellation_interrupts_drawing_generation(image_run):
    run, _, child, _, _ = image_run
    cancel = NotesCancellation()

    def interrupt(prompt, timeout):
        cancel.set()
        child.kill.assert_called_once()
        return b"", None

    child.communicate.side_effect = interrupt
    with notes_display(ChapterProgress(), "ch001", cancel), pytest.raises(ChapterCancelled):
        run()
