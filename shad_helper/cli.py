"""Command line entry points; heavyweight model imports stay in worker processes."""

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Local Russian lecture extraction and LaTeX notes")
    result.add_argument("--state-dir", type=Path, default=Path(".shad-helper/auth"),
                        help="App-owned ChatGPT credentials (default: .shad-helper/auth)")
    result.add_argument("--openai-proxy", help="HTTP/HTTPS proxy URL for OpenAI requests "
                        "(or SHAD_HELPER_OPENAI_PROXY; required for online commands)")
    commands = result.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="Check local dependencies and account-available models")
    check.add_argument("--smoke", type=Path, help="Transcribe a thirty-second audio or video excerpt")
    login = commands.add_parser("login", help="Authorize this local app to use your ChatGPT plan")
    extract = commands.add_parser("extract", help="Extract resumable local lecture evidence")
    extract.add_argument("media", type=Path, help="Audio file or video soundtrack")
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument("--start", type=float, default=0)
    extract.add_argument("--duration", type=float, help="Process a pilot interval, in seconds")
    extract.add_argument("--language", default="ru")
    notes = commands.add_parser("notes", help="Generate notes or export/import a ChatGPT handoff")
    notes.add_argument("job", type=Path)
    mode = notes.add_mutually_exclusive_group()
    mode.add_argument("--manual", action="store_true", help="Export handoff without any network calls")
    mode.add_argument("--import", dest="imported", type=Path, help="JSON response or chapter replies folder")
    notes.add_argument("--rebuild", action="store_true", help="Invalidate synthesized note checkpoints")
    for command in (check, login, notes):
        command.add_argument("--openai-proxy", default=argparse.SUPPRESS,
                             help="HTTP/HTTPS proxy URL for OpenAI requests")
    render = commands.add_parser("render", help="Compile editable LaTeX and create PDF QA images")
    render.add_argument("job", type=Path)
    review = commands.add_parser("review", help="Show review issues or record a verified correction")
    review.add_argument("job", type=Path)
    review.add_argument("--resolve", help="Review issue ID")
    review.add_argument("--resolution", help="Describe the correction made to evidence or notes")
    anki = commands.add_parser("anki", help="Reserved for a later, separate reviewed-notes step")
    anki.add_argument("job", type=Path)
    return result


def doctor(state_dir: Path = Path(".shad-helper/auth"), *, proxy: str | None = None) -> bool:
    packages = {name: importlib.util.find_spec(name) is not None for name in (
        "httpx", "jwt", "pydantic", "pymupdf")}
    # find_spec on a dotted name imports its parent, so check MLX top-level only.
    packages.update({name: importlib.util.find_spec(name) is not None
                     for name in ("mlx", "mlx_whisper")})
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe", "latexmk", "lualatex")}
    openai = {"status": "not_checked", "models": [], "default_model": "gpt-6-astra",
              "default_model_available": None}
    configured_proxy = os.environ.get("SHAD_HELPER_OPENAI_PROXY") if proxy is None else proxy
    if not (Path(state_dir) / "credentials.json").is_file():
        openai["message"] = "Run shad-helper login to authorize ChatGPT plan access."
    elif not configured_proxy:
        openai["message"] = "Configure --openai-proxy or SHAD_HELPER_OPENAI_PROXY to check models."
    elif not packages["httpx"] or not packages["jwt"]:
        openai["message"] = "Install HTTP/auth dependencies with uv sync before checking models."
    else:
        try:
            from .openai_client import PlanClient
            client = PlanClient(state_dir, proxy=proxy)
            openai["default_model"] = client.model
            openai["models"] = client.models()
            openai["status"] = "available"
            openai["default_model_available"] = client.model in openai["models"]
            if not openai["default_model_available"]:
                openai["message"] = "The default notes model is unavailable for this account."
        except ModuleNotFoundError:
            openai["message"] = "Install HTTP/auth dependencies with uv sync before checking models."
        except ValueError:
            openai.update(status="error", message="Invalid OpenAI proxy configuration; use "
                          "--openai-proxy or SHAD_HELPER_OPENAI_PROXY with an HTTP/HTTPS proxy URL.")
        except (OSError, RuntimeError):
            openai.update(status="error", message="Could not check the ChatGPT account model catalog. "
                          "Verify proxy connectivity and account access; run shad-helper login "
                          "if needed, then retry.")
    print(json.dumps({"python": sys.version.split()[0], "platform": platform.platform(),
                      "architecture": platform.machine(), "packages": packages,
                      "tools": tools, "cache": os.environ["HF_HOME"], "openai": openai}, indent=2))
    return all(packages.values()) and all(tools[name] for name in
                                          ("ffmpeg", "ffprobe", "latexmk", "lualatex"))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    # Keep downloads and model caches in this workspace, not the user's global caches.
    cache = Path(".cache/huggingface").resolve()
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        if args.command == "doctor":
            healthy = doctor(args.state_dir, proxy=args.openai_proxy)
            if args.smoke:
                from .extraction import extract
                extract(args.smoke, Path(".shad-helper/smoke"), duration=30)
            return 0 if healthy else 1
        if args.command == "login":
            from .auth import login
            account = login(args.state_dir, proxy=args.openai_proxy)
            print(f"Signed in: {account.get('email') or account.get('subject')}")
        elif args.command == "extract":
            from .extraction import extract
            print(extract(args.media, args.out, language=args.language,
                          start=args.start, duration=args.duration))
        elif args.command == "notes":
            from .notes import generate_notes
            print(generate_notes(args.job, args.state_dir, imported=args.imported,
                                 manual=args.manual, rebuild=args.rebuild, proxy=args.openai_proxy))
        elif args.command == "render":
            from .rendering import render
            print(render(args.job))
        elif args.command == "review":
            from .jobs import check_audio_job, read_json, validate_audio_evidence
            check_audio_job(args.job)
            validate_audio_evidence(read_json(args.job / "evidence.json"))
            if args.resolve:
                if not args.resolution:
                    raise ValueError("--resolve requires --resolution describing your correction")
                from .notes import resolve_issue
                resolve_issue(args.job, args.resolve, args.resolution)
            path = args.job / "review.json"
            data = read_json(path) if path.exists() else read_json(args.job / "evidence.json")
            issues = data.get("issues", [])
            for issue in issues:
                print(f"[{issue['status']}] {issue['id']}: {issue['description']}")
                if issue.get("resolution"):
                    print(f"  Correction: {issue['resolution']}")
            print(f"{sum(i['status'] == 'open' for i in issues)} unresolved issue(s)")
        elif args.command == "anki":
            print("Anki generation is not implemented. The future step will require reviewed notes.")
            return 2
        return 0
    except ModuleNotFoundError as exc:
        print(f"Missing dependency {exc.name}. Run: UV_CACHE_DIR=.cache/uv uv sync --extra local", file=sys.stderr)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
    return 1
