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
    check = commands.add_parser("doctor", help="Check Python, local packages, and external tools")
    check.add_argument("--smoke", type=Path, help="Run all local models on a short video excerpt")
    login = commands.add_parser("login", help="Authorize this local app to use your ChatGPT plan")
    extract = commands.add_parser("extract", help="Extract resumable local lecture evidence")
    extract.add_argument("video", type=Path)
    extract.add_argument("--source", type=Path, action="append", default=[])
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
    for command in (login, notes):
        command.add_argument("--openai-proxy", default=argparse.SUPPRESS,
                             help="HTTP/HTTPS proxy URL for OpenAI requests")
    render = commands.add_parser("render", help="Compile editable LaTeX and create PDF QA images")
    render.add_argument("job", type=Path)
    render.add_argument("--images", type=Path, help="Manual illustrations named FIGURE_ID.png/jpg/pdf")
    review = commands.add_parser("review", help="Show review issues or record a verified correction")
    review.add_argument("job", type=Path)
    review.add_argument("--resolve", help="Review issue ID")
    review.add_argument("--resolution", help="Describe the correction made to evidence or notes")
    anki = commands.add_parser("anki", help="Reserved for a later, separate reviewed-notes step")
    anki.add_argument("job", type=Path)
    return result


def doctor() -> bool:
    packages = {name: importlib.util.find_spec(name) is not None for name in (
        "httpx", "jwt", "pydantic", "fitz", "cv2", "numpy", "PIL", "pptx")}
    # find_spec on a dotted name imports its parent, so check MLX top-level only.
    packages.update({name: importlib.util.find_spec(name) is not None
                     for name in ("mlx", "mlx_whisper", "mlx_vlm")})
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe", "latexmk", "lualatex")}
    tools["libreoffice"] = shutil.which("libreoffice") or shutil.which("soffice")
    app = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")
    if not tools["libreoffice"] and app.exists():
        tools["libreoffice"] = str(app)
    print(json.dumps({"python": sys.version.split()[0], "platform": platform.platform(),
                      "architecture": platform.machine(), "packages": packages,
                      "tools": tools, "cache": os.environ["HF_HOME"]}, indent=2))
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
            healthy = doctor()
            if args.smoke:
                from .extraction import extract
                extract(args.smoke, [], Path(".shad-helper/smoke"), duration=30)
            return 0 if healthy else 1
        if args.command == "login":
            from .auth import login
            account = login(args.state_dir, proxy=args.openai_proxy)
            print(f"Signed in: {account.get('email') or account.get('subject')}")
        elif args.command == "extract":
            from .extraction import extract
            print(extract(args.video, args.source, args.out, language=args.language,
                          start=args.start, duration=args.duration))
        elif args.command == "notes":
            from .notes import generate_notes
            print(generate_notes(args.job, args.state_dir, imported=args.imported,
                                 manual=args.manual, rebuild=args.rebuild, proxy=args.openai_proxy))
        elif args.command == "render":
            from .rendering import render
            print(render(args.job, images_dir=args.images))
        elif args.command == "review":
            from .jobs import read_json
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
