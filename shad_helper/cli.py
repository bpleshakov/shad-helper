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
    from .vision_config import DEFAULT_VISION_MODEL, VISION_MODELS

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
    extract.add_argument("media", type=Path, help="Audio file or lecture video")
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument("--start", type=float, default=0)
    extract.add_argument("--duration", type=float, help="Process a pilot interval, in seconds")
    extract.add_argument("--language", default="ru")
    extract.add_argument("--vision-model", choices=(*VISION_MODELS, "off"),
                         default=DEFAULT_VISION_MODEL,
                         help="Local video model: 35b (default), 9b, 4b, or off for audio only")
    extract.add_argument("--vision-mode", choices=("frames", "video"), default="frames",
                         help="Separate frame requests (default) or native video batches")
    notes = commands.add_parser("notes", help="Generate notes or export/import a ChatGPT handoff")
    notes.add_argument("jobs", nargs="+", type=Path, metavar="JOB",
                       help="Project directories to process sequentially, in order")
    mode = notes.add_mutually_exclusive_group()
    mode.add_argument("--manual", action="store_true", help="Export handoff without any network calls")
    mode.add_argument("--import", dest="imported", type=Path,
                      help="JSON response or chapter replies folder for a single project")
    notes.add_argument("--rebuild", action="store_true",
                       help="Delete notes/render artifacts before starting, then regenerate notes and PDF")
    render = commands.add_parser("render", help="Compile editable LaTeX and create PDF QA images")
    render.add_argument("job", type=Path)
    review = commands.add_parser("review", help="Show review issues or record a verified correction")
    review.add_argument("job", type=Path)
    review.add_argument("--resolve", help="Review issue ID")
    review.add_argument("--resolution", help="Describe the correction made to evidence or notes")
    anki = commands.add_parser("anki", help="Create an importable Anki deck from a finished PDF")
    anki.add_argument("jobs", nargs="+", type=Path, metavar="JOB",
                      help="Project directories to process sequentially, in order")
    anki.add_argument("--pdf", type=Path,
                      help="Input PDF for a single project (defaults to the rendered lecture PDF)")
    anki.add_argument("--rebuild", action="store_true",
                      help="Delete all Anki artifacts before generating a fresh deck")
    anki.add_argument("--anki-collection", type=Path,
                      help="Anki collection.anki2 to read positions from (default: unique local profile)")
    run = commands.add_parser("run", help="Run extract, notes, render, and Anki for each recording")
    run.add_argument("media", nargs="+", type=Path, help="Audio files or lecture videos, in order")
    run.add_argument("--out", type=Path, required=True, help="Parent directory for separate jobs")
    run.add_argument("--start", type=float, default=0)
    run.add_argument("--duration", type=float, help="Process this interval length in each file")
    run.add_argument("--language", default="ru")
    run.add_argument("--vision-model", choices=(*VISION_MODELS, "off"),
                     default=DEFAULT_VISION_MODEL,
                     help="Local video model: 35b (default), 9b, 4b, or off for audio only")
    run.add_argument("--vision-mode", choices=("frames", "video"), default="frames",
                     help="Separate frame requests (default) or native video batches")
    run.add_argument("--rebuild", action="store_true",
                     help="Delete notes/render and Anki artifacts before rebuilding each stage")
    run.add_argument("--anki-collection", type=Path,
                     help="Anki collection.anki2 to read positions from (default: unique local profile)")
    for command in (check, login, notes, anki, run):
        command.add_argument("--openai-proxy", default=argparse.SUPPRESS,
                             help="HTTP/HTTPS proxy URL for OpenAI requests")
    return result


def doctor(state_dir: Path = Path(".shad-helper/auth"), *, proxy: str | None = None) -> bool:
    from .openai_config import load_api_key

    packages = {name: importlib.util.find_spec(name) is not None for name in (
        "httpx", "jwt", "pydantic", "pymupdf")}
    # find_spec on a dotted name imports its parent, so check MLX top-level only.
    packages.update({name: importlib.util.find_spec(name) is not None
                     for name in ("mlx", "mlx_whisper", "mlx_vlm", "numpy", "cv2", "PIL")})
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe", "latexmk", "lualatex")}
    openai = {}
    configured_proxy = os.environ.get("SHAD_HELPER_OPENAI_PROXY") if proxy is None else proxy
    for source in ("api_key", "chatgpt_login"):
        result = {"status": "not_checked", "models": [], "default_model": None,
                  "default_model_available": None}
        openai[source] = result
        try:
            api_key = load_api_key() if source == "api_key" else ""
            if source == "api_key" and not api_key:
                result["message"] = "Set OPENAI_API_KEY in the environment or config/openai.env."
            elif source == "chatgpt_login" and not (Path(state_dir) / "credentials.json").is_file():
                result["message"] = "Run shad-helper login to authorize ChatGPT plan access."
            elif not configured_proxy:
                result["message"] = "Configure --openai-proxy or SHAD_HELPER_OPENAI_PROXY to check models."
            elif not packages["httpx"] or not packages["jwt"]:
                result["message"] = "Install HTTP/auth dependencies with uv sync before checking models."
            else:
                from .openai_client import PlanClient
                client = PlanClient(state_dir, proxy=proxy, api_key=api_key)
                result["default_model"] = client.model
                result["models"] = client.models(allow_fallback=False)
                result["status"] = "available"
                result["default_model_available"] = client.model in result["models"]
                if not result["default_model_available"]:
                    result["message"] = "The default notes model is unavailable for this account."
        except ModuleNotFoundError:
            result["message"] = "Install HTTP/auth dependencies with uv sync before checking models."
        except ValueError:
            result.update(status="error", message="Invalid OpenAI configuration; check "
                          "credentials and the --openai-proxy or SHAD_HELPER_OPENAI_PROXY URL.")
        except (OSError, RuntimeError):
            result.update(status="error", message="Could not check the OpenAI model catalog. "
                          "Verify proxy connectivity, API key and account access; run shad-helper login "
                          "if needed, then retry.")
    print(json.dumps({"python": sys.version.split()[0], "platform": platform.platform(),
                      "architecture": platform.machine(), "packages": packages,
                      "tools": tools, "cache": os.environ["HF_HOME"], "openai": openai}, indent=2))
    return all(packages.values()) and all(tools[name] for name in
                                          ("ffmpeg", "ffprobe", "latexmk", "lualatex"))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from .openai_usage import command_usage

    with command_usage(args.command):
        return _run(args)


def _run(args) -> int:
    # Keep downloads and model caches in this workspace, not the user's global caches.
    cache = Path(".cache/huggingface").resolve()
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        if args.command == "doctor":
            healthy = doctor(args.state_dir, proxy=args.openai_proxy)
            if args.smoke:
                from .extraction import extract
                extract(args.smoke, Path(".shad-helper/smoke"), duration=30, vision_model="off")
            return 0 if healthy else 1
        if args.command == "login":
            from .auth import login
            account = login(args.state_dir, proxy=args.openai_proxy)
            print(f"Signed in: {account.get('email') or account.get('subject')}")
        elif args.command == "extract":
            from .extraction import extract
            print(extract(args.media, args.out, language=args.language,
                          start=args.start, duration=args.duration, vision_model=args.vision_model,
                          vision_mode=args.vision_mode))
        elif args.command == "notes":
            from .notes import generate_notes
            from .run_diagnostics import diagnostic_run
            batch = len(args.jobs) > 1
            if batch and args.imported is not None:
                raise ValueError("--import requires a single project directory")
            destinations = set()
            for job in args.jobs:
                destination = job.resolve()
                if destination in destinations:
                    raise ValueError(f"Repeated project directory: {job}")
                destinations.add(destination)
            completed = 0
            for number, job in enumerate(args.jobs, 1):
                label = f"[{number}/{len(args.jobs)}] {job}"
                stage = "notes"
                try:
                    if batch:
                        print(f"{label}: {stage}", flush=True)
                    with diagnostic_run(job, "notes"):
                        result = generate_notes(job, args.state_dir, imported=args.imported,
                                                manual=args.manual, rebuild=args.rebuild,
                                                proxy=args.openai_proxy)
                        if (not args.manual and (batch or args.rebuild)
                                and result.resolve() != (job / "notes" / "main.tex").resolve()):
                            raise RuntimeError(f"Notes require manual completion: {result}")
                        if args.rebuild and not args.manual:
                            from .rendering import render
                            stage = "render"
                            if batch:
                                print(f"{label}: {stage}", flush=True)
                            result = render(job, rebuild=True)
                except (OSError, RuntimeError, ValueError, subprocess.SubprocessError,
                        ModuleNotFoundError) as exc:
                    if not batch:
                        raise
                    print(f"{label}: {stage} failed: {exc}", file=sys.stderr, flush=True)
                    continue
                completed += 1
                print(f"{label}: completed — {result}" if batch else result, flush=True)
            if batch:
                print(f"Notes: {completed} completed, {len(args.jobs) - completed} failed.",
                      flush=True)
            return 0 if completed == len(args.jobs) else 1
        elif args.command == "render":
            from .rendering import render
            print(render(args.job))
        elif args.command == "review":
            from .jobs import check_job, read_json, validate_evidence
            check_job(args.job)
            validate_evidence(read_json(args.job / "evidence.json"))
            if args.resolve:
                if not args.resolution:
                    raise ValueError("--resolve requires --resolution describing your correction")
                from .notes import resolve_issue
                resolve_issue(args.job, args.resolve, args.resolution)
            path = args.job / "review.json"
            if not path.exists():
                path = args.job / "notes.json"
            data = read_json(path) if path.exists() else {"issues": []}
            issues = data.get("issues", [])
            for issue in issues:
                print(f"[{issue['status']}] {issue['id']}: {issue['description']}")
                if issue.get("resolution"):
                    print(f"  Correction: {issue['resolution']}")
            print(f"{sum(i['status'] == 'open' for i in issues)} unresolved issue(s)")
        elif args.command == "anki":
            from .anki import generate_anki
            from .jobs import read_json
            if len(args.jobs) == 1:
                print(generate_anki(args.jobs[0], args.state_dir, pdf=args.pdf,
                                    rebuild=args.rebuild, proxy=args.openai_proxy,
                                    anki_collection=args.anki_collection))
                return 0
            if args.pdf is not None:
                raise ValueError("--pdf requires a single project directory")
            destinations = set()
            for job in args.jobs:
                destination = job.resolve()
                if destination in destinations:
                    raise ValueError(f"Repeated project directory: {job}")
                destinations.add(destination)
            completed, next_position = 0, 1
            for number, job in enumerate(args.jobs, 1):
                label = f"[{number}/{len(args.jobs)}] {job}"
                try:
                    print(f"{label}: anki", flush=True)
                    package = generate_anki(job, args.state_dir, pdf=None,
                                            rebuild=args.rebuild, proxy=args.openai_proxy,
                                            anki_collection=args.anki_collection,
                                            minimum_position=next_position)
                    next_position = read_json(job / "anki" / "report.json")["end_position"] + 1
                except (OSError, RuntimeError, ValueError, subprocess.SubprocessError,
                        ModuleNotFoundError) as exc:
                    print(f"{label}: anki failed: {exc}", file=sys.stderr, flush=True)
                    continue
                completed += 1
                print(f"{label}: completed — {package}", flush=True)
            print(f"Anki: {completed} completed, {len(args.jobs) - completed} failed.", flush=True)
            return 0 if completed == len(args.jobs) else 1
        elif args.command == "run":
            from .pipeline import run_pipeline
            completed = run_pipeline(args.media, args.out, args.state_dir,
                                     language=args.language, start=args.start,
                                     duration=args.duration, vision_model=args.vision_model,
                                     vision_mode=args.vision_mode,
                                     rebuild=args.rebuild, proxy=args.openai_proxy,
                                     anki_collection=args.anki_collection)
            return 0 if completed else 1
        return 0
    except ModuleNotFoundError as exc:
        print(f"Missing dependency {exc.name}. Run: UV_CACHE_DIR=.cache/uv uv sync --extra local", file=sys.stderr)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
    return 1
