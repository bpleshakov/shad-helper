"""Run a fixed visual holdout and qualify models from explicit human reviews."""

import argparse
import gc
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

from .jobs import fingerprint, read_json, sha256_file, write_json
from .local_worker import has_repetition, parse_model_json, validate_visual_result
from .sparse_worker import (
    ANALYSIS_PROMPT,
    ROUTING_PROMPT,
    SETTINGS,
    _load,
    _stream,
    _validate_route,
)

MODELS = ("mlx-community/Qwen3.8-27B-4bit", "mlx-community/Qwen3.5-9B-4bit",
          "mlx-community/Qwen3.5-4B-MLX-4bit")
GENERATION = {**SETTINGS, "max_tokens": 1024}
ROUTING = {**SETTINGS, "max_tokens": 128, "max_pixels": 262144}
METRICS = ("prompt_tokens", "generation_tokens", "prompt_tps", "generation_tps", "peak_memory")


def _cases(manifest):
    manifest = Path(manifest)
    cases = read_json(manifest).get("cases", [])
    if (not isinstance(cases, list) or len(cases) != 40
            or any(not isinstance(case, dict) or not isinstance(case.get("id"), str)
                   or not case["id"] or not isinstance(case.get("image"), str)
                   or not case["image"] for case in cases)
            or len({case["id"] for case in cases}) != 40
            or Counter(case.get("split") for case in cases) != {"tuning": 20, "holdout": 20}):
        raise ValueError("Benchmark requires 40 unique case IDs: 20 tuning and 20 holdout images.")
    return [{"id": case["id"], "split": case["split"],
             "image": str((manifest.parent / case["image"]).resolve()),
             "sha256": sha256_file(manifest.parent / case["image"])} for case in cases]


def _release():
    gc.collect()
    mlx = sys.modules.get("mlx.core")
    if mlx is not None:
        mlx.clear_cache()


def _infer(loaded, prompt, image, settings, validate, failure=None):
    raw, last, parsed, error, stream = "", None, None, failure, None
    started = time.monotonic()
    if loaded is not None:
        try:
            stream = _stream(*loaded, prompt, image, settings)
            for response in stream:
                raw += response.text
                last = response
                if has_repetition(raw):
                    raise ValueError("Repeated visual output")
            if getattr(last, "finish_reason", None) != "stop":
                raise ValueError("Visual response did not finish naturally")
            parsed = validate(raw)
        except (RuntimeError, ValueError) as generation_error:
            error = str(generation_error)
        finally:
            if stream is not None and hasattr(stream, "close"):
                stream.close()
    telemetry = {name: float(value) for name in METRICS
                 if isinstance(value := getattr(last, name, None), (int, float))
                 and math.isfinite(value)}
    return {"raw": raw, "parsed": parsed, "valid": error is None, "error": error,
            "seconds": time.monotonic() - started,
            "finish_reason": getattr(last, "finish_reason", None), "telemetry": telemetry}


def run(manifest, out, models=MODELS):
    """Write an incremental audit; JSON validity alone never qualifies a model."""
    models = tuple(models)
    if len(models) != 3 or len(set(models)) != 3:
        raise ValueError("Provide one baseline and two distinct candidate models.")
    cases = _cases(manifest)
    inputs = {"version": 1, "cases": cases, "models": list(models),
              "prompt": ANALYSIS_PROMPT, "generation": GENERATION,
              "routing_prompt": ROUTING_PROMPT, "routing_generation": ROUTING}
    report = {**inputs, "benchmark_fingerprint": fingerprint(inputs), "results": [], "loads": []}
    path = Path(out) / "report.json"
    for model_id in models:
        loaded, failure = None, None
        started = time.monotonic()
        try:
            loaded = _load(model_id)
        except (RuntimeError, ValueError) as error:
            failure = str(error)
        report["loads"].append({"model": model_id, "seconds": time.monotonic() - started,
                                "error": failure})
        write_json(path, report)
        for case in cases:
            routing = _infer(loaded, ROUTING_PROMPT, case["image"], ROUTING, _validate_route, failure)
            analysis = _infer(loaded, ANALYSIS_PROMPT, case["image"], GENERATION,
                              lambda raw: validate_visual_result(parse_model_json(raw)), failure)
            report["results"].append({**analysis, "model": model_id, "case_id": case["id"],
                                      "split": case["split"], "case_sha256": case["sha256"],
                                      "routing": routing, "analysis_seconds": analysis["seconds"],
                                      "seconds": routing["seconds"] + analysis["seconds"]})
            write_json(path, report)
        loaded = None
        _release()
    return path


def _pilot_passed(pilot, primary, benchmark_fingerprint):
    if not isinstance(pilot, dict):
        return False
    required = ("video_seconds", "baseline_frame_count", "exported_frame_count", "qwen_seconds",
                "deferred_count", "failed_count", "final_content_omissions")
    if (pilot.get("reviewed") is not True
            or pilot.get("primary_model") != primary
            or pilot.get("benchmark_fingerprint") != benchmark_fingerprint
            or any(isinstance(pilot.get(name), bool) or not isinstance(pilot.get(name), (int, float))
                   or not math.isfinite(pilot[name]) or pilot[name] < 0 for name in required)):
        return False
    if any(type(pilot[name]) is not int for name in required
           if name not in {"video_seconds", "qwen_seconds"}):
        return False
    return (pilot["video_seconds"] >= 3600 and pilot["baseline_frame_count"] > 0
            and pilot["exported_frame_count"] > 0
            and pilot["exported_frame_count"] <= pilot["baseline_frame_count"] * 0.1
            and pilot["qwen_seconds"] <= pilot["video_seconds"] * 0.5
            and pilot["deferred_count"] == pilot["failed_count"] == pilot["final_content_omissions"] == 0)


def _parsed_result(result, *, routing=False):
    try:
        value = (_validate_route(result["raw"]) if routing else
                 validate_visual_result(parse_model_json(result["raw"])))
        return value == result.get("parsed")
    except (ValueError, TypeError, KeyError):
        return False


def qualify(report_path, reviews_path, out, pilot_path=None):
    """Require every held-out image to be reviewed before choosing a candidate."""
    report, reviews = read_json(report_path), read_json(reviews_path).get("reviews", [])
    models, cases = report.get("models", []), report.get("cases", [])
    if (not isinstance(models, list) or len(models) != 3
            or any(not isinstance(model, str) for model in models) or len(set(models)) != 3
            or not isinstance(cases, list) or len(cases) != 40
            or any(not isinstance(case, dict) or not isinstance(case.get("id"), str) for case in cases)
            or len({case.get("id") for case in cases}) != 40
            or Counter(case.get("split") for case in cases) != {"tuning": 20, "holdout": 20}
            or not isinstance(reviews, list) or any(not isinstance(x, dict) for x in reviews)):
        raise ValueError("Invalid benchmark report or human review manifest.")
    inputs = {key: report[key] for key in ("version", "cases", "models", "prompt", "generation",
                                         "routing_prompt", "routing_generation")}
    if fingerprint(inputs) != report.get("benchmark_fingerprint"):
        raise ValueError("Benchmark fingerprint does not match its cases, models, or settings.")
    holdout = {case["id"]: case for case in cases if case["split"] == "holdout"}
    passed, means = {}, {}
    for model_id in models:
        results = [x for x in report.get("results", []) if x.get("model") == model_id
                   and x.get("case_id") in holdout]
        human = [x for x in reviews if x.get("model") == model_id and x.get("case_id") in holdout]
        complete = (len(results) == len(human) == 20
                    and {x["case_id"] for x in results} == set(holdout)
                    and {x["case_id"] for x in human} == set(holdout))
        valid = complete and all(
            x.get("valid") is True and x.get("finish_reason") == "stop"
            and _parsed_result(x)
            and x.get("routing", {}).get("valid") is True
            and x["routing"].get("finish_reason") == "stop"
            and _parsed_result(x["routing"], routing=True)
            and x.get("case_sha256") == holdout[x["case_id"]].get("sha256")
            and isinstance(x.get("seconds"), (int, float)) and not isinstance(x["seconds"], bool)
            and math.isfinite(x["seconds"]) and x["seconds"] >= 0 for x in results)
        reviewed = complete and all(x.get("reviewed") is True and x.get("routing_correct") is True and all(
            type(x.get(name)) is int and x[name] == 0 for name in
            ("additional_critical_errors", "missed_regions", "missing_final_content")) for x in human)
        passed[model_id] = bool(valid and reviewed)
        means[model_id] = sum(x["seconds"] for x in results) / 20 if valid else None
    candidates = [model for model in models[1:] if passed[model] and passed[models[0]]]
    primary = min(candidates, key=lambda model: means[model]) if candidates else models[0]
    pilot = read_json(pilot_path) if pilot_path else None
    profile = {"version": 1, "qualified": passed[primary], "primary_model": primary,
               "smaller_model_qualified": bool(candidates),
               "benchmark_fingerprint": report["benchmark_fingerprint"],
               "sparse_default": passed[primary] and _pilot_passed(
                   pilot, primary, report["benchmark_fingerprint"]),
               "holdout_passed": passed, "holdout_mean_seconds": means,
               "pilot": pilot, "review_fingerprint": fingerprint(reviews)}
    write_json(out, profile)
    return profile


def main(argv=None):
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("cases", type=Path)
    runner.add_argument("--out", type=Path, required=True)
    runner.add_argument("--models", nargs=3, default=MODELS)
    gate = commands.add_parser("qualify")
    gate.add_argument("report", type=Path)
    gate.add_argument("reviews", type=Path)
    gate.add_argument("--out", type=Path, default=Path(".shad-helper/vision-profile.json"))
    gate.add_argument("--pilot", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        print(run(args.cases, args.out, args.models))
    else:
        print(qualify(args.report, args.reviews, args.out, args.pilot)["primary_model"])


if __name__ == "__main__":
    main()
