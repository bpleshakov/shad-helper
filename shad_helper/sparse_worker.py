"""Analyze a sparse evidence queue with a shared, resumable Qwen allowance."""

import gc
import time
from collections import defaultdict
from pathlib import Path

from .jobs import fingerprint, read_json, sha256_file, write_json
from .vision_budget import Budget, BudgetExhausted, deferred

ROUTING_PROMPT = """Identify the lecture medium, not its content. Return JSON only: kind (whiteboard, slide, notebook, other), lecture_surface_bbox (normalized [left,top,right,bottom] enclosing the whole physical board or lecture screen, or null), lecturer_bbox (live person standing before a physical board only, or null), issues (brief uncertainty strings). Ignore text transcription. Blank or transition images are other. Never infer a board from blank pixels."""
ANALYSIS_PROMPT = """Extract visible lecture evidence in Russian. Return JSON only: kind (slide, notebook, whiteboard, other), summary (at most one short sentence identifying the subject, without copying visible text), regions (kind text, code, formula, table, chart or diagram; normalized bbox [left,top,right,bottom]), diagram (compact visible entities, labels, arrows, spatial relations, axes and values, or null), issues (brief ambiguity or unreadability strings). Group adjacent ordinary text into a few regions. Preserve meaningful diagram relations. Do not transcribe ordinary text or solve formulas: a separate OCR stage reads regions. Do not invent missing content."""
SETTINGS = {"temperature": 0.0, "enable_thinking": False}


def _load(model):
    from .local_worker import _vlm_model

    return _vlm_model(model)


def _stream(model, processor, prompt, image, settings):
    from mlx_vlm import stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.stopping_criteria.reset(model.config.eos_token_id)
    formatted = apply_chat_template(processor, model.config, prompt, num_images=1,
                                    enable_thinking=False)
    return stream_generate(model, processor, formatted, image=[image], **settings)


def _validate_route(raw):
    from .local_worker import parse_model_json

    value = parse_model_json(raw)
    if value.get("kind") not in ("slide", "notebook", "whiteboard", "other"):
        raise ValueError("Invalid routing kind")
    if not isinstance(value.get("issues"), list) or any(not isinstance(x, str) for x in value["issues"]):
        raise ValueError("Invalid routing issues")
    for name in ("lecture_surface_bbox", "lecturer_bbox"):
        if name not in value:
            raise ValueError("Missing routing box")
        box = value[name]
        if box is not None and (not isinstance(box, list) or len(box) != 4
                               or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                      or not 0 <= x <= 1 for x in box)
                               or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError("Invalid routing box")
    return {key: value[key] for key in ("kind", "lecture_surface_bbox", "lecturer_bbox", "issues")}


def _ordered(items):
    buckets = defaultdict(list)
    for item in items:
        timestamp = item.get("timestamp")
        buckets[int(timestamp // 300) if timestamp is not None else -1].append(item)
    groups = [buckets[key] for key in sorted(buckets)]
    return [group[index] for index in range(max((len(group) for group in groups), default=0))
            for group in groups if index < len(group)]


def _primary_model(request, job):
    if request.get("primary_model"):
        return request["primary_model"]
    local = job / "sparse-model-profile.json"
    for path in (local, Path(".shad-helper/vision-profile.json")):
        if path.exists():
            profile = read_json(path)
            if (type(profile.get("version")) is int and profile["version"] == 1
                    and isinstance(profile.get("primary_model"), str) and profile["primary_model"]):
                allowed = (profile.get("qualified") is True
                           or path == local and profile["primary_model"] == request["model"])
                if allowed:
                    if path != local:
                        write_json(local, profile)
                    return profile["primary_model"]
    write_json(local, {"version": 1, "qualified": False, "primary_model": request["model"],
                       "reason": "Reference model retained until a smaller model is qualified."})
    return request["model"]


def run_sparse(request):
    from .local_worker import has_repetition, parse_model_json, validate_visual_result
    from .person_segmentation import PersonSegmenter
    from .sparse_pipeline import prepare_sparse_images

    if type(request.get("sparse_selection_version")) is not int or request["sparse_selection_version"] != 1:
        raise ValueError("Unsupported sparse selection version; use a new output directory.")
    job = Path(request["job"])
    budget = Budget(job, request)
    primary = _primary_model(request, job)
    profile_path = job / "sparse-model-profile.json"
    model_profile = read_json(profile_path) if profile_path.exists() else {}
    fallback_model = request["model"]
    loaded, loaded_id = None, None
    calls = []
    generated = 0
    cache_hits = 0

    def infer(item, model_id, *, routing=False):
        nonlocal loaded, loaded_id, generated, cache_hits
        from .local_worker import _pixel_signature

        prompt = ROUTING_PROMPT if routing else ANALYSIS_PROMPT
        settings = {**SETTINGS, "max_tokens": 128 if routing else 1024}
        if routing:
            settings["max_pixels"] = 262144
        inputs = {"model": model_id, "prompt": prompt, "settings": settings,
                  "image": _pixel_signature(item["image"]), "version": 1}
        key = fingerprint(inputs)
        path = job / "checkpoints" / "sparse-content" / f"{key}.json"
        def validate(raw):
            if routing:
                return _validate_route(raw)
            value = validate_visual_result(parse_model_json(raw))
            return {name: value[name] for name in ("kind", "summary", "regions", "diagram", "issues")}
        if path.exists():
            cached = read_json(path)
            try:
                result = validate(cached["raw"])
            except (ValueError, TypeError, KeyError):
                path.unlink()
            else:
                cache_hits += 1
                return result
        budget.check()
        if loaded_id != model_id:
            loaded = None
            gc.collect()
            started = time.monotonic()
            load_error = None
            try:
                with budget.active("model_load"):
                    loaded = _load(model_id)
                    loaded_id = model_id
            except (RuntimeError, ValueError) as error:
                load_error = str(error)
                raise
            finally:
                calls.append({"id": item["id"], "model": model_id, "phase": "model_load",
                              "seconds": time.monotonic() - started, "error": load_error})
        raw = ""
        last = None
        finished = False
        failure = None
        started = time.monotonic()
        try:
            with budget.active("routing" if routing else "analysis"):
                generated += 1
                stream = _stream(*loaded, prompt, item["image"], settings)
                try:
                    for response in stream:
                        budget.check()
                        raw += response.text
                        last = response
                        if has_repetition(raw):
                            raise ValueError("Repeated visual output")
                    finished = getattr(last, "finish_reason", None) == "stop"
                finally:
                    close = getattr(stream, "close", None)
                    if close is not None:
                        close()
            if not finished:
                raise ValueError("Visual output did not finish before its token limit")
            result = validate(raw)
        except (BudgetExhausted, ValueError, RuntimeError) as error:
            failure = str(error)
            raise
        finally:
            telemetry = {"id": item["id"], "model": model_id, "routing": routing,
                         "seconds": time.monotonic() - started, "error": failure}
            if last is not None:
                telemetry.update({name: getattr(last, name, None) for name in
                                  ("prompt_tokens", "generation_tokens", "prompt_tps",
                                   "generation_tps", "peak_memory", "finish_reason")})
            calls.append(telemetry)
            write_json(job / "raw" / "sparse" / f"{key}.json", {"raw": raw, "telemetry": telemetry})
        write_json(path, {"inputs": inputs, "raw": raw, "result": result})
        return result

    def route(timestamp, image):
        try:
            return infer({"id": f"route_{timestamp:.6f}", "image": image}, primary, routing=True)
        except BudgetExhausted:
            raise
        except (RuntimeError, ValueError) as error:
            return {"kind": "other", "lecture_surface_bbox": None, "lecturer_bbox": None,
                    "issues": [f"Routing unavailable: {error}"]}

    native_issue = None
    try:
        segmenter = PersonSegmenter(allow_empty_masks=True)
    except RuntimeError as error:
        segmenter = None
        native_issue = str(error)
    selected, audit = prepare_sparse_images({**request, "primary_model": primary}, route, segmenter)
    if native_issue:
        audit["native_segmentation_issue"] = native_issue
    all_items = [*selected, *request.get("items", [])]
    outputs = {}
    results_path = job / "sparse-vision-results.json"

    def persist():
        audit.update(qwen_generated_calls=generated, qwen_cache_hits=cache_hits,
                     qwen_calls=calls, qwen_budget=budget.summary(), primary_model=primary,
                     benchmark_fingerprint=model_profile.get("benchmark_fingerprint"),
                     budget_exhausted=budget.remaining <= 0)
        result = {"items": [outputs.get(item["id"], deferred(item, "Visual analysis is pending."))
                            for item in all_items], "audit": audit}
        write_json(results_path, result)
        return result

    def process(item, model_id):
        try:
            analysis = infer(item, model_id)
        except BudgetExhausted:
            outputs[item["id"]] = deferred(item)
            return False
        except (RuntimeError, ValueError) as error:
            outputs.setdefault(item["id"], {**deferred(item, f"Visual analysis failed: {error}"),
                                             "status": "failed"})
            return False
        for field in ("id", "image", "timestamp", "page_id", "page_start", "page_end", "status",
                      "selection_issues", "selection_role", "complement_of"):
            analysis.pop(field, None)
        outputs[item["id"]] = {**analysis, **item, "status": "processed", "analysis_model": model_id,
                               "sha256": sha256_file(Path(item["image"])),
                               "issues": list(dict.fromkeys([
                                   *analysis["issues"], *item.get("selection_issues", [])]))}
        return not analysis["issues"]

    main = _ordered([item for item in all_items if item.get("selection_role") not in ("complement", "complementary")])
    complements = _ordered([item for item in all_items if item.get("selection_role") in ("complement", "complementary")])
    persist()
    escalations = []
    for item in main:
        okay = process(item, primary)
        if not okay and outputs[item["id"]]["status"] != "deferred" and primary != fallback_model:
            escalations.append(item)
        persist()
    # Release the primary model once before a single bounded fallback batch.
    for item in escalations:
        process(item, fallback_model)
        persist()
    for item in complements:
        process(item, primary)
        persist()
    return persist()
