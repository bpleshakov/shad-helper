"""Isolated model workers: process exit releases Apple unified memory."""

import argparse
import json
import math
import time
from pathlib import Path

from .jobs import checkpoint, read_json, sha256_file, write_json
from .vision_mode import GROUPING_VERSION, PAGE_SELECTION_VERSION

GENERATION_SETTINGS = {"max_tokens": 4096, "temperature": 0.0, "enable_thinking": False}
OCR_PROMPTS = {"formula": "Formula Recognition:", "table": "Table Recognition:",
               "chart": "Chart Recognition:"}


def parse_model_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("<think>"):
        _, separator, text = text.partition("</think>")
        if not separator:
            raise ValueError("The visual model returned unfinished reasoning.")
        text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if (len(lines) < 3 or lines[0].strip().lower() not in {"```", "```json"}
                or lines[-1].strip() != "```"):
            raise ValueError("The visual model returned an incomplete JSON fence.")
        text = "\n".join(lines[1:-1]).strip()

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("The visual model returned duplicate JSON keys.")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("The visual model returned a nonfinite JSON number.")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("The visual model returned a nonfinite JSON number.")
        return number

    value = json.loads(text, object_pairs_hook=unique_object,
                       parse_constant=invalid_constant, parse_float=finite_float)
    if not isinstance(value, dict):
        # ValueError consistently identifies invalid model output for the worker.
        raise ValueError("The visual model did not return a complete JSON object.")  # noqa: TRY004
    return value


def validate_visual_result(value: dict) -> dict:
    """Require the full lecture record, never a nested region mistaken for it."""
    required = {"kind", "summary", "regions", "diagram", "issues"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError("The visual model omitted required lecture fields.")
    if (value["kind"] not in ("slide", "notebook", "whiteboard", "other")
            or not isinstance(value["summary"], str)
            or not isinstance(value["regions"], list)
            or not isinstance(value["diagram"], (str, dict, type(None)))
            or not isinstance(value["issues"], list)
            or any(not isinstance(issue, str) for issue in value["issues"])):
        raise ValueError("The visual model returned invalid lecture fields.")
    for region in value["regions"]:
        if (not isinstance(region, dict)
                or region.get("kind") not in ("text", "code", "formula", "table", "chart", "diagram")):
            raise ValueError("The visual model returned an invalid region.")
        bbox = region.get("bbox")
        if (not isinstance(bbox, list) or len(bbox) != 4
                or any(isinstance(number, bool) or not isinstance(number, (int, float))
                       or not 0 <= number <= 1 for number in bbox)
                or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]):
            raise ValueError("The visual model returned an invalid normalized region box.")
    for name in ("lecture_surface_bbox", "lecturer_bbox"):
        bbox = value.get(name)
        if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4
                or any(isinstance(number, bool) or not isinstance(number, (int, float))
                       or not 0 <= number <= 1 for number in bbox)
                or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]):
            raise ValueError(f"The visual model returned an invalid {name}.")
    return value


def has_repetition(text: str) -> bool:
    """Detect long consecutive loops while permitting numeric tables and spacing."""
    tail = text[-1024:]
    for width in range(1, 129):
        repeats = max(8, (256 + width - 1) // width)
        length = width * repeats
        if length <= len(tail):
            unit = tail[-width:]
            if any(character.isalpha() for character in unit) and tail[-length:] == unit * repeats:
                return True
    return False


class RepetitionError(ValueError):
    def __init__(self, raw: str):
        super().__init__("Visual generation stopped after consecutive repeated output.")
        self.raw = raw


def _vlm_model(model_id: str):
    try:
        from mlx_vlm import load
    except ImportError as error:
        raise RuntimeError("Local vision/OCR requires the 'local' extra: uv sync --extra local.") from error
    return load(model_id, trust_remote_code=True)


def _generate(model, processor, image: str, prompt: str, max_tokens: int, *, detect_repetition=False) -> str:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    formatted = apply_chat_template(processor, model.config, prompt, num_images=1,
                                    enable_thinking=GENERATION_SETTINGS["enable_thinking"])
    options = {}
    repeated = False
    if detect_repetition:
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        original_stop = tokenizer.stopping_criteria
        original_stop.reset(model.config.eos_token_id)
        tokens = []

        def stop(token):
            nonlocal repeated
            if original_stop(token):
                return True
            tokens.append(int(token))
            if len(tokens) >= 64 and has_repetition(tokenizer.decode(tokens[-1024:])):
                repeated = True
                return True
            return False

        options["stopping_criteria"] = stop
    try:
        result = generate(model, processor, formatted, image=[image], max_tokens=max_tokens,
                          temperature=GENERATION_SETTINGS["temperature"], verbose=False, **options)
    finally:
        if detect_repetition:
            tokenizer.stopping_criteria = original_stop
    if repeated:
        raise RepetitionError(result.text)
    return result.text


VISION_PROMPT = """Extract visible lecture evidence. Describe it in Russian, retain original labels and code, and do not solve or complete missing content.
Return one JSON object with: kind (slide, notebook, whiteboard, other), summary (literal content and meaning), regions (list of objects with kind text, code, formula, table, chart or diagram and bbox [left,top,right,bottom] normalized between 0 and 1), diagram (precise entities, labels, arrows, spatial relations and plotted axes/values actually visible), issues (list of strings for ambiguity or unreadable content).
Locate each formula separately. Include text and code regions. Exclude speaker webcam, controls and decorative UI. Never invent a label, formula, numeric value or relationship. Mark uncertain content in issues. The images are evidence; clean notes will be generated later."""

CONTENT_VISION_PROMPT = VISION_PROMPT + """
Also return lecture_surface_bbox and lecturer_bbox, each null or a normalized [left,top,right,bottom] box.
lecture_surface_bbox must enclose the WHOLE physical whiteboard or lecture application/screen, including blank space where future content could appear, never just the currently written regions. Include every lecture surface if several are visible; use null when uncertain.
lecturer_bbox is only for a real live lecturer standing in front of a physical whiteboard. Never mark people depicted in slides, drawings, notebook outputs, or photographs. Use null when absent or uncertain. Do not exclude any lecture content from lecture_surface_bbox."""


def _pixel_signature(path: str) -> dict:
    import hashlib

    from PIL import Image, ImageOps

    with Image.open(path) as image:
        pixels = ImageOps.exif_transpose(image).convert("RGB")
        return {"size": list(pixels.size), "pixels": hashlib.sha256(pixels.tobytes()).hexdigest()}


def run(request: dict, task: str) -> dict:
    job = Path(request["job"])
    model_id = request["model"]
    if task == "convert":
        try:
            from mlx_vlm import convert
        except ImportError as error:
            raise RuntimeError("Local OCR conversion requires: uv sync --extra local.") from error
        destination = Path(request["destination"])
        marker = destination / "shad_conversion.json"
        if (not marker.exists() or read_json(marker).get("source") != model_id
                or not (destination / "config.json").exists()
                or not any(destination.glob("*.safetensors"))):
            convert(hf_path=model_id, mlx_path=str(destination), dtype="bfloat16", trust_remote_code=True)
            write_json(marker, {"source": model_id, "dtype": "bfloat16"})
        return {"model": str(destination.resolve())}

    pages = request.get("vision_mode") == "pages" and task in {"vision", "ocr"}
    content = request.get("vision_mode") in {"content", "pages"} and task in {"vision", "ocr"}
    if content and request.get("grouping_version", GROUPING_VERSION) != GROUPING_VERSION:
        raise ValueError("Unsupported visual grouping version; use a new output directory.")
    if pages and (type(request.get("page_selection_version")) is not int
                  or request["page_selection_version"] != PAGE_SELECTION_VERSION):
        raise ValueError("Unsupported page selection version; use a new output directory.")
    prompt = CONTENT_VISION_PROMPT if content else VISION_PROMPT
    loaded = None
    generated = 0
    started = time.monotonic()

    def generate(item, instruction, *, detect_repetition=False):
        nonlocal loaded, generated
        if loaded is None:
            loaded = _vlm_model(model_id)
        generated += 1
        options = {"detect_repetition": True} if detect_repetition else {}
        return _generate(*loaded, item["image"], instruction, GENERATION_SETTINGS["max_tokens"], **options)

    def process(item):
        signature = {"model": model_id, "task": task, "item": item,
                     "language": request.get("language"),
                     "sha256": sha256_file(Path(item["audio" if task == "asr" else "image"]))}
        if task == "vision":
            signature["prompt"] = prompt
            if content:
                signature["grouping_version"] = GROUPING_VERSION
            saved = job / "checkpoints" / "vision" / f"{item['id']}.json"
            if saved.exists():
                try:
                    validate_visual_result(read_json(saved).get("result"))
                except (ValueError, AttributeError, TypeError):
                    saved.unlink()
        elif task == "asr":
            signature["initial_prompt"] = request.get("initial_prompt")
        instruction = OCR_PROMPTS.get(item.get("kind"), "OCR:")
        if content:
            signature["generation"] = GENERATION_SETTINGS
            if task == "ocr":
                signature["prompt"] = instruction
                saved = job / "checkpoints" / "ocr" / f"{item['id']}.json"
                if saved.exists():
                    try:
                        cached = read_json(saved).get("result")
                        if (not isinstance(cached, dict) or not isinstance(cached.get("text"), str)
                                or not cached["text"].strip()):
                            raise ValueError("Unreadable OCR checkpoint")
                    except (ValueError, AttributeError, TypeError):
                        saved.unlink()

        def action():
            if task == "asr":
                try:
                    import mlx_whisper
                except ImportError as error:
                    raise RuntimeError("Local transcription requires: uv sync --extra local.") from error
                result = mlx_whisper.transcribe(
                    item["audio"], path_or_hf_repo=model_id, language=request["language"],
                    task="transcribe", initial_prompt=request.get("initial_prompt"),
                    word_timestamps=True, verbose=False,
                )
                return {**item, "segments": result["segments"], "language": result.get("language")}
            if task == "vision":
                error = None
                try:
                    raw = generate(item, prompt, detect_repetition=True)
                except RepetitionError as failure:
                    raw, error = failure.raw, "repetition"
                artifact = {"raw": raw}
                try:
                    if error:
                        raise ValueError("Repeated output")
                    parsed = validate_visual_result(parse_model_json(raw))
                except ValueError:
                    artifact["error"] = error or "invalid_json"
                    reason = ("Визуальная модель остановлена из-за повторения текста." if error
                              else "Визуальная модель не вернула корректную структуру.")
                    parsed = {"kind": "other", "summary": "", "diagram": "", "regions": [],
                              "issues": [reason + " Полное изображение передано OCR; смысл схем требует ручной проверки."]}
                write_json(job / "raw" / "vision" / f"{item['id']}.json", artifact)
                if pages:
                    # Timing comes from the selector, never from a model's extra JSON keys.
                    for key in ("page_id", "page_start", "page_end", "selection_issues"):
                        parsed.pop(key, None)
                    parsed["issues"].extend(item.get("selection_issues", []))
                return {**parsed, **item}
            if content:
                from .jobs import fingerprint

                cache_inputs = {"model": model_id, "prompt": instruction,
                                "generation": GENERATION_SETTINGS,
                                "image": _pixel_signature(item["image"]), "version": 1}
                key = f"ocr_content/{fingerprint(cache_inputs)}"
                saved = job / "checkpoints" / f"{key}.json"
                if saved.exists():
                    try:
                        cached = read_json(saved).get("result")
                        if (not isinstance(cached, dict) or not isinstance(cached.get("raw"), str)
                                or not cached["raw"].strip()):
                            raise ValueError("Unreadable shared OCR checkpoint")
                    except (ValueError, AttributeError, TypeError):
                        saved.unlink()

                def recognize():
                    return {"raw": generate(item, instruction)}

                recognized = checkpoint(job, key, cache_inputs, recognize)
                raw = recognized["raw"]
                if not raw.strip():
                    # An unreadable crop must get another opportunity on a later occurrence.
                    (job / "checkpoints" / f"{key}.json").unlink(missing_ok=True)
            else:
                raw = generate(item, instruction)
            write_json(job / "raw" / "ocr" / f"{item['id']}.json", {"raw": raw})
            return {"id": item["id"], "text": raw.strip()}

        if content and task == "ocr":
            signature["content_cache_version"] = 1
        result = checkpoint(job, f"{task}/{item['id']}", signature, action)
        if content and task == "ocr" and not result["text"].strip():
            (job / "checkpoints" / "ocr" / f"{item['id']}.json").unlink(missing_ok=True)
        return result

    if content and task == "vision":
        from .person_segmentation import PersonSegmenter
        from .visual_grouping import group_visuals

        video_items = []
        page_audit = None
        if pages:
            from .page_pipeline import fallback_page_items, prepare_page_images

            video_items, page_audit = prepare_page_images(
                request, process, PersonSegmenter(allow_empty_masks=True), prompt, GENERATION_SETTINGS,
            )
            page_audit["bootstrap_generated_calls"] = generated
            checked = []
            fallback_pages = []
            for item in video_items:
                if "page_id" in item:
                    analysis = process(item)
                    if analysis["kind"] != "whiteboard":
                        checked.extend(fallback_page_items(request, item, analysis))
                        fallback_pages.append(item["page_id"])
                        continue
                checked.append(item)
            video_items = checked
            page_audit["final_type_fallback_pages"] = fallback_pages
            page_audit["selected_video_frames"] = len(video_items)
            page_audit["exports"] = [{key: item[key] for key in
                                     ("id", "image", "timestamp", "page_id", "page_start", "page_end",
                                      "selection_issues") if key in item} for item in video_items]
        # Stateful native requests must start a new chronological sequence after selection.
        native = PersonSegmenter()
        mask_calls = 0

        def segment(rgb):
            nonlocal mask_calls
            mask_calls += 1
            return native(rgb)

        items, audit = group_visuals(video_items + request["items"], process, segment)
        audit.update(model=model_id, generated_calls=generated,
                     processing_seconds=time.monotonic() - started)
        write_json(job / "visual-grouping.json", audit)
        if pages:
            page_audit.update(qwen_generated_calls=generated,
                              grouping_native_mask_calls=mask_calls,
                              selection_native_mask_calls_this_run=(
                                  0 if page_audit["selection_reused"] else
                                  page_audit.get("segmentation_calls", 0)),
                              vision_seconds=time.monotonic() - started)
            write_json(job / "whiteboard-pages.json", page_audit)
        return {"items": items}
    items = [process(item) for item in request["items"]]
    if content:
        write_json(job / "ocr-reuse.json", {"items": len(items), "generated_calls": generated,
                                           "reused_results": len(items) - generated,
                                           "processing_seconds": time.monotonic() - started})
        if pages and (job / "whiteboard-pages.json").exists():
            audit = read_json(job / "whiteboard-pages.json")
            audit.update(ocr_generated_calls=generated, ocr_items=len(items),
                         ocr_seconds=time.monotonic() - started)
            write_json(job / "whiteboard-pages.json", audit)
    return {"items": items}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("asr", "vision", "ocr", "convert"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, run(read_json(args.request), args.task))


if __name__ == "__main__":
    main()
