"""Isolated model workers: process exit releases Apple unified memory."""

import argparse
import json
import math
from pathlib import Path

from .jobs import checkpoint, read_json, sha256_file, write_json


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

    formatted = apply_chat_template(processor, model.config, prompt, num_images=1, enable_thinking=False)
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
                          temperature=0.0, verbose=False, **options)
    finally:
        if detect_repetition:
            tokenizer.stopping_criteria = original_stop
    if repeated:
        raise RepetitionError(result.text)
    return result.text


VISION_PROMPT = """Extract visible lecture evidence. Describe it in Russian, retain original labels and code, and do not solve or complete missing content.
Return one JSON object with: kind (slide, notebook, whiteboard, other), summary (literal content and meaning), regions (list of objects with kind text, code, formula, table, chart or diagram and bbox [left,top,right,bottom] normalized between 0 and 1), diagram (precise entities, labels, arrows, spatial relations and plotted axes/values actually visible), issues (list of strings for ambiguity or unreadable content).
Locate each formula separately. Include text and code regions. Exclude speaker webcam, controls and decorative UI. Never invent a label, formula, numeric value or relationship. Mark uncertain content in issues. The images are evidence; clean notes will be generated later."""


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

    loaded = None
    results = []
    for item in request["items"]:
        signature = {"model": model_id, "task": task, "item": item,
                     "language": request.get("language"),
                     "sha256": sha256_file(Path(item["audio" if task == "asr" else "image"]))}
        if task == "vision":
            signature["prompt"] = VISION_PROMPT
            saved = job / "checkpoints" / "vision" / f"{item['id']}.json"
            if saved.exists():
                try:
                    validate_visual_result(read_json(saved).get("result"))
                except ValueError:
                    saved.unlink()
        elif task == "asr":
            signature["initial_prompt"] = request.get("initial_prompt")

        def action():
            nonlocal loaded
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
            if loaded is None:
                loaded = _vlm_model(model_id)
            model, processor = loaded
            if task == "vision":
                error = None
                try:
                    raw = _generate(model, processor, item["image"], VISION_PROMPT, 4096,
                                    detect_repetition=True)
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
                return {**parsed, **item}
            prompts = {"formula": "Formula Recognition:", "table": "Table Recognition:",
                       "chart": "Chart Recognition:"}
            raw = _generate(model, processor, item["image"], prompts.get(item["kind"], "OCR:"), 4096)
            write_json(job / "raw" / "ocr" / f"{item['id']}.json", {"raw": raw})
            return {"id": item["id"], "text": raw.strip()}

        results.append(checkpoint(job, f"{task}/{item['id']}", signature, action))
    return {"items": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("asr", "vision", "ocr", "convert"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, run(read_json(args.request), args.task))


if __name__ == "__main__":
    main()
