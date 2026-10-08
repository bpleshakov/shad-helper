"""Raw GLM-OCR pilot: --request request.json --output response.json.

Items specify id, image and task (text or formula), with optional source metadata.
Only the model's documented recognition task prefix enters its prompt. Code and
comments are retained as OCR evidence; theory filtering is a separate experiment.
"""

import argparse
import os
import re
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from shad_helper.jobs import checkpoint, fingerprint, read_json, sha256_file, write_json

MODEL = "mlx-community/GLM-OCR-bf16"
TASK_PROMPTS = {"text": "Text Recognition:", "formula": "Formula Recognition:"}
MAX_IMAGE_PIXELS = 2_200_000


def _versions():
    result = {}
    for package in ("mlx-vlm", "mlx", "transformers", "Pillow"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


class _GLMRuntime:
    def __init__(self, model_id, revision):
        from mlx_vlm import load

        self.model, self.processor = load(model_id, revision=revision)
        if self.model.config.model_type != "glm_ocr":
            raise ValueError("This pilot requires a GLM-OCR model")
        processor = self.processor.image_processor
        # GLM's pixel budget includes duplicated temporal frames for each still image.
        # Per-call image-size kwargs are ignored by this adapter in MLX-VLM 0.7.6.
        processor.max_pixels = MAX_IMAGE_PIXELS * processor.temporal_patch_size
        processor.use_transformers_backend = False

    def infer(self, image, prompt, max_tokens):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens
        from PIL import Image

        started = time.monotonic()
        formatted = apply_chat_template(self.processor, self.model.config, prompt, num_images=1)
        inputs = prepare_inputs(
            self.processor, images=[image], prompts=formatted,
            image_token_index=getattr(self.model.config, "image_token_index", None),
            add_special_tokens=should_add_special_tokens(
                self.model.config.model_type, self.processor,
            ),
        )
        processor = self.processor.image_processor
        grid = inputs["image_grid_thw"].tolist()[0]
        processed_size = [int(grid[2] * processor.patch_size),
                          int(grid[1] * processor.patch_size)]
        with Image.open(image) as source:
            source_size = list(source.size)
        inputs["mask"] = inputs.pop("attention_mask", None)
        result = generate(self.model, self.processor, formatted, temperature=0.0,
                          max_tokens=max_tokens, verbose=False, **inputs)
        return result.text, {
            "seconds": time.monotonic() - started,
            "source_size": source_size, "processed_size": processed_size,
            "image_grid_thw": grid,
            "image_processor": {"patch_size": processor.patch_size,
                                "temporal_patch_size": processor.temporal_patch_size,
                                "merge_size": processor.merge_size,
                                "min_pixels": processor.min_pixels,
                                "max_pixels": processor.max_pixels,
                                "backend": "numpy_pil"},
            "prompt_tokens": result.prompt_tokens, "generation_tokens": result.generation_tokens,
            "prompt_tps": result.prompt_tps, "generation_tps": result.generation_tps,
            "peak_memory_gb": result.peak_memory, "finish_reason": result.finish_reason,
        }


class _IncompleteOCR(ValueError):
    def __init__(self, result):
        super().__init__("OCR response is empty or truncated")
        self.result = result


def run(request):
    started = time.monotonic()
    job = Path(request["job"])
    model, revision = request.get("model", MODEL), request["revision"]
    if not isinstance(revision, str) or not revision:
        raise ValueError("A pinned model revision is required")
    max_tokens = request.get("max_tokens", 4096)
    if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
        raise ValueError("max_tokens must be an integer between 1 and 4096")
    settings = {"max_tokens": max_tokens, "temperature": 0.0,
                "max_image_pixels": MAX_IMAGE_PIXELS, "image_processor_backend": "numpy_pil",
                "retry_count": 0}
    packages = _versions()
    runtime = None
    model_load_seconds = 0.0
    fresh_items = 0
    identifiers = set()

    for item in request["items"]:
        item_id = item["id"]
        if (not isinstance(item_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", item_id)
                or item_id in {".", ".."} or item_id in identifiers):
            raise ValueError("OCR item IDs must be unique safe filenames")
        identifiers.add(item_id)
        if item["task"] not in TASK_PROMPTS:
            raise ValueError("OCR task must be text or formula")

    def process(item):
        prompt = TASK_PROMPTS[item["task"]]
        signature = {"model": model, "revision": revision, "prompt": prompt,
                     "settings": settings, "packages": packages,
                     "sha256": sha256_file(Path(item["image"]))}

        def action():
            nonlocal runtime, model_load_seconds, fresh_items
            if runtime is None:
                load_started = time.monotonic()
                runtime = _GLMRuntime(model, revision)
                model_load_seconds = time.monotonic() - load_started
            fresh_items += 1
            raw, metrics = runtime.infer(item["image"], prompt, max_tokens)
            if (metrics.get("finish_reason") == "length"
                    or (not metrics.get("finish_reason")
                        and metrics.get("generation_tokens", 0) >= max_tokens)):
                status = "truncated"
            else:
                status = "complete" if raw.strip() else "empty"
            result = {"raw": raw, "status": status, "diagnostics": metrics}
            write_json(job / "diagnostics" / "ocr" / f"{item['id']}.json", {
                "fingerprint": fingerprint(signature), "inputs": signature, "item": item,
                **result,
            })
            if status != "complete":
                raise _IncompleteOCR(result)
            return result

        try:
            result = checkpoint(job, f"ocr/{item['id']}", signature, action)
        except _IncompleteOCR as error:
            result = error.result
        return {**item, **result}

    items = [process(item) for item in request["items"]]
    return {"items": items, "diagnostics": {
        "model": model, "revision": revision, "settings": settings, "packages": packages,
        "task_prompts": TASK_PROMPTS, "seconds": time.monotonic() - started,
        "model_load_seconds": model_load_seconds, "fresh_items": fresh_items,
        "cached_items": len(items) - fresh_items,
        "complete_items": sum(item["status"] == "complete" for item in items),
        "quality": "Unscored raw recognition; complete does not mean factually correct.",
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    write_json(args.output, run(read_json(args.request)))


if __name__ == "__main__":
    main()
