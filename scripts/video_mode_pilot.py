"""Compare native Qwen3.5 video input with independent calls on identical sampled frames."""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from shad_helper.jobs import sha256_file, write_json
from shad_helper.vision_config import VISION_MODELS
from shad_helper.vision_worker import (
    MAX_PIXELS,
    PROMPT,
    RETRY_PROMPT,
    _MLXRuntime,
    _versions,
    parse_output,
)

VIDEO_PROMPT = PROMPT.replace("этого кадра лекции", "этого видеофрагмента лекции") + "\n" + (
    "Объедини читаемое содержание всех кадров без повторов. Сохрани значимые промежуточные "
    "этапы рисунков и вычислений. Не объединяй разные состояния в выдуманное изображение."
)


def infer_video(runtime, frames, metadata, clip, max_tokens, *, retry=False):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

    started = time.monotonic()
    processor = runtime.processor
    # The video processor budgets pixels across all frames, unlike the image processor.
    processor.video_processor.min_pixels = 1024 * len(frames)
    processor.video_processor.max_pixels = MAX_PIXELS * len(frames)
    prompt = apply_chat_template(
        processor, runtime.model.config, VIDEO_PROMPT + (RETRY_PROMPT if retry else ""),
        num_images=0, video=str(clip), enable_thinking=False,
    )
    inputs = prepare_inputs(
        processor, videos=[frames], video_metadata=[asdict(metadata)], prompts=prompt,
        add_special_tokens=should_add_special_tokens(runtime.model.config.model_type, processor),
    )
    grid = inputs["video_grid_thw"].tolist()
    tokens = inputs["input_ids"].reshape(-1).tolist()
    video_tokens = tokens.count(runtime.model.config.video_token_index)
    assert video_tokens > 0 and "pixel_values_videos" in inputs
    assert "image_grid_thw" not in inputs
    inputs["mask"] = inputs.pop("attention_mask", None)
    result = generate(runtime.model, processor, prompt, max_tokens=max_tokens,
                      temperature=0.0, verbose=False, **inputs)
    patch = processor.video_processor.patch_size
    return result.text, {
        "seconds": time.monotonic() - started,
        "source_size": [frames.shape[3], frames.shape[2]],
        "processed_size": [grid[0][2] * patch, grid[0][1] * patch],
        "video_grid_thw": grid, "video_tokens": video_tokens,
        "prompt_tokens": result.prompt_tokens, "generation_tokens": result.generation_tokens,
        "prompt_tps": result.prompt_tps, "generation_tps": result.generation_tps,
        "peak_memory_gb": result.peak_memory, "finish_reason": result.finish_reason,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path, help="A short, unrotated lecture clip")
    parser.add_argument("--mode", choices=("video", "frames"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4)
    args = parser.parse_args()
    if args.frames < 4 or args.frames % 2:
        parser.error("--frames must be an even number of at least four")
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from mlx_vlm.utils import load_video
    from PIL import Image

    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frames, metadata = load_video(str(args.clip.resolve()), nframes=args.frames)
    assert len(frames) == args.frames
    images = []
    for index, frame in enumerate(frames):
        path = args.out / f"frame-{index:02d}.png"
        Image.fromarray(frame.transpose(1, 2, 0)).save(path)
        images.append({"path": str(path), "sha256": sha256_file(path),
                       "clip_timestamp": metadata.timestamps[index]})
    preparation_seconds = time.monotonic() - started
    model = VISION_MODELS["35b"]
    report = {"model": model, "mode": args.mode, "clip": str(args.clip.resolve()),
              "clip_sha256": sha256_file(args.clip), "sampling": asdict(metadata),
              "frames": images, "packages": _versions(), "results": [],
              "settings": {"max_pixels_per_frame": MAX_PIXELS, "max_tokens": 2048,
                           "retry_max_tokens": 4096, "enable_thinking": False,
                           "temperature": 0.0}, "preparation_seconds": preparation_seconds,
              "quality": "Requires source inspection; parser success is not an accuracy score."}
    (args.out / "prompt.txt").write_text(
        VIDEO_PROMPT if args.mode == "video" else PROMPT, encoding="utf-8")
    (args.out / "retry-prompt.txt").write_text(RETRY_PROMPT, encoding="utf-8")
    write_json(args.out / "report.json", report)
    loaded = time.monotonic()
    runtime = _MLXRuntime(model["model"], model["revision"])
    report["model_load_seconds"] = time.monotonic() - loaded
    for index in range(1 if args.mode == "video" else len(images)):
        record = {"index": index, "status": "incomplete", "attempts": []}
        for attempt, limit in enumerate((2048, 4096)):
            if args.mode == "video":
                raw, metrics = infer_video(runtime, frames, metadata, args.clip, limit,
                                           retry=bool(attempt))
            else:
                raw, metrics = runtime.infer(images[index]["path"], limit, retry=bool(attempt))
            attempt_record = {**metrics, "max_tokens": limit, "raw": raw}
            record["attempts"].append(attempt_record)
            try:
                if metrics["finish_reason"] == "length" or metrics["generation_tokens"] >= limit:
                    raise ValueError("Output reached token limit")
                record["content"] = parse_output(raw)
                record["status"] = "complete"
                break
            except (ValueError, TypeError) as error:
                attempt_record["error"] = str(error)
        report["results"].append(record)
        write_json(args.out / "report.json", report)
        print(f"{args.mode} {index + 1}: {record['status']}", flush=True)
    report["inference_seconds"] = sum(a["seconds"] for r in report["results"]
                                      for a in r["attempts"])
    report["wall_seconds"] = time.monotonic() - started
    write_json(args.out / "report.json", report)
    print(json.dumps({k: report[k] for k in ("mode", "model_load_seconds",
                                            "inference_seconds", "wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
