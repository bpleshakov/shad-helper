"""Bounded native-video inputs with observations mapped to exact source frames."""

import copy
import json
import re
import time

from .vision_worker import MAX_PIXELS, PROMPT, RETRY_PROMPT, parse_output

MAX_FRAMES = 4
MAX_SECONDS = 20
VIDEO_PROMPT = PROMPT.replace(
    "этого кадра лекции", "выбранных кадров этого видеофрагмента лекции",
).replace(
    '{"kind":"slides', '{"observations":[{"frame_indices":[0],"kind":"slides',
).replace(
    '"место и описание нечитаемого или закрытого фрагмента"]}.',
    '"место и описание нечитаемого или закрытого фрагмента"]}]}.',
) + """
frame_indices — индексы исходных кадров из списка ниже, начиная с 0.
Охвати каждый кадр: все доски, записи, читаемые формулы, рисунки и значимые изменения.
Можно дать несколько наблюдений для одного кадра, например отдельно для левой и правой
доски. Не пропускай вторую доску. Каждый индекс должен встретиться хотя бы один раз.
Для разных состояний доски или слайда создай отдельные наблюдения.
Объединяй индексы в одном наблюдении только если ВСЁ описанное
содержание видно на КАЖДОМ из этих кадров и рамки illustrations подходят к каждому кадру.
Не переноси поздние записи на ранние кадры и не восстанавливай закрытое по соседним кадрам.
Если кадр нечитаем, всё равно включи его индекс и укажи неполноту в unreadable.
"""


def group_frames(items: list[dict]) -> list[list[dict]]:
    from PIL import Image

    batches = []
    batch = []
    size = None
    for item in items:
        with Image.open(item["image"]) as image:
            current_size = image.size
        if batch and (len(batch) == MAX_FRAMES or current_size != size
                      or item["timestamp"] - batch[0]["timestamp"] > MAX_SECONDS):
            batches.append(batch)
            batch = []
        batch.append(item)
        size = current_size
    if batch:
        batches.append(batch)
    return batches


def parse_video_output(raw: str, frame_count: int) -> list[dict]:
    fenced = re.fullmatch(r"\s*```(?:json)?\s*\n?(.*?)\n?```\s*", raw, re.DOTALL)
    output = json.loads(fenced.group(1) if fenced else raw)
    if (not isinstance(output, dict) or set(output) != {"observations"}
            or not isinstance(output["observations"], list)):
        raise ValueError("Invalid video observations")
    contents = {}
    for observation in output["observations"]:
        if not isinstance(observation, dict):
            raise ValueError("Invalid video observation")  # noqa: TRY004 - invalid wire format
        indices = observation.get("frame_indices")
        if not isinstance(indices, list) or not indices:
            raise ValueError("Video observation requires frame_indices")
        content = parse_output(json.dumps({key: value for key, value in observation.items()
                                          if key != "frame_indices"}))
        seen = set()
        for index in indices:
            if type(index) is not int or not 0 <= index < frame_count or index in seen:
                raise ValueError("Invalid or repeated video frame index")
            seen.add(index)
            if index not in contents:
                contents[index] = copy.deepcopy(content)
                continue
            # A model may describe separate boards/regions as separate observations.
            # Combine only observations explicitly assigned to this source frame.
            combined = contents[index]
            if combined["kind"] != content["kind"]:
                combined["kind"] = "other"
            if content["text"] and content["text"] != combined["text"]:
                combined["text"] = "\n\n".join(filter(None, (combined["text"], content["text"])))
            for key in ("formulas", "diagrams", "illustrations", "unreadable"):
                combined[key].extend(copy.deepcopy(value) for value in content[key]
                                     if value not in combined[key])
    if len(contents) != frame_count:
        raise ValueError("Video observations must cover every source frame")
    return [contents[index] for index in range(frame_count)]


def infer_video(runtime, items: list[dict], max_tokens: int, *, retry: bool = False):
    import numpy as np
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs, should_add_special_tokens
    from PIL import Image

    started = time.monotonic()
    processor = runtime.processor
    images = []
    for item in items:
        with Image.open(item["image"]) as image:
            images.append(np.asarray(image.convert("RGB")).transpose(2, 0, 1))
    # Qwen fuses temporal patches. Repeating each source frame prevents it from
    # fusing two different board states into a single timestamp/illustration box.
    temporal_patch = processor.video_processor.temporal_patch_size
    frames = np.repeat(np.stack(images), temporal_patch, axis=0)
    # Predecoded frames have irregular source PTS. Use microsecond ticks, not an
    # invented constant sampling rate, for the processor's index/fps timebase.
    ticks = [round(item["timestamp"] * 1_000_000) for item in items]
    metadata = {"fps": 1_000_000, "total_num_frames": ticks[-1] + 1,
                "frames_indices": [tick for tick in ticks for _ in range(temporal_patch)]}
    processor.video_processor.min_pixels = 1024 * len(frames)
    processor.video_processor.max_pixels = MAX_PIXELS * len(frames)
    frame_list = "\n".join(f"Кадр {index}: {item['timestamp']:.6f} секунд от начала записи."
                           for index, item in enumerate(items))
    prompt = apply_chat_template(
        processor, runtime.model.config,
        VIDEO_PROMPT + frame_list + (RETRY_PROMPT if retry else ""),
        num_images=0, video="selected-frames", enable_thinking=False,
    )
    inputs = prepare_inputs(
        processor, videos=[frames], video_metadata=[metadata], prompts=prompt,
        add_special_tokens=should_add_special_tokens(runtime.model.config.model_type, processor),
    )
    grid = inputs["video_grid_thw"].tolist()
    video_tokens = inputs["input_ids"].reshape(-1).tolist().count(runtime.model.config.video_token_index)
    if not video_tokens or "pixel_values_videos" not in inputs:
        raise RuntimeError("Vision processor did not produce native video inputs")
    inputs["mask"] = inputs.pop("attention_mask", None)
    result = generate(runtime.model, processor, prompt, max_tokens=max_tokens,
                      temperature=0.0, verbose=False, **inputs)
    patch = processor.video_processor.patch_size
    return result.text, {
        "seconds": time.monotonic() - started,
        "source_size": [frames.shape[3], frames.shape[2]],
        "processed_size": [grid[0][2] * patch, grid[0][1] * patch],
        "source_timestamps": [item["timestamp"] for item in items],
        "video_grid_thw": grid, "video_tokens": video_tokens,
        "prompt_tokens": result.prompt_tokens, "generation_tokens": result.generation_tokens,
        "prompt_tps": result.prompt_tps, "generation_tps": result.generation_tps,
        "peak_memory_gb": result.peak_memory, "finish_reason": result.finish_reason,
    }
