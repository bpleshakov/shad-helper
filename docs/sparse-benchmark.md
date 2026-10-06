# Qualifying sparse vision

Sparse mode stays opt-in until both recognition quality and a complete lecture
pilot pass. A budget-limited partial run does not qualify it.

## Model comparison

Prepare 40 real lecture images: 20 tuning examples and 20 separate holdout
examples. Include handwriting, complex formulas, faint/small symbols, diagrams,
slides, notebook code/output, and lecturer occlusion. Use tuning images when
adjusting selection or prompts; keep the holdout separate for final review.

Create a JSON manifest with exactly 40 unique case IDs. Image paths are relative
to the manifest directory, or absolute. Each entry has this shape:

```json
{"cases": [
  {"id": "tuning-01", "image": "images/tuning-01.png", "split": "tuning"},
  {"id": "holdout-01", "image": "images/holdout-01.png", "split": "holdout"}
]}
```

The excerpt shows two entries; the actual manifest needs 20 of each split.
Run from the repository root in your ordinary terminal:

```sh
UV_CACHE_DIR=.cache/uv uv run python -m shad_helper.vision_benchmark run \
  benchmarks/cases.json --out benchmarks/vision
```

The utility loads models sequentially: the installed Qwen3.8-27B-4bit reference,
Qwen3.5-9B-4bit, then Qwen3.5-4B-MLX-4bit. Missing weights may be downloaded.
Each image receives the same compact routing prompt and full-image analysis
prompt used by sparse mode. `report.json` records raw/parsed results, validity,
termination reason, routing and analysis time, model loading time, token rates,
and peak memory. It is saved after each result; rerunning the command repeats
inference.

Review all 20 holdout images for **each** model against the source image and
the 27B reference. Check board/screen bounds, final content, formula symbols,
and meaningful diagram regions and relationships. Create `reviews.json` with
60 entries, each shaped as follows:

```json
{"reviews": [
  {
    "model": "mlx-community/Qwen3.5-9B-4bit",
    "case_id": "holdout-01",
    "reviewed": true,
    "routing_correct": true,
    "additional_critical_errors": 0,
    "missed_regions": 0,
    "missing_final_content": 0
  }
]}
```

Record actual errors rather than copying zeroes. These counts must be integers.
Every holdout response must also have valid JSON and finish naturally. The
fastest passing smaller model is selected only if the reference also passes;
otherwise the reference remains selected.

```sh
UV_CACHE_DIR=.cache/uv uv run python -m shad_helper.vision_benchmark qualify \
  benchmarks/vision/report.json benchmarks/reviews.json \
  --out .shad-helper/vision-profile.json
```

A qualified profile chooses the model for subsequent sparse jobs. Each new job
pins its model profile; changing the workspace profile does not change resumed
jobs. Qualification without a pilot leaves sparse mode opt-in.

## Full lecture pilot

Use a fresh output directory and at least one full hour of representative video:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper extract lecture.mp4 \
  --vision-mode sparse --out jobs/sparse-pilot
```

Review the selected final pages against the original video. Use
`audit.selected_video_frames` from the final `sparse-selection.json` for the
exported video frame count; attached reference images are counted separately.
Count deferred/failed items from
`sparse-vision-results.json`. Use cumulative Qwen time from
`sparse-qwen-budget.json`; include resumes, loading, routing, analysis, and
escalation. ASR, cheap selection/export, OCR, and note synthesis are outside
this allowance. Compare exports with the dense selector on the same video.

Create `pilot.json` with measured values. The following is a format example,
not an actual passing result:

```json
{
  "reviewed": true,
  "primary_model": "MODEL_ID_FROM_PROFILE",
  "benchmark_fingerprint": "FINGERPRINT_FROM_PROFILE",
  "video_seconds": 3600,
  "baseline_frame_count": 900,
  "exported_frame_count": 40,
  "qwen_seconds": 1200,
  "deferred_count": 0,
  "failed_count": 0,
  "final_content_omissions": 0
}
```

Promotion requires at least 90% fewer exported images, cumulative Qwen time at
most half the video duration, and zero deferred/failed visuals or reviewed
final-content omissions. Model and benchmark fingerprint must match the
qualified profile. Re-run qualification with the pilot:

```sh
UV_CACHE_DIR=.cache/uv uv run python -m shad_helper.vision_benchmark qualify \
  benchmarks/vision/report.json benchmarks/reviews.json \
  --pilot benchmarks/pilot.json --out .shad-helper/vision-profile.json
```

Only a passing profile with `sparse_default: true` promotes sparse mode for
new jobs. Existing saved `pages`, `content`, and `baseline` jobs retain their
mode. An unsuccessful pilot keeps the current default and its evidence for
review.
