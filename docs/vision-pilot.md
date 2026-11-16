# Local lecture vision pilot

The tested Qwen models are now connected to `extract` and note generation through
`--vision-model 35b|9b|4b|off`; see the [usage guide](../README.md). Existing audio
jobs remain schema v3; video evidence uses schema v4. The pilot results below
remain relevant: models have unresolved content errors.
The [OCR-first and 35B follow-up](vision-specialist-pilot.md) reports newer results
and replaces the original blanket coverage gate with separate quality measures.

## Implemented components

- `shad_helper.video`: probes video separately from audio, scans gray 720-pixel
  proxies at up to two frames per second, and retrieves selected native frames.
  Selection tracks accumulated changes, settled content, pre-erasure states,
  scrolling and a 15-second guard during sustained change. Timestamps come from
  source PTS. Successful selections and images can be reused after interruption.
- `shad_helper.vision_worker`: runs one MLX vision model in an isolated process,
  preserves native 1080p input, extracts text, formulas, diagram descriptions and
  unreadable regions, and caches successful results by image, model revision,
  prompt, settings and package versions. Invalid or truncated JSON gets one retry;
  a second failure remains explicitly incomplete. A valid response is not proof
  of factual accuracy.
- `scripts/vision_pilot.py`: reproduces a fixed-frame experiment and records
  prompts, requests, raw responses, actual input dimensions and timing metrics.
  Reference answers never enter model prompts.
- `tests/fixtures/vision-pilot-gold.json`: 43 source-grounded reference units
  (14 definitions, 21 formulas, 8 diagram structures), including 33 critical units.

## Baseline experiment

Run on the target Apple M5 Pro with 48 GB unified memory, using MLX-VLM 0.7.6,
MLX 0.32.3 and Transformers 5.19.0. Inference was sequential, temperature zero,
thinking disabled, with a 2,048-token initial limit and 4,096-token retry limit.
The image processor received native-size frames, including patch alignment
(1920 × 1080 became 1920 × 1088).

| Model | Pinned revision | 16-frame wall time | Peak MLX allocation |
| --- | --- | ---: | ---: |
| `mlx-community/Qwen3.5-4B-4bit` | `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c` | 123.6 s | 4.91 GB |
| `mlx-community/Qwen3.5-9B-4bit` | `8b2b98c00a6b4d291155e4890773ca8f769aee53` | 295.4 s | 7.72 GB |

Wall time includes native frame extraction, model loading, inference and retries;
it excludes model download. Peak MLX allocation is not total process/system memory.
These small pilots do not establish a one-hour lecture runtime.

Selector version 3 retained 21, 11, 16, 21 and 17 frames in separate two-minute
stress-testing, regression, Gaussian, notebook and slide samples. Each scan
processed 231 proxy frames in 1.04–1.34 seconds, excluding native frame retrieval
and inference. Sustaining those selection rates would require 330–630 model
inputs per hour before recovery. The proxy scan is cheap; selection precision
and model inference still need improvement to meet the runtime target.

| Baseline model | Definitions | Timestamp-grounded formulas | Diagram structures | Failed critical units |
| --- | ---: | ---: | ---: | ---: |
| 4B | 13/14 (92.9%) | 13/21 (61.9%) | 2/8 (25.0%) | 13/33 |
| 9B | 13/14 (92.9%) | 19/21 (90.5%) | 3/8 (37.5%) | 8/33 |

A generous 9B formula comparison accepting a correct expression at an occluded
timestamp raises that category to 20/21, but does not resolve the grounding error.

Observed failures include reversed stress-testing arrows, invented curve/axis
relationships, a wrong Gaussian likelihood normalization exponent, and omission
of the notebook comment forbidding fitting the encoder on test data. The smaller
model also transcribed notebook code and corrupted mathematical notation.

A second, more explicit prompt separated literal transcription, notebook theory
comments and diagram geometry. On 9B this regressed to 7/16 schema-valid responses
and 8/14 definitions, 8/21 formulas, 0/8 diagram structures (386.8 seconds).
Relationships and unreadable regions were emitted as objects instead of strings,
and entire boards were omitted. On 4B the revised prompt produced 8/16 valid
responses, 10/14 definitions, 8/21 formulas and 1/8 diagram structures in 320.2
seconds. The baseline prompt is retained in the worker.

Six manually chosen native crops were then tried with each baseline model:
the stress-testing diagram, left/right Gaussian boards, the Gaussian plot,
the vector-derivative region and notebook prose plus its code cell. Inference
took 128.4 seconds on 9B and 30.9 seconds on 4B, excluding crop selection and
creation. Crops added no pixels or source detail. This was exploratory recovery
on known failures, not a new independent test set or an automatic crop policy.

The 9B crop recovered the Gaussian normalization exponent but read partial
derivatives as `2 l / (2 a)` and `2 l / (2 b)`. The 4B crop invented a linear
chain of stress-testing steps and interpreted density heights as integrals.
Both still omitted the visible notebook warning about test data. These crops
do not provide a reliable repair for the baseline results.

The reference was annotated through direct assistant inspection before reading
model outputs; it is not human-adjudicated. It covers 16 fixed frames from five
short intervals plus one code-exclusion example. It cannot certify complete
interval coverage, lecture-wide recall, digital whiteboards, or all slide styles.
Scores use the best faithful observation at a readable source timestamp, so they
also do not mean every emitted formula or diagram was correct.

## Reproduce

Install FFmpeg and synchronize the local dependencies:

```sh
uv sync --extra local
uv run --extra local python scripts/vision_pilot.py \
  --media-dir /path/to/lecture-videos \
  --model mlx-community/Qwen3.5-9B-4bit \
  --revision 8b2b98c00a6b4d291155e4890773ca8f769aee53 \
  --out .cache/vision-pilot/qwen9b
```

The fixture names the required recordings and timestamps. The first run downloads
the selected model into `.cache/huggingface`; subsequent runs can use
`HF_HUB_OFFLINE=1`. Review `response.json` against the actual source frames and
reference before assigning quality scores. Reusing an output directory can reuse
successful checkpoints, so compare fresh directories for performance measurements.

Local experiment artifacts are under `.cache/vision-pilot/`: `qwen4b-native`,
`qwen9b-native`, `qwen4b-revised`, `qwen9b-revised`, `qwen4b-crops` and
`qwen9b-crops`. Each stores the exact prompts and request/response files;
`quality-review.json` or `crop-review.json` records the source-grounded review.
These directories and downloaded weights are ignored by Git.
`frame-selection.json` records the version-3 sample selection diagnostics.

## Code verification

The complete suite passed: **270 tests and 85 subtests**, including real FFmpeg
tests for container/stream offsets and a requested timestamp between frames.
New vision files pass Ruff. Worker tests cover interruption, per-frame diagnostic
persistence, resume, invalid JSON, truncation, cache invalidation and actual
processor-input dimensions. Selector tests include incremental writing,
pre-erasure retention, brief equal-area formula replacement, independent boards,
scrolling/revisits, cursor motion and missing-frame recovery. These checks verify
software behavior; model quality is assessed separately above.

## Integration status

Automatic extraction, visual evidence schema v4, notes/handoff integration and
visual citations are implemented with configurable Qwen presets. The original
95%-per-category gate was too
coarse for this small fixture: it required all 14 definitions and all eight diagram
units. The follow-up assesses coverage, mathematical correctness, unsupported
relationships, formatting and review burden separately. Existing errors still
require correction or explicit review; a lower coverage threshold alone does not
make the outputs reliable. Complete board-heavy and mixed hour-long intervals,
timestamp alignment, interruption/resume, and editable TikZ/PGFPlots output still
need validation. Notebook extraction must retain theory while excluding code.
The under-one-hour runtime target must be measured across the complete vision
stage, including selection and recovery.

The selector is intentionally conservative and may retain extra frames when the
lecturer moves. It does not yet detect independent board regions or score
occlusion. Rotated streams are explicitly unsupported. No confidence threshold
currently detects the models' confidently wrong outputs; retries based only on
JSON validity or self-reported unreadability cannot solve that problem.

## Native video-input pilot (2026-11-15)

`scripts/video_mode_pilot.py` runs the pinned Qwen3.5-35B-A3B checkpoint through
MLX-VLM's native video input, including video tokens and frame timestamps. It also
runs independent image requests on SHA256-identical sampled frames. Each mode
loads the model in a separate process and records inputs, prompts, responses,
processed dimensions, retries, package versions, time and peak MLX allocation.

Two 20-second clips were tested: `gaussian-mle` starting at 5029 seconds and
`stress-test` starting at 2005 seconds in the recordings named by the existing
gold fixture. Four frames were sampled uniformly per video. Spatial resolution
matched the image controls: 1920 × 640 and 1920 × 1088 after patch alignment.
Thinking was disabled, temperature was zero, and the existing 2048/4096-token
initial/retry limits and 2.2 MP per-frame budget were retained.

| Clip | Native video | Four matched image requests | Current selector + image worker (inference only) |
| --- | ---: | ---: | ---: |
| Gaussian MLE | 27.2 s | 44.1 s | 43.5 s, 4 selected frames |
| Stress-test diagram | 9.2 s | 28.5 s | 14.4 s, 2 selected frames |

Times include preprocessing and formatting retries, but exclude model loading,
clip preparation, selection, ASR and notes generation. The Gaussian video needed
one formatting retry; the stress-test video passed on its first attempt. Peak
video MLX allocation was 22.4–22.5 GB. These are single-run short-clip results,
about 1.6× faster than the selected-image inference baselines, not measured
whole-lecture savings.

Source inspection found unresolved errors in both modes. Video changed the
Gaussian graph's x axis to theta and invented a maximum arrow. In the stress-test
diagram it invented `yours → slow! and correct` and `generator → checker`, where
the source arrows enter a central block. Both videos and all eight matched image
responses omitted illustration boxes despite visible instructional diagrams.
This standalone pilot's prompt produces one aggregate object, without per-frame
evidence IDs, timestamps or illustration ownership. Its timings do not measure the
integrated mode below, which uses a different prompt and temporal input layout.
No full-lecture quality or notes run was made in this pilot.

Artifacts and the detailed comparison are in `.cache/video-mode-pilot/`.
To try another short, unrotated clip, choose a fresh output directory:

```sh
HF_HOME=.cache/huggingface .venv/bin/python -m scripts.video_mode_pilot clip.mkv --mode video --out .cache/video-mode-pilot/new-video
HF_HOME=.cache/huggingface .venv/bin/python -m scripts.video_mode_pilot clip.mkv --mode frames --out .cache/video-mode-pilot/new-frames
```

## Full-cycle video mode

The pipeline now accepts `--vision-mode video` on `extract` and `run`; independent
frame requests remain the default. For the full cycle:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper run lecture.mp4 --out jobs-video --vision-model 35b --vision-mode video
```

The normal API/login and OpenAI proxy setup is required for notes and Anki.
Use `--start 0 --duration 120` for a short end-to-end trial, or omit both for the
entire recording. The same flag also works with the existing `4b` and `9b` presets.

The existing selector supplies native source images. Requests contain at most
four images of the same dimensions within a 20-second span. Each source image is
repeated within its temporal patch so Qwen does not fuse different board states
under one timestamp. These arrays go through `videos`, `video_metadata`,
`pixel_values_videos` and `video_grid_thw`, with native video tokens. Source PTS
are represented using microsecond ticks. Spatial resolution retains the existing
2.2 MP per-source-frame cap; generation uses 4096 tokens, with one 8192-token retry.

The model assigns each observation to source frame indices. Multiple observations
for a frame, such as separate boards, are combined; invalid indices, repeated
indices within an observation, missing frames and malformed content trigger a
retry. The worker expands the result into the existing per-frame evidence schema,
so citations, illustration crops, notes, PDF and Anki need no new format.
Batch checkpoints are separate from independent-frame checkpoints and include
all source image hashes and timestamps. Failed batches remain explicitly incomplete
and are retried on the next extraction.

Integration checks used the same two recording intervals with the actual CLI,
ASR, selector, pinned 35B model, native-video worker and illustration materializer.
The final prompt passed on the first attempt for both batches: four Gaussian
frames in 23.7 s and two stress-test frames in 11.9 s of preprocessing/inference,
with about 22.5 GB peak MLX allocation. These are single runs, not a benchmark.
Both boards' formulas were retained in the Gaussian output; source IDs, timestamps,
managed crops and the offline notes handoff were verified. A repeat invocation
reused the video batch without loading the model. Artifacts and code hashes are
in `.cache/video-mode-integration/verification.json`.

Generated diagram descriptions still contain reading/inference errors, so these
checks establish integration and resumption, not whole-lecture content quality.
The full-cycle stage wiring is covered by automated tests; no live online
notes → PDF → Anki run was performed for these clips.
The runner defaults to four sampled frames; `--frames` accepts even counts of at
least four. The pilot was verified with actual GPU inference and Ruff.
