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
