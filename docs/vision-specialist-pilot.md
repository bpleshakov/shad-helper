# OCR-first and larger-model experiments

These local experiments ran on the Apple M5 Pro with 48 GB unified memory.
The Qwen models are now available in the main pipeline through configurable
presets; see [usage](../README.md). The OCR-first cascade remains an experiment.
The results below test two alternatives to the original vision
pilot: raw OCR followed by notebook theory filtering, and a larger vision model
for handwriting and diagrams. See [the original pilot](vision-pilot.md) for the
source fixture, selector measurements and baseline models.

## Protocol

All runs used MLX-VLM 0.7.6, MLX 0.32.3 and Transformers 5.19.0. Model downloads
finished before inference; GPU workers ran sequentially with fresh output
directories and no cached inference results. Model downloads are excluded from
the timings. Peak memory below is MLX allocation, not total system memory.

| Stage | Model | Pinned revision |
| --- | --- | --- |
| Raw text/formula OCR | `mlx-community/GLM-OCR-bf16` | `24f15402e83baa0a80eeeaecf5480e172abc6f2e` |
| Text-only notebook theory filter | `mlx-community/Qwen3.5-4B-4bit` | `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c` |
| Larger vision model | `mlx-community/Qwen3.5-35B-A3B-4bit` | `1e20fd8d42056f870933bf98ca6211024744f7ec` |

GLM received the same 16 native frames as the baseline and six manually selected
formula crops. It used only its documented `Text Recognition:` or
`Formula Recognition:` task prefix, temperature zero and a 4,096-token limit,
without retries. Raw code and comments were deliberately retained for the next
stage. Diagram topology is outside this OCR task and is not scored as OCR failure.

The GLM adapter in this MLX-VLM version ignores per-call image-size settings.
The runner instead sets its processor budget to 4.4 million temporal pixels,
equivalent to 2.2 million spatial pixels for duplicated still frames, and uses
the NumPy/PIL backend. Actual processed dimensions and token grids are logged;
patch alignment can slightly resize even native crops.

The 4B theory filter received only the raw OCR from three notebook theory frames
and one notebook/code frame. It saw no images, reference annotations, or
test-specific hints. It used temperature zero, thinking disabled, a 4,096-token
limit and no retry. Source OCR and its hash are retained with each output.

The 35B model used the baseline vision prompt and settings: thinking disabled,
temperature zero, 2,048 tokens initially and one 4,096-token formatting retry.
Its 16 freshly extracted images were SHA256-verified identical to the baseline
images. The new pilot runner records actual source PTS; older requests used
nominal timestamps. It then processed the same six recovery crops previously
tried on 4B and 9B.

The reference has 43 units: 14 definitions, 21 formulas and 8 diagram structures.
It was annotated through assistant source inspection, not human adjudication.
Coverage uses the best faithful observation at a readable source timestamp.
It does not measure the correctness of every generated statement. Manual crops
target known failures; they are neither an independent holdout nor an automatic
region-selection policy. Quality review considers mathematical meaning, literal
symbols, coverage, unsupported additions and formatting separately.

## Runtime

| Stage | Inputs | Measured time | Peak MLX allocation |
| --- | ---: | ---: | ---: |
| GLM raw OCR | 16 frames + 6 formula crops | 52.7 s | 3.37 GB |
| 4B theory filter | 4 OCR texts | 17.7 s | 4.15 GB |
| 35B native vision | 16 frames | 123.0 s | 22.40 GB |
| 35B recovery | 6 crops | 37.9 s | 21.48 GB |

GLM and theory timings include model load and processing but exclude preparation
of the already available images. The 35B native timing also includes frame
extraction; crop timing excludes manual selection and creation. GLM spent 45.2 s
on full-frame inference, 5.8 s on crop inference and 1.6 s loading. The theory
filter spent 15.3 s generating and 2.3 s loading. All 22 OCR outputs and all four
theory outputs finished without truncation; completion does not establish accuracy.

The 35B native run was faster than the prior 9B run (295.4 s), and approximately
matched the prior 4B run (123.6 s). All 16 native 35B responses passed the parser
on their first attempt. This removes the formatting/retry penalty on this sample,
but does not establish that their formulas are ready to render.

At the selector's previously observed 330–630 frames/hour, the 35B mean inference
cost of 7.24 s/frame extrapolates to roughly 40–76 minutes of inference alone.
GLM's full-frame mean of 2.83 s extrapolates to 16–30 minutes, before theory
filtering or recovery. These are arithmetic estimates from a small, fixed sample,
not full-hour measurements. The one-hour end-to-end target remains unverified.

## Quality results

GLM's native OCR recovered **19/35 applicable units**: 11/14 definitions and
8/21 formulas. The eight diagram topology units are excluded. Printed slides and
notebooks were substantially more useful than handwritten boards. It recovered
the exact comment forbidding `fit_transform` on test data, which the direct
vision runs omitted. However, all three notebook theory frames lost the explicit
definition of the loss function, despite readable source text.

Handwritten Gaussian density, derivatives, matrix indices and gradient notation
remain unreliable. The six formula crops recovered 6/13 target expressions; only
two crops had all their target formulas correct. Cropping recovered the Gaussian
joint-likelihood exponent `n/2`, but still changed sigma into `6`/`sqrt(6)` and
corrupted the variance estimate. This is not a dependable whiteboard OCR backend.

The text-only 4B filter retained **10/11 notebook reference units**, including all
three formula units and all three concepts in the code example. The missing loss
definition was already absent from OCR; the filter lost no additional reference
units. Nevertheless, it leaked two executable statements, changed "non-sparse
matrix" to "sparse matrix", and left garbled OCR prose unflagged. Coverage alone
would therefore overstate the quality of this stage.

For the same native-frame reference and source-qualified scoring rule:

| Vision model | Definitions | Formulas | Diagram structures | Failed critical units |
| --- | ---: | ---: | ---: | ---: |
| Prior 4B | 13/14 | 13/21 | 2/8 | 13/33 |
| Prior 9B | 13/14 | 19/21 | 3/8 | 8/33 |
| New 35B | 12/14 | 18/21 | 5/8 | 6/33 |

35B recovered the Gaussian normalization and all ten Gaussian formula units
semantically, but lost the regression vectors/matrix, row derivative and completed
gradient column. It also omitted both the test-data warning and the visible
sentence about combining real and encoded categorical features. Formula strings
mix ASCII, Unicode and LaTeX, including a malformed `y_\hat(x)` in one regression
frame; their rendering has not been validated. Accepted and raw-recoverable
coverage coincide because all outputs passed the parser on their first attempt.

Diagram extraction improved but remains unsafe to reconstruct without review.
35B correctly described more arrows entering the stress-test block, yet invented
a submissions/validation/verdict chain. It described the Gaussian density sketch
as a likelihood maximum at an estimated parameter, rather than heights of
different densities at the same observation.

All six 35B crops also passed the parser without retries. Taking the best faithful
observation across native frames and manually chosen crops raises coverage from
35/43 to **37/43**, with diagram coverage 7/8 and four critical units still not
fully covered. Crops recover the stress-test output arrow and density-height
comparison, but do not repair the regression notation or missing notebook warning.
The plot-only crop introduces a false common intersection, and other crops still
invent relationships or misread labels. These optimistic combined coverage
counts do not justify merging every crop response into notes.

## Decision

Keep GLM as a candidate for printed-content OCR, and 35B as the stronger candidate
for selective visual interpretation. Neither is a reliable universal extractor.
The present OCR-to-4B theory stage needs correction before its output is used as
final notebook theory. Diagram relationships and handwritten symbols remain the
main quality risks, rather than JSON formatting or memory capacity.

The original blanket 95% coverage gate is not used for this comparison: with only
14 definitions and eight diagram units it effectively required perfection in
those categories. Report coverage, mathematical errors, unsupported claims,
formatting and review burden independently. Lowering a percentage threshold would
not fix the observed wrong statements. A subsequent integration trial should
measure a representative complete lecture, automatic selection/recovery cost,
and editable TikZ/PGFPlots reconstruction; no raster diagram fallback is proposed.

## Reproduce

The scripts are standalone experiments and do not change `extract` or notes.
Install the repository's `local` dependencies, then run:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python scripts/ocr_pilot.py \
  --request .cache/vision-pilot/specialists/glm-ocr/request.json \
  --output .cache/vision-pilot/specialists/glm-ocr/response.json

HF_HUB_OFFLINE=1 .venv/bin/python scripts/ocr_theory_pilot.py \
  --ocr-response .cache/vision-pilot/specialists/glm-ocr/response.json \
  --job .cache/vision-pilot/specialists/ocr-theory \
  --output .cache/vision-pilot/specialists/ocr-theory/response.json \
  --item-id notebook-theory-4969-text \
  --item-id notebook-theory-4979-text \
  --item-id notebook-theory-4989-text \
  --item-id notebook-code-exclusion-5015-text

HF_HUB_OFFLINE=1 .venv/bin/python scripts/vision_pilot.py \
  --media-dir /Users/user/Downloads \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --revision 1e20fd8d42056f870933bf98ca6211024744f7ec \
  --out .cache/vision-pilot/specialists/qwen35b-native

HF_HUB_OFFLINE=1 .venv/bin/python -m shad_helper.vision_worker \
  --request .cache/vision-pilot/specialists/qwen35b-crops/request.json \
  --output .cache/vision-pilot/specialists/qwen35b-crops/response.json
```

The offline commands require downloaded weights. Every job directory retains its
exact request, response, per-item raw diagnostics and quality review. Successful
checkpoints resume automatically; use fresh job directories for timing.
Artifacts and weights are ignored by Git.

For a new OCR request, specify `job`, `model`, pinned `revision`, `max_tokens`
and `items`. Each item has a unique filename-safe `id`, local `image` path and
`task` (`text` or `formula`); optional source metadata passes through unchanged.
The existing fixture identifies the native frames. The formula crops below use
Pillow `(left, top, right, bottom)` coordinates without added pixels:

| Crop | Source frame | Box |
| --- | --- | --- |
| Gaussian derivatives | `gaussian-mle-5039` | `(990, 280, 1400, 510)` |
| Gaussian density/likelihood | `gaussian-mle-5039` | `(990, 65, 1300, 315)` |
| Regression matrix | `linear-regression-1661` | `(1010, 145, 1455, 250)` |
| Regression gradient | `linear-regression-1681` | `(1010, 250, 1290, 485)` |
| Regression loss | `linear-regression-1661` | `(495, 395, 945, 735)` |
| Notebook objective | `notebook-theory-4979` | `(1000, 545, 1350, 640)` |

## Software verification

The complete suite passed: **284 tests and 102 subtests**. Both new runners pass
Ruff. Their tests cover raw text preservation, text-only filtering, source
provenance, cache invalidation, resumption, truncation and invalid input. Actual
local GPU runs above verify model loading and generation beyond mocked tests.
