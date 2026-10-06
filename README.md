# shad-helper

An accuracy-first CLI for Russian lecture videos: local timed transcription and
visual extraction, followed by source-grounded study notes in LaTeX and PDF.
Anki is a separate, intentionally unimplemented command.

## Install

Use an Apple Silicon Mac and standard CPython 3.14. `uv` manages the interpreter
and virtual environment. Run from this project directory:

```sh
UV_CACHE_DIR=.cache/uv uv sync --extra local --group dev
```

Install FFmpeg (`ffmpeg` and `ffprobe`), LibreOffice for `.pptx` imports, and TeX
Live/MacTeX with `latexmk` and LuaLaTeX. The LaTeX template needs `fontspec`,
`unicode-math`, `polyglossia`, `cm-unicode`, `fvextra`, `fancyhdr`, `pgf`, and
`pgfplots`, plus the standard geometry, graphics, color, and hyperlink packages.
Make `/Library/TeX/texbin` available on `PATH` if your TeX installation uses it.
LibreOffice's standard macOS app location is recognized automatically.

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper doctor
```

The initial model run downloads Whisper large-v3, Qwen3.8-27B-4bit, and
PaddleOCR-VL-1.6. The Qwen weights alone are approximately 16 GB. Model caches live
under `.cache/huggingface`; Paddle's BF16 conversion is stored next to the jobs.
Workers run sequentially, releasing model memory when each process exits.
Once weights and tools are installed, extraction can run offline; set
`HF_HUB_OFFLINE=1` when intentionally disabling Hugging Face network access.

The initial development environment had no external tools or dependencies, and
its sandbox blocked dependency downloads. Dependency-free tests were run;
the real model, OAuth service, and PDF integration still need an installed
environment. This repository does not claim a measured M5 processing speed or
verified whiteboard recognition accuracy.

## Start with a pilot

Process a representative ten-minute interval before committing to a full lecture:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper extract lecture.mp4 \
  --source slides.pdf --source demo.ipynb \
  --start 1800 --duration 600 --out jobs/pilot
```

The same command without `--start` and `--duration` processes the complete video.
Repeat `--source` for PDF, PowerPoint, or notebook files. Notebook v4 JSON is read
directly; cells are never executed. Native code and saved outputs are preserved.
Source files are copied into the job for reproducibility; the video is identified
by its original path and SHA-256 hash.

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper extract lecture.mp4 \
  --source slides.pptx --source demo.ipynb --out jobs/lecture-01
```

Re-running a command reuses successful checkpoints for matching inputs. Changed
audio, source files, images, model settings, and synthesis prompts invalidate the
corresponding results. An interrupted worker resumes at the last completed item.

The job contains timed JSON, `transcript.srt`, `transcript.vtt`, original frames,
OCR crops, source assets, raw model replies, and `evidence.json`. Pilot timestamps
refer to the original video. The preview scan uses native temporal cadence,
retains stable/incremental states, and keeps periodic evidence during motion.
Visual generation stops early on long consecutive text repetitions. Incomplete
JSON or invalid visual fields produce a review warning and full-image OCR fallback;
the raw reply is retained. On resume, invalid visual checkpoints are regenerated
while valid results are reused.
Occlusion, unreadable handwriting, and imperfect source matching can still
require manual correction.

### Visual selection

For the sparse strategy, use a **new output directory**:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper extract lecture.mp4 \
  --vision-mode sparse --out jobs/lecture-sparse
```

Sparse mode samples small previews at two frames per second, supplemented by
scene events. These previews are not saved as individual PNGs or sent to Qwen.
It selects completed board pages, distinct slides, and notebook content before
exporting full-resolution images. A page has at most three real views: one main
frame and useful complementary views or explicitly uncertain earlier evidence.
Uncertain selection retains bounded evidence and review
warnings; it never restores every intermediate writing pause.
If the final writing is fully covered, it retains an earlier readable frame
with an explicit warning that the final content state could not be verified.

The Qwen allowance defaults to **30 minutes per video hour**. It includes model
loading, routing, analysis, and retries. Selection/export, Whisper, and PaddleOCR
have separate timings. The allowance is cumulative across resumes. Budget
exhaustion preserves partial results and selected images, marks deferred visual
intervals explicitly, and skips their automatic OCR. Notes retain visible review
gaps. To deliberately allow more Qwen work, raise the total job allowance:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper extract lecture.mp4 \
  --out jobs/lecture-sparse --vision-budget-minutes 45
```

Inspect `sparse-selection.json`, `sparse-vision-results.json`, and
`sparse-qwen-budget.json` for selections, call/token timings, coverage warnings,
and cumulative model time. `extraction-timings.json` separates current-run
audio/ASR, vision, and OCR wall time from cumulative Qwen usage.
A deadline-limited partial run is not a completed
30-minute quality benchmark.

The installed 27B model remains the reference until a smaller model passes a
reviewed handwriting/diagram holdout. The benchmark utility compares it with
Qwen3.5-9B-4bit and Qwen3.5-4B-MLX-4bit; run
`uv run python -m shad_helper.vision_benchmark --help` for commands. See the
[benchmark instructions](docs/sparse-benchmark.md) for input and review formats.
A qualified model profile can be saved as `JOB/sparse-model-profile.json` or
`.shad-helper/vision-profile.json` for subsequent jobs. Sparse becomes the default
only when that profile also records a reviewed, passing full-hour pilot.

Until the sparse benchmark qualifies, new jobs default to `--vision-mode pages`.
For a physical whiteboard, the
processor tracks writing and corrections and selects readable completed pages
for Qwen and OCR. Partial writing and intermediate corrections are discarded
from separate model analysis when a later completed page captures the content.
The original frames remain internal evidence. Ambiguous erasures, incomplete
pages, unreadable final views, and unresolved occlusion use conservative fallback
frames and review warnings instead of assuming the content is complete.

Apple Vision person masks help distinguish lecturer motion from board changes;
hidden areas remain unknown until a later view confirms them. The current Qwen
model, full image resolution, and native-cadence candidate scan are kept. Inspect
`whiteboard-pages.json` for selected pages, discarded intermediate states, page
boundaries, and fallback decisions before accepting a pilot's notes.

Use `--vision-mode content` on a new job to keep every changed or uncertain content
state and group only confirmed repeated content. This mode compares against a
fixed reference, tolerates up to three intensity levels per color channel as
noise, and retains an analysis for larger co-visible changes, including outside
the detected surface. `--vision-mode baseline` on a new job disables both page
selection and content grouping.

Install the macOS Vision and Quartz bindings with the local extra. Run native
segmentation in a normal macOS terminal: a restricted execution sandbox can
reject CoreVideo operations even when the bindings import successfully.
Existing saved `content` and `baseline` jobs keep their recorded behavior;
legacy jobs without `extraction-config.json` resume in baseline mode. Saved
`pages` jobs also keep their selector version. Modes and unsupported selector
versions cannot be changed in place; use a new output directory to compare modes.

Shared analyses retain occurrence-specific IDs/timestamps and `original_image`
provenance. Their `image` and crop coordinates refer to the actual canonical
analysis frame. Identical OCR inputs reuse results by decoded pixels, model,
recognition prompt, and generation settings. Both automatic notes and manual
handoff include each shared image once per chapter.

Inspect `visual-grouping.json` for original/representative frames, grouping and
fallback decisions, and Qwen call counts; `ocr-reuse.json` reports OCR reuse and
processing time. Savings depend on visible repeated content. Review the audit
on a representative pilot, especially handwriting, erasures, and brief content.
The original and optimization plans are recorded in [docs/plan.md](docs/plan.md).

## Generate notes with your ChatGPT plan

Set the proxy before online commands. Every app-owned OpenAI request requires it:
token exchange and refresh, identity discovery/signing keys, model discovery,
and streamed note generation.

```sh
export SHAD_HELPER_OPENAI_PROXY=http://127.0.0.1:7890
```

Alternatively, pass `--openai-proxy http://127.0.0.1:7890` before the command name or after `login`/`notes`.
The flag takes precedence over `SHAD_HELPER_OPENAI_PROXY`. HTTP and HTTPS proxies
are supported, including URL-encoded credentials (`http://user:password@host:port`).
For an HTTP proxy tunneling HTTPS destinations, keep `http://` in the proxy URL.
Proxy URLs are not saved to authentication files or printed in diagnostics.

Requests fail when the proxy is missing, invalid, or unavailable; they never retry
directly. `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, and `NO_PROXY` cannot override
or bypass this setting. Local extraction, rendering, and manual handoff/import
work without a proxy. This setting does not route Hugging Face downloads.

The sign-in page runs in your existing browser. **Configure that browser to use
the same proxy before running `login`.** The app cannot enforce browser routing.
Keep `127.0.0.1`/`localhost` direct so the local OAuth callback can return to the app.

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper login
UV_CACHE_DIR=.cache/uv uv run shad-helper notes jobs/lecture-01
```

Login registers this app through the official Sign in with ChatGPT flow. It uses
its own protected credentials in `.shad-helper/auth`, never Codex credentials.
To change their location, place `--state-dir PATH` before the command name.

The app checks that `gpt-6.1-sol` is available to your signed-in account. Access
depends on eligibility, preview availability, and usage limits; Pro does not
guarantee access. Inference uses your authorized ChatGPT plan, with no separately
billed API fallback. Image generation is not supported by this route.

The model creates a complete outline and then chapter-sized notes with a shared
notation glossary. Core lecture blocks have source references; explanations added
by the model are marked **Пояснение**, and tentative reconstructions are marked
**Требует проверки**. Existing extraction issues cannot be cleared by generation.
Unmatched source material cannot silently become lecture content.

The outputs are `notes.json`, `review.json`, and an editable `notes/` LaTeX project.
Running `notes` regenerates that project; keep manual TeX edits for the later
`render` step. `--rebuild` invalidates synthesized-note checkpoints while keeping
local extraction results.

## Manual ChatGPT fallback

If plan access is unavailable, the app exports a handoff automatically. You can
also choose the manual path without making any network calls:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper notes jobs/lecture-01 --manual
```

For each `handoff/chNNN/` directory, attach `evidence.pdf` and `context.json` in
ChatGPT and paste `prompt.txt`. The PDF deliberately includes original evidence
images for inspection. Save the JSON reply as `chNNN.json` in a replies directory:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper notes jobs/lecture-01 --import replies/
```

All expected chapter files must be present. A complete combined response can
also be imported with `--import response.json`; it uses the same response schema
and validation as automatic generation. `handoff/response-schema.json` documents
the format. JSON replies must not have Markdown fences.

The outline can be edited in `outline.json` before exporting a new handoff.
Chapters must be consecutive, cover the entire extracted interval, and be at
most twenty minutes long. Smaller chapters are useful for very dense visual
material. Never add an unmatched source as if it appeared in the lecture.

## Review, illustrations, and PDF

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper review jobs/lecture-01
```

Correct extraction errors in `evidence.json` before regenerating notes. For note
corrections, edit the chapter/TikZ source files directly, or edit `notes.json` and
import it to regenerate the LaTeX project. Resolve a review item explicitly after
making and checking the correction:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper review jobs/lecture-01 \
  --resolve issue_0001 --resolution "Checked the original formula; corrected the subscript."
```

An `uncertain` block stays visibly tentative until you correct its content and
change its kind. Recording a resolution alone does not repair the mathematical
content or automatically rewrite your chapter edits. The review appendix and
draft footer are refreshed while edited chapter sources are preserved. Explicit
resolutions survive later imports only while their issue and evidence still
match. An `uncertain` block keeps the document in draft until its kind is corrected.

Equations are LaTeX; technical diagrams are TikZ; appropriate plots use PGFPlots.
Video screenshots are kept only as internal evidence. Unknown plotted values are
not invented; qualitative reconstructions must be labeled as schematics.

Optional supplementary illustration prompts are in `notes/illustrations.json`.
Generate these images manually in ChatGPT, save them as `FIGURE_ID.png`, `.jpg`,
or `.pdf`, and import them when compiling:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper render jobs/lecture-01 --images illustrations/
```

Without illustrations, run `render JOB`. It compiles existing editable sources,
preserves manual TeX changes, and creates `notes/main.pdf`, `notes/qa/` page images,
and `notes/render-report.json`. Missing illustrations appear as visible pending
items. Compilation failures retain logs and sources. Inspect the QA images and
report for glyphs, clipped content, diagram labels, and math before accepting the
PDF. Generated TeX uses restricted file references and no shell escape.
If extraction evidence has changed, rendering requires regenerating or importing
notes first, so a previous document cannot silently use stale evidence.

## Checks

The lightweight suite runs without model downloads or network access:

```sh
python -m unittest discover -s tests -v
```

Use the project's Python 3.14 environment. Schema/PDF integration checks skip if
their dependencies are absent; the signed-JWT and mocked-HTTP pytest cases require
the installed development environment.

After installing the environment:

```sh
UV_CACHE_DIR=.cache/uv uv run pytest
UV_CACHE_DIR=.cache/uv uv run ruff check shad_helper tests main.py
UV_CACHE_DIR=.cache/uv uv run shad-helper doctor --smoke short-lecture.mp4
```

The smoke command processes thirty seconds through all local models and can
download their weights. It is not a benchmark for a three-hour lecture. The test
suite uses synthetic data and mocked HTTP services; the PDF integration test runs
only when TeX and PyMuPDF are installed. Evaluate a manually checked Russian
sample containing jargon, formulas, notebook output, silence, and board erasure
before processing the full lecture.

The example evidence and manual response under `examples/` are synthetic and
exist solely to exercise note import and rendering. They are not an actual
transcribed lecture.

To try note import without downloading models, after installing core dependencies:

```sh
mkdir -p jobs/example
cp examples/demo-evidence.json jobs/example/evidence.json
UV_CACHE_DIR=.cache/uv uv run shad-helper notes jobs/example --import examples/demo-response.json
UV_CACHE_DIR=.cache/uv uv run shad-helper render jobs/example
```

## References

- [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
- [PaddleOCR Apple Silicon guide](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL-Apple-Silicon.html)
- [ChatGPT-plan integration](https://developers.openai.com/siwc/token-sharing-open-source)
- [Plan integration limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)
