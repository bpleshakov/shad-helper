# shad-helper

A CLI that turns Russian lecture recordings into detailed, evidence-grounded LaTeX
notes and PDF. Audio files and video soundtracks share local transcription;
video frames also supply text, formulas, and cropped instructional illustrations.
Notes include equations, code, generated teaching illustrations, timestamps, and review items.

Video extraction uses a configurable local Qwen model. Handwritten formulas
and diagram relationships can still be wrong and need review. See
[the baseline pilot](docs/vision-pilot.md) and the
[OCR-first and 35B experiments](docs/vision-specialist-pilot.md).

## Install

Use an Apple Silicon Mac and CPython 3.14. From this project directory:

```sh
UV_CACHE_DIR=.cache/uv uv sync --extra local --group dev
UV_CACHE_DIR=.cache/uv uv run shad-helper doctor
```

Install FFmpeg (`ffmpeg` and `ffprobe`) and TeX Live/MacTeX with `latexmk` and
LuaLaTeX. The template uses `fontspec`, `unicode-math`, `polyglossia`,
`cm-unicode`, `fvextra`, `fancyhdr`, `pgf`, `pgfplots`, `microtype`, and
standard geometry, color, and hyperlink packages. Put `/Library/TeX/texbin` on
`PATH`.

The first transcription downloads Whisper large-v3 through MLX Whisper; the first
video extraction downloads the selected Qwen vision model.
Caches live under `.cache/huggingface`. The ASR worker exits after transcription
before the vision worker loads, so their model memory is not held together.
Once tools and weights are installed, extraction works
offline; use `HF_HUB_OFFLINE=1` to disable Hugging Face network access.

With a configured OpenAI proxy, `doctor` checks models independently for the API key
and saved ChatGPT login. Its JSON contains `openai.api_key` and `openai.chatgpt_login`,
each with `status`, `models`, `default_model`, and `default_model_available`.
The API key comes from `OPENAI_API_KEY` or `config/openai.env`; the login check always
uses app-owned credentials. Missing setup is reported as `not_checked`; an error in
one check does not prevent the other. Catalog checks never fall back from the API key
to login, start login, or generate notes. Local checks determine the command's exit status.

## Run the complete pipeline

For hand-drawn illustrations, install the Codex CLI on `PATH` and sign in to
ChatGPT with `codex login`. You can check that login with `codex login status`.

After installing dependencies, configuring an API key or signing in, and setting the OpenAI proxy, run:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper run lecture.mp4 --out jobs
UV_CACHE_DIR=.cache/uv uv run shad-helper run lecture.mp4 --out jobs-video --vision-mode video
UV_CACHE_DIR=.cache/uv uv run shad-helper run 'Lecture 1.mp4' 'Lecture 2.mkv' --out jobs --vision-model 9b
```

Each file gets its own job under `--out`, named `<filename-stem>-<path-hash>`.
The hash uses the resolved absolute input path, so recordings with identical names
in different directories remain separate. Repeating the command with the same
files and output directory reuses the same jobs, regardless of input order.
Repeated paths or symlink aliases in one invocation are rejected before work starts.
Audio files are also supported.

Files run in the supplied order. For video, one Qwen worker inspects selected frames
while one GPT worker drafts and verifies ready chapters in lecture order. Each chapter
includes its neighboring evidence. For a first chapter ending at 15:00, notes can start
when the selected frames through 16:00 and their crops are ready. Interactive terminals
show each worker's current chapter, activity, and elapsed processing time for the chapter
and the lecture so far. Redirected output records chapter and phase transitions.
Audio-only runs proceed from transcription to notes.

Once all chapters and extraction finish, the pipeline renders `JOB/notes/main.pdf` and
generates `JOB/anki/deck.apkg`. If a notes chapter needs manual input, Qwen finishes the
lecture and the job saves a complete handoff. The batch then advances to the next input.
The final summary reports completed and failed jobs. Any failure sets exit status 1.
[Chapter scheduling, progress and resume](docs/chapter-pipeline.md)

The extraction options `--start`, `--duration`, `--language`, `--vision-model`, and `--vision-mode`
apply to every input. `--rebuild` clears each notes/Anki stage's generated files,
caches, and diagnostics before that stage starts, including the Anki deck identity,
then regenerates them and compiles the PDF from scratch. Otherwise existing stage
checkpoints are reused. Rerunning the full pipeline
regenerates the editable TeX sources, so use the individual commands to retain
manual TeX corrections. `--anki-collection` selects the local Anki profile, and
`--openai-proxy` and the global `--state-dir` work as for individual commands.

Each job produces its own Anki deck. Within one invocation, successive decks use
non-overlapping position ranges even before import; all cards from one note still
share a position. Each export also checks the live Anki allocation. Import the
completed decks before starting a separate batch, since separate invocations do
not reserve positions for each other.

Each pipeline invocation writes `JOB/run-diagnostics.json`, also when calling
`extract`, `notes`, `render`, or `anki` separately (including their Python APIs).
It records the full source-checkout `commit_hash`, a `git_dirty` flag for uncommitted
changes, UTC start/end timestamps, and total `elapsed_seconds`. Git fields are
`null` when the checkout or Git metadata is unavailable.

The ordered `steps` tree records elapsed seconds and status for each executed stage
and its substeps, including chapter checkpoints, illustration verification and
reconciliation, LaTeX compilation, and Anki generation.
Each step also records its `start_offset_seconds`. In video runs, the extraction and
notes intervals overlap. `chapter_worker_overlap_seconds` measures the time spent in
chapter notes jobs while the vision subprocess was active. Use the top-level elapsed
time for wall-clock comparisons.
`generation_requests` counts fresh notes requests by operation, including retries and
direct comparisons of original figures with their proposed final replacements.
Cached checkpoints record this run's lookup time with status `cached`; they do not
reuse the original generation duration. Failed/interrupted steps retain elapsed time.
The report updates as steps finish and replaces the previous invocation's report;
no history directories are created. A full `run` keeps all four stages together,
and `notes --rebuild` keeps notes generation and PDF rendering together. Even a
rejected stage invocation writes its failed diagnostic report without publishing
new pipeline output. Existing `handoff/run.json` retains the notes-specific result.

## Extract lecture evidence

Start with a representative ten-minute interval:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper extract lecture.m4a --start 1800 --duration 600 --out jobs/audio-pilot
```

For a full recording, omit the interval options. Video defaults to Qwen 35B:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper extract lecture.mp4 --out jobs/video-lecture
UV_CACHE_DIR=.cache/uv uv run shad-helper extract lecture.mp4 --out jobs/video-lecture --vision-model 9b
UV_CACHE_DIR=.cache/uv uv run shad-helper extract lecture.mp4 --out jobs/video-lecture --vision-model 4b
UV_CACHE_DIR=.cache/uv uv run shad-helper extract lecture.mp4 --out jobs/audio-lecture --vision-model off
```

`--vision-model` accepts `35b`, `9b`, `4b`, or `off`. The default is `35b` on
every invocation; repeat the flag when continuing a job with another preset.
Audio-only files and album-cover images do not load a vision model.

`--vision-mode frames|video` works with both `extract` and `run`. The default,
`frames`, processes selected frames independently. `video` sends up to four
selected frames spanning at most 20 seconds together through Qwen3.5's native
video input, with source timestamps. It uses the same frame selector and spatial
resolution budget; it does not send every frame of the recording. Transcription
still runs separately. Original frame IDs and illustration crops are preserved,
so the remaining notes → PDF → Anki stages run normally.

Video mode is experimental: short-clip checks do not establish whole-lecture
quality or speed. See the [video pilot and integration notes](docs/vision-pilot.md).
Repeat `--vision-mode video` on subsequent runs; omitting it selects `frames`.
With `--vision-model off` or audio-only input, the mode has no effect.

| Preset | Local model (4-bit) | Peak MLX memory in the 16-frame pilot |
| --- | --- | ---: |
| `35b` (default) | `mlx-community/Qwen3.5-35B-A3B-4bit` | 22.4 GB |
| `9b` | `mlx-community/Qwen3.5-9B-4bit` | 7.7 GB |
| `4b` | `mlx-community/Qwen3.5-4B-4bit` | 4.9 GB |

Presets pin the tested model revisions in `shad_helper/vision_config.py`.
35B matched 4B's speed and was faster than 9B on this small pilot; model size
alone does not predict runtime. The under-one-hour target for a one-hour video
has not been established. Frames are selected from content changes, settled
states, pre-erasure states and scrolling, then processed at native resolution
up to the vision processor's 2.2-megapixel budget. Rotated streams must first be
normalized. The recording still needs an audio stream.

Frame selection scans proxies at 2 fps. Broad, bright rectangular board/screen
surfaces guide change detection and sharpness scoring, suppressing movement
outside them while keeping every detected panel and a margin around its edges.
Uncertain layouts and settled changes outside those regions use whole-frame
comparison. Pre-erasure, intermediate drawing, scrolling and time guards remain
active. Selection diagnostics include `region_frames`,
`full_frame_frames`, and per-image `content_regions` in proxy-pixel
`[left, top, right, bottom]` coordinates. The updated selection version rescans
older selections while retaining reusable transcription and vision checkpoints.

A second pass suppresses candidates that add no visible instructional content.
It uses the pinned, approximately 6 MB
[PPHumanSeg model from OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg)
to mask foreground people, including people crossing a board. Comparisons
compensate for small camera movements and smooth lighting changes, with a
one-proxy-pixel tolerance and a separate check for small local annotations.
Masks also handle lecturers cropped by an inset camera pane and reduce false
detections on broad board diagrams. Small camera shifts no
longer create artificial unseen strips at the image borders.
Newly revealed writing, changed/erased strokes,
scrolling boundaries, and clearer unobstructed views retain their native frames.
Nearby additive writing stages are combined within a five-second window when a
later clear frame preserves the visible marks from every earlier stage in the
group. This can omit the order of intermediate strokes or labels. The first
view, erasures, lost visibility, and scrolling boundaries prevent combining.
Comparisons use aligned pixels and visibility masks; the lecture vision model
still receives original images with their original timestamps. All native
candidates remain on disk for inspection.

The person model downloads once into the existing Hugging Face cache. Local
dependencies pin OpenCV below 5 because 5.0 misaligned this model's masks in
validation; run `uv sync --extra local` when upgrading an existing environment.
`checkpoints/video/novelty.json` records every retained/suppressed candidate and
its reason, including the final `represented_by` source ID for combined stages.
Its cache tracks image bytes, model revision/checksum, policy and
OpenCV version. `extraction-diagnostics.json` reports candidate/suppression counts
as well as `merged_frames` and filtering time under `selection.novelty`. A changed
filter policy recomputes these decisions while reusing matching vision results.

To measure candidate reduction on saved extraction images without rerunning ASR
or the lecture vision model:

```bash
.venv/bin/python -m scripts.frame_selection_benchmark out/lecture-job \
  --out .cache/frame-selection-benchmark
```

The benchmark writes its own checkpoints and `report.json` outside the source
job. Counts measure candidates that would skip inference. Full lecture runtime
and content coverage still require evaluation on the recordings being processed.

FFmpeg selects the first audio stream and produces 16 kHz mono WAV chunks.
Ten-minute core intervals have five-second overlaps; each word belongs to one
core interval. JSON, SRT, and VTT timestamps refer to the original recording.
Russian is the default; override transcription language with `--language`.
Notes remain in Russian.

Jobs contain `extraction-config.json`, `evidence.json`,
`transcript.json`, `transcript.srt`, `transcript.vtt`, audio chunks, worker
requests/responses, successful checkpoints, and current-run extraction timings.
Audio evidence remains schema version 3, with title, media identity and interval,
and speech segments containing `id`, `start`, `end`, and `text`. Video evidence
uses version 5 and adds timestamped `visuals`: source IDs, completion status,
content kind, text, formulas, diagram descriptions, unreadable regions, and
`illustrations` candidates with source kind, crop bounds, stable IDs, and image hashes.
Managed PNG crops and their manifest live under `illustrations/`. Crops include
a three-percent frame margin to preserve nearby labels. Failed visual extraction
retains an unknown source-image candidate for review instead of losing the frame.
Frames are saved under `frames/` for local inspection. Model metadata, raw
responses, timing/token/memory measurements, selection details, ASR confidence
and extraction warnings stay in worker files and `extraction-diagnostics.json`.
Notes do not read that report, and warnings do not block generation.

Reruns reuse matching audio, ASR, frame selection and vision checkpoints. Changing
`--vision-model` reuses transcription and selected frames while recomputing visual
content. Frame and video modes have separate vision checkpoints; switching modes
reuses transcription and frame selection, and returning to a previously used mode
can reuse its matching results. Video checkpoints also include the IDs, timestamps,
and image hashes of every frame in the batch. Changes to media, intervals, language,
model, or vision prompt/settings
invalidate affected results. Deleted audio/frame artifacts are rebuilt. Interrupted
work resumes from completed chunks/frames. Invalid vision JSON gets one retry;
an unsuccessful frame remains explicitly `incomplete` and is retried on rerun.
In video mode, retries and checkpoints cover a batch; a malformed response leaves
all its frames `incomplete`. Raw batch responses live in `diagnostics/vision-video/`.

Version 3 audio and version 4/5 video jobs can be reused, including enabling or
disabling vision in an existing supported job. Older formats require a **new
output directory**; extraction rejects them before changing their files. Re-extraction
preserves existing notes/PDFs, but changed evidence requires regenerating notes
before rendering. Extraction records completion in `extraction-status.json` and
publishes `evidence.json` only after the required stages finish. A failed rerun
preserves the previous complete evidence and saves partial speech evidence in
`evidence.partial.json`. Notes reject incomplete extraction and video jobs whose
evidence lacks the visual stage; resume extraction or explicitly select vision off.
Silent extraction keeps inspectable artifacts; notes require speech, useful
visual content, or an illustration candidate. Re-extract version 4 jobs to obtain
version 5 crops; existing transcription and frame selection are reused.

A transcription smoke check processes thirty seconds and may download Whisper:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper doctor --smoke short-lecture.wav
```

## Generate notes with an API key or your ChatGPT plan

Set `OPENAI_API_KEY` in the process environment or in `config/openai.env`, relative
to the working directory:

```dotenv
OPENAI_API_KEY=your-api-key
```

The process environment takes precedence. The file supports plain or quoted values,
comments, and an optional `export` prefix; it is read without executing shell code.
The key file is ignored by Git. The key is never written to request archives or logs.

Notes, Anki, and visual verification prefer the API key. When OpenAI reports
`credit_balance_exhausted` or the legacy `insufficient_quota` error, the client
retries the same request with the app's saved ChatGPT login and keeps using login for that client's remaining
requests. A new client tries the API key again, allowing topped-up credits to be used.
There is no dollar-balance check or $5 threshold. Other errors, including ordinary
rate limits, invalid keys, and network failures, do not switch authentication.
See the [OpenAI error-code reference](https://developers.openai.com/api/docs/guides/error-codes).

Run `shad-helper login` once to prepare the fallback. If no key is configured,
requests use login as before. Fallback never opens a browser automatically; without
saved login credentials it asks you to run `login`. Set an explicitly empty
`OPENAI_API_KEY` to use login even when the file contains a key.

Drawing generation uses the Codex CLI's own ChatGPT login. The pipeline starts
a read-only `codex exec` session for each uncached drawing and passes the
`--openai-proxy` or `SHAD_HELPER_OPENAI_PROXY` setting to the subprocess.

Every CLI command prints a spending summary to the console (stderr) when it ends,
including failures and keyboard interrupts. For example:

```text
OpenAI spending [notes]: estimated API cost $0.026400; API requests: 2; ChatGPT-plan requests: 0.
```

The total includes all generation attempts across the command's projects and stages,
including retries and replies that fail validation. Reused local artifacts and
model-list requests add no API generation cost. ChatGPT-plan requests are counted separately
as subscription usage, without attributing an API-key dollar charge to them.
`doctor` keeps its JSON on stdout; its summary goes to stderr like other commands.

No admin key is needed or used. Estimates use each response's token usage and
the published `gpt-6-astra` rates checked on 2026-11-08, including cached input,
30-minute/12-hour cache writes, reasoning output, long-context rates, and the
reported Standard/Flex/Fast processing tier. Standard rates are assumed if the
response omits its tier. These are public-rate estimates, not reconciled billing
totals; account-specific prices and future rate changes can differ. See
[model pricing](https://developers.openai.com/api/docs/models/gpt-6-astra) and
[cache accounting](https://developers.openai.com/api/docs/guides/prompt-caching).
Missing or malformed usage, interrupted streams, unsupported models, and unsupported
tiers produce an explicitly unknown total with a known subtotal, rather than a zero
charge. Each Codex drawing invocation counts as one ChatGPT-plan request.
Commands start a fresh total without querying the billing API.

Every app-owned OpenAI request requires an explicit HTTP/HTTPS proxy:

```sh
export SHAD_HELPER_OPENAI_PROXY=http://127.0.0.1:7890
UV_CACHE_DIR=.cache/uv uv run shad-helper login
UV_CACHE_DIR=.cache/uv uv run shad-helper notes jobs/audio-lecture
# Process several projects sequentially:
UV_CACHE_DIR=.cache/uv uv run shad-helper notes jobs/lecture-1 jobs/lecture-2
# Regenerate every project's notes and illustrations, then rebuild its PDF:
UV_CACHE_DIR=.cache/uv uv run shad-helper notes jobs/lecture-1 jobs/lecture-2 --rebuild
```

Multiple projects run in the supplied order. A failed project or an automatic
generation that requires manual completion is reported; later projects still run.
The batch prints a summary and exits with status 1 if any project failed.
Repeated destinations, including symlink aliases, are rejected before work starts.
`--manual` exports a handoff for each project; `--import` requires one project.

Alternatively, pass `--openai-proxy URL` before the command or after
`doctor`/`login`/`notes`/`anki`/`run`. The flag overrides the environment variable.
URL-encoded proxy credentials are supported; URLs are not saved or printed.
Requests fail if the proxy is missing, invalid, or unavailable, without direct
retries. Other proxy environment variables cannot bypass this setting.
Hugging Face downloads are configured separately.

Configure your existing sign-in browser to use the same proxy before `login`;
keep localhost direct for the OAuth callback. This app's protected credentials
live in `.shad-helper/auth`. Change the location with `--state-dir PATH` before
the command name.

Automatic synthesis checks availability of `gpt-6-astra` for the selected credentials
and uses the existing reasoning configuration through the OpenAI Responses API.
Failed chapter requests, invalid replies, and failures of the chapter's LaTeX check
share three retries after the initial attempt, waiting 5, 10, and 20 seconds. Each
retry sends the original request again and validates the newly generated chapter.
Completed chapters remain cached.
Unavailable access or failure after all four attempts exports a manual handoff.

The app uses `outline.json`, or creates consecutive fifteen-minute chapters.
`notes --rebuild` deletes the saved outline and starts from the default plan.
Each request contains one chapter, neighboring speech and visual context, a
shared glossary, and the broad topic names used in preceding replies. Speech
inputs contain only `id`, `start`, `end`, and `text`.
Video inputs contain the extracted content and timestamps, including the last
preceding frame when a static slide spans a chapter boundary. Version 5 also sends
labeled illustration crops, including source-image fallbacks, so the notes model
can inspect their content directly. Local paths, confidence and extraction diagnostics
are not sent. Only
uncertainties discovered during note generation become note review issues.

Local preflight thresholds split chapters above 1,200,000 UTF-8 bytes of
instructions, context, and response schema, or 40 MB of encoded image attachments.
These are application thresholds, not verified limits of the model endpoint.
There is no local image-count cap. Byte-identical crops share one labeled
attachment. A typed `context_length_exceeded` response also triggers a split and
retry. Splitting stops before creating parts shorter than thirty seconds.
Irreducible context is exported manually without dropping speech or visuals.
Editable outlines must cover the entire interval consecutively in chapters
of at most twenty minutes.

Notes focus on facts, definitions, formulas, theorem conditions and proof steps,
and useful explanations. Speaker narration, administrative discussion, repetition,
and redundant glossary entries are omitted. The editorial target is about ten
pages per three hours, up to twenty for dense lectures. Each chapter receives
duration-scaled guidance of 3,000/6,000 words per three hours (`writing_budget`),
including captions and glossary. This is a soft output allowance, separate from
the input byte cap; no proof or evidence is truncated to hit a page count. Actual
pagination depends on formulas and diagrams. Administrative-only chapters may
have `blocks: []`; their intervals stay in the outline but get no PDF heading.

PDF sections follow lecture chronology: chapters are ordered by their start time,
and blocks retain their sequence within each chapter. Every new block has a broad
`topic`, such as nearest neighbors, linear regression, or merge sort. Only adjacent
blocks with matching topics share a section, including across chapter boundaries.
Returning to a topic after another topic starts a later section. Matching ignores
case and extra whitespace. Chapter intervals remain in `notes.json` for provenance.
Legacy blocks without a topic use their chapter title.

Definitions, theorem statements, and proofs use `definition`, `theorem`, and
`proof` blocks, rendered with LaTeX `amsthm` environments. Definitions and theorems
share numbering within each section; proofs have a Russian heading. Examples,
clarifications, and explanations remain plain prose. The model supplies prose and
inline math, while the renderer supplies the environments.

Notes and Anki share support for the public built-in mathematical commands and
environments of `amsmath` (including its `amsopn`, `amsbsy`, and `amstext`
dependencies), following the [official user guide](https://www.latex-project.org/help/documentation/amsldoc.pdf).
This includes labeled arrows such as `\xrightarrow`, generalized fractions,
operator names, accents, and the `alignat`, `flalign`, and `subequations`
environments. This expansion excludes preamble declarations such as
`\DeclareMathOperator`, document configuration, and package internals, and does not
add commands from separately loaded packages; use `\operatorname{...}` for custom
names. The existing file-access, shell-command, and macro-definition restrictions
remain in effect. Commands must still be used in a valid LaTeX context.

The shared allowlist also covers the public formula commands defined by core
LaTeX's [`fontmath.ltx` (`fontdef.dtx`)](https://github.com/latex3/latex2e/blob/develop/base/fontdef.dtx),
including `\sqcup`, `\sqcap`, the standard symbols, delimiters, and accents. TeX's
eight math atom commands (`\mathord`, `\mathop`, `\mathbin`, `\mathrel`,
`\mathopen`, `\mathclose`, `\mathpunct`, and `\mathinner`) are supported too.
Neither `\mathbin` nor `\sqcup` requires an extra package. This does not enable
all TeX primitives or LaTeX declarations; the same safety restrictions apply.
The legacy symbols `\arrowvert`, `\Arrowvert`, and `\bracevert` are safe and render
with Anki's preamble, but Notes' `unicode-math` package explicitly rejects them.

Lecture blocks cite speech or visual evidence IDs. Questionable phrases or whole inline formulas are listed in
`uncertain_fragments` with their exact text and supporting evidence IDs. The PDF
colors only these fragments and adds one short `hh:mm:ss` footnote per affected
paragraph, combining and deduplicating its timestamps. Longer sets become a time
range. Detailed uncertainty explanations remain in local review JSON instead of
large “Требует проверки” passages. Missing essential formula/diagram details must
be flagged rather than invented. Notebook visuals
contribute theory, formulas and explanatory comments, not executable code.
For a hand-drawn figure, the notes model identifies the teaching step and cites
the lecturer’s explanation. Codex's built-in image-generation tool receives the
selected drawing, up to three distinct views of its construction, and the relevant
timestamped transcript. A visual check compares the generated PNG with the drawing
and speech. A failed or uncertain verdict triggers one correction. Codex receives
the rejected PNG, the verifier’s reason, and the original evidence. The verifier
checks the new PNG against the board and speech. Only a passing illustration goes
into the notes. If a verified illustration is unavailable, the pipeline checks the
source crop for screenshot publication and opens a review issue. Printed figures
use their original pixels. When Vision
leaves a figure unclassified, the crop verifier checks its source type. A confirmed
hand drawing goes to image generation. An ambiguous figure keeps its verified
screenshot and a review issue.
[Codex image generation](https://learn.chatgpt.com/docs/image-generation).
Completed drawings and meaningful intermediate stages are preserved. The notes
model first checks candidate pixels: logos, decoration, lecturer/clothing, code/UI,
and handwriting or formulas without a figure are excluded with an explicit reason.
Useful text and equations remain ordinary note content. Every chapter-owned
candidate has a coverage decision: a figure block or an exclusion reason.
A whole-frame fallback is a candidate, not a mandatory screenshot.
Repeated views share a figure. A text catalog of earlier figures enables reuse
across chapters without reattaching all earlier images. Merging different crops
requires a visual check focused on the actual figure and its integral labels.
Failed matches try already retained alternatives before adding another screenshot.
An automatic fallback is excluded only when native-frame inspection explicitly
finds no meaningful figure content, such as plot headings with only a thin edge.
The observation stays in the evidence, with an exclusion decision and a review
issue. A later chapter can rescue an earlier unresolved observation and restore its
prose references. When a figure already represents another figure, redirecting that
original requires a direct comparison with the new target. An uncertain result
keeps its last verified target in the notes. Repaired crops are compared again
against accepted alternatives before adding a screenshot. Unavailable verification
is retried automatically from the original candidates. If bounded recovery still
fails, source assets and diagnostics retain `unresolved` coverage and an open
issue; the notes contain no manual-recovery placeholders or dangling figure references.
A final comparison retrieves up to three plausible earlier figures by source, topic and description, including
nonconsecutive figures and differently worded topics. It consolidates verified duplicates,
preferring a complete view over an unfinished drawing or partially visible plot.
Distinct teaching stages remain separate; uncertain matches retain both figures.
Selected figures are cropped again from their native frames: include the complete
drawing, axes, labels and legend, excluding surrounding notes, code and people.
This can tighten a large crop or recover content clipped by the initial crop.
During notes finalization, each proposed crop undergoes cheap checks for blank or
thin content and suspicious boundaries, followed by an independent visual check
of the actual crop against the native frame and intended figure. The checks cover
identity and drawing stage, completeness, readability, and unrelated content;
heuristic warnings alone never reject a crop. Before cropping, a bounded visual
selection ranks the original and up to six saved frames spanning 30 seconds on
either side. Prefer sharp direct capture to a filmed projection of the same
state; projector lag must not change the selected parameters or drawing stage.
Completeness is checked against the intended caption: complete named panels do
not require neighboring panels of a scrolling notebook. A harmless mouse cursor
does not invalidate a readable screenshot. For figures taller than the viewport,
matching scroll views can be joined using their original pixels and independently
verified against both source frames before publication.
Failed crops get at most two repairs. Every alternate is independently verified.
Hands, fingers and held objects are excluded from published screenshots without
cropping away the figure. The notes model also receives up to four native context
frames for ambiguous candidates and selects complete representatives before
reconstructing their meaningful stages. See [crop verification](docs/illustration-crops.md)
for the repair limits, saved diagnostics, and validation plan.
Refined crops are separate managed assets; original frames and evidence stay intact.
Actual illustrations are never discarded to satisfy the editorial page target.
Use the [label benchmark](docs/illustration-benchmark.md) to freeze reviewed outputs
and compare later runs without treating removed or changed bad images as fixes.
Ordinary source citations create no footnotes; only marked uncertainty does.
The PDF has a compact title followed by topical sections, and no review
appendix. Review issues remain available in the JSON files and through `review`.
Figures stay in source order beside the surrounding text and do not introduce
sections by themselves. Prose uses `figure_refs` with stable figure block IDs;
the renderer supplies numbered links and updates them after consolidation.
Transcription warnings are informational: inspect or correct them before notes
when needed. They are ignored by note generation rather than carried into review.

Outputs are `notes.json`, `review.json`, and editable `notes/` LaTeX sources.
`notes` regenerates the project. `--rebuild` first deletes `notes/`, `notes.json`,
`review.json`, `outline.json`, the entire `handoff/` including old run archives,
`checkpoints/notes/`, `checkpoints/notes_candidates/`, `checkpoints/notes_final/`,
`illustration-crops/`, and `illustration-verification/`. This happens before job
compatibility or input validation and before model calls. Manual TeX edits, the
old PDF, and custom outline changes in these paths are removed. A failed rebuild
does not restore any deleted files.
Extraction artifacts and checkpoints, original illustrations, and the explicit
`review-resolutions.json` correction ledger remain unchanged. Successful synthesis
creates fresh checkpoints and forces PDF compilation, QA, and source maps; each
project's PDF finishes before the next project starts. Later normal runs can reuse
the fresh checkpoints.
With `--import --rebuild`, imported notes are rendered after import. Copy replies
outside the cleared paths first: an import from `JOB/notes.json` or `JOB/handoff/`
would be deleted before it is read. Directory imports use the freshly defaulted
outline; a complete lecture JSON import can restore its custom chapter plan.
`--manual --rebuild` performs the same cleanup and exports a fresh handoff.
Keep manual TeX edits for subsequent ordinary `render` commands.

Automatic generation compiles each chapter locally with the final notes preamble
before saving its synthesis checkpoint. This requires `latexmk` and LuaLaTeX.
A compilation failure during a generation attempt retries the whole attempt, up
to the shared limit above. Missing compilers stop immediately. If all attempts fail,
the handoff reports the chapter and compiler error, with a block ID when available.
Each attempt's `response.json` and `latex-check/notes/` retain its response, editable
sources, and compiler logs locally.
Compiler diagnostics are not sent to OpenAI. Checkpoints created before this check
are regenerated once; successfully checked chapters are reused on later runs.

Every `notes` run saves a handoff, including successful automatic generation,
cache reuse, and imports. `handoff/inputs/` contains the current chapter prompts,
contexts, and response schema. Current diagnostics are saved directly under
`handoff/`, whose path is printed when the command starts:

- `inputs/`, `README.txt`, and `outline.json`: the current handoff and chapter plan;
  automatic contexts include the actual glossary and topic names accumulated before each chapter.
- `run.json` and `chapters.json`: run mode/result and generated versus cached chapters.
- `requests/<chapter ID>/attempt-01/call.json`: instructions, exact context string,
  application response schema, and model for each generation attempt.
- `http-*.request.json` and `http-*.body.json` within each attempt: request method/URL
  and exact outgoing JSON bytes, including model settings and the API-normalized schema.
- `verification/` and `figure-reconciliation/`: current illustration checks and
  consolidation diagnostics.

Requests are saved before transmission, including retries and failed parent
requests that lead to chapter splitting. If saving fails, the request is not sent.
Manual export makes no model calls. Cached reconstruction/consolidation verdicts
are reused, but imported drawings and new consolidation decisions require visual
verification. Each ordinary `notes` invocation replaces the generated handoff inputs
and current diagnostics while retaining existing `handoff/runs/` archives and
unrelated operator files. `--rebuild` removes the entire handoff directory first.
No new run archives are created. Authentication
headers, tokens, proxy credentials, and OAuth exchanges are excluded. Failures
after the job compatibility check receive a failed `run.json`; cleanup or
compatibility failures can leave no current handoff record.

## Manual ChatGPT workflow

Export without network calls:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper notes jobs/audio-lecture --manual
```

For each chapter listed in the job's `outline.json`, attach
`handoff/inputs/CHAPTER_ID/context.json` and the shared
`handoff/inputs/response-schema.json` in ChatGPT, then paste that chapter's
`prompt.txt`. If `images.json` is present, also attach its listed PNGs with their
illustration IDs. The text inputs supply instructions, lecture data, and response
structure. The context contains the lecture title, glossary, topics, writing budget, chapter
metadata, speech and (for video jobs) extracted visuals once. `handoff/README.txt`
and any `status.json` are operator
instructions and should not be sent to the model.
Initial manual inputs have an empty `topics` list; use consistent broad topic
names across replies. Automatic generation accumulates names from both generated
and cached chapters.
Save JSON replies without Markdown fences as `CHAPTER_ID.json`:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper notes jobs/audio-lecture --import replies/
```

All expected replies must be present. A complete combined response can also
be imported with `--import response.json`. Automatic and manual responses use
the same schema and evidence validation. Imports undergo the same illustration
verification as automatic notes; unavailable verification preserves source crops
and flags them for review. Inputs, candidates, comparison requests, previews and
accepted results are saved in the current handoff diagnostics. Transient
verification failures retry on rerun without regenerating cached candidate chapters.

## Review and render PDF

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper review jobs/audio-lecture
```

Correct transcription in `evidence.json` before regenerating notes.
Edit chapter/TikZ sources directly for note corrections, or edit `notes.json`
and import it to regenerate the project. Record a verified correction explicitly:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper review jobs/audio-lecture --resolve ch001_missing_detail --resolution "Checked the lecture and corrected the missing detail."
UV_CACHE_DIR=.cache/uv uv run shad-helper render jobs/audio-lecture
```

Resolving an issue refreshes the project metadata and draft footer while
preserving chapter edits. A marked fragment or legacy uncertain block stays
tentative until its content is corrected and its markers removed or kind updated.
Resolutions survive imports only while issue and evidence match.
Review commands resolve note issues only; they do not edit evidence or extraction
warnings. Editing the diagnostic report does not invalidate notes or rendering.

Rendering preserves sources and produces `notes/main.pdf`, `notes/qa/` page
previews, and `notes/render-report.json`. Inspect previews for clipped text,
labels, equations, and glyphs. Compilation errors retain logs and sources.
Changed evidence requires regeneration/import before rendering.
TeX references are confined to declared sources, code, and copied PNG assets;
shell escape and arbitrary external files are disabled. The notes project remains
renderable without the original frame/crop directories. Rendering preserves edits
and does not make model calls.

## Checks and synthetic example

```sh
UV_CACHE_DIR=.cache/uv uv run pytest
UV_CACHE_DIR=.cache/uv uv run ruff check shad_helper tests main.py
```

Tests mock model and HTTP responses. FFmpeg integration verifies matching audio-only
and multistream-video soundtracks, delayed audio, and video extending beyond audio.
All three vision presets were also exercised through the main CLI on a real
15-second lecture clip, including model switching, cached resume, and manual
notes handoff. PDF integration uses real LuaLaTeX and PyMuPDF
when installed. These checks do not establish real-lecture ASR accuracy or speed;
assess a representative Russian recording before a full run.

Try note import/rendering without models or network calls:

```sh
mkdir -p jobs/audio-example
cp examples/demo-evidence.json jobs/audio-example/evidence.json
UV_CACHE_DIR=.cache/uv uv run shad-helper notes jobs/audio-example --import examples/demo-response.json
UV_CACHE_DIR=.cache/uv uv run shad-helper render jobs/audio-example
```

The synthetic example exercises Russian text, mathematics, a transcript-based
diagram, and code. See [the pipeline design](docs/plan.md) and the
[stage contracts](docs/stage-contracts.md) for file structures and validation rules.
## Create an Anki deck from the finished PDF

After checking and rendering the notes, run:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper anki jobs/audio-lecture
# Process several projects sequentially:
UV_CACHE_DIR=.cache/uv uv run shad-helper anki jobs/lecture-1 jobs/lecture-2 jobs/lecture-3
# Use a separately edited PDF:
UV_CACHE_DIR=.cache/uv uv run shad-helper anki jobs/audio-lecture --pdf notes.pdf
# Select a profile when more than one Anki collection exists:
UV_CACHE_DIR=.cache/uv uv run shad-helper anki jobs/audio-lecture --anki-collection '/path/to/collection.anki2'
```

With multiple project directories, `anki` processes them one at a time in the
supplied order, saving each deck in its own `JOB/anki/` directory. It continues
after a project fails, prints a completion summary, and exits with status 1 if
any project failed. Options apply to every project; `--pdf` requires a single
project. Repeated destinations, including symlink aliases, are rejected before
generation starts.

The default input is `JOB/notes/main.pdf`. The command sends the complete PDF to
`gpt-6-astra` using the API key or ChatGPT login described above and the explicit
OpenAI proxy. It includes text and page images. For generated lecture PDFs it also
sends a catalog of the saved note blocks and their PDF page ranges to identify
sources. Figures reference their original figure block IDs. Generated illustrations
and screenshots reuse the exact managed PNGs embedded in the notes. Existing TikZ
diagrams render from their saved source. Missing or changed assets stop export rather than falling
back to PDF crops. Raw transcripts, recording paths, timestamps, and pipeline diagnostics
are not model inputs. The PDF must be
unencrypted and smaller than 50,000,000 bytes. If the model cannot process the
whole document or return a complete valid response, the command fails without
publishing a partial deck. Temporary failures get three retries, after 5, 10 and
20 seconds.

Each generated answer and proof/definition step carries its own required source
in the model response. The exporter derives the matching Anki field locally;
the model does not have to keep a separate list of field names in sync.
Raw responses and validation errors are retained locally in
`JOB/diagnostics/anki/<run>/attempt-XX/response.json` and `status.json`.
The terminal prints the diagnostic path on failure. These files are not sent
to the model on retries.

Import `JOB/anki/deck.apkg` into Anki. Each job has a distinct, persistent deck
name ending in a short job ID. The package reuses the four existing note types
`Basic`, `Multistep definition`, `Theorem Full proof`, and `Theorem step by step`,
with their original fields, model IDs, and template order. It includes the
templates, original lecture figures, and precompiled LaTeX images. Standalone PDFs
without a lecture source map still use PDF crops. All math uses native
Anki LaTeX: `[$]…[/$]` inline and `[$$]…[/$$]` for display equations. The editable
LaTeX remains in the fields. Export requires `latex` and `dvipng` (TeX Live/MacTeX);
the included images display after import without compiling them again.

Field export encodes each source newline as a single `<br>`. Anki converts it
back to one newline for LaTeX, preserving multiline formulas without introducing
blank paragraphs in math. This also applies when repackaging a cached response.

Before caching or exporting, Anki math normalization braces row-leading intervals:
`\\[-1,1]` becomes `\\{[-1,1]}`. The same repair applies to cached responses.
Explicit row spacing, including decimal-comma lengths such as `\\[1,2pt]`, is
preserved. Stored fields and precompiled image names use the corrected LaTeX.

Card generation separates statements, complete proofs, proof plans and individual
steps. Definition steps show the common context and complete outline; proof steps
show the statement and complete proof plan. Up to 15 populated logical steps are
supported by the existing templates; unused fields create no cards. Alternative
proofs can be separate steps. The model may supply missing reasoning, marked
**Дополнено моделью** in the affected answer and tagged `model_supplement`.
Unresolved content is reported and tagged `needs_review`.

New-card positions follow **first appearance in the PDF**, including order within
a page. Generated lecture PDFs preserve the chronological chapter/block sequence;
separately supplied PDFs define their own reading order. All cards belonging to
one note share one position. Immediately before
export, the command reads the local Anki collection, including its current WAL,
without modifying it. The first note uses the next free position: the larger of
Anki's allocation counter and the highest existing new-card position plus one.
Later notes increment this position by one. Anki need not be running, but its
collection must be readable; missing or ambiguous profiles require
`--anki-collection`, and export never silently falls back to position 1.
Positions reflect the collection at export time: with separate `anki` commands,
import each deck before exporting the next one. `run` and multi-project `anki`
keep position ranges separate across decks in the same batch. Regenerate packages
if new cards were added in between export and import.
In Anki's deck options, choose **New card gather order: Ascending position** and **New card sort
order: Order gathered**. Sibling burying and daily limits may postpone cards;
reviews continue to use Anki's scheduling. See the
[Anki display-order documentation](https://docs.ankiweb.net/manual/deck-options#display-order).

`JOB/anki/notes.json` preserves the structured generated content;
`JOB/anki/sources.md` links **every exported card**, including individual definition
and proof steps, to a PDF page and a heading, formula, or excerpt. Generated lecture
PDFs also have clickable block anchors and recording timestamps resolved locally
from the block's original speech/frame references. Multiple supporting intervals
are listed separately; these are source-block intervals, not inferred phrase
boundaries. Model additions are marked and their references identify the premises.
A separately supplied PDF has no assumed connection to the job's recording; its
report explicitly marks timestamps as unavailable.

`render` writes `JOB/notes/source-map.json`, binding block IDs and page ranges to
the exact PDF hash and lecture evidence. Missing, stale, or invalid references
stop lecture export. For projects generated before source maps were introduced,
back up any manual TeX corrections, run `notes JOB --import JOB/notes.json` to
recreate the project, reapply the corrections while retaining the block anchors,
then run `render JOB` and `anki JOB`.

`JOB/anki/report.json` records the same per-card sources, note positions, model
additions, warnings, and image coordinates. `JOB/anki/media/` contains figures and
compiled formulas. The deck, JSON sidecars, and `sources.md` are published together;
a failed ordinary export keeps the previous versions.
Successful generation is cached against the PDF bytes, block catalog, model, prompt and schema;
a cached export needs no model request. Use `anki JOB --rebuild` to regenerate only this
stage: before PDF, collection, or other input validation, it deletes the entire
`JOB/anki/`, `JOB/checkpoints/anki.json`, and `JOB/diagnostics/anki/`. This removes
the prior package, media, diagnostics, and saved deck identity. A rebuilt export
gets a new deck identity and note GUIDs, so importing it creates a new deck and
notes rather than updating the previous import. Failure cannot restore the old
package. Validated generation writes a fresh checkpoint; a later packaging failure
can leave that checkpoint reusable. Keep a custom `--pdf` outside these cleared
paths, copying it beforehand if necessary. Notes, their PDF, and extraction
artifacts remain unchanged.
Reimporting the same output creates no duplicates and preserves existing
positions, review progress and local template edits. It does not reposition cards
already imported into Anki. Without `--rebuild`, changing the PDF creates new note
identities in the same job's deck; remove obsolete imported notes manually.

Anki export tests include real imports into disposable collections, covering
positions, media, model reuse and repeat-import scheduling. They never modify
your personal Anki collection.
