# shad-helper

A CLI that turns Russian lecture recordings into detailed, evidence-grounded LaTeX
notes and PDF. Audio files and video soundtracks share local transcription;
video frames also supply text, formulas, and cropped instructional illustrations.
Notes include equations, code, TikZ/PGFPlots diagrams, timestamps, and review items.

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

With app-owned credentials and a configured OpenAI proxy, `doctor` also lists
account-available models and reports default-model availability. It does not
start login or generate notes. Local checks determine its exit status.

## Run the complete pipeline

After installing dependencies, signing in, and configuring the OpenAI proxy, run:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper run lecture.mp4 --out jobs
UV_CACHE_DIR=.cache/uv uv run shad-helper run 'Lecture 1.mp4' 'Lecture 2.mkv' --out jobs --vision-model 9b
```

Each file gets its own job under `--out`, named `<filename-stem>-<path-hash>`.
The hash uses the resolved absolute input path, so recordings with identical names
in different directories remain separate. Repeating the command with the same
files and output directory reuses the same jobs, regardless of input order.
Repeated paths or symlink aliases in one invocation are rejected before work starts.
Audio files are also supported.

Files run in the supplied order. For each file, `run` completes **extract → notes
→ render → anki** before starting the next file. It prints the job directory and
current stage; outputs include `JOB/notes/main.pdf` and `JOB/anki/deck.apkg`.
An error stops that file's remaining stages and processing continues with the next
file. A manual notes handoff also counts as an incomplete job, so an older PDF is
never used in its place. The final summary reports completed/failed jobs, and the
command exits with status 1 if any job failed.

The extraction options `--start`, `--duration`, `--language`, and `--vision-model`
apply to every input. `--rebuild` regenerates notes and Anki model responses;
otherwise existing stage checkpoints are reused. Rerunning the full pipeline
regenerates the editable TeX sources, so use the individual commands to retain
manual TeX corrections. `--anki-collection` selects the local Anki profile, and
`--openai-proxy` and the global `--state-dir` work as for individual commands.

Each job produces its own Anki deck. Within one invocation, successive decks use
non-overlapping position ranges even before import; all cards from one note still
share a position. Each export also checks the live Anki allocation. Import the
completed decks before starting a separate batch, since separate invocations do
not reserve positions for each other.

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
content. Changes to media, intervals, language, model, or vision prompt/settings
invalidate affected results. Deleted audio/frame artifacts are rebuilt. Interrupted
work resumes from completed chunks/frames. Invalid vision JSON gets one retry;
an unsuccessful frame remains explicitly `incomplete` and is retried on rerun.

Version 3 audio and version 4/5 video jobs can be reused, including enabling or
disabling vision in an existing supported job. Older formats require a **new
output directory** and are rejected before changing their files. Re-extraction
preserves existing notes/PDFs, but changed evidence requires regenerating notes
before rendering. A fatal vision error preserves the completed transcript and
audio evidence and returns an error; rerun the extraction to complete video.
Silent extraction keeps inspectable artifacts; notes require speech, useful
visual content, or an illustration candidate. Re-extract version 4 jobs to obtain
version 5 crops; existing transcription and frame selection are reused.

A transcription smoke check processes thirty seconds and may download Whisper:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper doctor --smoke short-lecture.wav
```

## Generate notes with your ChatGPT plan

Every app-owned OpenAI request requires an explicit HTTP/HTTPS proxy:

```sh
export SHAD_HELPER_OPENAI_PROXY=http://127.0.0.1:7890
UV_CACHE_DIR=.cache/uv uv run shad-helper login
UV_CACHE_DIR=.cache/uv uv run shad-helper notes jobs/audio-lecture
```

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

Automatic synthesis checks account availability of `gpt-6.1-sol` and requests
`high` reasoning effort through the authorized ChatGPT-plan integration.
Each failed chapter request or invalid reply gets three retries after the initial
attempt, waiting 5, 10, and 20 seconds. Completed chapters remain cached.
Unavailable access or failure after all four attempts exports a manual handoff.

The app uses `outline.json`, or creates consecutive fifteen-minute chapters.
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

Lecture blocks cite speech or visual evidence IDs. Questionable phrases or whole inline formulas are listed in
`uncertain_fragments` with their exact text and supporting evidence IDs. The PDF
colors only these fragments and adds one short `hh:mm:ss` footnote per affected
paragraph, combining and deduplicating its timestamps. Longer sets become a time
range. Detailed uncertainty explanations remain in local review JSON instead of
large “Требует проверки” passages. Missing essential formula/diagram details must
be flagged rather than invented. Notebook visuals
contribute theory, formulas and explanatory comments, not executable code.
Drawings become editable TikZ/PGFPlots and undergo a separate visual comparison
between the rendered drawing and its source crop. Only an explicit pass accepts
the reconstruction. Uncertain, failed, uncompilable, or unverifiable drawings use
the source crop and create a review issue; slide illustrations use crops directly.
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
Transient comparison failures are retried. A final comparison of neighboring
figures within each topic also consolidates independently generated duplicates,
preferring a complete view over an unfinished drawing or partially visible plot.
Distinct teaching stages remain separate; uncertain matches retain both figures.
Selected figures are cropped again from their native frames: include the complete
drawing, axes, labels and legend, excluding surrounding notes, code and people.
This can tighten a large crop or recover content clipped by the initial crop.
Refined crops are separate managed assets; original frames and evidence stay intact.
Actual illustrations are never discarded to satisfy the editorial page target.
Ordinary source citations create no footnotes; only marked uncertainty does.
The PDF has a compact title followed by topical sections, and no review
appendix. Review issues remain available in the JSON files and through `review`.
Figures stay in source order beside the surrounding text and do not introduce
sections by themselves. Prose uses `figure_refs` with stable figure block IDs;
the renderer supplies numbered links and updates them after consolidation.
Transcription warnings are informational: inspect or correct them before notes
when needed. They are ignored by note generation rather than carried into review.

Outputs are `notes.json`, `review.json`, and editable `notes/` LaTeX sources.
`notes` regenerates the project; `--rebuild` invalidates synthesis checkpoints.
Keep manual TeX edits for the subsequent `render` step.

Automatic generation compiles each chapter locally with the final notes preamble
before saving its synthesis checkpoint. This requires `latexmk` and LuaLaTeX.
A compilation failure stops that chapter without another model request; the handoff
reports the chapter and compiler error, with a block ID when available. `response.json` and
`latex-check/notes/` retain the response, editable sources, and compiler logs locally.
Compiler diagnostics are not sent to OpenAI. Checkpoints created before this check
are regenerated once; successfully checked chapters are reused on later runs.

Every `notes` run saves a handoff, including successful automatic generation,
cache reuse, and imports. `handoff/inputs/` contains the current chapter prompts,
contexts, and response schema. Each run also gets a separate archive under
`handoff/runs/<run ID>/`, whose path is printed when the command starts:

- `inputs/`, `README.txt`, and `outline.json`: the run's handoff and chapter plan;
  automatic contexts include the actual glossary and topic names accumulated before each chapter.
- `run.json` and `chapters.json`: run mode/result and generated versus cached chapters.
- `requests/<chapter ID>/attempt-01/call.json`: instructions, exact context string,
  application response schema, and model for each generation attempt.
- `http-*.request.json` and `http-*.body.json` within each attempt: request method/URL
  and exact outgoing JSON bytes, including model settings and the API-normalized schema.

Requests are saved before transmission, including retries and failed parent
requests that lead to chapter splitting. If saving fails, the request is not sent.
Manual export makes no model calls. Cached reconstruction/consolidation verdicts
are reused, but imported drawings and new consolidation decisions require visual
verification. Earlier
run archives survive later runs and `--rebuild`. Authentication headers, tokens,
proxy credentials, and OAuth exchanges are excluded. A run that fails before
inputs can be prepared still has a failed `run.json` record.

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
accepted results remain in local archives. Transient verification failures retry
on rerun without regenerating cached candidate chapters.

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
# Use a separately edited PDF:
UV_CACHE_DIR=.cache/uv uv run shad-helper anki jobs/audio-lecture --pdf notes.pdf
# Select a profile when more than one Anki collection exists:
UV_CACHE_DIR=.cache/uv uv run shad-helper anki jobs/audio-lecture --anki-collection '/path/to/collection.anki2'
```

The default input is `JOB/notes/main.pdf`. The command sends the complete PDF to
`gpt-6.1-sol` with high reasoning through the existing ChatGPT sign-in and explicit
OpenAI proxy. It includes text and page images. For generated lecture PDFs it also
sends a catalog of the saved note blocks and their PDF page ranges to identify
sources. Raw transcripts, recording paths, timestamps, and pipeline diagnostics
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
templates, cropped PDF figures, and precompiled LaTeX images. All math uses native
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
import each deck before exporting the next one. `run` keeps position ranges
separate across decks in the same batch. Regenerate packages if new cards were
added in between export and import.
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
a failed export keeps the previous versions.
Successful generation is cached against the PDF bytes, block catalog, model, prompt and schema;
a cached export needs no model request. Use `anki JOB --rebuild` to regenerate only this
stage. Reimporting the same output creates no duplicates and preserves existing
positions, review progress and local template edits. It does not reposition cards
already imported into Anki. Changing the PDF creates new note
identities in the same job's deck; remove obsolete imported notes manually.

Anki export tests include real imports into disposable collections, covering
positions, media, model reuse and repeat-import scheduling. They never modify
your personal Anki collection.
