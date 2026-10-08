# shad-helper

A CLI that turns Russian lecture recordings into detailed, evidence-grounded LaTeX
notes and PDF. Audio files and video soundtracks share local transcription;
video frames also supply text, formulas, and diagram descriptions.
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
uses version 4 and adds timestamped `visuals`: source IDs, completion status,
content kind, text, formulas, diagram descriptions and unreadable regions.
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

Version 3 audio and version 4 video jobs can be reused, including enabling or
disabling vision in an existing supported job. Older formats require a **new
output directory** and are rejected before changing their files. Re-extraction
preserves existing notes/PDFs, but changed evidence requires regenerating notes
before rendering. A fatal vision error preserves the completed transcript and
audio evidence and returns an error; rerun the extraction to complete video.
Silent extraction keeps inspectable artifacts; notes require speech or useful
visual content.

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
`doctor`/`login`/`notes`. The flag overrides the environment variable.
URL-encoded proxy credentials are supported; URLs are not saved or printed.
Requests fail if the proxy is missing, invalid, or unavailable, without direct
retries. Other proxy environment variables cannot bypass this setting.
Hugging Face downloads are configured separately.

Configure your existing sign-in browser to use the same proxy before `login`;
keep localhost direct for the OAuth callback. This app's protected credentials
live in `.shad-helper/auth`. Change the location with `--state-dir PATH` before
the command name.

Automatic synthesis checks account availability of `gpt-6-astra` and requests
`high` reasoning effort through the authorized ChatGPT-plan integration.
Each failed chapter request or invalid reply gets three retries after the initial
attempt, waiting 5, 10, and 20 seconds. Completed chapters remain cached.
Unavailable access or failure after all four attempts exports a manual handoff.

The app uses `outline.json`, or creates consecutive fifteen-minute chapters.
Each request contains one chapter, neighboring speech and visual context, and a
shared glossary. Speech inputs contain only `id`, `start`, `end`, and `text`.
Video inputs contain the extracted content and timestamps, including the last
preceding frame when a static slide spans a chapter boundary. Raw frame images,
local frame paths, confidence and extraction diagnostics are not sent. Only
uncertainties discovered during note generation become note review issues.

Chapters split above 120,000 UTF-8 bytes of instructions, context, and response
schema. A typed `context_length_exceeded` response also triggers a split and
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

Lecture blocks cite speech or visual evidence IDs. Added explanations are
**Пояснение**. Questionable phrases or whole inline formulas are listed in
`uncertain_fragments` with their exact text and supporting evidence IDs. The PDF
colors only these fragments and adds one short `hh:mm:ss` footnote per affected
paragraph, combining and deduplicating its timestamps. Longer sets become a time
range. Detailed uncertainty explanations remain in local review JSON instead of
large “Требует проверки” passages. Missing essential formula/diagram details must
be flagged rather than invented. Notebook visuals
contribute theory, formulas and explanatory comments, not executable code.
Diagrams remain editable TikZ/PGFPlots; source screenshots are not placed in PDFs.
Ordinary source citations create no footnotes; only marked uncertainty does.
The PDF has a compact title followed by content, chapter intervals, and no review
appendix. Review issues remain available in the JSON files and through `review`.
Transcription warnings are informational: inspect or correct them before notes
when needed. They are ignored by note generation rather than carried into review.

Outputs are `notes.json`, `review.json`, and editable `notes/` LaTeX sources.
`notes` regenerates the project; `--rebuild` invalidates synthesis checkpoints.
Keep manual TeX edits for the subsequent `render` step.

Every `notes` run saves a handoff, including successful automatic generation,
cache reuse, and imports. `handoff/inputs/` contains the current chapter prompts,
contexts, and response schema. Each run also gets a separate archive under
`handoff/runs/<run ID>/`, whose path is printed when the command starts:

- `inputs/`, `README.txt`, and `outline.json`: the run's handoff and chapter plan;
  automatic contexts include the actual glossary accumulated before each chapter.
- `run.json` and `chapters.json`: run mode/result and generated versus cached chapters.
- `requests/<chapter ID>/attempt-01/call.json`: instructions, exact context string,
  application response schema, and model for each generation attempt.
- `http-*.request.json` and `http-*.body.json` within each attempt: request method/URL
  and exact outgoing JSON bytes, including model settings and the API-normalized schema.

Requests are saved before transmission, including retries and failed parent
requests that lead to chapter splitting. If saving fails, the request is not sent.
Cached, manual, and imported chapters create no HTTP request records. Earlier
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
`prompt.txt`. These are the three model inputs: instructions, lecture data, and
response structure. The context contains the lecture title, glossary, writing budget, chapter
metadata, speech and (for video jobs) extracted visuals once. `handoff/README.txt`
and any `status.json` are operator
instructions and should not be sent to the model.
Save JSON replies without Markdown fences as `CHAPTER_ID.json`:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper notes jobs/audio-lecture --import replies/
```

All expected replies must be present. A complete combined response can also
be imported with `--import response.json`. Automatic and manual responses use
the same schema and evidence validation. Success and imports refresh and archive
the handoff without making additional model calls.

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
TeX references are confined to declared sources and code; shell escape and
external images are disabled.

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
Anki remains a separate, unimplemented command.
