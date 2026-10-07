# shad-helper

A CLI that turns Russian lecture audio into detailed, transcript-grounded LaTeX
notes and PDF. Audio files and video soundtracks share local transcription.
Notes include equations, code, TikZ/PGFPlots diagrams, timestamps, and review items.

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

The first transcription downloads Whisper large-v3 through MLX Whisper.
Caches live under `.cache/huggingface`. The ASR worker exits after transcription
to release model memory. Once tools and weights are installed, extraction works
offline; use `HF_HUB_OFFLINE=1` to disable Hugging Face network access.

With app-owned credentials and a configured OpenAI proxy, `doctor` also lists
account-available models and reports default-model availability. It does not
start login or generate notes. Local checks determine its exit status.

## Extract audio evidence

Start with a representative ten-minute interval:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper extract lecture.m4a --start 1800 --duration 600 --out jobs/audio-pilot
```

For a full recording, omit the interval options. Video files work the same way:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper extract lecture.mp4 --out jobs/audio-lecture
```

FFmpeg selects the first audio stream and produces 16 kHz mono WAV chunks.
Ten-minute core intervals have five-second overlaps; each word belongs to one
core interval. JSON, SRT, and VTT timestamps refer to the original recording.
Russian is the default; override transcription language with `--language`.
Notes remain in Russian.

Jobs contain `extraction-config.json`, schema-version-2 `evidence.json`,
`transcript.json`, `transcript.srt`, `transcript.vtt`, audio chunks, worker
requests/responses, successful checkpoints, and current-run extraction timings.
Evidence stores media identity and interval, timed speech, local diagnostics,
review issues, and the ASR model.

Reruns reuse matching audio and ASR checkpoints. Changes to media, intervals,
language, or model invalidate affected results. Deleted audio artifacts are
rebuilt, and interrupted ASR resumes at the last completed chunk.

Use a **new output directory**. Legacy jobs are rejected before modifying their
files. Existing jobs, PDFs, and downloaded model caches are preserved.
Silent extraction keeps inspectable artifacts; note generation requires speech.

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
Unavailable access or generation failure exports a manual handoff.

The app uses `outline.json`, or creates consecutive fifteen-minute chapters.
Each request contains one chapter, neighboring speech context, and a shared
glossary. Speech inputs contain only `id`, `start`, `end`, and `text`.
Confidence, diagnostics, and existing issues stay local and are restored in
resulting review records.

Chapters split above 120,000 UTF-8 bytes of instructions, context, and response
schema. A typed `context_length_exceeded` response also triggers a split and
retry. Splitting stops before creating parts shorter than thirty seconds.
Irreducible context is exported manually without dropping speech.
Editable outlines must cover the entire interval consecutively in chapters
of at most twenty minutes.

Lecture blocks cite transcript IDs. Added explanations are **Пояснение**;
uncertain claims are **Требует проверки**. Unspoken details of a referenced
formula, diagram, or code display must be flagged rather than invented.
Generation cannot clear existing transcription issues.

Outputs are `notes.json`, `review.json`, and editable `notes/` LaTeX sources.
`notes` regenerates the project; `--rebuild` invalidates synthesis checkpoints.
Keep manual TeX edits for the subsequent `render` step.

## Manual ChatGPT workflow

Export without network calls:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper notes jobs/audio-lecture --manual
```

For each chapter in `handoff/outline.json`, attach its `context.json` and shared
`handoff/response-schema.json` in ChatGPT, then paste its `prompt.txt`.
`transcript.txt` is an optional readable copy for local inspection.
Save JSON replies without Markdown fences as `CHAPTER_ID.json`:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper notes jobs/audio-lecture --import replies/
```

All expected replies must be present. A complete combined response can also
be imported with `--import response.json`. Automatic and manual responses use
the same schema and evidence validation. Success and imports skip handoff export.

## Review and render PDF

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper review jobs/audio-lecture
```

Correct transcription in `evidence.json` before regenerating notes.
Edit chapter/TikZ sources directly for note corrections, or edit `notes.json`
and import it to regenerate the project. Record a verified correction explicitly:

```sh
UV_CACHE_DIR=.cache/uv uv run shad-helper review jobs/audio-lecture --resolve issue_0000 --resolution "Checked the audio and corrected the subscript."
UV_CACHE_DIR=.cache/uv uv run shad-helper render jobs/audio-lecture
```

Resolving an issue refreshes the review appendix and footer while preserving
chapter edits. An uncertain block stays tentative until its content and kind
are corrected. Resolutions survive imports only while issue and evidence match.

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

Tests mock ASR and HTTP responses. FFmpeg integration verifies matching audio-only
and multistream-video soundtracks. PDF integration uses real LuaLaTeX and PyMuPDF
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
diagram, and code. See [the pipeline design](docs/plan.md).
Anki remains a separate, unimplemented command.
