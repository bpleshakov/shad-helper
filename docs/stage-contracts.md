# Recording to PDF stage contracts

The pipeline turns a recording into timestamped speech and visual evidence,
Russian notes, an editable LaTeX project, and a PDF. Each boundary below specifies
the files produced, their structure, and the rules the next stage applies before
using them.

This contract describes the current implementation. It covers audio-only version
3 jobs, audio/video version 4/5 jobs, automatic synthesis, manual ChatGPT handoff
and import, review, and rendering. Video extraction adds local frame selection
and recognition of slides, physical or digital whiteboards, and notebook theory.
Version 5 supplies clean evidence and labeled source crops to synthesis; recognition
diagnostics stay local. Drawings use visually verified TikZ/PGFPlots, with source
crops as the fallback. Importing original slide or notebook files is unsupported.
The version 4 field tables below describe the retained legacy format; version 5
extends them as specified in the following section.

## Illustration contract (version 5)

Explicit video re-extraction writes version 5 while reusing audio, transcription,
and frame-selection caches. Existing version 3/4 jobs remain readable. Version 4
inputs require re-extraction before illustration coverage can be enforced.

Each visual adds `illustrations`, an array of objects with exactly `id`,
`source_kind` (`drawn`, `printed`, or `unknown`), `box` (normalized left/top/right/bottom),
`description`, and `image_sha256`. IDs append `_illNN` to the frame ID. Crop bounds
include a three-percent frame margin, clamped and rounded to native pixels. Worker
output contains only source kind, detected box, and description; extraction assigns
IDs, hashes, and candidate crop bounds. Diagram descriptions alone do not require a
crop: formulas or matrices may be described there without an illustration.
Exhausted extraction retries retain an unknown full-frame
candidate instead of silently dropping instructional content.

`illustrations/manifest.json` contains `{version: 1, assets: {ID: {path, sha256}}}`.
Paths are job-relative managed PNGs. Every read validates confinement, the evidence
and manifest hashes, and PNG decoding. These local paths do not enter model context.

Chapter context adds `required_illustration_ids` for candidates owned by the core
interval and a text catalog `prior_illustrations` of accepted figures from all
earlier chapters. Each chapter returns
`illustration_coverage: [{illustration_id, block_id, exclusion_reason}]`.
Every required ID occurs exactly once. Real figures target a block in this or an
earlier chapter, with an empty exclusion reason. False candidates have an empty
block ID and one explicit reason: `decoration`, `lecturer_or_clothing`,
`code_or_interface`, `text_or_formulas`, or `no_instructional_visual`. Excluded
candidates cannot produce figure blocks. Both/neither decisions and unsupported
reasons are invalid. Older represented decisions default to an empty reason.
A text citation alone does not preserve a real illustration. Repeated views share
a block; meaningful stages remain separate. Full-frame fallbacks undergo the same
classification and are not automatically included.

`diagram` and new `screenshot` blocks select their source using `illustration_id`
and cite it in `evidence_ids`. Screenshot content is a description, not a path or
TeX command. Printed/unknown sources use screenshots; drawn sources prefer TikZ.
The chapter model receives labeled PNG attachments (`input_image`, high detail).
Prior catalog images are attached only when already visible or byte-identical to
current context; proposed reuse is checked separately using both source images.
Identical hashes share one attachment; `images.json` records all its observation
IDs and the relative PNG filename in handoffs and current request records.

Before accepting a drawing, the pipeline compiles an isolated preview and compares
it with the source in a separate model call. Only `pass` preserves TikZ; `fail`,
`uncertain`, compilation failure, or unavailable verification preserves a source
screenshot and opens a review issue. Consolidation of different source hashes also
requires visual verification focused on the figure and integral labels, ignoring
unrelated surrounding text, formulas, code, people and decoration. A failed match
tries retained alternative representatives from its group before creating another
crop block. Excluded candidates skip reconstruction and consolidation checks.
Imported drawings receive the same checks. Candidates, exact credential-free model
requests, previews, verdicts, and accepted blocks are saved in the current handoff diagnostics.

Candidate chapters use `checkpoints/notes_candidates/`; completed chapter compilation
uses `checkpoints/notes_final/`. Verification caches bind source pixels, drawing code,
rendering/verifier policies, model settings, and preview hashes. Temporary verifier
failures are retried on rerun from the cached candidate. Ordinary legacy chapters
retain `checkpoints/notes/`.

The version 2 project manifest adds optional `image_files`. The renderer copies
selected PNGs into `notes/figures/illustration-<ID>.png`, emits confined
`includegraphics` references, and preserves source anchors. The copied project
renders without original crops. Model-authored drawing code cannot load files.
Chapter checks accept the original asset job as their asset source.

Accepted figures undergo native-frame crop refinement. The model sees the full
frame and original candidate crop, and selects tight pixel bounds containing the
whole figure, axes, labels and legend. Bounds may expand beyond the original crop
to recover clipped content. The block's `illustration_crop_id` selects a separately
managed derivative under `illustration-crops/`; empty selects the original crop.
Generated notes leave this field empty until refinement fills it. Refinement caches
bind frame and candidate hashes, selected figure, prompt/schema and model. Resolvers
check source-ID/hash binding, path confinement and derivative PNG integrity.
The renderer copies the selected derivative into the portable project; original
evidence and crops are preserved. Unavailable or uncertain refinement retains the
original crop with a review issue.

The editorial page target never removes illustrations. Local request thresholds
are 1,200,000 text bytes and 40 MB of base64 image data; these are application
choices, not verified model-endpoint limits. There is no local image-count cap.
Existing time splitting and complete manual handoff handle oversized contexts.

## Contents

- [Stage boundaries](#stage-boundaries)
- [Shared conventions](#shared-conventions)
- [Recording to audio chunks](#recording-to-audio-chunks)
- [Audio chunks to raw transcription](#audio-chunks-to-raw-transcription)
- [Video to visual evidence](#video-to-visual-evidence)
- [Raw transcription to evidence](#raw-transcription-to-evidence)
- [Evidence to chapter inputs](#evidence-to-chapter-inputs)
- [Chapter inputs to validated notes](#chapter-inputs-to-validated-notes)
- [Review state and resolutions](#review-state-and-resolutions)
- [Validated notes to LaTeX](#validated-notes-to-latex)
- [LaTeX to PDF](#latex-to-pdf)
- [Resume and artifact freshness](#resume-and-artifact-freshness)
- [Implementation references](#implementation-references)

## Stage boundaries

| Stage | Entry point | Reads | Produces | Completion result |
| --- | --- | --- | --- | --- |
| Prepare audio | Internal part of `extract` | Original recording | Job marker, WAV chunks, audio checkpoints | Every requested core interval has a WAV descriptor |
| Transcribe audio | Local worker started by `extract` when the clip overlaps audio | Worker request, WAV chunks, ASR checkpoints | Worker response, ASR checkpoints | Worker exits successfully and returns all requested items, or is skipped for an empty audio intersection |
| Extract video | Local selection and vision worker started by `extract`, after ASR exits or is skipped | Original recording, selected frames, vision checkpoints | Frame images, recognized content, separate local diagnostics | Every selected frame has complete or incomplete extraction status |
| Assemble evidence | Internal part of `extract` | Raw worker responses | Clean evidence, separate diagnostics, transcript copies, extraction timings | `extract` returns the path to `evidence.json` |
| Prepare chapters | Before vision in a full video run, or during standalone `notes` | Evidence, optional saved outline | Saved outline and in-memory chapter contexts | A consecutive outline covers the extracted interval |
| Synthesize or import | `notes`, optionally `--manual` or `--import` | Chapter contexts, schema, optional replies and resolutions | Current handoff and diagnostics replaced on every run; notes, review summary, and LaTeX project on success | A project path means notes were generated; a handoff path means replies are still required |
| Review | `review`, optionally `--resolve` | Clean evidence, notes, review files and project | Resolution ledger, updated note issues and review metadata | A named note issue has an explicit correction recorded |
| Render | `render` | Existing LaTeX project, manifest, current evidence | PDF, page previews, report and compiler files | PDF compilation and page preview generation both complete |

The main stage boundary is `evidence.json` → `notes.json` → the files declared
by `notes/project.json` → `notes/main.pdf`. The worker files and checkpoints
support extraction and resume. Transcript copies support inspection. The handoff
files provide an alternative way to obtain the same note response format.

Full video runs also publish chapter readiness under `checkpoints/chapter-pipeline/ready/`.
The notes thread can read those snapshots during extraction. Publication of the combined
LaTeX project requires the completed extraction and a verified checkpoint for every
notes chapter. [Chapter pipeline](chapter-pipeline.md)

Review is optional in the command sequence. Open issues do not prevent rendering;
they produce a draft footer and remain available in the review JSON files and
the `review` command. The PDF contains no review appendix. Only questionable
fragments receive source-time footnotes, combined per paragraph.
Extraction warnings are separate informational data. Notes ignore them without
requiring a resolution or an acknowledgment. A note issue instead records a
missing explanation, formula detail, or other uncertainty discovered during
synthesis; it remains relevant even when the transcription is accurate.

### Job directory

Paths in this tree are relative to the job passed through `--out` or `JOB`.
Files marked conditional appear only when that branch runs or that content exists.
`run-diagnostics.json` describes the latest pipeline/stage invocation: source commit
hash, dirty working tree flag, UTC start/end, total elapsed seconds, and ordered,
nested step timings with completion/cache/failure/interruption status. Parent times
include their children. Cache hits measure current lookup time, not saved generation
time. This file is replaced per invocation, updated as steps finish, and also written
on stage validation failures. A directory containing only this file can start extraction.

~~~text
JOB/
├── run-diagnostics.json                      # latest commit and per-step wall times
├── extraction-config.json
├── audio/
│   └── chunk_0000.wav                         # one file per audio core interval
├── workers/
│   ├── asr.request.json
│   ├── asr.response.json
│   ├── vision.request.json                   # video extraction only
│   └── vision.response.json                  # video extraction only
├── frames/                                  # video extraction only; native PNGs
├── diagnostics/vision/                      # local per-frame raw attempts
├── checkpoints/
│   ├── audio/chunk_0000.json
│   ├── asr/chunk_0000.json
│   ├── video/selection.json                  # video selection checkpoint
│   ├── vision/<frame ID>.json                # successful visual extraction only
│   └── notes/ch001.json                       # automatic synthesis only
├── evidence.json
├── extraction-diagnostics.json
├── transcript.json
├── transcript.srt
├── transcript.vtt
├── extraction-timings.json
├── outline.json
├── handoff/                                  # current inputs and diagnostics, replaced each notes run
│   ├── README.txt
│   ├── status.json                           # fallback or unresolved size warning
│   ├── run.json                              # current mode and completion state
│   ├── outline.json                          # current final chapter plan
│   ├── chapters.json                         # generated/cached automatic chapters
│   ├── verification/                         # current illustration checks
│   ├── figure-reconciliation/                # current consolidation checks
│   ├── requests/<chapter ID>/attempt-01/
│   │   ├── call.json                         # application-level model call
│   │   ├── status.json                       # attempt result
│   │   ├── http-001.request.json             # method, URL, body filename
│   │   └── http-002.body.json                # exact POST bytes; GET has no body
│   └── inputs/
│       ├── response-schema.json
│       └── ch001/
│           ├── context.json
│           └── prompt.txt
├── notes.json
├── review.json
├── review-resolutions.json                   # after an explicit resolution
└── notes/
    ├── project.json
    ├── main.tex
    ├── chapters/section-001.tex
    ├── figures/diagram-ch001_diagram.tex      # diagram blocks only
    ├── code/ch001_code.txt                    # code blocks only
    ├── main.pdf                              # successful compiler output
    ├── render-report.json
    ├── compiler-output.log
    ├── main.log                              # compiler diagnostic log
    ├── main.aux                              # compiler-managed files
    ├── main.out
    ├── main.toc
    ├── main.fls
    ├── main.fdb_latexmk
    └── qa/page-001.png                        # one preview per PDF page
~~~

Manual reply files are user-created inputs, usually in a separate `replies/`
directory. They are not created by the pipeline. Credentials under
`.shad-helper/auth` and model downloads under `.cache/huggingface` are runtime
dependencies, not files passed between these stages. The original recording is
referenced rather than copied into the job.

## Shared conventions

### Encoding and JSON

Text artifacts use UTF-8. The common JSON writer uses two-space indentation,
preserves Russian characters, and appends a newline. It writes a temporary
sibling named `.<filename>.<process ID>.tmp`, then replaces the destination.
The replacement is atomic for that file; a group of stage outputs is not a
transaction.

The LaTeX manifest and render report use the same readable JSON formatting but
are written directly, rather than through the atomic JSON helper. Plain text,
LaTeX, WAV, PDF, compiler files, and previews also have independent writes.

JSON objects described below are complete producer formats unless explicitly
called an excerpt. Evidence is a dictionary with enforced root and segment field
boundaries. Diagnostics and worker formats are ordinary dictionaries.
Outline and note objects are validated with Pydantic and forbid additional
fields at every modeled object level. Pydantic can perform normal type
coercion; these models do not enable strict primitive-type validation.

“Required” in the note tables means required when importing JSON. Fields with
defaults may be omitted during import and are filled during normalization.
Automatic API responses use a stricter transport schema that requires every
declared field. Saved normalized notes include the defaulted fields.

### Time coordinates

All durations and timestamps are seconds, represented as JSON numbers.

| Location | Coordinate system |
| --- | --- |
| Media bounds, chunk `offset`, `core_start`, `core_end` | Absolute seconds from the beginning of the original recording |
| Raw Whisper segment and word `start` and `end` | Seconds from the beginning of that chunk's WAV |
| Clean evidence segment and diagnostic word `start` and `end` | Absolute seconds in the original recording |
| Visual evidence `timestamp` | Selected frame's source time relative to the original container start |
| Outline, notes and handoff chapter bounds | Absolute seconds in the original recording |
| Subtitle cues | Absolute recording timestamps, rounded to milliseconds |
| Human-readable timestamps and PDF source labels | Absolute recording timestamps, floored to whole seconds |

A pilot extraction does not reset its timestamps to zero. A clip starting at
1800 seconds produces source labels beginning around `00:30:00`.

Core ownership uses a half-open interval: `core_start <= midpoint < core_end`.
Chapter context selection instead uses inclusive intersection with an expanded
chapter interval. This distinction allows neighboring context without assigning
the same recognized word to two adjacent audio cores.

### Identifiers and paths

Generated chunk IDs are zero-based `chunk_0000`, `chunk_0001`, and so on.
Normalized segment IDs are zero-based `segment_00000`. Extraction warning IDs
are zero-based `issue_0000`. These numbers are minimum display widths, not
limits on the number of items.

Default chapter IDs are one-based `ch001`, `ch002`, and so on. Note, chapter,
outline chapter, and review issue IDs must match:

~~~text
^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$
~~~

They contain at most 80 ASCII characters and start with a letter or digit.
Within a complete note object, chapter, block, and issue IDs must all be unique
across those three categories. A chapter prefix on block IDs is requested in the
prompt but is not a separate validator requirement.

Segment IDs identify the current normalized extraction. They are assigned by
output order and can change after retranscription. They are not persistent
identities for a spoken sentence across different extractions.

Visual IDs have the form `frame_<stream index>_<12-digit timestamp in microseconds>`.
The numeric width is a minimum, and timestamps are rounded to microseconds for
the ID. Version 4 requires IDs to be unique across speech and visuals.

Media and WAV paths stored in extraction artifacts are absolute local paths.
Manifest paths are relative to `JOB/notes/`. Moving a job does not automatically
rewrite absolute paths.

### Hashes and checkpoints

A file SHA-256 hashes raw bytes. A JSON fingerprint hashes UTF-8 encoding of:

~~~python
json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
~~~

Object key order and indentation therefore do not change a JSON fingerprint.
Array order and string contents do. Fingerprints and file hashes are 64 lowercase
hexadecimal characters. Hashes in the examples are illustrative placeholders.

| Hash | Input | Purpose |
| --- | --- | --- |
| `media.sha256` | Original recording bytes | Invalidate audio extraction when recording bytes change |
| ASR input `sha256` | WAV bytes | Invalidate transcription when the chunk audio changes |
| Checkpoint `fingerprint` | Stage-specific input object | Reuse only a matching stage result |
| Diagnostic `evidence_fingerprint` | Entire parsed clean `evidence.json` | Identify the extraction to which words, scores, and warnings belong |
| Project `evidence_fingerprint` | Entire parsed clean `evidence.json` | Reject rendering a project bound to different evidence |
| Resolution `evidence_fingerprint` | Entire parsed clean `evidence.json` | Preserve explicit resolutions only for matching source material |

Diagnostics are not part of any note checkpoint, resolution, or render freshness
fingerprint. Editing that report cannot invalidate notes, reopen note issues, or
make a project stale. Editing clean evidence changes its fingerprint, even if
only its title or media metadata changes. The diagnostic report can then remain
on disk with its old fingerprint; it is not a gate on note generation.

Every checkpoint file has this envelope:

| Field | Type | Meaning |
| --- | --- | --- |
| `fingerprint` | string | Fingerprint of the inputs listed for that stage |
| `seconds` | number | Elapsed time for the successful action that originally produced this result |
| `result` | stage-specific JSON | Audio descriptor, transcribed chunk, video selection, visual extraction, or validated single-chapter note object |

Only successful actions create a new checkpoint. Reuse returns the saved result
without rerunning the action or updating its recorded time. A matching
fingerprint is not a full revalidation of the saved result. Checkpoints are
pipeline-owned cache files, rather than an interface for manual corrections.

## Recording to audio chunks

**Producer:** `shad_helper.extraction`. **Consumers:** the local ASR worker and
later extraction reruns.

The input must be a readable media file with a first audio stream. `ffprobe`
selects `a:0`. For formats other than Matroska/WebM, audio start is the stream
start minus the container start, clamped to zero; missing container start
defaults to zero, and missing stream start defaults to the container start.
When stream duration exists, audio end is this relative start plus stream
duration. If stream duration is missing or `N/A`, container duration is used
directly as the audio end, without adding the start. The audio bounds must be
finite and describe a positive interval.

When `ffprobe` identifies Matroska or WebM, audio and video probes derive their
bounds from a streaming packet scan instead of stream duration/start metadata.
The scan uses the minimum valid packet presentation timestamp and the maximum
timestamp plus packet duration, relative to container start; a missing packet
duration contributes zero. Invalid or empty packet bounds fail probing. This
targets containers whose reported stream bounds can match the container bounds
despite delayed or shorter streams; other formats retain the metadata route.

With an enabled vision model and a real video stream, the recording end is the
later of audio end and video end. With `--vision-model off` or audio-only input,
the recording end is audio end. These bounds preserve video that continues
after audio finishes, including requested clips wholly within that visual tail.

`--start` defaults to zero and must be finite, nonnegative, and less than the
media duration. An optional `--duration` must be finite and positive. The saved
end is `min(media duration, start + requested duration)`, or the media duration
when no duration is requested. Here media duration is the recording end defined
above; all bounds are relative to the original container start, not the clip.

### extraction-config.json

This job marker is written after input probing and interval validation, before
chunk extraction. For audio-only inputs or `--vision-model off`, its complete
structure is:

~~~json
{
  "schema_version": 3,
  "pipeline": "audio"
}
~~~

For an input with a real video stream and vision enabled, the marker is:

~~~json
{
  "schema_version": 4,
  "pipeline": "audio_video",
  "vision_model": "35b"
}
~~~

`extract --vision-model` accepts `35b`, `9b`, `4b`, or `off`, defaulting to `35b`.
Enabled aliases select pinned Qwen3.5 4-bit MLX checkpoints; `35b` selects
Qwen3.5-35B-A3B. Attached cover art is not a real video stream. An audio-only
recording remains version 3 even when an enabled vision model is selected.

Each marker must have exactly its displayed keys; the version 4 model alias
must be `35b`, `9b`, or `4b`. The marker permits an interrupted extraction to
resume before `evidence.json` exists.

The job guard accepts a new or empty directory, a directory with a supported marker,
or a directory containing supported version 3 or 4 evidence. An incompatible marker,
legacy evidence, or a nonempty directory with neither marker nor evidence is
rejected before writing pipeline outputs. The current `run-diagnostics.json` is
ignored when deciding whether the directory is empty; rejected stage invocations
may still replace that report. This includes version 1 and 2 jobs. The commands
do not migrate them; use a fresh directory.

### audio/chunk_0000.wav

Each WAV has one channel and a 16,000 Hz sample rate. FFmpeg chooses its WAV
encoding; the command does not explicitly select a PCM subtype. Only audio
stream `0:a:0` is mapped, and video output is disabled.

The requested interval is intersected with the available audio interval. An
empty intersection creates no audio chunks and skips the ASR worker, leaving
empty speech evidence while video extraction can proceed. Otherwise this audio
intersection is divided into core intervals of at most 600 seconds. For a core
`[lower, upper)`:

~~~text
offset    = max(0, audio stream start, lower - 5)
audio_end = min(requested audio intersection end, upper + 5)
WAV spans offset through audio_end
~~~

Thus adjacent full chunks can share ten seconds of audio. The first chunk of
a pilot clip may include five seconds before the requested start, bounded by
the actual audio stream start. Audio never extends past the available audio
or extracted clip end. Delayed audio therefore retains its recording timestamp
without a leading overlap into a region containing no audio samples. The final
core may be shorter than 600 seconds.

The descriptor passed to ASR is:

~~~json
{
  "id": "chunk_0000",
  "audio": "/path/to/JOB/audio/chunk_0000.wav",
  "offset": 1795.0,
  "core_start": 1800.0,
  "core_end": 1920.0
}
~~~

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Chunk ID used in filenames and checkpoints |
| `audio` | string | Absolute path to the generated WAV |
| `offset` | number | Recording timestamp corresponding to WAV time zero |
| `core_start` | number | Lower bound of the speech owned by this chunk |
| `core_end` | number | Exclusive upper bound of that ownership |

`audio_end` is used to generate and fingerprint the WAV but is not saved in its
descriptor. Chunk filenames and IDs restart at zero for the requested interval.

### checkpoints/audio/chunk_0000.json

The checkpoint result is the exact descriptor above. Its input fingerprint is
computed from:

| Input key | Value |
| --- | --- |
| `media` | SHA-256 of the original recording |
| `lower` | Core start |
| `upper` | Core end |
| `offset` | WAV start in the recording |
| `audio_end` | WAV end in the recording |

A complete example envelope is:

~~~json
{
  "fingerprint": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "seconds": 0.42,
  "result": {
    "id": "chunk_0000",
    "audio": "/path/to/JOB/audio/chunk_0000.wav",
    "offset": 1795.0,
    "core_start": 1800.0,
    "core_end": 1920.0
  }
}
~~~

After reuse, extraction checks that the recorded WAV path exists. A missing
file causes checkpoint removal and regeneration. This check does not verify WAV
contents or encoding; the ASR stage separately hashes the file bytes.

## Audio chunks to raw transcription

**Producer:** `shad_helper.local_worker`. **Consumer:** the extraction process
that restores recording timestamps.

The worker runs in a separate Python process. Successful process exit releases
its model memory and allows extraction to read its response. Each item requests
Whisper transcription with word timestamps.

### workers/asr.request.json

The parent process writes one request containing all current chunk descriptors:

~~~json
{
  "job": "/path/to/JOB",
  "model": "mlx-community/whisper-large-v3-mlx",
  "language": "ru",
  "items": [
    {
      "id": "chunk_0000",
      "audio": "/path/to/JOB/audio/chunk_0000.wav",
      "offset": 1795.0,
      "core_start": 1800.0,
      "core_end": 1920.0
    }
  ]
}
~~~

| Field | Type | Meaning |
| --- | --- | --- |
| `job` | string | Absolute job path used for guards and checkpoint storage |
| `model` | string | Whisper repository or model identifier; the displayed value is the extraction default |
| `language` | string | Requested ASR language, `ru` by default |
| `items` | array of chunk descriptors | Chunks to transcribe in request order |

This request has no independent schema-version field. These are internal
dictionaries rather than a Pydantic-validated public worker API.

### workers/asr.response.json

The response contains an `items` array. Each result preserves all descriptor
fields and adds raw Whisper `segments` and a detected `language`:

~~~json
{
  "items": [
    {
      "id": "chunk_0000",
      "audio": "/path/to/JOB/audio/chunk_0000.wav",
      "offset": 1795.0,
      "core_start": 1800.0,
      "core_end": 1920.0,
      "segments": [
        {
          "start": 5.25,
          "end": 8.5,
          "text": "Событие A имеет вероятность p.",
          "words": [
            {"word": "Событие", "start": 5.25, "end": 5.7, "probability": 0.8},
            {"word": " A", "start": 5.7, "end": 6.1, "probability": 0.8},
            {"word": " имеет", "start": 6.1, "end": 6.8, "probability": 0.8},
            {"word": " вероятность", "start": 6.8, "end": 7.9, "probability": 0.8},
            {"word": " p.", "start": 7.9, "end": 8.5, "probability": 0.8}
          ],
          "avg_logprob": -1.2,
          "no_speech_prob": 0.01,
          "compression_ratio": 1.1
        }
      ],
      "language": "ru"
    }
  ]
}
~~~

Raw segment timestamps are relative to the WAV, so the example's 5.25 seconds
becomes 1800.25 seconds after adding `offset`.

Whisper segment objects are preserved as returned by the installed library.
They can contain additional fields such as token IDs or decoder statistics.
The stitcher consumes `start`, `end`, `text`, and optional `words`; each word
needs `word`, `start`, and `end`. Optional `probability` feeds warnings.
Only the three displayed segment diagnostics are copied into the separate
diagnostic report. A chunk's detected `language` can be null when Whisper omits
it; detected language is not copied into final evidence or the report.

The response is written only after all requested items succeed. The request and
response filenames are reused on reruns. A failed worker can leave an older
response, but the parent checks process success before reading it.

### checkpoints/asr/chunk_0000.json

`result` is one complete transcribed chunk object from the response's `items`,
not the outer response. Its fingerprint inputs are:

| Input key | Value |
| --- | --- |
| `model` | Requested model identifier |
| `item` | Complete chunk descriptor, including its absolute WAV path |
| `language` | Requested language |
| `sha256` | SHA-256 of the current WAV bytes |

The worker persists successful chunks individually. If a later chunk fails,
earlier checkpoints remain reusable even though the full response was not
written. Model, language, descriptor, or WAV-byte changes invalidate reuse.
Moving the job and rewriting WAV paths also changes the descriptor fingerprint.

## Video to visual evidence

**Producers:** `shad_helper.video` and `shad_helper.vision_worker`.
**Consumer:** the extraction process that assembles version 4 evidence.

Video processing runs after the ASR subprocess exits and releases its model
memory, or immediately when the clip's audio intersection is empty. The requested
interval uses the combined recording bounds defined above; frame selection
intersects it with the first non-cover-art video stream. Matroska/WebM uses the
packet-derived bounds described above; other formats use stream metadata, with
container duration as the fallback when stream duration is unavailable.
An input must therefore contain an audio stream even when it contains no speech.
An empty video intersection produces an empty `visuals` array. Rotated video
streams are rejected and require orientation normalization before extraction.

### Frame selection and images

FFmpeg streams grayscale proxies at up to 720 pixels wide and two samples per
second, retaining source timestamps. The selector tracks accumulated changes,
settled frames, frames before erasure or transitions, and overlapping scroll
positions. A 15-second change guard retains slowly developing content. Unchanged
frames are omitted; lecturer motion can retain extra frames. Selection is a
content-preservation heuristic, not a semantic guarantee or a fixed frame cap.

`checkpoints/video/selection.json` stores selected timestamps, reasons, sharpness,
proxy count, and original scan time. Its fingerprint includes media bytes, the
intersected interval, video metadata, and selection settings. Reuse skips the
proxy scan. Missing or empty selected PNGs are retrieved again from the source.

Native frames are saved under `frames/<frame ID>_<first 12 media hash digits>.png`.
Each local worker item has `id`, `image` (absolute path), `timestamp`, `reason`,
and `sharpness`. Images and these selection details are local extraction artifacts.

### Vision worker request, response, and checkpoints

`workers/vision.request.json` contains `job`, pinned model repository `model`,
`revision`, and the selected `items`. The worker loads one MLX-VLM model lazily
and processes frames sequentially. It exits after writing
`workers/vision.response.json`, releasing the model memory. If selection returns
no frames, extraction does not start the worker.

Recognition uses thinking disabled, temperature zero, a 2,200,000-pixel ceiling,
and an initial output limit of 2,048 tokens. Invalid or truncated JSON receives
one retry with a corrective prompt and a 4,096-token limit. Both attempts use
the same source frame. A response that still fails becomes an `incomplete`
frame with an unknown full-frame illustration candidate; extraction continues with
the other frames. Notes can exclude it explicitly if no instructional figure is present.
Model-loading, inference, and worker-process failures propagate instead.

For slides and physical or digital whiteboards, the prompt requests readable
text, exact LaTeX formulas, and diagram descriptions with labels and relationships.
Notebook frames contribute theory, concepts, formulas, and meaningful plots;
program code, service output, and application interface are excluded. Hidden or
unreadable fragments are described explicitly rather than reconstructed.

The response contains `items` and aggregate `diagnostics`. Each response item
preserves its local input fields and adds `content` and `diagnostics`. `content`
contains `kind`, `text`, `formulas`, `diagrams`, and `unreadable`; these become the
clean visual fields below. Per-item diagnostics contain `status` and `attempts`.
Attempts include raw output, output limits, generation metrics, and validation
errors when present. Aggregate diagnostics include model/revision, settings,
package versions, elapsed and load times, and fresh/cached item counts.

`diagnostics/vision/<frame ID>.json` preserves the latest attempts and their
input fingerprint even when extraction is incomplete. Successful frame results
are checkpointed at `checkpoints/vision/<frame ID>.json`; incomplete results
are retried on a later extraction. The checkpoint fingerprint includes image
bytes, model/revision, both prompts, generation settings, and package versions.
Changing the vision preset invalidates visual inference without invalidating
matching audio, ASR, or selection checkpoints.

## Raw transcription to evidence

**Producer:** `shad_helper.extraction`. **Consumers:** chapter preparation and
project freshness checks read clean evidence; human inspection reads either
clean evidence or the separate diagnostic report.

### Segment normalization

Chunks are processed in order of `core_start`. For a segment with a nonempty
word list, the stitcher adds `offset` to each word's bounds and retains only
words whose midpoint belongs to this chunk's core. It joins their `word`
strings without an added separator, then strips surrounding whitespace.
Whisper's leading spaces therefore participate in reconstructing the sentence.

Segment bounds come from the first and last retained words and are clamped to
the core. Retained word bounds themselves are not clamped and can slightly
extend across a core boundary. Extra word fields are preserved.

If a segment has no words or an empty word list, ownership is decided by the
midpoint of the complete segment instead. Its text is stripped, timestamps are
restored and clamped, and its normalized word list is empty. Empty text and
segments with no owned words are omitted.

A stitched segment has these internal fields before extraction separates speech
from diagnostic data:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Current extraction's `segment_00000` ID |
| `start`, `end` | numbers | Absolute recording bounds, clamped to the core |
| `text` | string | Nonempty reconstructed speech |
| `words` | array | Retained words with absolute timestamps, or `[]` |
| `diagnostics` | object | Present subset of `avg_logprob`, `no_speech_prob`, `compression_ratio`; possibly `{}` |

Silence gaps are allowed. The segment array need not cover the entire clip,
and it may be empty for a silent recording.

### evidence.json

This is the authoritative input to synthesis. It contains clean speech and,
for version 4, clean visual content, without word data, scores, extraction
diagnostics, image paths, or model metadata. The complete audio-only shape is:

~~~json
{
  "schema_version": 3,
  "title": "lecture",
  "media": {
    "path": "/path/to/lecture.m4a",
    "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "duration": 2400.0,
    "clip_start": 1800.0,
    "clip_end": 1920.0
  },
  "segments": [
    {
      "id": "segment_00000",
      "start": 1800.25,
      "end": 1803.5,
      "text": "Событие A имеет вероятность p."
    }
  ]
}
~~~

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | integer | `3` for audio-only evidence; `4` when `visuals` is present |
| `title` | string | Recording filename stem by default; can be corrected before synthesis |
| `media` | object | Recording identity and extracted interval |
| `media.path` | string | Absolute path to the original recording |
| `media.sha256` | string | Original recording's byte hash |
| `media.duration` | number | Recording end relative to container start: audio end, or the later audio/video end when vision is enabled |
| `media.audio_start` | number, optional | Audio stream start relative to container start; omitted when zero |
| `media.clip_start`, `media.clip_end` | numbers | Requested interval after end clipping |
| `segments` | array of clean segments | Speech and source IDs; may be empty |
| `visuals` | array of clean visual objects, version 4 only | Timestamped recognized content; may be empty |

Every clean segment has exactly these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Current extraction's source identifier |
| `start`, `end` | numbers | Absolute recording bounds, clamped to the audio core |
| `text` | string | Nonempty reconstructed speech |

Version 3 requires exactly the four root keys shown above; version 4 requires
those four keys plus `visuals`. Both require `media` to be an object and
`segments` to be an array of objects
containing exactly `id`, `start`, `end`, and `text`. Missing or additional root or
segment fields are rejected,
including `issues`, `models`, `words`, and `diagnostics`. This enforces the
boundary between content and diagnostics; it is not full nested type or semantic
validation of media and timestamps. Downstream code expects the producer types
above.

Each version 4 visual has exactly these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | nonempty string | Frame source identifier, unique across speech and visuals |
| `timestamp` | finite nonnegative number | Absolute source frame time |
| `status` | `"complete"` or `"incomplete"` | Whether recognition produced a valid, non-truncated content object |
| `kind` | `"slides"`, `"whiteboard"`, `"notebook"`, or `"other"` | Recognized display category |
| `text` | string | Readable instructional text, or `""` |
| `formulas` | array of strings | Recognized LaTeX formulas |
| `diagrams` | array of diagram descriptions | Visible objects, labels, and relationships |
| `unreadable` | array of strings | Descriptions of obscured or unreadable source regions |

Each diagram has exactly `description` (string), `labels` (array of strings),
and `relationships` (array of strings). These are descriptions for later editable
diagram generation, not images or executable drawing code. Missing or extra
visual/diagram fields and invalid visual types are rejected.

`complete` means the response passed structural validation; it does not establish
recognition accuracy or imply an empty `unreadable` array. An `incomplete` item
has `kind: "other"`, empty text and arrays; failure explanations remain in local
diagnostics. The clean status lets synthesis see the gap without receiving raw
model output or implementation errors.

For lecture bounds, synthesis falls back to `clip_start = 0` and
`clip_end = duration` if those keys are missing. It rejects an empty interval.
New extraction always writes both clip bounds.

### extraction-diagnostics.json

This report contains the data removed from stitched segments and the warnings
computed from it. Extraction computes warnings before creating clean evidence,
so separation does not change word ownership or warning thresholds. The following
complete example is audio-only; video extraction adds the local fields below.

~~~json
{
  "schema_version": 1,
  "evidence_fingerprint": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  "models": {
    "asr": "mlx-community/whisper-large-v3-mlx"
  },
  "segments": [
    {
      "id": "segment_00000",
      "words": [
        {"word": "Событие", "start": 1800.25, "end": 1800.7, "probability": 0.8},
        {"word": " A", "start": 1800.7, "end": 1801.1, "probability": 0.8},
        {"word": " имеет", "start": 1801.1, "end": 1801.8, "probability": 0.8},
        {"word": " вероятность", "start": 1801.8, "end": 1802.9, "probability": 0.8},
        {"word": " p.", "start": 1802.9, "end": 1803.5, "probability": 0.8}
      ],
      "diagnostics": {
        "avg_logprob": -1.2,
        "no_speech_prob": 0.01,
        "compression_ratio": 1.1
      }
    }
  ],
  "warnings": [
    {
      "id": "issue_0000",
      "description": "Whisper отметил низкую уверенность транскрипции; сверить с аудио.",
      "evidence_ids": ["segment_00000"]
    }
  ]
}
~~~

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | integer | Diagnostic report format, currently `1` |
| `evidence_fingerprint` | string | Fingerprint of the complete clean evidence produced with this report |
| `models.asr` | string | Requested ASR model identifier |
| `segments` | array of diagnostic segment objects | One object per retained speech segment, in the same order as evidence |
| Segment `id` | string | Foreign key to the corresponding clean evidence segment |
| Segment `words` | array | Retained words with absolute timestamps and any extra Whisper word fields, or `[]` |
| Segment `diagnostics` | object | Present subset of `avg_logprob`, `no_speech_prob`, and `compression_ratio`, or `{}` |
| `warnings` | array of warning objects | Informational extraction warnings, or `[]` |
| Warning `id` | string | Generated `issue_0000` identity within this extraction |
| Warning `description` | string | Human-readable condition and suggested inspection |
| Warning `evidence_ids` | array of strings | Affected clean speech or visual IDs |

For video jobs, `models.vision` adds the selected preset, model repository, and
pinned revision. `selection` records sampling settings, interval, proxy/selected
counts, selection/retrieval times, and cache reuse. `vision` contains aggregate
worker diagnostics, or failure status and an error when the vision stage fails.
`visuals` preserves each worker item's local image path, selection details, status,
and raw attempts, excluding its clean `content`. Individual attempt files and
frame images remain under `diagnostics/vision/` and `frames/`.

A warning has no `status` or `resolution`. Its `issue_` ID prefix is a naming
convention inherited from extraction and does not make it a note review issue.
Diagnostic segment records omit speech text and segment bounds; join them to
clean evidence by ID when inspecting a source fragment. Word bounds can slightly
cross audio core boundaries as described under segment normalization.

Extraction creates a warning for each applicable condition:

| Condition | Warning |
| --- | --- |
| `avg_logprob < -1` | Low segment transcription confidence |
| `no_speech_prob > 0.6` with nonblank text | Possible text generated during silence |
| Any word `probability < 0.3` | Low word confidence; description lists at most five weak words |
| `compression_ratio > 2.4` or a four-word sequence repeats at least four times | Strong repetition |
| A complete visual contains `unreadable` descriptions | One inspection warning per described source region |
| Visual output is invalid or truncated after its retry | Incomplete frame extraction |

Warnings preserve the transcribed text; they do not silently correct it.
The report is an inspection artifact, not a synthesis or review input. Notes
do not read it, including when missing or edited, and there is no required
approval or ignore action. Correct recognized content in `evidence.json` if a
warning reveals an extraction error before generating notes. Clean visual
`status` and `unreadable` fields remain part of the source evidence available to
synthesis; raw attempts, metrics, image paths, and model metadata do not.

### transcript.json

This is a bare JSON array identical to `evidence.segments` at extraction time.
It has no outer object, version field, media metadata, words, diagnostics, or
warnings.

Synthesis reads `evidence.json`, not this file. Editing `transcript.json`
does not correct synthesis input. Editing evidence also does not automatically
update this transcript copy.

### transcript.srt and transcript.vtt

Both files contain one numbered cue per normalized segment, in the same order.
Cue numbers begin at 1 and are independent of segment IDs. Times remain relative
to the original recording.

`transcript.srt` uses comma-separated milliseconds:

~~~text
1
00:30:00,250 --> 00:30:03,500
Событие A имеет вероятность p.
~~~

`transcript.vtt` begins with `WEBVTT` and a blank line, and uses periods:

~~~text
WEBVTT

1
00:30:00.250 --> 00:30:03.500
Событие A имеет вероятность p.
~~~

They are inspection/export artifacts; no later pipeline stage parses them.
Silent extraction produces an empty JSON segment array, an empty SRT, and a
VTT containing just its header.

### extraction-timings.json

Audio-only example:

~~~json
{
  "scope": "current extraction invocation",
  "audio_asr_seconds": 12.34
}
~~~

`audio_asr_seconds` measures elapsed wall time from before chunk preparation
through transcription, stitching, and initial transcript/evidence writes.
It includes checkpoint lookup and reused work, and is not a sum of
historic checkpoint times or a measurement of model execution alone. It excludes initial probing and
original-media hashing. The file is written after evidence.

Video jobs add `vision_seconds` for selection, frame retrieval, model loading,
inference, retries, and evidence assembly, plus `total_seconds` from extraction
entry through the vision attempt, including initial probing and media hashing.
These fields are written even when the vision stage fails after audio extraction.
They measure current-run
wall time, so checkpoint reuse reduces them. The runtime target is assessed
from measurements, not enforced by dropping frames or stopping at one hour.

Matroska/WebM probing adds a sequential packet read for each selected stream:
one for audio and, when vision is enabled, one for video. The scan retains only
running timestamp bounds in memory. Extraction passes the probed video metadata
into frame selection, avoiding another packet scan there. These initial scans
are included in video-job `total_seconds`, but precede `audio_asr_seconds` and
`vision_seconds`; they still run when later stage checkpoints are reused.

## Evidence to chapter inputs

**Producer:** `shad_helper.notes`. **Consumers:** automatic synthesis, manual
handoff, and chapter reply validation.

`notes` requires nonblank speech or at least one complete visual with nonblank
text, a formula, or diagram description/label/relationship. Silence can therefore
produce notes from usable visual evidence. Empty, incomplete-only, or
unreadable-only visuals without speech do not satisfy this requirement.

### outline.json

This is the editable chapter plan and glossary seed:

~~~json
{
  "title": "lecture",
  "chapters": [
    {
      "id": "ch001",
      "title": "Часть 1: 00:30:00–00:32:00",
      "start": 1800.0,
      "end": 1920.0
    }
  ],
  "glossary": []
}
~~~

| Field | Type | Required or default | Meaning |
| --- | --- | --- | --- |
| `title` | string | Required | Lecture title used when merging chapter replies |
| `chapters` | nonempty array | Required | Ordered chapter plan |
| Chapter `id` | identifier string | Required | Reply filename and checkpoint identity |
| Chapter `title` | string | Required | Seed title; outline titles may be empty |
| Chapter `start` | number, at least 0 | Required | Chapter start |
| Chapter `end` | number, greater than 0 | Required | Chapter end |
| `glossary` | array of term objects | Default `[]` | Terms carried into prompts |

A glossary object contains only required string fields `term` and
`definition`. Neither has a schema minimum length.

The default outline uses consecutive 900-second chapters, with a shorter final
chapter. Validation requires unique chapter IDs, positive chapter durations,
and chapter lengths no greater than 1200 seconds. Starting at the extracted
clip start, each chapter must start within 0.01 seconds of the preceding end.
The final end must be within 0.01 seconds of the extracted clip end. The chapter
list order is significant.

On reuse, a saved outline whose first start or last end differs **exactly** from
the current clip bounds is replaced by a default outline. Otherwise, the saved
outline is validated. Internal gaps, duplicate IDs, and oversized chapters
are errors rather than silently repaired.
`notes --rebuild` deletes the saved outline before validation and starts with the
default fifteen-minute plan, losing any custom titles, glossary seed, or intervals.

Automatic and manual-export preparation can split this plan further. A
successful combined-file import replaces it with the imported title, chapter
metadata, and glossary. Automatic chapter titles and accumulated glossary
entries are saved in `notes.json`; the seed outline does not automatically
become a copy of the generated notes.

### Chapter context

The local context is an in-memory object:

| Field | Structure | Meaning |
| --- | --- | --- |
| `chapter` | Outline chapter object | Exact chapter identity and interval |
| `segments` | Array of `{id, start, end, text}` | Speech selected for this chapter and neighboring context |
| `visuals` | Array of clean visual objects, version 4 only | Selected visual source content, with exactly the eight evidence fields |

Default neighboring context is `min(60 seconds, chapter duration / 4)` on
each side. A segment is selected when its end is at or after the expanded
start and its start is at or before the expanded end. Selected segments are
included in full, not trimmed at chapter or context bounds. Adjacent chapters
can therefore receive the same context segments.

Visuals are sorted by `timestamp`, then `id`. The context includes every visual
within the inclusive expanded interval and the latest visual before its start,
when one exists. This carries a previously shown static slide or board into a
later chapter. It does not claim that the retained frame's content first
appeared at the current chapter boundary.

The JSON context has no image bytes, local image paths, raw model attempts, model metadata,
word probabilities, word timestamps, extraction diagnostics, extraction warnings,
or existing note issues. Note generation does not open
`extraction-diagnostics.json` or copy its contents into checkpoints or review.
Version 5 images are separate labeled attachments as described above.

The model-facing context adds lecture metadata to these fields. Its complete
audio-only shape, also saved as a manual `context.json`, is:

~~~json
{
  "lecture_title": "lecture",
  "glossary": [],
  "topics": [],
  "writing_budget": {"target_words": 34, "dense_words": 67},
  "chapter": {
    "id": "ch001",
    "title": "Часть 1: 00:30:00–00:32:00",
    "start": 1800.0,
    "end": 1920.0
  },
  "segments": [
    {
      "id": "segment_00000",
      "start": 1800.25,
      "end": 1803.5,
      "text": "Событие A имеет вероятность p."
    }
  ]
}
~~~

| Model context field | Structure | Meaning |
| --- | --- | --- |
| `lecture_title` | string | Root outline title used for the lecture |
| `glossary` | array of `{term, definition}` | Seed terms for manual export; seed plus accumulated prior chapter terms during automatic synthesis |
| `topics` | array of strings | Broad topic names from preceding generated or cached chapters; initially empty for manual exports |
| `writing_budget` | object with integer `target_words` and `dense_words` | Soft output allowance, scaled to the chapter's core duration at 3,000/6,000 words per three hours, rounded up with a minimum of one |
| `chapter` | Outline chapter object | Required single chapter and its bounds |
| `segments` | Array of clean speech segments | Chapter and neighboring speech, included once |
| `visuals` | Array of clean visual objects, version 4 only | Timestamped text, formulas, diagram descriptions, status, and unreadable regions |

There is no nested `context` wrapper. Automatic API input is a compact JSON
serialization of this object. Manual JSON is the same structure written with
readable indentation. Neither contains a response schema or instructions.

A block can cite any speech or visual ID selected for its chapter, including
neighboring context and the carried preceding frame. The prompt asks the model
to use neighboring content for understanding without duplicating it in the
chapter; that editorial rule is not enforced by timestamp filtering of the
reply's prose.

The normal editorial target is about ten pages per three hours, up to twenty
for dense lectures. Word allowances include all chapter content and glossary;
they exclude neighboring context duration. They are not minimum quotas, hard
truncation limits, or guarantees about rendered page counts. Definitions,
conditions, and proof logic take priority over exact length. Empty administrative
intervals retain their chapter bounds but return `blocks: []`. The prompt excludes
speaker narration, routine logistics, repeated explanations, and redundant
glossary entries. Mathematical notation uses LaTeX rather than Unicode math
glyphs; ordinary Russian text remains UTF-8.

Processing chapters do not determine the PDF sections. Each new block names its
broad `topic`, such as nearest neighbors or linear regression. The prompt asks
the model to reuse names from `topics` exactly and keep definitions, formulas,
examples, and proofs of the same topic together. A chapter may cover several
topics, and a topic may span or recur across chapters. Initial manual handoffs
cannot contain names discovered in later replies; their instructions require
consistent broad names across responses.

### Request size and chapter splitting

Preflight measures UTF-8 bytes of the synthesis instructions, serialized model
context, and serialized Pydantic response schema. The cap is 1,200,000 bytes. This
is a local text budget, not an exact token count or the serialized HTTP body size.

Oversized chapters are bisected in time. Child IDs have the form
`<first 60 characters of parent>_a_<8-character hash>` and the corresponding
`_b_` ID. The hash comes from the parent ID and child bounds. Titles are
initially inherited. Children must be at least 30 seconds long; a chapter
shorter than 60 seconds cannot be bisected. This does not impose a 30-second
minimum on an original short recording or an imported chapter.

Preparation persists the split outline. During automatic generation, the same
check repeats with the glossary and topic names accumulated from earlier replies. A typed
`context_length_exceeded` model failure also triggers splitting immediately,
instead of retrying the same input.

Other chapter request failures and invalid replies get up to three retries after
the initial attempt, with waits of 5, 10, and 20 seconds. Each retry uses the same
chapter, context, glossary, topic names, instructions, and response schema. Only a validated
successful reply is checkpointed. Completed chapters are reused; exhausting all
four attempts exports the manual handoff with the final failure in `status.json`.

If time splitting cannot reduce an oversized input, all selected evidence is
retained in the manual handoff and `status.json` explains the failure. Import
mode skips automatic request-size splitting because no model request is made.

## Chapter inputs to validated notes

**Producer:** automatic model responses or user-created ChatGPT reply files,
followed by local validation and merging. **Consumers:** review and LaTeX
generation.

### Automatic request and reply

Automatic synthesis submits text only. The user input is the compact JSON
model context: lecture title, current glossary, topic names, writing budget, chapter, speech, and optional
clean visuals. Images and local recognition artifacts are never attached. Russian
instructions request one chapter with the provided ID and timestamps,
evidence-grounded content, source citations, and explicit uncertainty.
The instructions are identical to the manual `prompt.txt`. They contain no
lecture data or embedded response schema. The output schema is supplied through
the client's response-format field rather than duplicated in the user input.

The current client default is model `gpt-6-astra` with reasoning effort
`max`. Its response format is named `lecture_notes`. For the API's strict
JSON Schema format, the client recursively removes schema defaults, requires
every declared object property, and sets `additionalProperties: false`.
Application-level provenance and review validators still run afterward.

Exact outgoing model request bodies are saved before transmission under
`handoff/requests/<chapter ID>/attempt-NN/`. Streaming events and
raw note replies are not archived; incomplete streams are discarded. Only a
successfully validated chapter result is checkpointed.

### Manual handoff files

Every notes run prepares the files below, including successful automatic runs,
cache reuse, manual mode, and imports. Automatic inputs are saved before client
creation and updated with each chapter's actual accumulated glossary and topic names before
checkpoint lookup. Imports save a reconstructed handoff after validating replies;
they do not claim those reconstructed inputs were sent to a model.

| File | Structure | Producer and consumer |
| --- | --- | --- |
| `handoff/inputs/response-schema.json` | Full JSON Schema generated by `LectureNotes.model_json_schema()` | Shared response constraints attached to ChatGPT for every chapter |
| `handoff/inputs/<chapter ID>/context.json` | Model-facing context shown above | Prepared context; automatic chapters visited in the run retain the actual accumulated glossary and topic names |
| `handoff/inputs/<chapter ID>/prompt.txt` | Russian synthesis instructions only, byte-identical to automatic API instructions | Pasted into ChatGPT |
| `handoff/README.txt` | Plain English instructions for attaching files, saving replies, and importing | Human operator |
| `handoff/status.json` | Status and explanatory reason | User or automation deciding whether manual work remains |

For a chapter, send only its prompt, its context, and the shared response schema.
Their responsibilities are distinct: instructions, lecture data, and response
structure. The root `outline.json` remains the authoritative chapter plan and
determines which replies must be imported. A current copy is saved as `handoff/outline.json`,
but is not attached to the model. There is no chapter `transcript.txt`; it would repeat speech already
included in context. Operator README and status files stay outside `inputs/` and
are not model inputs.

`response-schema.json` describes the **response**, not the context. It contains
root `properties`, `required`, `additionalProperties: false`, and `$defs`
for nested models. It is the Pydantic schema with import defaults, rather than
the client's stricter API-normalized schema. Cross-object uniqueness,
chapter interval relationships, evidence provenance, and resolution matching
also require the Python validators; JSON Schema alone is insufficient.

A fallback status file has this complete shape:

~~~json
{
  "status": "manual_required",
  "reason": "Automatic synthesis failed; import chapter replies from the exported handoff."
}
~~~

`reason` contains the actual exception message or chapter budget warnings;
the displayed message is illustrative. Normal manual export within the budget
does not need a status file. Successful synthesis or import removes a previous
status file. Those paths also refresh model inputs, so a current handoff exists
after successful generation and import.

Every ordinary export checks job and evidence compatibility, then deletes and rebuilds
the generator-owned `handoff/inputs/` subtree. This removes obsolete chapter
folders and duplicate files from earlier exports. Unrelated operator files are
preserved by the cleanup; `README.txt` is rewritten with current
instructions, while the calling workflow manages `status.json`. The current root
`outline.json` determines the exported chapter IDs. Send one chapter's three
inputs per request. An interrupted export can leave only part of the rebuilt
input tree; rerun export before attaching it.
With `notes --rebuild`, the entire `handoff/` is deleted before compatibility or
input checks, including operator files and legacy `runs/` archives.

### Current inputs and outbound request diagnostics

With `--rebuild`, stage cleanup precedes the initial job compatibility check.
After that check, each invocation writes
`handoff/run.json` and prints the handoff path. Once evidence and outline checks
pass, it replaces the generated current handoff state. On ordinary invocations,
imported replies are read before this current-diagnostics cleanup, so a prior
saved response can be imported directly. A rebuild deletes the entire stage
before reading replies; copy imports outside its deletion paths first.
`mode` is `automatic`, `manual`, or `import`; `status` begins as `running` and
ends as `completed`, `handoff`, or `failed`. A forcibly terminated process can
leave `running`; a failure before valid inputs exist can leave only the run record.
`rebuild` records the flag.

The current `handoff/inputs/`, `README.txt`, and `outline.json` describe the latest
preparation. The saved outline follows runtime splitting; rejected parent requests
remain in the current diagnostics even when only leaf chapters remain in the handoff.
During automatic generation, `chapters.json` maps completed chapter IDs to
`generated` or `cached`. Manual exports make no model requests, and cache hits
are never logged as transmissions.

Every generation attempt gets `requests/<chapter ID>/attempt-NN/call.json`
containing `instructions`, the exact serialized context `text`, application
`schema`, and `model`. Its `status.json` records `success` or `failed` plus the
exception type/code on handled failures. A crash or local write failure may leave
an incomplete attempt record. Retries have separate directories, even when their
inputs match byte for byte.

At the HTTP boundary, `PlanClient.request_recorder` saves the exact byte payload
passed to `client.stream` as `http-NNN.body.json`. Its paired
`http-NNN.request.json` contains `method`, `url`, and `body` (the filename, or null
for the model-catalog GET). The POST includes the real model/reasoning settings,
instructions, input, stream/store flags, and strict normalized response schema.
Saving happens before the HTTP operation; a saved request records an attempt,
not proof that OpenAI received it. A write failure prevents the HTTP operation.

Request records omit HTTP authorization headers, tokens, proxy credentials, OAuth
exchanges, and response streams. Raw extraction diagnostics remain outside both
handoff inputs and model payloads. Each new notes invocation replaces generated
handoff inputs, requests, verification and figure-reconciliation diagnostics, and
run/status records. Ordinary invocations leave unrelated operator files and
existing `handoff/runs/` archives unchanged. A rebuild deletes all of `handoff/`
before preparing new inputs; no new run archives are created.

### Manual reply inputs

For directory import, every current root outline chapter must have
`<chapter ID>.json` in the supplied directory, for example:

~~~text
replies/
├── ch001.json
└── ch002.json
~~~

Each file contains a complete `LectureNotes` object with exactly one chapter.
It must have the expected chapter ID and start/end within 0.01 seconds of the
outline values. Missing or invalid expected files fail import; unrelated extra
files are ignored. All expected replies are loaded and validated before final
notes are written.

Alternatively, `--import response.json` accepts one complete lecture object.
Its chapters must satisfy outline coverage and duration rules, and its title,
chapter metadata, and glossary become the saved outline. Both modes use the
same note and evidence validation as automatic replies.
With `--rebuild`, directory imports must match the freshly defaulted outline.
A complete lecture JSON import can replace that default with its own valid
chapter plan. Imports stored in `notes.json`, `handoff/`, or another cleared
stage path must be copied elsewhere before invoking the rebuild.

Reply files contain JSON only, without Markdown fences or commentary. Their
root shape is the same as the normalized notes example below. Imported files
may omit fields with defaults; normalization fills them.

### notes.json

The complete normalized structure is:

~~~json
{
  "title": "lecture",
  "language": "ru",
  "chapters": [
    {
      "id": "ch001",
      "title": "Вероятность события",
      "start": 1800.0,
      "end": 1920.0,
      "blocks": [
        {
          "id": "ch001_definition",
          "kind": "text",
          "content": "Событие \\(A\\) имеет вероятность \\(p\\).",
          "topic": "Вероятность события",
          "evidence_ids": ["segment_00000"],
          "caption": "",
          "language": "",
          "uncertain_fragments": []
        }
      ]
    }
  ],
  "glossary": [
    {
      "term": "Вероятность события",
      "definition": "Число, обозначенное в речи буквой p."
    }
  ],
  "issues": [
    {
      "id": "ch001_missing_detail",
      "description": "Лектор сослался на неприведенный вывод; проверить недостающие шаги.",
      "evidence_ids": ["segment_00000"],
      "status": "open",
      "resolution": ""
    }
  ]
}
~~~

#### Lecture and chapter fields

| Field | Type | Required or default | Validation and meaning |
| --- | --- | --- | --- |
| `title` | nonempty string | Required | Lecture title |
| `language` | `"ru"` | Default `"ru"` | Output language; other values are rejected |
| `chapters` | nonempty array | Required | Ordered chapter content |
| `glossary` | array of `{term, definition}` | Default `[]` | Both fields are required strings |
| `issues` | array of review issues | Default `[]` | Uncertainties discovered during note generation; no extraction warnings |
| Chapter `id` | identifier string | Required | Stable identity within the current outline and notes |
| Chapter `title` | nonempty string | Required | Generated thematic title |
| Chapter `start` | number, at least 0 | Required | Chapter interval start |
| Chapter `end` | number, greater than 0 | Required | Must exceed chapter start |
| Chapter `blocks` | array | Required | Ordered content blocks; `[]` omits an interval with no useful study material from the PDF |

Note chapter bounds must lie within the extracted interval with 0.01-second
tolerance. Each automatic or directory-import reply must also match its
designated outline chapter. Whole-file import additionally validates a
consecutive complete outline.

There is no `schema_version` field in `notes.json` or note replies. Adding
one is rejected as an extra field. There is also no saved `evidence_index`:
the source-label mapping is computed locally for project generation.

#### Block fields and content rules

| Field | Type | Required or default | Meaning |
| --- | --- | --- | --- |
| `id` | identifier string | Required | Unique block identity and possible asset filename |
| `kind` | one of nine values below | Required | Rendering and provenance behavior |
| `content` | nonempty string | Required | Text, LaTeX, or literal code |
| `topic` | string | Default `""` | Broad section title; the prompt requires a nonempty title for new blocks, while legacy empty values use the chapter title |
| `evidence_ids` | array of strings | Default `[]` | Existing speech or visual source IDs |
| `caption` | string | Default `""` | Diagram caption; other block renderers do not use it |
| `language` | string | Default `""` | Code-language metadata; current renderer uses plain verbatim output |
| `uncertain_fragments` | array of `{text, evidence_ids}` | Default `[]` | Exact questionable prose selections with source IDs; renderer supplies highlighting and timestamp footnotes |

| Kind | Content contract | Source requirement and rendering |
| --- | --- | --- |
| `text` | Ordinary prose; inline math uses only `\(...\)` | At least one permitted evidence ID; prose is escaped and supported inline math retained |
| `definition` | Definition as prose with supported inline math, without a manual heading or environment | At least one permitted evidence ID; wrapped in the numbered `definition` environment |
| `theorem` | Theorem statement with assumptions, as prose with supported inline math | At least one permitted evidence ID; wrapped in the numbered `theorem` environment |
| `proof` | Proof as prose with supported inline math; may contain several paragraphs | At least one permitted evidence ID; wrapped in the `proof` environment with a Russian heading |
| `math` | Prompt requests a bare LaTeX expression without display delimiters or Markdown | At least one permitted evidence ID; wrapped as display math unless already a supported display form |
| `code` | Literal code without Markdown fences | At least one permitted evidence ID; written to a text asset and displayed verbatim |
| `diagram` | Prompt requests a complete `tikzpicture`; PGFPlots may be embedded | At least one permitted evidence ID; diagram commands are validated and written to a TeX asset |
| `clarification` | Legacy explanatory prose, with supported inline math | May have no evidence IDs; rendered as plain prose without a label; new responses use `text` |
| `uncertain` | Legacy tentative prose, with supported inline math | At least one permitted evidence ID; colored with a compact timestamp footnote and generates an open issue; new responses use fragments |

Fragment annotations are allowed only in `text`, `definition`, `theorem`, `proof`,
`clarification`, and legacy `uncertain` blocks. Each nonblank `text` must occur exactly once in the content;
selections must not overlap or cross a blank-line paragraph boundary. A selection
containing inline math must include the entire `\(...\)` expression, including
delimiters. Each selection has a nonempty evidence list drawn from its block's
citations. Questionable formulas belong in an annotated prose paragraph rather
than a separate `math` block. The model supplies no footnotes or timestamps.

Definitions, theorem statements, and proofs contain ordinary prose and `\(...\)`
math, not raw `\begin`/`\end` commands. The renderer supplies their environments
and headings. The prompt places each proof immediately after its theorem under
the same topic and prohibits inventing a proof absent from the sources. Examples,
clarifications, and explanations use ordinary `text` blocks without special
headings or environments. These editorial relationships require review; the
schema does not establish that a proof proves the preceding theorem.

For every evidence ID cited by a block, the speech or visual source must exist
in that chapter's selected context. An empty citation list is accepted only for
clarification.
An issue's citations must exist in the available evidence. Citations validate
the reference relationship; they do not prove that the model's prose or code
accurately represents the evidence.

The instructions prohibit inventing formulas, labels, code, execution results,
or graph values absent from the sources. Notebook concepts and theory remain
ordinary text; notebook code, arrays, service output, and UI are excluded, and
theory must not be inferred from code alone. Diagram blocks use editable TikZ
or PGFPlots based on the described labels and relationships, without raster
fallback. Captions should identify a schematic plot when actual data are missing.
Unreadable or missing details require marking only when essential to the selected
study material; discarded diagnostics do not become notes. Those content rules
require review; they are not semantic facts that the JSON validator can establish.

The renderer accepts math beginning with `\[` or supported
`equation`, `align`, `gather`, and `multline` display environments without
adding another display wrapper. A diagram fragment is wrapped in a
`tikzpicture` unless it already begins with `\begin{tikzpicture}` or `\tikz`.
The canonical response requested from the model remains a bare math expression
and complete diagram environment.

#### Merging and uncertainty

Chapter replies are merged in outline order. The final title comes from the
outline, and the final language is Russian. Glossary entries are keyed by their
exact `term` string: later chapter definitions replace earlier definitions,
including glossary seeds. Note issues are combined by ID and their citations
are checked against clean evidence. Each model or imported issue is forced to
`open` with an empty resolution; generation cannot declare its own uncertainty
resolved. Only a matching explicit local resolution can restore resolved state.
No extraction warning is merged into notes.

Every block with fragments, and every legacy `uncertain` block, generates one
issue with ID `review_<block ID>`.
If that exceeds 80 characters, the ID becomes
`review_<first 60 characters of block ID>_<12-character block-ID hash>`.
For annotated blocks, its description joins the selected fragments and its
citations combine only their references. For legacy blocks they come from the
whole block. Saved matching explicit
resolutions can subsequently restore resolved issue status.

### checkpoints/notes/ch001.json

For automatic synthesis, `result` is a validated single-chapter
`LectureNotes` object, including note uncertainty and explicit resolution
handling. The fingerprint input object contains:

| Input key | Value |
| --- | --- |
| `context` | Internal chapter context containing chapter metadata, clean speech, and optional clean visuals |
| `prompt` | Exact compact model-context JSON with lecture title, current accumulated glossary, and topic names |
| `instructions` | Russian synthesis instructions |
| `model` | Client's model identifier |
| `reasoning_effort` | Current default effort, `max` |
| `schema` | Full Pydantic response schema |

These inputs are hashed but are not individually saved alongside the
checkpoint. The current handoff stores the prepared model inputs separately;
actual calls also retain their exact outbound API payload. A cached chapter result
can be reused while later chapters run.
Speech, visual content, prompt, model, outline context, glossary, topic names, or schema
changes can invalidate affected chapter reuse. Diagnostic-report changes cannot. Final
resolution checks use the complete clean evidence fingerprint even when a
single chapter checkpoint is reused.

`notes --rebuild` deletes these stage-owned paths before job compatibility checks,
input validation, or model calls:

- `notes/`, including editable TeX, figures/code, PDF, QA, source map, logs and auxiliaries;
- `notes.json`, `review.json`, and `outline.json`;
- the entire `handoff/`, including current diagnostics, operator files, and legacy run archives;
- `checkpoints/notes/`, `checkpoints/notes_candidates/`, and `checkpoints/notes_final/`;
- `illustration-crops/` and `illustration-verification/`.

Deleting `checkpoints/notes/` also removes historical `notes/outline.json`
checkpoints. The custom outline and manual edits to generated sources are lost.
Removal errors stop the rebuild; a later input or generation failure does not
restore deleted artifacts. Extraction outputs and checkpoints, original frames
and illustrations, and explicit `review-resolutions.json` corrections remain
unchanged. The flag is not hashed, so subsequent ordinary runs can reuse fresh
checkpoints written at the same paths.
The `notes --rebuild` command also renders the regenerated project with a clean
LaTeX build and refreshes PDF QA and source maps. Manual export still produces
only a handoff; an automatic handoff does not trigger rendering.
Neither import mode nor manual export creates synthesized chapter checkpoints.

## Review state and resolutions

### Review issue format

Note issues and review summaries share this object. Extraction warnings have
their separate, status-free format in `extraction-diagnostics.json`.

| Field | Type | Required or default in note imports | Meaning |
| --- | --- | --- | --- |
| `id` | identifier string | Required | Issue identity |
| `description` | string | Required | Problem requiring review |
| `evidence_ids` | array of strings | Default `[]` | Related speech or visual sources |
| `status` | `"open"` or `"resolved"` | Default `"open"` | Review state |
| `resolution` | string | Default `""` | User's correction or explanation |

A resolved issue requires a nonblank resolution. The schema does not require
a nonempty description or citation list.

### review.json

**Producer:** successful synthesis/import or explicit resolution.
**Consumer:** the review command and human inspection.

~~~json
{
  "issues": [
    {
      "id": "ch001_missing_detail",
      "description": "Лектор сослался на неприведенный вывод; проверить недостающие шаги.",
      "evidence_ids": ["segment_00000"],
      "status": "open",
      "resolution": ""
    }
  ],
  "ready": false
}
~~~

`ready` is true only when there are no open issues, uncertain fragments, or legacy
`uncertain` blocks.
Resolving an issue alone does not change the block's kind or content. A remaining
marked fragment or uncertain block keeps the notes in draft state even if its generated issue was
resolved.

Rendering uses the project's review flags rather than reading this summary as
a gate. Extraction does not refresh `review.json`, so it can be stale after
re-extraction. When no summary exists, `review` displays issues from `notes.json`,
or an empty list if notes do not yet exist. It never displays extraction warnings.

### review-resolutions.json

**Producer:** `review JOB --resolve ID --resolution TEXT`.
**Consumer:** later note validation and import.

The root is an object keyed by issue ID:

~~~json
{
  "ch001_missing_detail": {
    "description": "Лектор сослался на неприведенный вывод; проверить недостающие шаги.",
    "evidence_ids": ["segment_00000"],
    "resolution": "Проверил источник и добавил недостающие шаги в конспект.",
    "evidence_fingerprint": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
  }
}
~~~

Each entry contains the issue description, citation array, explicit resolution,
and the fingerprint of the complete clean evidence. It does not
contain an extra `id` or `status` field.

On validation, a saved resolution applies only when clean evidence fingerprint,
description, and citation array all match. Citation order participates in the
comparison. A mismatch reopens the issue and clears its resolution. Editing
source material therefore prevents an old correction from silently resolving
a different issue.

The resolution command updates matching issues in `notes.json`, writes the
ledger, and refreshes `review.json`. It does not edit `evidence.json` or
`extraction-diagnostics.json`. Notes must exist, and the ID must identify a note
issue. Unknown issue IDs and blank explanations are rejected.

Recording a resolution does not transcribe audio again or correct note content
automatically. Correct the affected note content explicitly, or correct evidence
and regenerate notes when recognized content is wrong. When the LaTeX project is
bound to the current clean evidence fingerprint, resolving an issue refreshes the marked
footer in `main.tex` and `project.json`, preserving chapter, diagram, and code
edits. If the project was already stale, resolution is saved but project refresh
is deferred until notes
are regenerated or imported.

## Validated notes to LaTeX

**Producer:** `shad_helper.rendering.write_project`, called after successful
synthesis or import. **Consumer:** `render`, with optional human source edits.

The project writer receives the validated notes directly. `evidence_index` is
used internally to validate source citations; it is neither persisted in
`notes.json` nor passed to the project writer. Block and issue `evidence_ids`
remain in the JSON contracts for validation and review. Ordinary citations produce
no footnotes. Uncertain fragments receive one timestamp-only footnote per affected
paragraph, derived from speech `start` and visual `timestamp` in local evidence.
Times are absolute, floored to whole seconds, deduplicated and sorted; more than
three times are compacted to their earliest/latest range. Review explanations
are not printed as an appendix or separate warning passages.

The writer orders chapters by `start` and preserves each chapter's block sequence.
Only adjacent blocks with matching `topic` names share a section, including across
processing chapters. Names are compared case-insensitively after collapsing
whitespace. Returning to a topic after another topic starts a later section;
blocks are never moved back to an earlier section. Different topic names are not
merged semantically. Blank or missing legacy topics fall back to the chapter title.
Chapters and timestamps remain in `notes.json`; section boundaries do not change
evidence or review provenance.

Generating or importing notes rewrites generated source paths. Rendering
compiles the existing project and preserves source edits. To edit content, use
either the LaTeX files directly or an edited note JSON imported through
`--import`. Directly editing `notes.json` without importing does not rebuild
the project.

### notes/project.json

The complete manifest shape is:

~~~json
{
  "version": 2,
  "draft": true,
  "tex_files": [
    "main.tex",
    "chapters/section-001.tex"
  ],
  "code_files": [],
  "source_blocks": [
    {
      "id": "ch001_missing_detail",
      "topic": "Основные понятия",
      "content": "Формулировка из конспекта.",
      "evidence_ids": ["segment_00000"]
    }
  ],
  "review": [
    {
      "id": "ch001_missing_detail",
      "description": "Лектор сослался на неприведенный вывод; проверить недостающие шаги."
    }
  ],
  "issues": [
    {
      "id": "ch001_missing_detail",
      "description": "Лектор сослался на неприведенный вывод; проверить недостающие шаги.",
      "evidence_ids": ["segment_00000"],
      "status": "open",
      "resolution": ""
    }
  ],
  "evidence_fingerprint": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
}
~~~

| Field | Type | Meaning |
| --- | --- | --- |
| `version` | integer | Manifest format; rendering and review refresh require `2` |
| `draft` | boolean | Whether generation found review flags |
| `tex_files` | array of relative paths | Declared TeX sources, including `main.tex`, topic sections under `chapters/`, and any diagrams |
| `code_files` | array of relative paths | Declared verbatim code assets under `code/` |
| `source_blocks` | array of `{id, topic, content, evidence_ids}` | Original note blocks for per-card PDF and recording provenance |
| `review` | array of `{id, description}` | Blocks with uncertain fragments or legacy uncertain kind, followed by unresolved issues |
| `issues` | array of full review issues | Review metadata, including resolved issues; not printed in the PDF |
| `evidence_fingerprint` | string or null | Fingerprint of complete evidence when the project was written |

Normal synthesis has evidence and writes a fingerprint string. The low-level
project writer can produce null if no evidence file exists.

Each block has invisible PDF destinations `shad-block-ID` and `shad-end-ID`.
After compilation, `notes/source-map.json` stores schema version `1`, the PDF's
SHA-256, `evidence_fingerprint`, and blocks keyed by ID with one-based
`page_start`, `page_end`, and the start `anchor`. Diagram anchors stay inside
the floating figure. A missing destination fails rendering. Old projects without
`source_blocks` can still render, but must be regenerated before lecture Anki
export can produce recording references.

Model-facing Anki answers (`back`, `overview`, `statement`, `proof`, `plan`, and
each `steps` item) are `{text, source: {page, location, block_ids}}` objects;
an absent proof remains null. This makes source coverage part of the response
schema. The exporter assigns Anki field names and step numbers locally.
Normalized `anki/notes.json` and checkpoints carry `card_sources` on every note: exactly one
`{field, page, location, block_ids}` record per populated answer field, including
individual steps. PDF page/order still defines note identity and chronology.
The model selects existing block IDs; recording times are resolved locally from
their speech/frame evidence. The card report uses the actual exported GUID and
template ordinal, not an assumed card count. `anki/report.json` schema version `2`
includes these records in `cards`; `anki/sources.md` presents them as Markdown.
Every model attempt retains its original JSON response and validation status
under `diagnostics/anki/<run>/attempt-XX/`. Failed validation does not publish
the deck or save a successful checkpoint, and diagnostics remain local.
`anki --rebuild` deletes all of `anki/`, `checkpoints/anki.json`, and
`diagnostics/anki/` before any input validation, PDF read, collection access, or
model request. This includes the prior package, all media and temporary build
files, and `anki/identity.json`. The next successful export creates a new
`job_uuid`, `deck_id`, and corresponding note GUIDs; importing it creates new
deck/notes rather than updating the earlier import. Deletion is not rolled back
after validation, generation, or packaging failure. Successful generation writes
a fresh checkpoint at the same path, which can remain reusable if later packaging
fails. Ordinary exports preserve the saved identity and keep the prior package
on failure. Notes and extraction artifacts are unchanged. A custom `--pdf`
within a cleared path must be copied elsewhere before the rebuild.

`review` entries for marked blocks use the block ID. Entries for unresolved
issues use the issue ID. An uncertain block and its generated issue can both
appear as separate flags. `draft` is derived from this list during generation;
the renderer also derives its report's draft flag from the list.

The writer appends diagram files and section files while traversing the
chronological sections; consumers must use the manifest rather than assume a sorted file
list. Ordinary regeneration may leave obsolete files on disk. Only declared
files belong to the current compilation contract. Regeneration removes the
obsolete generator-owned `review.tex` from earlier projects.
`notes --rebuild` first deletes the entire project directory, so no prior sources
or render artifacts survive that reset.

The manifest has no separate Pydantic model. Version, path confinement,
declared references, and source restrictions are checked during rendering.
Editing a fingerprint manually does not validate that a project represents
the current evidence.

### notes/main.tex

This is the LuaLaTeX entry point. It contains:

1. The fixed preamble and supported packages.
2. The marked draft footer region.
3. Document start and a compact centered escaped lecture title.
4. Body content on the same page, without a table of contents or forced page break.
5. `\input{chapters/section-001.tex}` entries in lecture chronology.
6. An inline glossary section when terms exist.
7. Document end.

The generated class is `article`, A4, 11 pt, with 23 mm margins. Text fonts are
`CMU Serif` and `CMU Typewriter Text`; mathematics uses `Latin Modern Math`.
The main language is Russian, with English also configured.
The `amsthm` package supplies theorem, definition, and proof environments.
Definitions and theorems share a counter reset within each section.

The review refresh operation depends on exactly one ordered pair of these
literal markers:

~~~tex
% shad-helper: draft footer begin
\fancyfoot[L]{\scriptsize ЧЕРНОВИК - требуется проверка}
% shad-helper: draft footer end
~~~

For ready notes, the region between the markers is empty. Preserve both markers
when editing `main.tex` so a later resolution can refresh the footer.

### notes/chapters/section-001.tex

Each section file contains one broad topic heading and its blocks in their
original relative order, without a processing-interval heading. Ordinary text is escaped for LaTeX.
Supported inline math is retained. Selected uncertain fragments are colored;
their source times are combined into a footnote. Unmarked source citations
remain invisible. Empty chapters contribute no blocks or heading.

A generated section excerpt is:

~~~tex
\section{Вероятность события}

Событие \(A\) имеет вероятность \(p\).
~~~

Definition and theorem blocks use `\begin{definition}` and `\begin{theorem}`,
respectively; proof blocks use `\begin{proof}[Доказательство]`. The matching end
commands close each block. Examples and explanations are unwrapped prose.

Section files contain body content rather than their own document preamble.
Their filenames are sequential by rendered topic, independent of processing
chapter IDs, and they must be declared in `project.json`.

### notes/figures/diagram-ch001_diagram.tex

There is one file per diagram block, named `diagram-<block ID>.tex`. Its content
is UTF-8 TeX for an inline TikZ picture, optionally containing PGFPlots axes,
expressions, and literal coordinates. It has no independent JSON metadata.

The section wraps the diagram reference in a `figure`, centers it, and writes
the escaped block caption. The manifest lists the diagram in `tex_files`.
External image files and external plot data are unsupported.

### notes/code/ch001_code.txt

There is one UTF-8 file per code block, named `<block ID>.txt`. Content is the
block's exact code string, without Markdown fences or an added newline. The
section displays it through `\VerbatimInput` with line wrapping and a frame.
The manifest lists it in `code_files`.

Code is rendered as text. The renderer does not execute it or apply
language-specific syntax highlighting.

### Project validation

Before compilation, the renderer requires all manifest references to remain
inside `JOB/notes/`. Absolute paths, `..` traversal, and symbolic-link
components are rejected. TeX inputs must reference declared TeX files, and
`\VerbatimInput` must reference declared code files. References use literal
braced paths; dynamically constructed paths are unsupported.

Each declared TeX file must have a `.tex` suffix and exist as readable UTF-8.
Declared code assets must exist under `code/`. `main.tex` must be declared.
Source limits are 2,000,000 UTF-8 bytes per TeX file, 8,000,000 bytes in aggregate,
and at most 40,000 parsed TeX commands per file. Character-code escapes
containing `^^` are rejected.

Commands and environments are checked against the implementation's allowlists.
Math and diagrams are also checked while the project is generated. The
supported project configuration is:

| Configuration | Allowed values |
| --- | --- |
| Document class | `article` |
| Packages | `geometry`, `fontspec`, `polyglossia`, `unicode-math`, `amsmath`, `amssymb`, `amsthm`, `tikz`, `pgfplots`, `fvextra`, `fancyhdr`, `xcolor`, `hyperref`, `microtype`, `graphicx` |
| TikZ libraries | `arrows.meta`, `positioning`, `calc`, `shapes.geometric`, `fit`, `matrix` |
| PGFPlots libraries | `groupplots` |
| Languages | Default `russian`, other `english` |
| Font names | `CMU Serif`, `CMU Typewriter Text`, `Latin Modern Math` |
| Command redefinition | Only `\renewcommand{\headrulewidth}{0pt}` |
| Theorem declarations | Only `\newtheorem{theorem}{Теорема}[section]`, `\theoremstyle{definition}`, and `\newtheorem{definition}[theorem]{Определение}` |

The renderer restricts `documentclass`, `usepackage`, `setmainfont`, and
`renewcommand`, plus `newtheorem` and `theoremstyle`, to `main.tex`. Supported arithmetic and drawing commands
operate on inline content; only renderer-managed PNG image references are permitted.
External plot files, arbitrary files, shell execution, and Lua are outside the contract. The exact command and
environment lists are maintained in [rendering.py](../shad_helper/rendering.py).

## LaTeX to PDF

**Producer:** `shad_helper.rendering.render`, LuaLaTeX/latexmk, and PyMuPDF.
**Consumers:** the reader, page inspection, and downstream use of the PDF.

Rendering checks job compatibility and manifest version, compares the current
full evidence fingerprint, validates the project, and requires `latexmk`
and `lualatex`. It does not regenerate source files from `notes.json`.

Compilation runs from `JOB/notes/`:

~~~sh
latexmk -norc -lualatex -latexoption=-no-shell-escape \
  -interaction=nonstopmode -halt-on-error -file-line-error main.tex
~~~

The environment sets `openin_any=p` and `openout_any=p`, and compilation
has a 600-second timeout. Existing output paths are checked for confinement
before invoking the compiler.
The PDF render requested by `notes --rebuild` adds `-gg` for a clean compilation.

### notes/main.pdf

This is the compiled PDF. It includes a compact title, broad topical sections,
and optional glossary, without a standalone title/contents page or review appendix.
Definitions and theorems are numbered within each section; proofs have their own
environment. Processing chapter intervals are not printed. Highlighted uncertain
fragments retain compact timestamp footnotes. Open review
flags produce the draft footer. A successful compiler exit and an existing PDF
are required, followed by successful page preview generation.

An older PDF can remain after a failed ordinary render. `notes --rebuild` deletes
the old project and PDF before synthesis, so its failure cannot leave the old PDF.
File existence alone is not
evidence that the current render succeeded. The report records the latest
attempt once rendering has entered its report-writing phase.

### notes/qa/page-001.png

PyMuPDF produces one PNG for every PDF page, numbered from 1 with a minimum
three-digit width. A scale of 1.5 pixels per PDF point gives a nominal 108 dpi.
The images preview complete pages and are not input to any model.

Before creating new previews, rendering removes matching `page-*.png` files.
If preview generation fails partway through, the directory can contain only
some pages from that attempt. Compilation and rasterization do not establish
correct mathematical content or a readable layout; inspect the page images
for clipping, labels, equations, and missing glyphs.

### notes/render-report.json

A successful report has this shape:

~~~json
{
  "draft": true,
  "review": [
    {
      "id": "ch001_missing_detail",
      "description": "Лектор сослался на неприведенный вывод; проверить недостающие шаги."
    }
  ],
  "missing_fonts": [],
  "missing_glyphs": [],
  "overfull": [],
  "status": "rendered",
  "pages": 3
}
~~~

| Field | Type | Meaning |
| --- | --- | --- |
| `draft` | boolean | Whether the manifest contains review flags |
| `review` | array of `{id, description}` | Manifest review flags for this project |
| `missing_fonts` | array of strings | Unique sorted log lines matching missing-font/fontspec error patterns |
| `missing_glyphs` | array of strings | Unique sorted `Missing character:` log lines |
| `overfull` | array of strings | Unique sorted overfull horizontal/vertical box log lines |
| `status` | `"rendered"` or `"failed"` | Whether compilation and preview generation completed |
| `pages` | integer, success only | Page count returned after successful preview generation |
| `error` | string, failure only | Caught rendering error or compiler timeout/OS error |

“Font names database not found” startup notices are excluded from
`missing_fonts`. Warnings do not by themselves change status to failed:
a rendered PDF may still contain missing-glyph or overfull-box warnings and
may still be a draft.

After report initialization, a `finally` block attempts to write the report
even on failure. Job guard, project path, missing/malformed manifest, and
unsupported manifest-version failures occur earlier and do not guarantee a
new report. Unexpected exceptions can also leave a failed report without an
`error` field. Do not treat an old report as the result of an attempt that
failed before initialization.

### notes/compiler-output.log and notes/main.log

`compiler-output.log` captures the combined stdout and stderr from the
latexmk invocation. It is replaced when compilation starts. `main.log` is
the TeX engine's own diagnostic log.

The renderer scans both logs for font, glyph, and overfull-box diagnostics.
On compiler failure, the logs and editable sources remain available for
inspection. A failure before compiler invocation can leave earlier logs
unchanged.

### Compiler working files

These files are managed by TeX and latexmk rather than shared note schemas:

| File | Structure and role |
| --- | --- |
| `main.aux` | TeX auxiliary commands carrying references and cross-run document state |
| `main.out` | Hyperref/bookmark auxiliary data |
| `main.toc` | TeX commands for the table of contents |
| `main.fls` | Compiler recorder text listing inputs and outputs, when produced |
| `main.fdb_latexmk` | latexmk dependency/state database for incremental compilation |
| `main.synctex.gz` | Optional compressed synchronization data if produced by compiler configuration |

Presence depends on compiler execution and configuration. The application
checks confinement for these expected output paths but does not parse their
structures as stage input. Their schemas belong to the installed compiler,
and they should not be edited to change lecture content.

## Resume and artifact freshness

### Authority and manual edits

| Desired change | Edit or command | Consequence |
| --- | --- | --- |
| Correct recognized speech | Edit `evidence.json`, then regenerate or import corrected notes | Evidence fingerprint changes; old project cannot render |
| Correct recognized visual content | Edit the matching `visuals` entry in `evidence.json`, then regenerate or import corrected notes | Clean evidence changes invalidate affected notes and old project freshness |
| Change chapter plan | Edit `outline.json`, then run notes | Current outline and reply IDs determine synthesis/import |
| Correct generated content through JSON | Edit a note response and use `--import` | Revalidates provenance and rewrites generated project files |
| Correct layout or content directly in TeX | Edit declared project sources, then render | Rendering preserves edits within supported TeX rules |
| Record a verified issue correction | Use `review --resolve ... --resolution ...` | Updates issue state, resolution ledger, and eligible review metadata |
| Inspect or change extraction diagnostic data | Read or edit `extraction-diagnostics.json` or local vision attempt files | No effect on synthesis, note review, resolution matching, or rendering |
| Change transcript export | Edit `transcript.json`, SRT, or VTT | Inspection copy changes; synthesis evidence does not |
| Change cached stage output | Do not edit checkpoints | Use source corrections, re-extraction, import, or `--rebuild` instead |

### Partial failures

Extraction first writes transcript JSON, SRT/VTT, version 3 speech evidence,
diagnostics, and audio timings. Video extraction then selects frames, runs
recognition, replaces evidence with version 4, and refreshes diagnostics and
timings. An interruption can leave only some files updated. Audio, ASR, selection,
and successful vision checkpoints remain useful for rerun. A fatal vision error
raises after preserving the audio artifacts and recording the failure; version
3 speech evidence can remain under a version 4 marker. The marker is therefore
not a completion flag or proof that video processing succeeded.

Notes preparation writes its outline before generation or import completes.
Automatic synthesis checkpoints chapters individually. Only after all chapter
replies validate and merge does it write `notes.json`, `review.json`, and
the project. These writes remain separate: a project-generation error can
leave updated notes with an incomplete or, on an ordinary invocation, older project.
With `--rebuild`, all old notes-stage outputs have already been deleted before
compatibility or evidence validation begins.

Automatic fallback returns a handoff path normally, so the CLI can exit
successfully while manual replies are still needed. Check the returned path
and handoff status rather than treating exit code zero or an older
`notes.json` as proof of a new synthesis result. Invalid evidence, lack of both
speech and usable visual content, and outline errors before the synthesis loop
fail directly instead.

Neither re-extraction nor ordinary handoff export removes an old note project or PDF.
Ordinary project generation does not purge obsolete TeX/code assets. A notes
rebuild removes the entire old project, even for manual export or failed input
validation. Rendering compares
evidence identity and validates declared files, but neither `notes.json` nor
the render report stores a complete hash of all manually edited TeX and PDF
bytes.

### Reuse boundaries

| Repeated action | What can be reused | What is refreshed or invalidated |
| --- | --- | --- |
| Extract same recording and interval | Matching audio, ASR, selection, and successful vision checkpoints | Worker request/response, clean transcript/evidence, and diagnostic report are rewritten |
| Change recording bytes or core bounds | Unaffected matching checkpoints only | Changed audio inputs and dependent ASR results |
| Change ASR language/model or WAV bytes | Existing WAV if its audio inputs still match | ASR checkpoint fingerprint |
| Change vision preset, revision, prompt, settings, package versions, or selected image bytes | Matching audio, ASR, and selection results | Affected visual inference checkpoints and clean visual evidence |
| Delete a selected frame image | Matching selection metadata | Missing PNG is retrieved from the original recording |
| Run automatic notes again | Matching validated chapter checkpoints | Final merged notes, review summary and generated project |
| Edit only the diagnostic report | Matching note checkpoints, resolutions, and project | No synthesis or render invalidation; report remains informational |
| Use notes `--rebuild` | Extraction artifacts/checkpoints and explicit review resolutions | All notes outputs, outline, handoff and notes/illustration caches deleted before validation; fresh notes/PDF generated |
| Use anki `--rebuild` | Notes/PDF and extraction artifacts | Anki outputs, identity, checkpoint and diagnostics deleted before validation; new deck and note identities generated |
| Export manual handoff | Existing extraction artifacts | Entire generated input subtree rebuilt for the current outline; final notes need replies/import |
| Import replies | Extraction artifacts and compatible resolutions | Final notes, review summary and project; no model calls |
| Render existing project | Compiler may use its working files | PDF, logs, report and previews; source edits are preserved |

Re-extraction rewrites informational warnings and diagnostics without touching
note issue statuses. The resolution ledger is retained; later note validation
restores a resolution only if its complete clean evidence and note issue match.

## Implementation references

- [jobs.py](../shad_helper/jobs.py): version guard, atomic JSON writes, fingerprints,
  checkpoints, and source timestamp labels.
- [extraction.py](../shad_helper/extraction.py): media probing, audio intervals,
  stitching, transcript exports, warnings, and evidence.
- [local_worker.py](../shad_helper/local_worker.py): worker request/response and ASR
  checkpoint results.
- [video.py](../shad_helper/video.py): source timestamps, frame selection, image
  retrieval, and selection checkpoints.
- [vision_worker.py](../shad_helper/vision_worker.py) and
  [vision_config.py](../shad_helper/vision_config.py): pinned presets, local model
  inference, strict content parsing, raw attempts, and visual checkpoints.
- [schemas.py](../shad_helper/schemas.py): outline, note, block, glossary, and issue
  fields and model validators.
- [notes.py](../shad_helper/notes.py): chapter contexts, budgets, handoff/import,
  merging, evidence validation, and explicit resolutions.
- [openai_client.py](../shad_helper/openai_client.py): automatic text request,
  transport schema normalization, and completed-response handling.
- [rendering.py](../shad_helper/rendering.py): generated files, manifest validation,
  review refresh, compilation, reports, and previews.
- [cli.py](../shad_helper/cli.py): command options, printed result paths, and exit
  behavior.
- [Pipeline design](plan.md) and [usage instructions](../README.md).
