# Audio to PDF stage contracts

The pipeline turns a recording into timestamped speech evidence, Russian notes,
an editable LaTeX project, and a PDF. Each boundary below specifies the files
produced, their structure, and the rules the next stage applies before using them.

This contract describes the current implementation. It covers the audio-only
version 3 job format, automatic synthesis, manual ChatGPT handoff and import,
review, and rendering. Video inputs contribute only their first audio stream.
There are no extracted frames, OCR results, source-file imports, or external
image assets. Diagrams are generated from speech as inline TikZ or PGFPlots.

## Contents

- [Stage boundaries](#stage-boundaries)
- [Shared conventions](#shared-conventions)
- [Recording to audio chunks](#recording-to-audio-chunks)
- [Audio chunks to raw transcription](#audio-chunks-to-raw-transcription)
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
| Transcribe audio | Local worker started by `extract` | Worker request, WAV chunks, ASR checkpoints | Worker response, ASR checkpoints | Worker exits successfully and returns all requested items |
| Assemble evidence | Internal part of `extract` | Raw worker response | Clean evidence, separate diagnostics, transcript copies, extraction timings | `extract` returns the path to `evidence.json` |
| Prepare chapters | Internal part of `notes` | Evidence, optional saved outline | Saved outline and in-memory chapter contexts | A consecutive outline covers the extracted interval |
| Synthesize or import | `notes`, optionally `--manual` or `--import` | Chapter contexts, schema, optional replies and resolutions | Notes, review summary, LaTeX project; or a manual handoff | A project path means notes were generated; a handoff path means replies are still required |
| Review | `review`, optionally `--resolve` | Clean evidence, notes, review files and project | Resolution ledger, updated note issues and review metadata | A named note issue has an explicit correction recorded |
| Render | `render` | Existing LaTeX project, manifest, current evidence | PDF, page previews, report and compiler files | PDF compilation and page preview generation both complete |

The main stage boundary is `evidence.json` → `notes.json` → the files declared
by `notes/project.json` → `notes/main.pdf`. The worker files and checkpoints
support extraction and resume. Transcript copies support inspection. The handoff
files provide an alternative way to obtain the same note response format.

Review is optional in the command sequence. Open issues do not prevent rendering;
they produce a draft footer and remain available in the review JSON files and
the `review` command. The PDF contains no review appendix or audio source
footnotes.
Extraction warnings are separate informational data. Notes ignore them without
requiring a resolution or an acknowledgment. A note issue instead records a
missing explanation, formula detail, or other uncertainty discovered during
synthesis; it remains relevant even when the transcription is accurate.

### Job directory

Paths in this tree are relative to the job passed through `--out` or `JOB`.
Files marked conditional appear only when that branch runs or that content exists.

~~~text
JOB/
├── extraction-config.json
├── audio/
│   └── chunk_0000.wav                         # one file per audio core interval
├── workers/
│   ├── asr.request.json
│   └── asr.response.json
├── checkpoints/
│   ├── audio/chunk_0000.json
│   ├── asr/chunk_0000.json
│   └── notes/ch001.json                       # automatic synthesis only
├── evidence.json
├── extraction-diagnostics.json
├── transcript.json
├── transcript.srt
├── transcript.vtt
├── extraction-timings.json
├── outline.json
├── handoff/                                  # manual export or automatic fallback
│   ├── README.txt
│   ├── status.json                           # fallback or unresolved size warning
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
    ├── chapters/ch001.tex
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
| `result` | stage-specific JSON | Audio descriptor, transcribed chunk, or validated single-chapter note object |

Only successful actions create a new checkpoint. Reuse returns the saved result
without rerunning the action or updating its recorded time. A matching
fingerprint is not a full revalidation of the saved result. Checkpoints are
pipeline-owned cache files, rather than an interface for manual corrections.

## Recording to audio chunks

**Producer:** `shad_helper.extraction`. **Consumers:** the local ASR worker and
later extraction reruns.

The input must be a readable media file with a first audio stream. `ffprobe`
selects `a:0`; the duration of that stream takes precedence over the container
duration. Container duration is used only when stream duration is missing or
`N/A`. The resulting duration must be finite and positive.

`--start` defaults to zero and must be finite, nonnegative, and less than the
media duration. An optional `--duration` must be finite and positive. The saved
end is `min(media duration, start + requested duration)`, or the media duration
when no duration is requested.

### extraction-config.json

This job marker is written after input probing and interval validation, before
chunk extraction. Its complete structure is:

~~~json
{
  "schema_version": 3,
  "pipeline": "audio"
}
~~~

The marker must equal this object exactly; extra keys make it unsupported.
It permits an interrupted extraction to resume before `evidence.json` exists.

The job guard accepts a new or empty directory, a directory with this marker,
or a directory containing supported version 3 evidence. An incompatible marker,
legacy evidence, or a nonempty directory with neither marker nor evidence is
rejected before writing. This includes version 2 audio jobs as well as older
visual-processing jobs. The commands do not migrate them; use a fresh directory.

### audio/chunk_0000.wav

Each WAV has one channel and a 16,000 Hz sample rate. FFmpeg chooses its WAV
encoding; the command does not explicitly select a PCM subtype. Only audio
stream `0:a:0` is mapped, and video output is disabled.

Audio is divided into core intervals of at most 600 seconds. For a core
`[lower, upper)`:

~~~text
offset    = max(0, lower - 5)
audio_end = min(extracted clip end, upper + 5)
WAV spans offset through audio_end
~~~

Thus adjacent full chunks can share ten seconds of audio. The first chunk of
a pilot clip may include five seconds before the requested start. Audio never
extends past the extracted clip end. The final core may be shorter than 600
seconds.

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

This is the authoritative input to synthesis. It contains clean speech without
word data, scores, warnings, or ASR model metadata. Its complete shape is:

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
| `schema_version` | integer | Must be `3` |
| `title` | string | Recording filename stem by default; can be corrected before synthesis |
| `media` | object | Recording identity and extracted interval |
| `media.path` | string | Absolute path to the original recording |
| `media.sha256` | string | Original recording's byte hash |
| `media.duration` | number | Probed full audio duration |
| `media.clip_start`, `media.clip_end` | numbers | Requested interval after end clipping |
| `segments` | array of clean segments | Speech and source IDs; may be empty |

Every clean segment has exactly these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Current extraction's source identifier |
| `start`, `end` | numbers | Absolute recording bounds, clamped to the audio core |
| `text` | string | Nonempty reconstructed speech |

The version guard requires version 3 and exactly the four root keys shown above.
It requires `media` to be an object and `segments` to be an array of objects
containing exactly `id`, `start`, `end`, and `text`. Missing or additional root or
segment fields are rejected,
including `issues`, `models`, `words`, and `diagnostics`. This enforces the
boundary between speech and diagnostics; it is not full nested type or semantic
validation of media and timestamps. Downstream code expects the producer types
above.

For lecture bounds, synthesis falls back to `clip_start = 0` and
`clip_end = duration` if those keys are missing. It rejects an empty interval.
New extraction always writes both clip bounds.

### extraction-diagnostics.json

This report contains the data removed from stitched segments and the warnings
computed from it. Extraction computes warnings before creating clean evidence,
so separation does not change word ownership or warning thresholds.

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
| `warnings` | array of warning objects | Informational ASR warnings, or `[]` |
| Warning `id` | string | Generated `issue_0000` identity within this extraction |
| Warning `description` | string | Human-readable condition and suggested inspection |
| Warning `evidence_ids` | array of strings | Affected clean speech segment IDs |

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

Warnings preserve the transcribed text; they do not silently correct it.
The report is an inspection artifact, not a synthesis or review input. Notes
do not read it, including when missing or edited, and there is no required
approval or ignore action. Correct source speech in `evidence.json` if a warning
reveals a transcription error before generating notes.

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

~~~json
{
  "scope": "current extraction invocation",
  "audio_asr_seconds": 12.34
}
~~~

`audio_asr_seconds` measures elapsed wall time from before chunk preparation
through transcription, stitching, and transcript, evidence, and diagnostic
writes. It includes checkpoint lookup and reused work, and is not a sum of
historic checkpoint times or a measurement of model execution alone. It excludes initial probing and
original-media hashing. The file is written after evidence.

## Evidence to chapter inputs

**Producer:** `shad_helper.notes`. **Consumers:** automatic synthesis, manual
handoff, and chapter reply validation.

At least one evidence segment must contain nonblank speech before `notes`
can run. Silence remains a valid extraction result but cannot produce notes.

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

Default neighboring context is `min(60 seconds, chapter duration / 4)` on
each side. A segment is selected when its end is at or after the expanded
start and its start is at or before the expanded end. Selected segments are
included in full, not trimmed at chapter or context bounds. Adjacent chapters
can therefore receive the same context segments.

The context has no word probabilities, word timestamps, ASR diagnostics,
extraction warnings, or existing note issues. Note generation does not open
`extraction-diagnostics.json` or copy its contents into checkpoints or review.

The model-facing context adds lecture metadata to those two fields. Its complete
shape, also saved as a manual `context.json`, is:

~~~json
{
  "lecture_title": "lecture",
  "glossary": [],
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
| `chapter` | Outline chapter object | Required single chapter and its bounds |
| `segments` | Array of clean speech segments | Chapter and neighboring speech, included once |

There is no nested `context` wrapper. Automatic API input is a compact JSON
serialization of this object. Manual JSON is the same structure written with
readable indentation. Neither contains a response schema or instructions.

A block can cite any segment selected for its chapter, including neighboring
context. The prompt asks the model to use neighboring speech for understanding
without duplicating it in the chapter; that editorial rule is not enforced by
timestamp filtering of the reply's prose.

### Request size and chapter splitting

Preflight measures UTF-8 bytes of the synthesis instructions, serialized model
context, and serialized Pydantic response schema. The cap is 120,000 bytes. This
is a local text budget, not an exact token count or the serialized HTTP body size.

Oversized chapters are bisected in time. Child IDs have the form
`<first 60 characters of parent>_a_<8-character hash>` and the corresponding
`_b_` ID. The hash comes from the parent ID and child bounds. Titles are
initially inherited. Children must be at least 30 seconds long; a chapter
shorter than 60 seconds cannot be bisected. This does not impose a 30-second
minimum on an original short recording or an imported chapter.

Preparation persists the split outline. During automatic generation, the same
check repeats with the glossary accumulated from earlier replies. A typed
`context_length_exceeded` model failure also triggers splitting immediately,
instead of retrying the same input.

Other chapter request failures and invalid replies get up to three retries after
the initial attempt, with waits of 5, 10, and 20 seconds. Each retry uses the same
chapter, context, glossary, instructions, and response schema. Only a validated
successful reply is checkpointed. Completed chapters are reused; exhausting all
four attempts exports the manual handoff with the final failure in `status.json`.

If time splitting cannot reduce an oversized input, all speech is retained in
the manual handoff and `status.json` explains the failure. Import mode skips
automatic request-size splitting because no model request is made.

## Chapter inputs to validated notes

**Producer:** automatic model responses or user-created ChatGPT reply files,
followed by local validation and merging. **Consumers:** review and LaTeX
generation.

### Automatic request and reply

Automatic synthesis submits text only. The user input is the compact JSON
model context: lecture title, current glossary, chapter, and speech. Russian
instructions request one chapter with the provided ID and timestamps,
speech-grounded content, source citations, and explicit uncertainty.
The instructions are identical to the manual `prompt.txt`. They contain no
lecture data or embedded response schema. The output schema is supplied through
the client's response-format field rather than duplicated in the user input.

The current client default is model `gpt-6-astra` with reasoning effort
`high`. Its response format is named `lecture_notes`. For the API's strict
JSON Schema format, the client recursively removes schema defaults, requires
every declared object property, and sets `additionalProperties: false`.
Application-level provenance and review validators still run afterward.

Raw HTTP requests, streaming events, and raw note replies are not written as
job artifacts. Incomplete streams are discarded. Only a successfully validated
chapter result is checkpointed.

### Manual handoff files

`notes JOB --manual` creates the files below without network calls. Automatic
synthesis also creates them when plan access, response processing, or generation
validation fails inside the synthesis loop.

| File | Structure | Producer and consumer |
| --- | --- | --- |
| `handoff/inputs/response-schema.json` | Full JSON Schema generated by `LectureNotes.model_json_schema()` | Shared response constraints attached to ChatGPT for every chapter |
| `handoff/inputs/<chapter ID>/context.json` | Model-facing context shown above | All lecture data attached to ChatGPT once |
| `handoff/inputs/<chapter ID>/prompt.txt` | Russian synthesis instructions only, byte-identical to automatic API instructions | Pasted into ChatGPT |
| `handoff/README.txt` | Plain English instructions for attaching files, saving replies, and importing | Human operator |
| `handoff/status.json` | Status and explanatory reason | User or automation deciding whether manual work remains |

For a chapter, send only its prompt, its context, and the shared response schema.
Their responsibilities are distinct: instructions, lecture data, and response
structure. The root `outline.json` remains the authoritative chapter plan and
determines which replies must be imported. It is not exported again or attached
to the model. There is no chapter `transcript.txt`; it would repeat speech already
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
status file. Those paths do not otherwise refresh model inputs, so an existing
handoff can become stale after successful generation or import.

Every export checks job and evidence compatibility, then deletes and rebuilds
the generator-owned `handoff/inputs/` subtree. This removes obsolete chapter
folders and duplicate files from earlier exports. Operator files outside that
subtree are preserved by the cleanup; `README.txt` is rewritten with current
instructions, while the calling workflow manages `status.json`. The current root
`outline.json` determines the exported chapter IDs. Send one chapter's three
inputs per request. An interrupted export can leave only part of the rebuilt
input tree; rerun export before attaching it.

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
          "evidence_ids": ["segment_00000"],
          "caption": "",
          "language": ""
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
| Chapter `blocks` | nonempty array | Required | Ordered content blocks |

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
| `kind` | one of six values below | Required | Rendering and provenance behavior |
| `content` | nonempty string | Required | Text, LaTeX, or literal code |
| `evidence_ids` | array of strings | Default `[]` | Existing source segment IDs |
| `caption` | string | Default `""` | Diagram caption; other block renderers do not use it |
| `language` | string | Default `""` | Code-language metadata; current renderer uses plain verbatim output |

| Kind | Content contract | Source requirement and rendering |
| --- | --- | --- |
| `text` | Ordinary prose; inline math uses only `\(...\)` | At least one permitted evidence ID; prose is escaped and supported inline math retained |
| `math` | Prompt requests a bare LaTeX expression without display delimiters or Markdown | At least one permitted evidence ID; wrapped as display math unless already a supported display form |
| `code` | Literal code without Markdown fences | At least one permitted evidence ID; written to a text asset and displayed verbatim |
| `diagram` | Prompt requests a complete `tikzpicture`; PGFPlots may be embedded | At least one permitted evidence ID; diagram commands are validated and written to a TeX asset |
| `clarification` | Added explanatory prose, with supported inline math | May have no evidence IDs; rendered with a “Пояснение” label |
| `uncertain` | Missing or ambiguous details, with supported inline math | At least one permitted evidence ID; rendered with a review label and generates an open issue |

For every evidence ID cited by a block, the source segment must exist in that chapter's
selected context. An empty citation list is accepted only for clarification.
An issue's citations must exist in the available evidence. Citations validate
the reference relationship; they do not prove that the model's prose or code
accurately represents the speech.

The instructions prohibit inventing unspoken formulas, code, execution results,
graph values, or unseen slide details. Diagram captions should identify a
schematic plot when actual data were not spoken. Those content rules require
review; they are not semantic facts that the JSON validator can establish.

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

Every `uncertain` block generates an issue with ID `review_<block ID>`.
If that exceeds 80 characters, the ID becomes
`review_<first 60 characters of block ID>_<12-character block-ID hash>`.
Its description and citations come from the block. Saved matching explicit
resolutions can subsequently restore resolved issue status.

### checkpoints/notes/ch001.json

For automatic synthesis, `result` is a validated single-chapter
`LectureNotes` object, including note uncertainty and explicit resolution
handling. The fingerprint input object contains:

| Input key | Value |
| --- | --- |
| `context` | Internal chapter context containing only chapter metadata and clean speech |
| `prompt` | Exact compact model-context JSON with lecture title and current accumulated glossary |
| `instructions` | Russian synthesis instructions |
| `model` | Client's model identifier |
| `rebuild` | Boolean for this invocation |
| `reasoning_effort` | Current default effort, `high` |
| `schema` | Full Pydantic response schema |

These inputs are hashed but are not individually saved alongside the
checkpoint. A cached chapter result can be reused while later chapters run.
Speech, prompt, model, outline context, glossary, or schema changes can
invalidate affected chapter reuse. Diagnostic-report changes cannot. Final
resolution checks use the complete clean evidence fingerprint even when a
single chapter checkpoint is reused.

`--rebuild` removes the entire notes checkpoint directory after the job and
nonempty-speech checks, then generates again. Since the flag is also hashed,
a later run without `--rebuild` does not match checkpoints created with it.
Neither import mode nor manual export creates synthesized chapter checkpoints.

## Review state and resolutions

### Review issue format

Note issues and review summaries share this object. Extraction warnings have
their separate, status-free format in `extraction-diagnostics.json`.

| Field | Type | Required or default in note imports | Meaning |
| --- | --- | --- | --- |
| `id` | identifier string | Required | Issue identity |
| `description` | string | Required | Problem requiring review |
| `evidence_ids` | array of strings | Default `[]` | Related speech segments |
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

`ready` is true only when there are no open issues and no `uncertain` blocks.
Resolving an issue alone does not change the block's kind or content. A remaining
uncertain block keeps the notes in draft state even if its generated issue was
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
and regenerate notes when speech is wrong. When the LaTeX project is bound to the
current clean evidence fingerprint, resolving an issue refreshes the marked
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
remain in the JSON contracts for validation and review, but produce no audio
source footnotes in the PDF. Review issues are not printed as an appendix.

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
    "chapters/ch001.tex"
  ],
  "code_files": [],
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
| `tex_files` | array of relative paths | Declared TeX sources, including `main.tex`, chapters, and any diagrams |
| `code_files` | array of relative paths | Declared verbatim code assets under `code/` |
| `review` | array of `{id, description}` | Uncertain blocks followed by unresolved issues |
| `issues` | array of full review issues | Review metadata, including resolved issues; not printed in the PDF |
| `evidence_fingerprint` | string or null | Fingerprint of complete evidence when the project was written |

Normal synthesis has evidence and writes a fingerprint string. The low-level
project writer can produce null if no evidence file exists.

`review` entries for uncertain blocks use the block ID. Entries for unresolved
issues use the issue ID. An uncertain block and its generated issue can both
appear as separate flags. `draft` is derived from this list during generation;
the renderer also derives its report's draft flag from the list.

The writer appends diagram files and chapter files while traversing the
chapters; consumers must use the manifest rather than assume a sorted file
list. A regenerated project may leave obsolete files on disk. Only declared
files belong to the current compilation contract. Regeneration removes the
obsolete generator-owned `review.tex` from earlier projects.

The manifest has no separate Pydantic model. Version, path confinement,
declared references, and source restrictions are checked during rendering.
Editing a fingerprint manually does not validate that a project represents
the current evidence.

### notes/main.tex

This is the LuaLaTeX entry point. It contains:

1. The fixed preamble and supported packages.
2. The marked draft footer region.
3. Escaped lecture title, empty author and date.
4. Document start, title page content, table of contents, and a new page.
5. `\input{chapters/<chapter ID>.tex}` entries in note order.
6. An inline glossary section when terms exist.
7. Document end.

The generated class is `article`, A4, 11 pt, with 23 mm margins. Text fonts are
`CMU Serif` and `CMU Typewriter Text`; mathematics uses `Latin Modern Math`.
The main language is Russian, with English also configured.

The review refresh operation depends on exactly one ordered pair of these
literal markers:

~~~tex
% shad-helper: draft footer begin
\fancyfoot[L]{\scriptsize ЧЕРНОВИК - требуется проверка}
% shad-helper: draft footer end
~~~

For ready notes, the region between the markers is empty. Preserve both markers
when editing `main.tex` so a later resolution can refresh the footer.

### notes/chapters/ch001.tex

Each chapter file contains a section heading, a human-readable audio interval,
then blocks in their original order. Ordinary text is escaped for LaTeX.
Supported inline math is retained. Source citations do not produce footnotes.

A generated chapter excerpt is:

~~~tex
\section{Вероятность события}

\noindent\textit{Аудио: 00:30:00 - 00:32:00}\par

Событие \(A\) имеет вероятность \(p\).
~~~

Chapter files contain body content rather than their own document preamble.
Their filename derives from the chapter ID, and they must be declared in
`project.json`.

### notes/figures/diagram-ch001_diagram.tex

There is one file per diagram block, named `diagram-<block ID>.tex`. Its content
is UTF-8 TeX for an inline TikZ picture, optionally containing PGFPlots axes,
expressions, and literal coordinates. It has no independent JSON metadata.

The chapter wraps the diagram reference in a `figure`, centers it, and writes
the escaped block caption. The manifest lists the diagram in `tex_files`.
External image files and external plot data are unsupported.

### notes/code/ch001_code.txt

There is one UTF-8 file per code block, named `<block ID>.txt`. Content is the
block's exact code string, without Markdown fences or an added newline. The
chapter displays it through `\VerbatimInput` with line wrapping and a frame.
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
| Packages | `geometry`, `fontspec`, `polyglossia`, `unicode-math`, `amsmath`, `amssymb`, `tikz`, `pgfplots`, `fvextra`, `fancyhdr`, `xcolor`, `hyperref`, `microtype` |
| TikZ libraries | `arrows.meta`, `positioning`, `calc`, `shapes.geometric`, `fit`, `matrix` |
| PGFPlots libraries | `groupplots` |
| Languages | Default `russian`, other `english` |
| Font names | `CMU Serif`, `CMU Typewriter Text`, `Latin Modern Math` |
| Command redefinition | Only `\renewcommand{\headrulewidth}{0pt}` |

The renderer restricts `documentclass`, `usepackage`, `setmainfont`, and
`renewcommand` to `main.tex`. Supported arithmetic and drawing commands
operate on inline content; commands for external images, external plot files,
shell execution, and Lua are outside the contract. The exact command and
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

### notes/main.pdf

This is the compiled PDF. It includes the title, contents, ordered chapters,
and optional glossary. It has no audio source footnotes or review appendix.
Chapter audio intervals and inline uncertainty labels remain. Open review
flags produce the draft footer. A successful compiler exit and an existing PDF
are required, followed by successful page preview generation.

An older PDF can remain after a failed render. File existence alone is not
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
| Change chapter plan | Edit `outline.json`, then run notes | Current outline and reply IDs determine synthesis/import |
| Correct generated content through JSON | Edit a note response and use `--import` | Revalidates provenance and rewrites generated project files |
| Correct layout or content directly in TeX | Edit declared project sources, then render | Rendering preserves edits within supported TeX rules |
| Record a verified issue correction | Use `review --resolve ... --resolution ...` | Updates issue state, resolution ledger, and eligible review metadata |
| Inspect or change ASR diagnostic data | Read or edit `extraction-diagnostics.json` | No effect on synthesis, note review, resolution matching, or rendering |
| Change transcript export | Edit `transcript.json`, SRT, or VTT | Inspection copy changes; synthesis evidence does not |
| Change cached stage output | Do not edit checkpoints | Use source corrections, re-extraction, import, or `--rebuild` instead |

### Partial failures

Successful extraction writes transcript JSON, SRT/VTT, evidence, diagnostics,
then timings.
An interruption can leave only some of those files updated. Chunk and ASR
checkpoints remain useful for rerun. The version marker is not a “complete”
status flag.

Notes preparation writes its outline before generation or import completes.
Automatic synthesis checkpoints chapters individually. Only after all chapter
replies validate and merge does it write `notes.json`, `review.json`, and
the project. These writes remain separate: a project-generation error can
leave updated notes with an incomplete or older project.

Automatic fallback returns a handoff path normally, so the CLI can exit
successfully while manual replies are still needed. Check the returned path
and handoff status rather than treating exit code zero or an older
`notes.json` as proof of a new synthesis result. Invalid evidence, silent
input, and outline errors before the synthesis loop fail directly instead.

Neither re-extraction nor handoff export removes an old note project or PDF.
Project generation does not purge obsolete TeX/code assets. Rendering compares
evidence identity and validates declared files, but neither `notes.json` nor
the render report stores a complete hash of all manually edited TeX and PDF
bytes.

### Reuse boundaries

| Repeated action | What can be reused | What is refreshed or invalidated |
| --- | --- | --- |
| Extract same recording and interval | Matching audio and ASR checkpoints | Worker request/response, clean transcript/evidence, and diagnostic report are rewritten |
| Change recording bytes or core bounds | Unaffected matching checkpoints only | Changed audio inputs and dependent ASR results |
| Change ASR language/model or WAV bytes | Existing WAV if its audio inputs still match | ASR checkpoint fingerprint |
| Run automatic notes again | Matching validated chapter checkpoints | Final merged notes, review summary and generated project |
| Edit only the diagnostic report | Matching note checkpoints, resolutions, and project | No synthesis or render invalidation; report remains informational |
| Use notes `--rebuild` | Extraction checkpoints | All automatic note checkpoints are removed |
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
