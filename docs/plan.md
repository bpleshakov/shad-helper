# Russian lecture-to-PDF pipeline

## Flow

`extract MEDIA --out JOB → notes JOB → review JOB → render JOB`

Audio files and video soundtracks share local Whisper large-v3 transcription.
Videos also use local Qwen vision extraction, configured with
`--vision-model 35b|9b|4b|off` (default `35b`).
Note synthesis uses the existing ChatGPT-plan integration with manual export/import
as an offline alternative. Outputs are editable Russian LaTeX and a PDF containing
evidence-grounded explanations, equations, code, and TikZ/PGFPlots diagrams.

## Extraction and evidence

- Select the first audio stream; prefer its duration and fall back to container
  duration. Reject missing audio, invalid duration, or invalid intervals.
- Produce 16 kHz mono WAV chunks with ten-minute core intervals and five-second
  overlaps. Assign words to one core interval and retain original timestamps.
- Run ASR separately with successful per-chunk checkpoints. Model, language, chunk
  identity, and audio-file hash determine reuse.
- Export JSON/SRT/VTT, confidence/repetition warnings, and current-run timings.
  Preserve silent artifacts; synthesis requires speech or useful visual content.
- After the ASR process exits, select native video frames using content changes,
  settled states, pre-erasure states and scrolling. Run one isolated vision
  worker using the selected pinned model. Save per-frame checkpoints; retry
  malformed JSON once and retain incomplete frames explicitly.

Audio `evidence.json` uses schema version 3 and contains exactly `schema_version`,
`title`, `media`, and `segments`. Each speech segment contains exactly `id`,
`start`, `end`, and `text`. Media includes path, SHA-256, duration, and extracted
interval. `transcript.json` is a copy of the clean segment array.
Video evidence uses version 4 and adds `visuals`, with `id`, `timestamp`, `status`,
`kind`, `text`, `formulas`, `diagrams`, and `unreadable`. Frames remain local;
their source PTS supply the visual timestamps.

`extraction-diagnostics.json` uses schema version 1 and contains the clean
evidence fingerprint, `models.asr`, per-segment words and diagnostic scores, and
informational warnings with `id`, `description`, and `evidence_ids`. It has no
review statuses or approval gate. Notes never read it.
Video diagnostics additionally retain the effective model/revision, selection
measurements and per-frame raw attempts; they do not enter clean visual evidence.

Before extraction, `extraction-config.json` records
`{"schema_version": 3, "pipeline": "audio"}` so interrupted jobs can resume.
Video uses `{"schema_version": 4, "pipeline": "audio_video", "vision_model": "35b"}`
with the selected preset. Supported v3/v4 jobs can switch presets or disable
vision while reusing matching audio/selection work. A fatal vision error returns
an error while preserving completed speech evidence for inspection and resumption.
Legacy jobs and nonempty unrecognized directories require a new output directory;
reject them before any rebuild, review, or rendering writes.

## Notes and rendering

- Use consecutive fifteen-minute chapters, bounded neighboring speech/visual context,
  an editable outline, shared glossary, accumulated topic names, and chapter checkpoints.
  These chapters are processing intervals; render them by start time, preserving
  the block sequence. Only adjacent blocks of the same broad topic share a section,
  including across intervals. Keep later returns to a topic in their chronological
  place and chapter provenance in JSON. Legacy blocks use chapter titles.
- Send speech IDs, timestamps, text and clean visual records. Include the latest
  preceding frame for a potentially continuing slide. Isolate extraction diagnostics from
  generation, checkpoints, review, and render freshness. Lecture blocks cite valid
  chapter-scoped speech or visual evidence.
- Split requests above 120,000 UTF-8 bytes and retry typed context-limit failures.
  Stop before parts become shorter than thirty seconds. Preserve all evidence in
  manual handoffs when splitting cannot help.
- Automatic and imported replies use the same schema. Render definitions,
  theorems, and proofs with LaTeX `amsthm` environments; keep examples,
  clarifications, and explanations as plain prose. Mark only the questionable
  fragments of useful study material; do not invent
  missing formula, code, or diagram details. Combine source times into one short
  footnote per affected paragraph; keep diagnostic explanations in local review.
  Cross-check fallible visual extraction against speech. Notebook visuals supply
  theory and explanatory comments, excluding executable code and service output.
- Focus on definitions, facts, formulas, theorem conditions, proof steps, and
  useful explanations, omitting narration, administration, and repetitions.
  Aim for ten pages per three hours, up to twenty for dense lectures, with a soft
  duration-scaled word allowance. Omit empty chapters from the PDF, preserve their
  outline intervals, and avoid a standalone title/contents page.
- Use three distinct manual inputs: shared instructions in `prompt.txt`, lecture
  title/glossary/topics/writing budget/chapter/speech/visuals in `context.json`, and output constraints in
  `response-schema.json`. Keep them under `handoff/inputs/`, with operator README
  and fallback status outside it. Do not duplicate the outline or transcript in model inputs.
  Initial manual topic lists are empty; use consistent broad names across replies.
- Save the handoff on every notes run, including cache hits and imports. Retain
  separate `handoff/runs/<run ID>/` archives with the run's outline and actual
  chapter contexts, including accumulated glossary and topic names. Save each attempt's exact
  outbound API JSON before transmission; keep retries and split-parent requests.
  Exclude credentials and OAuth exchanges; never send after an archive write fails.
- Preserve note-generated issues and explicit resolutions. Review operates on
  notes only. Bind resolution records and the rendered project to the complete
  clean evidence fingerprint; diagnostic-only changes leave them valid.
- Retain confined TeX/code references, disabled shell escape, review-driven draft
  footers, manual-edit preservation, and evidence fingerprints.
- Compile with LuaLaTeX, retain logs, and render PDF previews. Diagrams use inline
  TikZ/PGFPlots; external images are unsupported.

## Verification

Cover media routing, intervals, overlap ownership, resume, checkpoint invalidation,
silence, diagnostic isolation, distinct manual inputs, manual round trips,
chapter budgets, citations, review persistence, and legacy-file preservation.
Use generated audio and multistream video with mocked Whisper, and real
LuaLaTeX/PyMuPDF for PDF checks.
Assess real speech accuracy separately on a representative Russian recording.
Vision pilot results are documented separately. Integration does not certify
handwritten-symbol or diagram accuracy, or the one-hour vision runtime target.
