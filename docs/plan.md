# Russian audio-to-PDF pipeline

## Flow

`extract MEDIA --out JOB → notes JOB → review JOB → render JOB`

Audio files and video soundtracks share local Whisper large-v3 transcription.
Note synthesis uses the existing ChatGPT-plan integration with manual export/import
as an offline alternative. Outputs are editable Russian LaTeX and a PDF containing
speech-grounded explanations, equations, code, and TikZ/PGFPlots diagrams.

## Extraction and evidence

- Select the first audio stream; prefer its duration and fall back to container
  duration. Reject missing audio, invalid duration, or invalid intervals.
- Produce 16 kHz mono WAV chunks with ten-minute core intervals and five-second
  overlaps. Assign words to one core interval and retain original timestamps.
- Run ASR separately with successful per-chunk checkpoints. Model, language, chunk
  identity, and audio-file hash determine reuse.
- Export JSON/SRT/VTT, confidence/repetition warnings, and current-run timings.
  Preserve silent artifacts; reject synthesis without speech.

`evidence.json` uses schema version 2: `title`, `media`, `segments`, `issues`,
and `models.asr`. Media includes path, SHA-256, duration, and extracted interval.
Before extraction, `extraction-config.json` records
`{"schema_version": 2, "pipeline": "audio"}` so interrupted jobs can resume.
Legacy jobs and nonempty unrecognized directories require a new output directory;
reject them before any rebuild, review, or rendering writes.

## Notes and rendering

- Use consecutive fifteen-minute chapters, bounded neighboring speech context,
  an editable outline, shared glossary, and chapter checkpoints.
- Send speech IDs, timestamps, and text. Keep diagnostics and existing review
  issues local. Lecture blocks cite valid chapter-scoped speech evidence.
- Split requests above 120,000 UTF-8 bytes and retry typed context-limit failures.
  Stop before parts become shorter than thirty seconds. Preserve all speech in
  manual handoffs when splitting cannot help.
- Automatic and imported replies use the same schema. Mark added explanations
  and uncertain details; do not invent unspoken formula, code, or diagram details.
- Retain confined TeX/code references, disabled shell escape, review-driven draft
  footers, manual-edit preservation, and evidence fingerprints.
- Compile with LuaLaTeX, retain logs, and render PDF previews. Diagrams use inline
  TikZ/PGFPlots; external images are unsupported.

## Verification

Cover media routing, intervals, overlap ownership, resume, checkpoint invalidation,
silence, text-only requests, manual round trips, chapter budgets, citations,
review persistence, and legacy-file preservation. Use generated audio and
multistream video with mocked Whisper, and real LuaLaTeX/PyMuPDF for PDF checks.
Assess real speech accuracy separately on a representative Russian recording.
