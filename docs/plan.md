# Russian lecture notes pipeline for your M5 Pro

## Summary

Build an accuracy-first Python CLI that processes a three-hour lecture into **editable LaTeX sources and a PDF**.

Transcription, frame selection, visual interpretation, and OCR run locally. Note synthesis uses `gpt-6.1-sol` through the official ChatGPT-plan integration when available, with a complete manual ChatGPT fallback.

Notes contain detailed explanations, reconstructed equations and diagrams, code, timestamps, and clearly marked uncertainties. **Anki generation remains unimplemented, with a separate stub.**

## Models and runtime

| Task | Initial choice | Reason |
|---|---|---|
| Speech recognition | **Whisper large-v3, FP16 through mlx-whisper** | Quality-oriented multilingual transcription with segment and word timestamps. [MLX Whisper (https://github.com/ml-explore/mlx-examples/tree/main/whisper)](<https://github.com/ml-explore/mlx-examples/tree/main/whisper>) |
| Visual interpretation | **Qwen3.8-27B, MLX 4-bit** | Interpret slides, notebook displays, diagrams, and relationships between visual elements. [Official model (https://huggingface.co/Qwen/Qwen3.8-27B)](<https://huggingface.co/Qwen/Qwen3.8-27B>) |
| Text and mathematical OCR | **PaddleOCR-VL-1.6, BF16 through mlx-vlm** | Recognize text, formulas, tables, and charts; avoid aggressive quantization for the small OCR model. [Model documentation (https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)](<https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6>) |
| Note synthesis | **gpt-6.1-sol** | Combine extracted evidence into coherent notes and generate LaTeX/TikZ source. [Official inference documentation (https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference)](<https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference>) |

The Qwen checkpoint contains approximately 16 GB of weights. Running models sequentially with bounded image/context batches makes it a plausible fit for 48 GB; actual peak memory and runtime require measurement. [Checkpoint files (https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/main)](<https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/main>)

Use arm64 CPython 3.14 managed by `uv`, preserving the project’s current Python requirement. Lock dependencies after installation and inference checks. Required external tools are FFmpeg, LibreOffice for PowerPoint rendering, and TeX Live with LuaLaTeX and `latexmk`.

## Extraction pipeline

1. **Import video and optional source materials.** Preserve the original video timeline. Read PDF text and page layouts with PyMuPDF; extract PowerPoint text, tables, notes, and figures, then render slides through LibreOffice. Read notebook Markdown, LaTeX, code, and recorded outputs with `nbformat`, without executing cells.
2. **Transcribe Russian speech.** Extract 16 kHz mono audio. Process approximately ten-minute chunks with short overlaps, retaining absolute timestamps and deduplicating overlap text. Export JSON, SRT, and VTT. Use source materials to supply a short technical glossary; flag repetition, silence hallucinations, and suspicious decoding results.
3. **Select informative frames.** Scan a low-resolution stream at the video’s temporal cadence using inexpensive change detection. Retrieve sharp original-resolution frames around persistent changes, incremental slide reveals, notebook scrolling, and board erasures. Preserve intermediate derivation states rather than only completed boards.
4. **Interpret and recognize visual content.** Qwen identifies content regions and diagram entities, labels, and relationships. Paddle recognizes prose, code, mathematical expressions, and tables from readable crops. Keep both observations when they disagree; use matched native source content where available.
5. **Assemble timestamped evidence.** Store transcript segments, visual observations, formulas, source page/slide/cell references, and review issues with stable IDs. Match supplied materials to what appears in the lecture; unmatched material must not silently become lecture content.

Each expensive stage writes resumable checkpoints. Model workers exit between stages to release unified memory. After initial downloads, extraction works offline.

## Previous reliability update

- Stop long consecutive text repetitions during visual generation without rejecting numeric tables or whitespace.
- Parse one complete JSON object; reject duplicate keys, nonfinite numbers, invalid field types, and malformed region coordinates.
- Preserve failed raw replies, create an explicit review issue, and send the full image to OCR.
- Regenerate invalid visual checkpoints on resume while reusing valid results.
- Route every application-owned OpenAI/ChatGPT request through the configured proxy; the user configures the browser proxy separately.

## Note generation, rendering, and CLI

- Generate a lecture outline first, then synthesize bounded chapters with neighboring context and a shared notation glossary. Preserve definitions, assumptions, derivations, examples, and code.
- Label added explanations **“Пояснение”**. Label tentative reconstructions **“Требует проверки”**, retaining a best-effort draft and a separate issue list.
- Produce equations as LaTeX, technical diagrams as TikZ, and suitable plots as PGFPlots. Preserve known labels and relationships; flag unknown values rather than inventing numerical data. [PGFPlots (https://ctan.org/pkg/pgfplots)](<https://ctan.org/pkg/pgfplots>)
- Keep video screenshots in the evidence folder only. Finished notes contain clean rendered visuals and imported illustrations.
- Export prompts for supplementary images generated manually in ChatGPT. Import returned images using stable figure IDs and label them as explanatory additions.
- Compile with LuaLaTeX and `latexmk`, using Cyrillic-capable fonts, verbatim code blocks, source references, and visible review markers. Preserve editable chapter and figure sources.

The subscription integration uses this app’s own OAuth sign-in, checks account-specific availability of `gpt-6.1-sol`, and sends inline evidence with `store=false` and `stream=true`. Eligibility and limits are account-dependent. **Image generation is excluded from this API route**, so illustration creation stays manual. [Plan integration (https://developers.openai.com/siwc/token-sharing-open-source)](<https://developers.openai.com/siwc/token-sharing-open-source>), [preview limitations (https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)](<https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations>)

If automation is unavailable, export chapter-sized evidence PDFs, transcripts, prompts, and the expected response schema. Import manual ChatGPT responses through the same validation and rendering path.

Provide these CLI interfaces:

- `extract VIDEO --source FILE --out JOB`, with repeatable source arguments.
- `notes JOB`, supporting automatic synthesis or an imported manual response.
- `render JOB`, compiling editable LaTeX and incorporating imported illustrations.
- Supporting environment-check and login commands.
- `anki JOB`, returning an explicit “not implemented” result and reserving reviewed notes as its future input.

## Verification and acceptance

- Check installation and one small inference for every local model before processing a full lecture.
- Benchmark representative Russian material containing slides, notebook code, and handwritten mathematics. Record runtime, peak memory, transcription errors, and formula recognition errors; document quality does not establish whiteboard quality.
- Test timestamp rebasing, overlap deduplication, incremental frame retention, source matching, and interruption/resume behavior.
- Test unavailable Pro access, interrupted responses, and manual handoff/import.
- Inspect the rendered PDF for Cyrillic, mathematical notation, diagram labels, clipped content, code layout, and visible uncertainty markers.
- Validate one complete three-hour lecture: all selected evidence remains traceable, major lecture sections are represented, and corrections rebuild affected outputs without repeating completed extraction.

Initial model choices are evidence-supported starting points, not a guarantee of handwriting accuracy or processing speed. No model training, automatic illustration API, or Anki implementation is included in this version.

# Process repeated lecture content once

## Summary

Add content-aware grouping before expensive vision processing. Keep the current Qwen model, image resolution, and native-cadence scan. Analyze one clear, real frame per confirmed group and reuse its OCR results.

Enable this by default for **new jobs**. Leave existing jobs, including the current lecture, on their existing behavior. Savings will depend on how often the lecture content stays unchanged and how clearly the board is visible.

## Frame grouping

- Preserve every original candidate image, ID, and timestamp. Group consecutive candidates; compare each with a fixed reference frame to prevent gradual writing changes from disappearing.
- Extend the first required Qwen analysis with an optional bounding box for the **whole board or application surface, including blank areas**. Refresh it after layout or camera changes. Invalid or uncertain detection falls back to full-frame comparison.
- Use Apple Vision person segmentation through PyObjC, with accurate mode and revision 1. Apply masks only to confirmed lecturer foreground on physical whiteboards. People depicted in slides or notebooks remain lecture content. [Apple segmentation guidance (https://developer.apple.com/videos/play/wwdc2021/10040/)](<https://developer.apple.com/videos/play/wwdc2021/10040/>)
- Check local color and stroke changes at full resolution. Global similarity can shortlist candidates but cannot authorize reuse. New symbols, subscripts, diagram changes, highlighting, code outputs, and erasures create separate states.
- Track covered areas as **unknown**, rather than unchanged. Clear that uncertainty only when a later visible observation matches the reference. New writing, uncertain mask boundaries, a scene cut, the clip ending, or a 30-second timeout causes affected candidates to receive full analysis.
- Choose the clearest real frame within each verified group. Preserve brief content and pre-erasure views. Ignore ordinary body movement, pointing, and cursor movement when content remains unchanged.

## Integration and compatibility

- Add `extract --vision-mode content|baseline` and an equivalent Python argument. New jobs default to `content`; existing jobs without recorded settings resume with `baseline`. Changing an existing job’s mode requires a new output directory.
- Record the mode and grouping version in job configuration and checkpoint fingerprints.
- Keep original-image provenance separate from the canonical analysis image. Reused bounding boxes and crops always refer to that canonical image; each occurrence retains its own evidence IDs and timestamps.
- Cache OCR by image pixels, model, recognition prompt, and generation settings, independent of occurrence IDs. Failed analyses, malformed output, and uncertain visual results do not authorize group reuse.
- Send each canonical image and crop once per chapter to both automatic note generation and manual handoff. Preserve occurrence timestamps so later writing cannot appear as earlier evidence.
- Add the macOS Vision/Quartz bindings to the local dependency extra. Missing bindings produce installation guidance; unreliable segmentation retains candidates.

## Validation and acceptance

- Test tiny minus signs, decimal points, subscripts, faint colored writing, arrowheads, code operators, erasures, brief displays, occlusion, and camera changes.
- Keep `00079`/`00080` separate because their writing changes. Evaluate `00098`/`00099` for reuse; uncertain hand coverage must retain them.
- Test gradual changes against the fixed reference, unresolved occlusion, people inside illustrations, cache invalidation, resumed jobs, and chapter-boundary provenance.
- Produce an audit report containing grouped originals, representatives, comparison decisions, fallback reasons, Qwen/OCR call counts, and processing times.
- Validate against annotated whiteboard, slide, and notebook samples. **Any lost meaningful content fails acceptance.** Conservative extra analyses are acceptable; report measured savings without promising a fixed speedup.

Changed-crop reasoning, reconstructed board images, and smaller vision models remain deferred.
