# Chapter-by-chapter vision and notes

Implemented for full video runs. Runtime priority 4, 19 November 2026.

For a video lecture, `shad-helper run` starts drafting chapter 1 as soon as its visual evidence is ready. A single local Qwen worker continues through the remaining chapters while a single notes worker runs GPT, verifies illustrations, makes repairs, and checks chapter LaTeX. Both workers process chapters in lecture order. The last successful notes job publishes the combined project and releases it to the PDF renderer. [Optimization priority 4](speed_up.md#4-overlap-chapter-vision-and-notes)

## Execution order

Extraction saves the transcript before Qwen starts. The coordinator plans chapter inputs from the selected frames and sends ready evidence to the notes thread as inference progresses. Standalone extraction publishes the same final evidence and manifest. [Pipeline](../shad_helper/pipeline.py) · [Coordinator](../shad_helper/chapter_pipeline.py) · [Streaming extraction](../shad_helper/extraction.py)

After transcription and frame selection, the coordinator prepares the chapter plan and launches one long-lived vision subprocess. Qwen streams frame or batch completion events with IDs and paths. The parent drains those events while a single notes thread owns the OpenAI client, its request recorder and all illustration calls. A ready chapter goes onto a first-in, first-out queue of disk references. Qwen can finish the entire lecture while GPT is busy with chapter 1.

| When | Vision worker | Notes worker |
| --- | --- | --- |
| Extraction inputs are ready | Processes the frames needed by chapter 1 | Waits for chapter 1 |
| Chapter 1 evidence is ready | Starts chapter 2 | Starts chapter 1 |
| Chapter 2 evidence is ready while GPT is busy | Starts chapter 3 | Continues chapter 1, with chapter 2 queued |
| Chapter 2 exceeds the model context | Continues its own chapter order | Runs 2a, then 2b, before chapter 3 |

The schedule applies to `shad-helper run` in both `frames` and `video` mode. Audio-only inputs and `--vision-model off` use the existing audio path. Files in a batch run in the supplied order. Standalone `extract`, `notes`, manual import/export, and `render` keep their public entry points. [Full-pipeline CLI](../README.md#run-the-complete-pipeline)

## Chapter plan and evidence readiness

1. Use the requested clip's absolute start and end. Load a valid saved outline or create consecutive 15-minute chapters with the existing outline rules. Before Qwen inference, assign transcript segment IDs and selected frame IDs to each chapter. Keep the base chapter numbers for scheduling and progress. [Outline selection](../shad_helper/notes.py#L729)
2. Use the same context as `chapter_context`. Include speech segments intersecting the expanded interval, all selected frames within it, and the last selected frame before it. The extension on each side is the lesser of 60 seconds and one quarter of the chapter duration. A chapter owns illustrations in its core interval `[start, end)`. The last chapter also owns observations at the clip end. Overlapping chapters share the same frame ID, pixels and inference result. [Chapter context and ownership](../shad_helper/notes.py#L236)
3. Run frame selection and novelty filtering once over the clip. In video mode, build the existing batches over the whole selection before assigning work. Process each frame or batch on its *first* use by a chapter. If a batch straddles a boundary, finish it once and distribute its results by frame ID. Keep the Qwen model loaded across jobs and preserve the existing frame and batch cache keys. [Selection](../shad_helper/video.py#L418) · [Video grouping](../shad_helper/video_vision.py#L32) · [Vision checkpoints](../shad_helper/vision_worker.py#L210)
4. Release a chapter once every required frame has a result and its original crops and manifest entries are readable. A recorded `incomplete` result carries the existing reviewable fallback for this invocation. A chapter with no new frames can be ready from the transcript and shared evidence. For a first chapter spanning **00:00–15:00**, GPT can start when the required selected frames through **16:00**, inclusive, are ready. In video mode this includes finishing the batch that contains the last required frame.

Give crop repair the full index of selected native frames from the start. Nearby-frame selection can use pixels within 30 seconds, and scroll stitching can use pixels up to 90 seconds away. Both can locate those images by ID and timestamp before Qwen describes them. Keep this index separate from the timestamped observations admitted to GPT's chapter prompt. The same chapter should receive the same inputs whether Qwen is one chapter ahead or ten. [Nearby frames](../shad_helper/illustration_crops.py#L235) · [Scroll stitching](../shad_helper/illustration_stitching.py#L137)

## Work owned by a notes job

The worker builds the chapter prompt from its evidence snapshot, the running glossary, previous topic names and earlier figure IDs. It saves the exact request and candidate reply. It then refines crops, verifies reconstructed diagrams against their sources, attempts bounded repairs, and checks screenshot fallbacks. Each owned illustration observation gets a verified figure, an explicit exclusion, or an unresolved review record. Keep source-state grouping, batched coverage checks, alternatives and recovery in the same job. Flush the chapter's last batch when it is ready. [Chapter generation](../shad_helper/notes.py#L757) · [Illustration verification and coverage](../shad_helper/illustration_verification.py#L767)

The worker reconciles figures with completed chapters, validates the affected references, runs the chapter LaTeX check and checkpoints the result before taking the next notes job. Keep the original drafts and source IDs available for later recovery. A change to a figure link can then be applied without another chapter-writing request.

If the text or image size check fails, or GPT returns `context_length_exceeded`, split the chapter at its midpoint. Run the two children consecutively using their existing deterministic IDs and the parent's processed frames. Each child has its own context and illustration ownership. The second receives the first child's glossary, topics and figures. Persist the split tree and write its leaf IDs to `outline.json` and the handoff for retry or import. An oversized child can split again, down to the existing minimum duration for manual handoff. [Budget and split rules](../shad_helper/notes.py#L648) · [Runtime split](../shad_helper/notes.py#L800)

## Cross-chapter figures and request cost

A later notes job may update earlier chapters. A chapter boundary costs zero LLM requests by itself. Carry forward accepted figures, the matcher's verified examples, unresolved observations and prior verdicts. For each new figure, shortlist up to three plausible earlier figures. Resolve identical crops of the same intended figure locally. For different crops, reuse a matching verdict or make one GPT comparison before retries. Merge verified equivalent views or keep the figure verified to contain the other. Distinct or uncertain pairs keep both figures. Retry an unresolved observation when a new plausible candidate or a changed crop supplies new evidence. [Candidate selection](../shad_helper/illustration_verification.py#L487) · [Recovery](../shad_helper/illustration_verification.py#L915) · [Pair comparison](../shad_helper/illustration_verification.py#L958)

**Verify the original figure against its proposed final replacement.** Suppose A already points to B and a later chapter proposes C for B. Reuse a direct verdict for A and C on the exact inputs, or compare A's original crop and intended caption with C's publication crop or verified diagram preview. Redirect A only when C preserves its labels, conditions and meaningful intermediate stages. On a distinct, uncertain or unavailable verdict, record B as A's final target and keep B published. Apply this rule to each affected original. Cache the decision by the actual figure inputs, captions, comparison policy and model settings.

Commit the winning figure, coverage mappings, prose `figure_refs`, and associated review issues together. Keep the winning block in its original chapter and retain the source ID behind each mapping. The sorting seminar supplies a regression case. A board screenshot contained both a quicksort recursion tree and a Lomuto partition. Two successive merges left two chapter 2 references pointing at the Lomuto-only diagram. The direct check must retain a figure that actually shows the recursion tree. The archived case is in `out_algo/30-09-*/handoff/figure-reconciliation/replacements.json`, under `ch002_fig_quicksort`. [Current replacement logic](../shad_helper/illustration_verification.py#L1014)

| Read-only archive check on 19 November 2026 | Result |
| --- | --- |
| Recreate boundaries from clip metadata | All 216 intervals matched the saved notes in 19 video jobs |
| Cross-chapter figure comparisons in the 18 jobs with reconciliation records | 352 distinct request descriptions, median 19 per lecture, range 0–49 |
| Final cross-chapter replacements lacking a direct original-to-final comparison | 16 of 42 |
| Additional distinct cross-chapter pairs seen when checking replacements at every chapter boundary | 20 across the 18 jobs, range 0–7 per lecture, about 5.7% of the 352 comparisons |
| Recovery checks against figures from later chapters | 177 checks, including 104 with request contents already present in coverage and 73 others |

These counts come from `out/` and `out_algo/`, using the saved pair verdicts and replacement maps, and the latest coverage and recovery attempt per job. The comparisons span two model versions. The 20-pair estimate includes interim replacements and holds the archived decisions fixed. Rejections, changed candidates, chapter-sized coverage batches, cache hits and retries can change live totals. Count fresh requests separately for drafting, coverage, repairs, recovery, ordinary reconciliation and direct checks on replacements. An accuracy rate requires independent review of the figures.

## Publication, failure and resume

The job's `checkpoints/chapter-pipeline/` directory contains `plan.json`, incremental evidence records under `ready/`, verified leaf checkpoints under `notes/`, and the last committed leaf in `notes-state.json`. The coordinator publishes a readiness record atomically after staging its results, crops and manifest entries. Pass notes and verification a read-only view that resolves assets from that snapshot. The extractor stages original crops per completed inference batch and publishes the combined manifest when vision finishes. [Crop materialization](../shad_helper/illustrations.py#L57) · [Atomic JSON writes](../shad_helper/jobs.py#L164)

When Qwen finishes, publish the conventional `illustrations/manifest.json` and `evidence.json`, then mark extraction complete. Standalone `notes` keeps its existing completion check. The last notes job assembles the lecture locally, validates all chapters against the full evidence, applies review resolutions using its final fingerprint, and writes `notes.json`, `review.json` and the TeX project. Start the PDF render as soon as this run has both completed extraction and a validated project covering every notes child. Validated notes with open review issues can still render as a draft. Anki consumes the resulting PDF. [Extraction completion](../shad_helper/extraction.py#L31) · [Project publication](../shad_helper/notes.py#L976) · [PDF rendering](../shad_helper/rendering.py#L789)

| Event | Required behavior |
| --- | --- |
| Notes for a chapter or split child exhaust retries and require manual input | Report the failed chapter immediately and stop later GPT jobs for this lecture. **Qwen finishes every remaining chapter** and saves its work. With full evidence available, finish the ordinary manual handoff with the remaining inputs. Report the lecture as incomplete, skip PDF and Anki, then continue to the next file. |
| A vision process fails before supplying required frames | Wake the waiting notes worker, mark extraction failed and retain successful checkpoints for resume. The current lecture cannot enter rendering. A reviewable per-frame fallback follows the normal evidence path. |
| The user interrupts or the process exits mid-write | Stop and join workers on interruption. On restart, read the last committed records, retry incomplete work, and ignore an older PDF when determining current-run success. |
| The same lecture is resumed | Restore the base plan and its split tree. Reuse compatible frame, batch, chapter and verdict checkpoints. Resume notes at the first failed or invalid child in order. |
| A job predates the chapter scheduler | Build its plan from the saved outline or clip bounds. Seed ready work from matching extraction and notes checkpoints. Check existing figure replacements against the direct-verdict rule before reuse. |

Bind readiness to the media, clip, plan, selection, vision settings and exact frame results. A draft checkpoint depends on its prompt, schema, model settings and actual attachments. Its verified chapter state also depends on the crop and merge policies, source hashes and pair verdicts it used. This lets a corrected mapping reuse the draft and a corrected transcript reuse compatible vision work. Retry incomplete frames on a later invocation. Apply `--rebuild` cleanup once per stage, discard the derived notes plan with the saved outline, and give each shared handoff or diagnostics file a single writer. [Stage contracts](stage-contracts.md) · [Notes rebuild](../shad_helper/notes.py#L706)

## CLI progress

During overlap, the parent process owns two fixed `tqdm` rows on interactive stderr. Each row shows the worker, base chapter number and total chapters, and its current activity. A split adds the child label to the notes row. While notes are two chapters behind, the terminal might show

```text
Vision  ch004  4/12          [██████░░░░]  60/100 frames
Notes   ch002  2/12  part b  [████░░░░░░]   7/18 observations · coverage
```

Count newly assigned vision frames once, including cache hits. A completed video batch advances by its frame count. For notes, show the phase and its completed units when known. A pending GPT response shows the drafting phase and elapsed time. Show waiting, retry, manual handoff and completion on the corresponding row. Leave the notes failure visible while the vision row advances.

Use a dedicated event channel for progress. Route stdout, stderr, retry messages and subprocess logs through the parent during overlap. Print each log line once above the fixed rows and redraw them under the same terminal lock. Disable nested worker and verification bars in this mode. In redirected output, emit plain chapter and phase transitions plus warnings and errors. Attribute both workers' timings to the coordinator's combined job report and propagate the command's usage-accounting context to the notes thread. [Existing progress loops](../shad_helper/vision_worker.py#L278) · [Run diagnostics](../shad_helper/run_diagnostics.py) · [Request accounting](../shad_helper/openai_usage.py)

## Implementation and acceptance

1. **Expose chapter evidence.** Extend extraction and the vision worker to publish results progressively. Compare speech IDs, frame IDs, image hashes and coverage ownership with `chapter_context` on completed evidence. Exercise a boundary-spanning segment and video batch, a carried slide, a short chapter, a frame at clip end, zero new frames and a cache hit. Verify that overlapping chapters request each inference unit once.
2. **Schedule the two workers.** Use controlled worker events to hold GPT on chapter 1 while Qwen finishes chapters 2 and 3, and to hold Qwen while notes wait. Assert one active job of each kind, chronological starts, consecutive split children, and correct resume after a child succeeds. Force a manual handoff and a fatal vision error separately. Check publication, the PDF gate and multi-file ordering. [Pipeline tests](../tests/test_pipeline.py) · [Budget tests](../tests/test_notes_budget.py)
3. **Verify notes and cross-chapter outcomes.** On frozen evidence and prior replies, compare the actual prompts and attachments with the serial pipeline. Cover later recovery, a valid cumulative figure, a distinct teaching stage, the quicksort chain, a cached verdict and an unavailable verifier. After restart, check source coverage, figure links and review issues. On live lectures, review sampled explanations and formulas against their sources, and compare changed assets and captions with the reviewed baseline. [Reconciliation tests](../tests/test_illustration_verification.py) · [Illustration quality benchmark](illustration-benchmark.md)
4. **Verify terminal behavior and speed.** In a pseudo-terminal, interleave child stderr, a cache hit and a retry while both bars advance. Assert two live rows, one visible copy of each log line and clean exit on failure or interruption. Repeat with redirected output. Benchmark both schedules on the same recordings, models and settings, separating cold and warm caches. Acceptance requires lower elapsed time through PDF with verified retention of instructional figures and stages. Report actual worker overlap and attribute changes in fresh requests to their operations. End-to-end speedup remains to be measured.

The implemented scheduler was checked against 19 saved video lectures. All 216 chapter contexts matched the complete-evidence contexts, including the ownership of 10,544 observations across 27,593 selected frames. Every managed original crop matched its manifest and evidence hash. In a replay of the sorting seminar using its two archived pair verdicts, withholding the direct quicksort-to-Lomuto verdict kept the mixed board published and both recursion-tree references pointed to it. These checks used archived model output. [Scheduler tests](../tests/test_chapter_pipeline.py) · [Figure regressions](../tests/test_illustration_verification.py)

A controlled three-chapter run measured elapsed time through a real PDF render. With six frame responses delayed by 0.7 seconds each and three draft responses delayed by 1.2 seconds each, the sequential path took 19.15 seconds and the overlapping path took 16.82 seconds. With warm checkpoints the same paths took 0.28 and 0.16 seconds. Both schedules produced identical extracted PDF text. The cold runs each made three draft calls and the warm runs made zero. These are single-run fixture timings with simulated inference. Live full-lecture timing and independent quality review remain open.
