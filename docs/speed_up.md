# Pipeline runtime analysis and optimization priorities

Analysis date: 2026-11-12.

Goal: reduce lecture-processing time while preserving the current quality of lecture notes and illustrations. This document consolidates the read-only analysis of saved artifacts in `out/` and `out_algo/`. No pipeline changes, model calls, or performance experiments were made during the analysis. All proposed improvements remain unimplemented and unbenchmarked.

## Main findings

- The dominant extraction cost is inference over many selected frames, not video decoding or model loading.
- The dominant illustration-verification work is checking whether source observations can share a selected figure, not checking rendered diagrams.
- Coverage verification makes meaningful corrections, but it repeatedly checks similar source observations against the same representative.
- Revised priority: reduce repeated comparisons, improve candidate selection, and batch remaining checks before parallelizing them.
- Existing artifacts do not establish the exact number of safely removable calls or a guaranteed end-to-end speedup.

## Scope and measurement limitations

The audit covered 15 jobs containing `notes/main.pdf`: eight in `out/` and seven in `out_algo/`. The unfinished 21 October algorithms job was excluded. A rendered PDF establishes the presence of notes, not the absence of unresolved review issues.

Sources inspected:

- `extraction-timings.json`: timings for the latest extraction invocation.
- `workers/vision.response.json` and `diagnostics/vision/*.json`: frame-level inference attempts, outputs, errors, throughput, and cache information.
- `extraction-diagnostics.json` and `checkpoints/video/selection.json`: frame-selection settings, reasons, and timing.
- `illustration-verification/**/call.json`, `response.json`, and `verdict.json`: canonical comparison and repair records.
- `handoff/verification/attempt-*`: candidates, accepted coverage, alternatives, refined comparisons, recovery, and copied evidence.
- `notes.json`: final figure blocks and observation-to-figure mappings.

Important distinctions:

1. **Ten extraction invocations were fresh; five were resumed.** Fresh-run timing percentages exclude resumed invocations. Retained frame timings can include inference performed before the latest invocation.
2. **Verification has no explicit persisted elapsed-time measurements in the inspected artifacts.** Its durations below are filesystem-modification-time proxies. They can include pauses and are not reliable API latency or end-to-end wall-time measurements.
3. **Canonical caches, latest-attempt archives, and final notes represent different scopes.** Do not combine their denominators or add their timing estimates together.
4. **Archive records are not new API calls.** Cached requests and responses are copied into attempt directories. Canonical records also do not constitute an exhaustive billing history: HTTP retries and overwritten artifacts complicate that interpretation.
5. **Model verdicts are not independently verified ground truth.** A recorded failure or changed mapping is evidence of pipeline behavior, not proof that every change was necessary.

For example, the entire 15 September verification directory spans about 21 hours, while its latest attempt spans about 0.124 hours. Radix-sort request files postdate its PDF by about 16.5 hours. These examples rule out treating directory timestamp spans as uniform end-to-end runtimes.

## Extraction findings

Across the ten fresh runs:

- Vision accounted for approximately **91% of extraction time**.
- Frame inference accounted for approximately **97% of vision time**.
- Proxy scanning and native-frame retrieval together accounted for approximately **2.8% of vision time**.
- Model loading took approximately **5–9 seconds per invocation**.

This supports the visual-bottleneck diagnosis, but does not independently establish the user's approximate 80% visual share of the entire pipeline.

Across all 15 jobs, retained frame results contain:

| Metric | Observed value |
|---|---:|
| Selected frames | 21,657 |
| Typical selection rate | 8–13 frames per video minute |
| Sum of retained inference-attempt durations | 40.8 hours |
| Second inference attempts | 597 |
| Time in second attempts | 1.45 hours, or 3.6% of retained inference time |
| Frames with exactly repeated structured output within their job | 2,620 |
| Inference time associated with those repeated outputs | 3.54 hours |
| Exact duplicate full-frame hashes in inspected diagnostics | 0 |

Repeated model output suggests redundancy but does not prove visual equivalence. Two different source images can produce the same incomplete or erroneous extraction.

Frame-selection reasons:

| Reason | Frames | Approximate share |
|---|---:|---:|
| Before erasure | 9,212 | 42.5% |
| Before transition | 3,574 | 16.5% |
| Change-time guard | 6,397 | 29.5% |
| Settled change | 2,415 | 11.2% |
| Other | 59 | 0.3% |

The selector operates on whole-frame pixel differences. Its minimum change fraction is `0.0008`; the change-time guard is 15 seconds. Lecturer motion can retain extra frames. These counts identify where to investigate, not how many frames can safely be discarded.

Relevant code: [`video.py`](../shad_helper/video.py), particularly `SELECTION_SETTINGS` and `_select_proxy_frames`; [`vision_worker.py`](../shad_helper/vision_worker.py), particularly `run` and `_MLXRuntime.infer`.

### Representative job timings

| Lecture | Video duration | Retained frame inference | Verification interval proxy |
|---|---:|---:|---:|
| Regularization, 6 October | 2.99 h | 3.53 h | 2.03 h |
| Statistical properties, 13 October | 3.30 h | 3.96 h | 2.84 h + 0.48 h second attempt |
| Random forest, 3 November | 3.11 h | 3.28 h | 3.54 h + 1.00 h second attempt |
| Logistic regression, 29 September | 3.11 h | 2.88 h | 3.53 h + 1.43 h second attempt |
| Sorting and heaps, 14 October | 3.03 h | 3.12 h | 4.62 h |

Inference is the sum of retained frame-attempt timings, including retries. Verification intervals run from the earliest figure/coverage candidate artifact to the latest accepted artifact within an attempt. They exclude subsequent chapter validation and reconciliation and can include pauses. This table is not a complete end-to-end timing breakdown.

## Verification findings

### Work distribution in canonical caches

The following counts exclude copied handoff archives and include nested repair records:

| Operation | Retained call records | Sum of response-minus-call modification-time proxies |
|---|---:|---:|
| Coverage consolidation | 10,030; 9,975 with responses | 21.302 h |
| Final reconciliation | 625 | 1.293 h |
| Diagram comparison, including repair verification | 71 | 0.383 h |
| Repair generation | 60 | 0.764 h |
| **Total** | **10,786; 10,731 with responses** | **23.743 h** |

Coverage consolidation represents approximately **93% of retained call records and 90% of this duration proxy**. Its question is whether a source observation can be represented by a selected figure without losing meaningful content.

An initial root-only inventory contained 10,683 records: 10,655 sharing the consolidation/reconciliation instruction family and 28 direct reconstruction comparisons. The detailed classification above separates consolidation from reconciliation and includes nested repair calls; it supersedes treating all 10,655 as coverage calls.

### Coverage outcomes and usefulness

| Outcome | Retained coverage calls | Share |
|---|---:|---:|
| Pass | 7,178 | 71.6% |
| Fail | 2,769 | 27.6% |
| Uncertain | 28 | 0.3% |
| No saved response | 55 | 0.5% |

A pass is not automatically wasted work, and a failure is not automatically a distinct defect. Similar observations can repeatedly confirm or reject the same proposed match.

In the **latest saved verification attempt per job**:

- There were 7,309 coverage candidates and 7,274 accepted records; 35 candidates lacked an accepted record.
- Processing changed the selected figure for **1,027 accepted observations, or 14.1%**. These are mappings, not distinct figures or independently proven corrections.
- Initial verdicts were 6,190 pass, 455 fail, and eight uncertain; 656 candidates had no initial verdict. The code can skip the initial comparison when the representative is unpublished, so absence does not imply an API failure.
- There were **373 fallback figure blocks**, of which **30 remained in final notes**. Subsequent merging or removal does not by itself prove their creation was unnecessary.
- Candidate-to-final coverage differed for 2,455 observations, including 1,420 whose initial verdict passed. Later processing contributes to these changes; they cannot all be attributed to the initial coverage check.

Saved verdicts demonstrate substantive problems caught by verification:

- A random-forest error-versus-number-of-trees plot was matched to an error-versus-tree-depth plot.
- A regularization representative contained only one panel of a two-panel comparison; recovery found a complete representative.
- A random-forest comparison detected different decision surfaces and a missing feature label; its fallback block remained in the final notes.

### Repeated comparisons

- There were **353 distinct representative-image hashes counted separately within jobs**.
- **9,369 of 10,030 coverage calls** belong to representative groups containing at least ten comparisons.
- On 30 September, one representative received **385 comparisons: 379 pass and six fail**. Final coverage maps 391 observations to that figure.
- On 14 October, one representative received **524 comparisons: 287 pass and 237 fail**.

These results support grouping equivalent **source states**, not simply approving each target once. Different source observations can contain information absent from the same target.

### Alternatives, refinement, and recovery

Latest-attempt verdict records:

| Stage | Verdict records | Pass | Fail | Uncertain | Pass share |
|---|---:|---:|---:|---:|---:|
| Alternative matches | 2,107 | 571 | 1,520 | 16 | 27.1% |
| Matches after crop refinement | 642 | 140 | 502 | 0 | 21.8% |
| Recovery | 855 | 87 | 759 | 9 | 10.2% |

These are decision records, not counts of fresh API requests. One alternative pass was local because the images were identical. Across the inspected stages there were 10,256 archived `call.json` records but 9,600 unique request contents within jobs.

In recovery, 656 of 855 records copied requests from previous stages. Among the remaining 199 requests, 87 passed, 110 failed, and two were uncertain: **43.7% passed**. The raw 10.2% recovery pass share therefore substantially understates the usefulness of additional recovery comparisons.

### Retry, cache, and archive behavior

- Four jobs retain two finalization attempts. Some second-attempt artifact intervals span 0.5–1.4 hours despite substantial cache reuse.
- Saved second-pass archives show reuse of identical request/response contents and response modification times, including 1,329/1,374 records for random forest and 1,168/1,168 for statistical properties. These are archive records, not unique API calls.
- Canonical caches contain 496 retained attempt-error files. This is not a historical retry rate because filenames can be overwritten and caches survive reruns.
- Exact duplicate full request descriptions were not found within canonical job caches. Only seven extra root calls shared ordered image hashes and instructions while differing in other context. Basic exact-request deduplication has little demonstrated upside.
- Canonical verification caches occupy approximately 8.04 GB logically; verification handoff archives occupy 31.47 GB; reconciliation archives occupy 0.483 GB. Physical allocation and copy time were not measured.
- Repairs are useful: **36 of 43 completed repair records passed**.

## Revised optimization priorities

### 1. Group equivalent source states before coverage verification

Track each instructional figure across nearby observations. Group observations only when the instructional content is equivalent, choose a clear source representative, and verify that source state against the selected notes figure once. Preserve original observation IDs and their mapping to the group.

Split groups when labels, arrows, panels, geometry, or meaningful intermediate states change. Account for scrolling, occlusion, and source-crop boundaries. A common target, similar text, a shared topic, or matching model output alone is insufficient evidence for grouping.

This addresses both repeated passes and repeated rejected matches. The logs show substantial repetition, but do not prove a safe reduction percentage. Preserve a per-observation fallback for ambiguous grouping.

### 2. Improve alternative matching before remote verification

`_matching_figures` currently ranks candidates using source identity, topic equality, word overlap, and temporal distance, returning up to three candidates. Topic equality can admit different instructional figures into expensive comparison.

Use stronger visual matching and figure-state tracking to rank plausible alternatives. Clear incompatibilities in panels or axis labels can help reject poor candidates; uncertain extraction must not silently eliminate a valid candidate. Retain unmatched source content and the existing recovery path.

The 27.1% alternative pass share identifies a promising target, but is not itself a measured avoidable-call rate.

### 3. Batch remaining checks against a common representative

Send the target once with several labeled source images and require an independent verdict for each source. This reduces request overhead while retaining source inspection.

Purely as an arithmetic packaging exercise, grouping the retained comparisons by representative into batches of at most eight would yield approximately **1,417 requests instead of 10,030**. This is not a deployment estimate: conditional alternatives, recovery dependencies, image budgets, and accuracy may require smaller batches. Request-count reduction does not imply proportional reduction in image processing, cost, or latency.

Benchmark small batches before adoption, checking that difficult sources do not receive an unjustified group-wide pass.

### 4. Parallelize independent comparisons that remain

The current coverage loop waits for each remote comparison. Bounded concurrency can preserve the same prompts, model, evidence, and verdict requirements while overlapping independent requests.

Apply results deterministically and preserve dependencies in alternative selection and recovery. Do not simply thread the existing shared client: `_compare` mutates `client.request_recorder`.

Four concurrent requests have an ideal ceiling of 4× acceleration for the independent request portion, not the whole stage. Rate limits and dependencies determine actual gains; none were measured.

### 5. Reduce redundant frame inference upstream

Make selection sensitive to the board or screen content rather than whole-frame motion. Track instructional regions, suppress noninstructional movement, and retain clear representatives of unchanged states. Preserve content before erasure and meaningful intermediate drawings.

As a scenario, removing 30% of inference work from a lecture spending 3.3 hours in vision would save about one hour, before downstream effects. The achievable reduction has not been measured. Simply increasing the sampling interval risks losing content.

### 6. Retry failed operations rather than replaying full finalization

The outer finalization retry starts from the original candidate. Preserve successful decisions, retry failed operations, and recompute only affected dependencies.

Separate rendered-preview caching from verdict caching: a transport failure should not require deleting and rerendering an otherwise valid preview. Retain independent semantic verification.

Potential savings are most relevant to problematic runs, but saved artifacts do not isolate avoidable second-pass time.

### 7. Reduce formatting-induced vision retries

Of the first-attempt failures, 278 included extra illustration fields such as `labels` and `relationships`, rejected by the strict validator. Their second attempts cost approximately 31 minutes collectively.

Improve schema adherence or use narrowly specified normalization that preserves raw information and validates every required field. Do not broadly relax coordinate or semantic validation.

All second attempts account for only 3.6% of retained inference time, so this cannot independently solve the multi-hour slowdown.

### 8. Reduce repeated evidence copying

Whole verification caches are copied into attempt archives even on cache hits. Immutable shared evidence plus per-attempt references, or verified filesystem clones, could preserve auditability while reducing copying.

Disk and copying savings are plausible, but runtime savings remain unmeasured. Prioritize remote comparisons and frame inference first.

## Changes not supported as first priorities

- Optimizing FFmpeg or model loading: both are small relative to inference.
- Adding basic exact-match caching: existing reuse is effective and exact duplicates are uncommon.
- Removing repair or recovery: saved outcomes demonstrate useful recoveries.
- Switching models, lowering resolution, or lowering reasoning effort without evaluation: the inspected logs do not establish preservation of current quality.
- Treating every pass as redundant, every failure as useful, or every copied request archive as a fresh API call.

## Validation required before adopting an optimization

Use saved jobs as fixed baselines, including slides, physical boards, scrolling notebooks, and meaningful intermediate diagrams. Include high-pass groups such as 30 September and mixed pass/fail groups such as 14 October.

Measure:

- Actual wall time per stage and API operation, with run/attempt identifiers.
- Fresh requests, cache hits, retries, and copied archive records separately.
- Source observations, distinct source states, inspected images, and final figures separately.
- Lost labels, formulas, panels, arrows, intermediate states, and illustration coverage.
- Preservation of meaningful corrections and final fallback figures found in the baseline.

For state grouping and candidate filtering, inspect discarded distinctions against original images; final figure count alone is insufficient. For batching, check each source verdict independently. For concurrency, preserve deterministic application of decisions and reliable request archives.

Acceptance means fewer calls or lower elapsed time while preserving the quality and coverage of the saved notes. No such optimization has yet been validated by this audit.

## Code and artifact references

- [`video.py`](../shad_helper/video.py): selection settings, proxy-frame selection, native retrieval.
- [`vision_worker.py`](../shad_helper/vision_worker.py): sequential inference, schema parsing, retries, frame checkpoints.
- [`illustrations.py`](../shad_helper/illustrations.py): illustration-observation validation and source assets.
- [`illustration_verification.py`](../shad_helper/illustration_verification.py): `_consolidation`, `_matching_figures`, `_finalize_coverage`, `_recover_coverage`, `_compare`, and reconciliation.
- [`notes.py`](../shad_helper/notes.py): finalization retries and chapter generation.
- Regularization evidence: `out/06-10-2026-Лекция---Никита-Волков---Польза-регуляризации-39c721e69f77/`.
- Random-forest evidence: `out/03-11-2026-Семинар---Никита-Волков---Случайный лес-65e02303bfa8/`.
- Repeated confirmation example: `out_algo/30-09-2026-Семинар---Максим-Бабенко---Сортировки-и-порядковые-ст-7d1fe10b33a4/`.
- Mixed pass/fail example: `out_algo/14-10-2026-Лекция---Максим-Бабенко---Сортировки-и-кучи-3670fed2c021/`.
