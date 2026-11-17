# Pipeline runtime analysis and optimization priorities

Analysis date: 2026-11-12.

Goal: reduce lecture-processing time while preserving the current quality of lecture notes and illustrations. This document consolidates the read-only analysis of saved artifacts in `out/` and `out_algo/`. No pipeline changes, model calls, or performance experiments were made during that analysis. The implementation and local validation of priorities 1 and 2 on 2026-11-13 and priority 3 on 2026-11-16 are recorded below; the audit figures remain historical.

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

Implemented and revised on 2026-11-13 in [`illustration_states.py`](../shad_helper/illustration_states.py) and [`illustration_verification.py`](../shad_helper/illustration_verification.py):

- Group nearby, complete observations using image evidence. Descriptions, topics, selected targets, and model output do not establish equivalence. The first implementation required nearly identical crop dimensions and pixels and removed only 31 of 3,771 observations; changing crop boundaries, capture noise, and lighting defeated that policy.
- For original managed crops, locate their saved native frames and require the frame pixels to reproduce each crop exactly. Track overlapping figure boxes and give the track a fixed native region containing the union of its original boxes. This recovers content clipped by inconsistent crop boundaries without discarding anything visible in a member. Refined crops and missing or mismatched frames use their existing source files instead.
- Compare locally normalized native detail and colors, with separate limits for tiny strong marks and concentrated local changes. Estimate a small global camera/crop alignment from visual features when necessary. Check content outside the aligned overlap and, for interpolated alignment, compare on both original pixel grids to detect detail that interpolation might weaken. There is no person masking or overlap-only approval; detected changes in labels, panels, scrolling content, or occlusion keep observations separate.
- Allow a continuous state to last beyond the representative timestamp, with at most 180 seconds between matching observations. Every member must match the fixed first representative directly, then pass another direct check if a sharper representative is selected. This prevents transitive visual drift. Blank states, incomplete extraction, unknown source kinds, and separate observations from the same frame remain independent.
- Verify each state against each selected target once, reusing completed passes and rejections for initial and alternative matches. Send the full native state region actually used for grouping to the verifier. Preserve original coverage IDs, fallback placement, refinement, recovery, and the existing persistent verification cache. Uncertain native-state checks and request failures fall back to every original crop, including the representative's original crop.
- Save policy settings, original source hashes, frame hashes, native regions, crop IDs, and group membership in `illustration-source-states/groups.json` inside the finalization attempt. Save each verified native state as `source.png`; each observation retains its ordinary comparison archive and `source-state.json`. Groups are recomputed on every attempt. Without the optional local OpenCV dependencies, verification stays per observation.

Regression tests cover capture noise, illumination, crop boundaries, camera translation/zoom, changed labels, arrows, geometry, panels, colors, meaningful intermediate states, tiny and faint marks, clipping, occlusion, temporal separation, refined sources, frame provenance, cache reuse, and transient failures. Controlled rejection/fallback tests preserve identical final notes while reducing five mocked remote comparisons to one with original crops or two with native state regions. A visual spot-check of 18 representative/member pairs from the four saved jobs found no instructional differences; it is not an exhaustive quality evaluation.

A read-only local replay of **all original source crops and their native frames**, rather than finalization or live model calls, found:

| Saved job | Observations | First implementation states | Revised states | Reduction | Local replay time |
|---|---:|---:|---:|---:|---:|
| 22 September, linear regression | 342 | 334 | 279 | 18.4% | 53.57 s |
| 3 November, random forest | 1,023 | 1,005 | 874 | 14.6% | 135.16 s |
| 30 September, sorting seminar | 1,512 | 1,510 | 898 | 40.6% | 292.45 s |
| 14 October, sorting and heaps | 894 | 891 | 574 | 35.8% | 181.98 s |
| **Total** | **3,771** | **3,740** | **2,625** | **30.4%** | **663.17 s** |

The revised policy removes **1,146 redundant observations instead of 31**, approximately **37 times as many**. Replay time includes source validation, native-region preparation, and grouping: **11.1 minutes** across the four jobs. This added local work is not a measured API-call or end-to-end speedup: actual finalization uses the active coverage subset and may use refined sources, different targets, existing caches, or individual retries. Live verification and full lecture regeneration were not run.

Reproduce the replay with [`illustration_states_benchmark.py`](../scripts/illustration_states_benchmark.py), which validates managed sources and writes settings, source/frame hashes, groups, and stage timings outside the input jobs:

```sh
uv run python scripts/illustration_states_benchmark.py \
  --out /tmp/illustration-state-replay \
  out/22-09-* out/03-11-* out_algo/30-09-* out_algo/14-10-*
```

### 2. Improve alternative matching before remote verification

Before this optimization, `_matching_figures` ranked candidates using source identity, topic equality, word overlap, and temporal distance, returning up to three candidates. Topic equality could admit different instructional figures into expensive comparison.

Use stronger visual matching and figure-state tracking to rank plausible alternatives. Clear incompatibilities in panels or axis labels can help reject poor candidates; uncertain extraction must not silently eliminate a valid candidate. Retain unmatched source content and the existing recovery path.

The 27.1% alternative pass share identifies a promising target, but is not itself a measured avoidable-call rate.

Implemented on 2026-11-13 in [`illustration_matching.py`](../shad_helper/illustration_matching.py) and [`illustration_verification.py`](../shad_helper/illustration_verification.py), with an aggressive default shortlist:

- Rank the actual managed source crops using locally matched visual features and geometric alignment. Count distinct matched target features so repeated labels cannot inflate the score above one. Match partial views against larger figures at several scales as an additional safeguard. A strong visual match can qualify despite different topics or descriptions.
- Remember up to three source views already verified against each exact target crop. These examples help retrieve later stages or partial views that resemble an accepted observation more than the final figure. They influence retrieval only: they never authorize a merge or propagate a remote verdict.
- Drop a candidate from the initial alternative search only when both observations have complete extraction and known source kinds, both crops have enough visual features, and both the feature and partial-view scores are weak. Missing optional OpenCV dependencies, sparse images, unknown source kinds, and incomplete extraction retain the text fallback.
- Try just the leading alternative when its matched-feature fraction is at least `0.10` and it leads the second candidate by at least `0.02`; otherwise retain up to three alternatives. Uncertain candidates keep the broader budget. These are retrieval heuristics, not proof of equivalence.
- Preserve the initial coverage check, remote merge verification, and unmatched-source fallback. Recompute retrieval after crop refinement, using its actual pixels. Refinement, recovery, and final figure reconciliation use the broader three-candidate search without visual pruning or the one-candidate limit. Recovery of an alias uses that observation's source, not a crop belonging to the failed representative.
- Cache descriptors and pair scores within each stage. Save policy settings, candidate scores, verified source examples, crop IDs, and selected alternatives in `matching.json` beside the ordinary comparison diagnostics.

The local replay covers the latest saved attempts in **17 currently completed jobs**, a different population from the original 15-job audit. It reads archived image pairs and verdicts, validates image hashes, and learns only from earlier accepted comparisons. It does not feed a pruned historical pass back into the source-view tracker.

| Saved jobs | Archived alternative comparisons | Retained in shortlist | Reduction | Recorded passes retained |
|---|---:|---:|---:|---:|
| Eight ML jobs (`out/`) | 1,200 | 707 | 41.1% | 295 / 305 |
| Nine algorithms jobs (`out_algo/`) | 1,874 | 1,377 | 26.5% | 843 / 863 |
| **Total** | **3,074** | **2,084** | **32.2%** | **1,138 / 1,168 (97.4%)** |

The replay took **59.15 seconds** locally. Two alternative verdict records without saved requests were excluded. Stopping at the first recorded pass among the retained candidates would evaluate 2,037 pairs, but that is also a retrospective replay count, not a measured API-call reduction.

**Aggressive-policy tradeoff:** 30 recorded successful pairs fell outside the initial shortlist. Their source observations remain eligible for separate screenshots, refinement, recovery, and final reconciliation, but this can add figures or recovery work. Retaining source content does not establish unchanged final deduplication or lecture quality. The replay considers only historically evaluated alternatives; new candidates, changed fallback paths, wider candidate pools, existing caches, and remote retries can change real costs. No live verification or full lecture regeneration was performed, and **32.2% is not an end-to-end runtime improvement**.

Regression tests exercise scaled and partial views, strong candidates outside the text shortlist, uncertain extraction, missing dependencies, actual refined crops, crop-specific tracking, recovery, and archive integrity. Controlled complete-finalization tests produce identical notes with **four mocked remote comparisons reduced to two (50%)** for a successful alternative or a visually similar but semantically rejected one, and **four reduced to one (75%)** when every alternative is incompatible. The latter cases retain the unmatched source screenshot. The focused matching/verification suite passes all 115 tests; the broader illustration, notes, crop, and figure suite passes 532 tests with a writable temporary TeX cache.

The full-suite run with that cache reported 1,399 passed, 56 failed, and one skipped. Every failing test also failed on an isolated, untouched `HEAD` checkout; the remaining failures concern existing handoff/OpenAI client and usage expectations. No new failing test was introduced. Ruff and `git diff --check` pass.

Reproduce the read-only replay with [`illustration_matching_benchmark.py`](../scripts/illustration_matching_benchmark.py):

```sh
uv run python scripts/illustration_matching_benchmark.py \
  --out /tmp/illustration-matching-replay out/* out_algo/*
```

The output contains aggregate reports and per-pair scores, retained/pruned decisions, and historical verdicts. Input jobs are not modified.

### 3. Batch remaining checks against a common representative

Send the target once with several labeled source images and require an independent verdict for each source. This reduces request overhead while retaining source inspection.

Purely as an arithmetic packaging exercise, grouping the retained comparisons by representative into batches of at most eight would yield approximately **1,417 requests instead of 10,030**. This is not a deployment estimate: conditional alternatives, recovery dependencies, image budgets, and accuracy may require smaller batches. Request-count reduction does not imply proportional reduction in image processing, cost, or latency.

Benchmark small batches before adoption, checking that difficult sources do not receive an unjustified group-wide pass.

Implemented on 2026-11-16 in [`illustration_verification.py`](../shad_helper/illustration_verification.py), with **eight sources per request by default**:

- Batch the independent initial coverage checks after source-state grouping. Every batch shares the exact target pixels, crop identity, caption, and content. Send that target and its context once, alongside individually labeled, full-resolution source images and descriptions. Limit the batch to eight sources and 20 MB of base64-encoded image data. Oversized pairs and singleton groups use the existing individual path; images are never resized to fit a batch.
- Skip identical source/target images, existing completed pair-cache entries, and repeated state/target pairs before packing. Preserve existing single-pair cache keys. Native state regions and refined source crops remain the actual inputs to verification.
- Require an independent `pass`, `fail`, or `uncertain`, with a source-specific reason, under every exact source key. Validate the whole response before caching any result; missing, extra, or malformed results invalidate the batch. Response order cannot associate a verdict with a different source. The prompt explicitly prohibits using other sources to fill in missing details or justify a group-wide pass.
- A failed or malformed batch falls back to ordinary individual verification. An otherwise valid batch's uncertain sources are rechecked individually; its passes and failures remain cached. Existing state-member fallback, refinement, recovery, and retry behavior then applies.
- Prefill the per-pair cache, then apply decisions in the original coverage order. Alternatives, refined comparisons, recovery, and final reconciliation remain sequential because earlier decisions can change their inputs or candidate lists. This implementation does not speculatively request conditional alternatives or add concurrency.
- Bind cached batch results to the batch prompt/schema policy, exact pair inputs, source image hash, raw response, and source-to-pair mapping. Save shared images, labeled membership, schema, raw response, exact HTTP request when using `PlanClient`, and elapsed request-wrapper time under `illustration-verification/batches/<id>`. Initial coverage archives copy each shared batch once into `illustration-batches/<id>`; each pair retains its own source/target, verdict, and `artifacts/batch.json` reference. Existing cached batch evidence is also copied on subsequent initial-coverage attempts.

Controlled complete-finalization tests produce identical notes, including fallback placement, coverage IDs, and prose references:

| Controlled workload | Individual calls | Batched calls | Reduction |
|---|---:|---:|---:|
| 16 independently accepted sources | 16 | 2 | 87.5% |
| 16 sources with mixed pass/fail outcomes and reuse of a newly created fallback | 17 | 3 | 82.4% |

Other regression cases cover shuffled response keys, missing/extra results, invalid individual verdicts, context-limit failures, per-source uncertain retries, cache invalidation and provenance, already cached/identical/excluded sources, different target captions, exact refined pixels, byte budgets, full-resolution HTTP image attachments, and request-recorder restoration.

Validation: **576 tests pass** across the illustration, notes, crop-validation, Anki-figure, chapter-LaTeX, and video-notes suites, using a writable temporary TeX cache. Ruff and `git diff --check` also pass. The full repository suite was not run for this change.

A read-only packaging replay of the latest saved attempts in **18 completed jobs** (nine ML and nine algorithms jobs; one has no initial pair requests) found:

| Saved jobs | Distinct archived initial pairs | Packaged requests | Reduction |
|---|---:|---:|---:|
| ML (`out/`) | 4,544 | 696 | 84.7% |
| Algorithms (`out_algo/`) | 4,077 | 545 | 86.6% |
| **Total** | **8,621** | **1,241** | **85.6%** |

The replay validates the archived image hashes, collapses 27 repeated pair records from 8,648 initial records, and uses the production packer with its eight-source and byte limits. All **8,621 source images remain in the requests**. Target images fall from 8,621 to 1,241, and total PNG attachment bytes fall from **2,570,888,417 to 1,726,630,047 (32.8%)**. It finds **267 batches containing historically mixed verdicts**, preserving every recorded per-source outcome in the report; no pair has conflicting recorded verdicts. Ten historically uncertain pairs would need additional individual checks if the batched model returned the same uncertainties.

Leaving all **3,953 archived conditional checks** sequential gives **12,574 → 5,194 packaged coverage requests (58.7% fewer)** before uncertainty retries. The high-pass 30 September job packages 1,478 initial pairs into 193 requests; the mixed-outcome 14 October job packages 733 into 98.

**Measurement limits:** this is a cold-cache packaging replay, not an executed pipeline or a live model benchmark. It does not rerun source-state grouping, alternative selection, or finalization; it only collapses state reuse already present in archived requests. It preserves recorded single-pair verdicts as evidence rather than predicting batched verdicts. Different batch judgments, uncertainty, failures, warm caches, and later recovery can change actual request counts and final notes. No live latency, cost, batched-model accuracy, or end-to-end speedup has been measured. The local replay took **8.36 seconds**, which measures archive reading and packaging only.

Reproduce with [`illustration_batching_benchmark.py`](../scripts/illustration_batching_benchmark.py):

```sh
uv run python scripts/illustration_batching_benchmark.py \
  --out /tmp/illustration-batching-replay out/* out_algo/*
```

The script writes aggregate reports and per-pair assignments outside the source jobs. It requires historical single-pair archives and rejects already-batched attempts instead of silently excluding their checks.

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

Acceptance means fewer calls or lower elapsed time while preserving the quality and coverage of the saved notes. The original audit did not validate an optimization; priorities 1–3 have the local and regression evidence listed above, without an end-to-end benchmark.

## Code and artifact references

- [`video.py`](../shad_helper/video.py): selection settings, proxy-frame selection, native retrieval.
- [`vision_worker.py`](../shad_helper/vision_worker.py): sequential inference, schema parsing, retries, frame checkpoints.
- [`illustrations.py`](../shad_helper/illustrations.py): illustration-observation validation and source assets.
- [`illustration_states.py`](../shad_helper/illustration_states.py): conservative local source-state grouping and fixed representative selection.
- [`illustration_verification.py`](../shad_helper/illustration_verification.py): `_consolidation`, `_matching_figures`, `_finalize_coverage`, `_recover_coverage`, `_compare`, and reconciliation.
- [`notes.py`](../shad_helper/notes.py): finalization retries and chapter generation.
- Regularization evidence: `out/06-10-2026-Лекция---Никита-Волков---Польза-регуляризации-39c721e69f77/`.
- Random-forest evidence: `out/03-11-2026-Семинар---Никита-Волков---Случайный лес-65e02303bfa8/`.
- Repeated confirmation example: `out_algo/30-09-2026-Семинар---Максим-Бабенко---Сортировки-и-порядковые-ст-7d1fe10b33a4/`.
- Mixed pass/fail example: `out_algo/14-10-2026-Лекция---Максим-Бабенко---Сортировки-и-кучи-3670fed2c021/`.
