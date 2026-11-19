# Illustration source selection, verification and publication

Notes generation sees candidate crops and up to four native context frames for
ambiguous candidates. It chooses complete, readable representatives of each
instructional drawing and preserves independently meaningful stages. Ordinary
construction steps and scrolling fragments may share a representative; coverage
still records every observation. Captions and adjacent prose explain the figure's
specific teaching role. Automatic and manual handoffs use the same images.

## Source selection and bounded recovery

Before refining an uncached crop, visually rank its native frame and up to six
saved frames within 30 seconds, sampled near, midway and farther away on each
side. Prefer a complete, sharp direct screen capture to a filmed projection of
the same content. Projector lag can change parameters or drawing stages: similarity
and temporal proximity do not establish identity. Selection uses existing frames;
it does not extract new video frames or invent unseen content.

The original remains a fallback. An unavailable ranking may still publish an
independently verified original, but records retry state instead of caching the
ranking as complete. The initial crop proposal has at most two repairs. A repair
uses concrete defects to recrop or try another ranked frame. Every alternate
must independently preserve the exact figure, model/data condition and meaningful
stage. Never crop away mathematical content to hide a lecturer.

For printed figures clipped by a scrolling viewport, bounded recovery can align
nearby frames by a vertical translation. The overlap must match informative pixels;
static views, ambiguous alignment and changed content are rejected. The assembly
keeps original pixels without scaling, drawing or blending. It is only a candidate:
crop selection and independent verification still inspect both contributing frames.

## Independent verification and reconstruction

The verifier receives the unmarked native frame, a copy with proposed bounds,
the actual cropped PNG, and the intended caption/description. An alternate also
includes the original native frame. It does not receive the selector's reasoning.
Checks cover identity, completeness, readability and cleanliness. Pixel checks
for blank/thin content and suspicious boundaries remain advisory.
The caption defines the intended panels. A complete named subset does not require
every neighboring panel of the same notebook grid. Incidental mouse cursors are
allowed if they obscure no instructional content. Tiny architecture details need
to be legible when the explanation relies on them; an overview can use readable
blocks and arrows to show sequence or branching.

The purpose is explicit and part of the cache key:

- Screenshot publication requires complete, readable content without unrelated
  interface elements, people, hands, fingers or held objects. An equivalent,
  clearly better direct capture should replace an inferior projected view.
- A reconstruction source may contain harmless surrounding clutter if all required
  mathematical geometry, relationships and labels remain observable. Occluded or
  missing mathematical content still fails.

For a hand-drawn figure, the image-generation tool receives the verified primary
crop and up to three distinct views from its first continuous drawing interval.
The request includes speech overlapping a 30-second margin around those observations
and the selected frame, plus the exact speech segments cited by the figure block.
A separate request checks the generated PNG against the same images and speech.
It accepts changes in layout that preserve the teaching step. A failed or unavailable
generation or check falls back to the stricter screenshot publication check.
Explicitly imported TikZ diagrams retain their render-and-compare workflow.

## Publication, coverage and duplicates

A verified screenshot or reconstruction can be published. All chapters are
finalized together so later complete figures can rescue earlier observations.
Transient verification failures trigger one automatic retry from the original
candidates, reusing completed checks. After bounded recovery fails, original
assets and candidate diagnostics retain `unresolved` coverage with an open issue
for every affected observation. No manual-recovery messages are inserted into
the notes. A native-frame finding of no instructional visual permits an explicit
exclusion. An unresolved or excluded observation does not count as a recovered figure.

Failed coverage matches try a bounded set of retained alternatives before
creating another illustration, and recheck those alternatives after source repair.
Earlier failures get another chance to reuse newly recovered complete figures.
Final reconciliation also retrieves up to three
plausible earlier figures by source identity, topic, description and time, including
nonconsecutive figures and different topic wording. Only visual equivalence or
containment can merge them; meaningful stages remain separate. References and
coverage are remapped to the retained representative.

## Provenance and evaluation

Original frames, observations and managed assets stay intact. Derivatives bind
source hashes, selected frame, bounds, purpose and independent verification.
Cache inputs include relevant frames, prompts/schemas, model and policy. Archives
retain source selection, proposals, repairs, rendered reconstructions and verdicts.
Scroll assemblies also bind both native source hashes, viewport and translation;
their source pixels are reconstructed and checked when resolving a derivative.
The relaxed crop policy can reuse a previous policy's independently verified pass
only when all nonprompt inputs match. Failed decisions are checked again.

Regression tests exercise publication decisions, bounded repair, stage mismatch,
cache invalidation, unresolved references and portable rendering. Mocked judgments
test control flow; they do not establish real model accuracy. Use the
[label benchmark](illustration-benchmark.md) to freeze human ratings and compare
later outputs. Changed images require review; removed bad images do not count as
fixes, and missing good figures and missing-illustration lectures remain separate.
