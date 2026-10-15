# Illustration crop verification and repair

## Scope and execution plan

This change applies during notes finalization, after the notes model selects an
illustration and before screenshot acceptance or TikZ reconstruction verification.
Automatic notes and imported handoffs use the same finalization path. Candidate
extraction and missing-illustration detection are outside its scope.

Implementation order:

1. Add independent verification of actual crop pixels and cheap warning checks.
   Verify that bad crops cannot become successful managed refinements, while
   heuristic warnings alone cannot reject a correct crop.
2. Add at most two repairs after the initial proposal. Verify correction feedback,
   nearby-frame provenance, same-stage checks, and the attempt limit.
3. Integrate with screenshot/TikZ acceptance and existing caches. Verify stale
   imported crops cannot bypass the check and failures preserve source evidence.
4. Run focused illustration tests, then the project regression suite and lint.

## Verification

The verifier receives the unmarked native frame, a copy with the proposed box
marked, the actual cropped PNG, and the intended figure caption/description.
When another frame is used, it also receives the original native frame. It does
not receive the crop selector's justification or earlier verdicts.

It judges figure identity (including the intended drawing stage), completeness
of integral labels/axes/legends/panels, readability/occlusion, and substantial
unrelated content. Acceptance requires an explicit pass on every criterion.
Failures and uncertainty include concrete defects and a repair recommendation.

Cheap checks flag nearly uniform images, thin fragments, contact with the native
frame edge, and varying content at crop boundaries that may indicate cut strokes
or text. They are advisory signals for the verifier, not OCR, correctness proofs,
or independent rejection rules. Every proposed crop receives visual verification.

## Bounded repair

There are at most three proposals per finalization attempt: the initial crop and
two repairs. A repair receives the previous crop and specific verification defects
to expand or relocate its box. Source-frame clipping or occlusion can instead
trigger selection of an existing nearby native frame.

Only the nearest available earlier and later distinct frames within 30 seconds
are candidates, ordered by distance from the original timestamp. Recovery does
not rerun vision or extract additional video frames. Temporal proximity is not
proof of identity: the verifier must establish the same figure, model/data
condition, and meaningful drawing stage against the original frame.

If no safe repair succeeds, preserve the original screenshot and an open review
issue. Do not accept a TikZ reconstruction against an unverified source, reuse an
incoming stale crop ID, or discard an explicitly selected figure. Transport and
invalid-response failures retain the existing retry behavior. Exhausted semantic
failures remain reviewable without requesting an unlimited retry loop.

## Assets, cache, and diagnostics

Original frames, extracted observations, IDs, and source crops stay intact.
Successful derivative records include original and selected frame provenance,
actual crop hash, box, and the independent verification result. Cache inputs
include target context, relevant frame hashes, prompts/schemas, model, and policy.
Previously generated assets remain resolvable, but old refinement successes do
not bypass the new verification policy during finalization.

Attempt archives retain requests/responses, source and selected frames, proposed
crops, marked boxes, heuristic warnings, and verification decisions. Only a
verified proposal is published as a successful managed refinement.

Regression tests cover wrong neighboring figures, clipped labels, bounded repair,
alternate-frame stage mismatch, verifier failure, cache invalidation, advisory
warnings, stale imported crops, and exact preservation of original pixels/assets.
Mocked model tests validate control flow and acceptance rules; they do not measure
real model detection accuracy on lecture images.
