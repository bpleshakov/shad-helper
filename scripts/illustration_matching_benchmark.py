"""Replay archived alternative checks locally; never modify jobs or call a model.

Only recorded candidate pairs have known verdicts. This measures retrospective
retrieval, not new candidates, fallback cost, model accuracy, or end-to-end speed.
"""

import argparse
from pathlib import Path
from time import perf_counter

from shad_helper.illustration_matching import (
    MATCHING_SETTINGS,
    VisualMatcher,
    candidate_limit,
    visually_plausible,
)
from shad_helper.illustrations import illustration_index
from shad_helper.jobs import fingerprint, read_json, sha256_file, write_json


def replay(job, destination):
    attempts = sorted((job / "handoff/verification").glob("attempt-*"))
    attempt = attempts[-1]
    evidence = read_json(job / "evidence.json")
    observations = illustration_index(evidence)
    blocks = {}
    for path in [*attempt.glob("illustration-verification/*/accepted.json"),
                 *attempt.glob("illustration-coverage/*/accepted-block.json")]:
        block = read_json(path)
        if block:
            blocks[block["id"]] = block
    images = {}

    def resolve(digest, crop_id):
        path = images[digest]
        if sha256_file(path) != digest:
            raise ValueError(f"Archived image hash mismatch: {path}")
        return path

    matcher = VisualMatcher(resolve)

    def pair(folder):
        call = folder / "artifacts/call.json"
        if not call.is_file():
            return None
        record = read_json(call)
        result = []
        for image in record["images"]:
            images.setdefault(image["sha256"], call.parent / image["file"])
            result.append({"illustration_id": image["sha256"]})
        return result

    started = perf_counter()
    records = []
    examined = kept = old_passes = kept_passes = replayed = missing = 0
    candidates = sorted(attempt.glob("illustration-coverage/*/candidate.json"),
                        key=lambda path: (path.stat().st_mtime_ns, str(path)))
    for candidate in candidates:
        folder = candidate.parent
        source = observations[folder.name]
        pending = []
        for verdict_file in sorted(folder.glob("alternatives/*/verdict.json")):
            comparison = pair(verdict_file.parent)
            if comparison is None:
                missing += 1
                continue
            target_id = verdict_file.parent.name
            target = observations.get(blocks.get(target_id, {}).get("illustration_id"), {})
            reliable = all(item.get("status") == "complete"
                           and item.get("source_kind") in {"drawn", "printed"}
                           for item in (source, target))
            score = matcher.score(*comparison)
            eligible = not reliable or visually_plausible(score)
            verdict = read_json(verdict_file)["verdict"]
            record = {"archive": str(verdict_file.parent), "source_id": folder.name,
                      "target_id": target_id, "score": score, "eligible": eligible,
                      "reliable": reliable, "verdict": verdict}
            records.append(record)
            examined += 1
            old_passes += verdict == "pass"
            if eligible:
                pending.append((score["inlier_fraction"] if score else 0,
                                verdict_file.stat().st_mtime_ns, record, comparison))
        pending.sort(key=lambda item: (-item[0], item[1]))
        limit = candidate_limit([item[2]["score"] for item in pending],
                                broad=any(not item[2]["reliable"] for item in pending))
        for _, _, record, _ in pending[limit:]:
            record["eligible"] = False
        pending = pending[:limit]
        kept += len(pending)
        kept_passes += sum(item[2]["verdict"] == "pass" for item in pending)
        # Do not teach retrieval a historical pass that this policy pruned.
        for _, _, record, comparison in pending:
            replayed += 1
            if record["verdict"] == "pass":
                matcher.remember(*comparison)
                break
        initial = folder / "verdict.json"
        if initial.is_file() and read_json(initial).get("verdict") == "pass":
            comparison = pair(folder)
            if comparison:
                matcher.remember(*comparison)
    summary = {"job": str(job.resolve()), "attempt": attempt.name,
               "source_evidence": fingerprint(evidence), "settings": MATCHING_SETTINGS,
               "archived_comparisons": examined, "retained_comparisons": kept,
               "pruned_fraction": (1 - kept / examined) if examined else 0,
               "recorded_passes": old_passes, "retained_passes": kept_passes,
               "replayed_checks_until_pass": replayed, "records_without_call": missing,
               "seconds": perf_counter() - started}
    write_json(destination / "comparisons.json", records)
    write_json(destination / "report.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if any(args.out.resolve().is_relative_to(job.resolve()) for job in args.jobs):
        parser.error("Benchmark output must be outside every source job.")
    reports = []
    for job in args.jobs:
        if not (job / "notes/main.pdf").is_file():
            continue
        if not list((job / "handoff/verification").glob("attempt-*")):
            continue
        report = replay(job, args.out / job.name)
        reports.append(report)
        write_json(args.out / "report.json", reports)
        print(f"{job.name}: {report['archived_comparisons']} -> "
              f"{report['retained_comparisons']} ({report['pruned_fraction']:.1%} pruned); "
              f"passes {report['retained_passes']}/{report['recorded_passes']}; "
              f"{report['seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
