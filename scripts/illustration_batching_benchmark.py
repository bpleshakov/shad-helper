"""Package historical single-pair coverage checks without modifying jobs or calling a model.

This is a cold-cache packaging replay, not an accuracy or latency measurement.
Conditional alternatives/refinement stay sequential. Existing source-state aliases
are deduplicated by their archived request, but local grouping is not rerun.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

from shad_helper.illustration_verification import BATCH_SETTINGS, _consolidation_batches
from shad_helper.jobs import fingerprint, read_json, rendered_pdf, sha256_file, write_json


def replay(job, destination):
    attempt = max((job / "handoff/verification").glob("attempt-*"))
    if list(attempt.glob("illustration-coverage/**/artifacts/batch.json")):
        raise ValueError(f"Replay requires an archive from before batching: {attempt}")
    started = perf_counter()
    pairs, checked = {}, {}
    records = conditional = 0
    for call_path in sorted(attempt.glob("illustration-coverage/**/artifacts/call.json")):
        if len(call_path.relative_to(attempt).parts) != 4:
            conditional += 1
            continue
        record = read_json(call_path)
        images = {}
        for image in record["images"]:
            path = call_path.parent / image["file"]
            if path not in checked:
                checked[path] = sha256_file(path)
            if checked[path] != image["sha256"]:
                raise ValueError(f"Archived image hash mismatch: {path}")
            images[image["label"]] = path
        context = read_json(call_path.parent.parent / "candidate.json")
        description = json.loads(record["text"].split("Контекст: ", 1)[1])
        # Block ID is conservative: archived calls do not record the target crop ID.
        target = {"block_id": context["block_id"],
                  "image": checked[images["RECONSTRUCTION"]],
                  "figure": {key: value for key, value in description.items() if key != "observation"},
                  **{key: record[key] for key in ("model", "instructions", "schema")}}
        key = fingerprint({"target": target, "source": checked[images["SOURCE"]],
                           "observation": description["observation"]})
        pair = pairs.setdefault(key, {"key": key, "source": images["SOURCE"],
                                     "representative": images["RECONSTRUCTION"],
                                     "target": fingerprint(target), "archives": [], "verdicts": []})
        pair["archives"].append(str(call_path.parent.parent.relative_to(attempt)))
        result = read_json(call_path.parent.parent / "verdict.json")["verdict"]
        if result not in pair["verdicts"]:
            pair["verdicts"].append(result)
        records += 1
    batches = list(_consolidation_batches(pairs.values()))
    mixed = uncertain = source_bytes = target_bytes = repeated_target_bytes = 0
    assignments = []
    for index, batch in enumerate(batches, 1):
        verdicts = {verdict for item in batch for verdict in item["verdicts"]}
        mixed += len(batch) > 1 and "pass" in verdicts and bool(verdicts & {"fail", "uncertain"})
        uncertain += sum(len(batch) > 1 and "uncertain" in item["verdicts"] for item in batch)
        target_size = batch[0]["representative"].stat().st_size
        target_bytes += target_size
        repeated_target_bytes += target_size * len(batch)
        for item in batch:
            source_bytes += item["source"].stat().st_size
            assignments.append({"pair": item["key"], "batch": index,
                                "archives": item["archives"], "recorded_verdicts": item["verdicts"],
                                "source_sha256": checked[item["source"]],
                                "target_sha256": checked[item["representative"]]})
    report = {"job": str(job.resolve()), "attempt": attempt.name, "settings": BATCH_SETTINGS,
              "archived_initial_records": records, "distinct_initial_pairs": len(pairs),
              "repeated_pair_records": records - len(pairs),
              "conflicting_recorded_verdicts": sum(len(item["verdicts"]) > 1 for item in pairs.values()),
              "packaged_requests": len(batches), "mixed_verdict_batches": mixed,
              "recorded_uncertain_pairs_in_batches": uncertain,
              "conditional_records_left_sequential": conditional,
              "request_reduction_fraction": 1 - len(batches) / len(pairs) if pairs else 0,
              "source_images": len(pairs), "target_images_before": len(pairs),
              "target_images_after": len(batches),
              "image_bytes_before": source_bytes + repeated_target_bytes,
              "image_bytes_after": source_bytes + target_bytes,
              "seconds": perf_counter() - started}
    write_json(destination / "pairs.json", assignments)
    write_json(destination / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if any(args.out.resolve().is_relative_to(job.resolve()) for job in args.jobs):
        parser.error("Benchmark output must be outside every source job.")
    reports = []
    for job in args.jobs:
        if (not rendered_pdf(job).is_file()
                or not list((job / "handoff/verification").glob("attempt-*"))):
            continue
        report = replay(job, args.out / job.name)
        reports.append(report)
        write_json(args.out / "report.json", reports)
        print(f"{job.name}: {report['distinct_initial_pairs']} -> {report['packaged_requests']} "
              f"({report['request_reduction_fraction']:.1%} fewer packaged requests); "
              f"{report['seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
