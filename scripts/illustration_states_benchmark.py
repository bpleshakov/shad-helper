"""Replay source-state grouping on saved crops and frames without model calls."""

import argparse
from pathlib import Path
from time import perf_counter

from shad_helper.illustration_states import (
    SOURCE_STATE_SETTINGS,
    group_source_states,
    source_contexts,
)
from shad_helper.illustrations import illustration_index, resolve_illustration
from shad_helper.jobs import fingerprint, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    destinations = [args.out.resolve() / job.name for job in args.jobs]
    if any(destination.is_relative_to(job.resolve())
           for destination in destinations for job in args.jobs):
        parser.error("Benchmark output must be outside every source job.")
    report = []
    for job, destination in zip(args.jobs, destinations):
        evidence = read_json(job / "evidence.json")
        observations = illustration_index(evidence)
        started = perf_counter()
        sources = {identifier: resolve_illustration(job, evidence, identifier)
                   for identifier in observations}
        resolved = perf_counter()
        contexts = source_contexts(observations, sources, job, evidence)
        prepared = perf_counter()
        groups = group_source_states(observations, sources, contexts=contexts)
        finished = perf_counter()
        write_json(destination / "groups.json", {
            "settings": SOURCE_STATE_SETTINGS, "groups": groups, "contexts": contexts,
            "source_hashes": {identifier: item["image_sha256"]
                              for identifier, item in observations.items()},
        })
        summary = {"job": str(job.resolve()), "source_evidence": fingerprint(evidence),
                   "observations": len(sources), "states": len(groups),
                   "reduction_fraction": 1 - len(groups) / max(1, len(sources)),
                   "resolve_seconds": resolved - started,
                   "context_seconds": prepared - resolved,
                   "grouping_seconds": finished - prepared, "seconds": finished - started}
        report.append(summary)
        write_json(args.out / "report.json", report)
        print(f"{job.name}: {len(sources)} -> {len(groups)} "
              f"({summary['reduction_fraction']:.1%} fewer), {summary['seconds']:.2f}s",
              flush=True)


if __name__ == "__main__":
    main()
