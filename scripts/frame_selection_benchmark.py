"""Replay novelty filtering over saved native frames without rerunning lecture models."""

import argparse
import os
from pathlib import Path

from shad_helper.frame_novelty import filter_frames
from shad_helper.jobs import fingerprint, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float, default=float("inf"))
    args = parser.parse_args()
    if not 0 <= args.start < args.end:
        parser.error("Use 0 <= start < end.")
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    report = []
    for job in args.jobs:
        evidence = read_json(job / "evidence.json")
        diagnostics = read_json(job / "extraction-diagnostics.json")
        if diagnostics.get("evidence_fingerprint") != fingerprint(evidence):
            raise ValueError(f"Evidence/diagnostics mismatch: {job}")
        keys = ("id", "timestamp", "image", "reason", "sharpness", "content_regions")
        frames = [{key: visual[key] for key in keys if key in visual}
                  for visual in diagnostics["visuals"]
                  if args.start <= visual["timestamp"] < args.end]
        if not frames:
            raise ValueError(f"No saved frames in the requested interval: {job}")
        destination = args.out.resolve() / job.name
        if destination.is_relative_to(job.resolve()):
            raise ValueError("Benchmark output must be outside the source job.")
        selected, result = filter_frames(destination, frames)
        summary = {"job": str(job.resolve()), "source_evidence": fingerprint(evidence),
                   **result, "retained_frames": len(selected),
                   "reduction_fraction": 1 - len(selected) / len(frames)}
        report.append(summary)
        write_json(args.out / "report.json", report)
        print(f"{job.name}: {len(frames)} -> {len(selected)} "
              f"({summary['reduction_fraction']:.1%} fewer), {result['seconds']:.2f}s",
              flush=True)


if __name__ == "__main__":
    main()
