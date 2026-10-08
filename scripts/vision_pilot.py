"""Reproduce the fixed-frame local vision pilot; references never enter model prompts."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=Path("tests/fixtures/vision-pilot-gold.json"))
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    from shad_helper.video import probe_video
    from shad_helper.vision_worker import PROMPT, RETRY_PROMPT

    (out / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    (out / "retry-prompt.txt").write_text(RETRY_PROMPT, encoding="utf-8")
    frames = out / "frames"
    frames.mkdir(exist_ok=True)
    gold = json.loads(args.gold.read_text())
    started = time.monotonic()
    items = []
    for case in gold["cases"]:
        media = args.media_dir / case["source_filename"]
        metadata = probe_video(media)
        if metadata is None or metadata["rotation"] % 360:
            raise ValueError(f"Pilot requires an unrotated video stream: {media}")
        for timestamp in case["inspected_timestamps_seconds"]:
            identifier = f"{case['id']}-{timestamp}"
            image = frames / f"{identifier}.png"
            result = subprocess.run([
                "ffmpeg", "-hide_banner", "-nostdin", "-y", "-ss", str(timestamp),
                "-copyts", "-noautorotate", "-i", str(media),
                "-map", f"0:{metadata['stream_index']}", "-vf", "showinfo",
                "-frames:v", "1", str(image),
            ], check=True, capture_output=True, text=True)
            base = re.search(r"config in time_base:\s*(\d+)/(\d+)", result.stderr)
            pts = re.search(r"\bpts:\s*(-?\d+)\s+pts_time:", result.stderr)
            if not base or not pts:
                raise RuntimeError(f"Pilot did not retrieve a frame at {timestamp}: {media}")
            actual = (int(pts.group(1)) * int(base.group(1)) / int(base.group(2))
                      - metadata["container_start"])
            items.append({"id": identifier, "image": str(image), "timestamp": actual,
                          "requested_timestamp": timestamp})
    request = {"job": str(out), "model": args.model, "revision": args.revision, "items": items}
    request_path = out / "request.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    subprocess.run([
        sys.executable, "-m", "shad_helper.vision_worker", "--request", str(request_path),
        "--output", str(out / "response.json"),
    ], check=True)
    report = {"seconds": time.monotonic() - started, "frames": len(items),
              "model": args.model, "revision": args.revision,
              "reference": str(args.gold.resolve()),
              "quality": "Pending independent comparison with reference; not scored automatically."}
    (out / "pilot-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
