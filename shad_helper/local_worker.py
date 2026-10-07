"""Isolated transcription worker: process exit releases Apple unified memory."""

import argparse
from pathlib import Path

from .jobs import check_audio_job, checkpoint, read_json, sha256_file, write_json


def run(request: dict) -> dict:
    job = Path(request["job"])
    check_audio_job(job)
    model_id = request["model"]

    def process(item):
        signature = {"model": model_id, "item": item, "language": request["language"],
                     "sha256": sha256_file(Path(item["audio"]))}

        def action():
            try:
                import mlx_whisper
            except ImportError as error:
                raise RuntimeError("Local transcription requires: uv sync --extra local.") from error
            result = mlx_whisper.transcribe(
                item["audio"], path_or_hf_repo=model_id, language=request["language"],
                task="transcribe", word_timestamps=True, verbose=False,
            )
            return {**item, "segments": result["segments"], "language": result.get("language")}

        return checkpoint(job, f"asr/{item['id']}", signature, action)

    return {"items": [process(item) for item in request["items"]]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, run(read_json(args.request)))


if __name__ == "__main__":
    main()
