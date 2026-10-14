"""Managed source crops shared by extraction, notes, and portable rendering."""

import math
import os
import re
from pathlib import Path

from .jobs import read_json, sha256_file, write_json

ILLUSTRATION_POLICY_VERSION = 2
CROP_FRAME_MARGIN = 0.03


def validate_illustration_observation(value: dict) -> dict:
    if (not isinstance(value, dict)
            or set(value) != {"source_kind", "box", "description"}
            or value["source_kind"] not in ("drawn", "printed", "unknown")
            or not isinstance(value["description"], str)
            or not value["description"].strip()
            or not isinstance(value["box"], list) or len(value["box"]) != 4
            or any(type(number) not in (int, float) or not math.isfinite(number)
                   or not 0 <= number <= 1 for number in value["box"])):
        raise ValueError("Invalid illustration observation")
    x0, y0, x1, y1 = value["box"]
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Invalid illustration box")
    return value


def illustration_index(evidence: dict) -> dict[str, dict]:
    return {item["id"]: {**item, "visual_id": visual["id"],
                         "timestamp": visual["timestamp"], "status": visual["status"]}
            for visual in evidence.get("visuals", [])
            for item in visual.get("illustrations", [])}


def materialize_illustrations(job: Path, items: list[dict]) -> dict[str, list[dict]]:
    """Crop native frames and publish the manifest only after every crop is saved."""
    from PIL import Image

    job = Path(job).resolve()
    root = job / "illustrations"
    root.mkdir(parents=True, exist_ok=True)
    if root.resolve() != root:
        raise ValueError("Illustration directory must not be a symlink")
    assets, result = {}, {}
    for item in items:
        frame_id = item["id"]
        if (not isinstance(frame_id, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,73}", frame_id)):
            raise ValueError("Invalid visual item ID")
        result[frame_id] = []
        observations = item["content"]["illustrations"]
        if not isinstance(observations, list):
            raise ValueError("Invalid illustration observations")  # noqa: TRY004 - invalid wire format
        if not observations:
            continue
        with Image.open(item["image"]) as image:
            width, height = image.size
            margin_x, margin_y = math.ceil(width * CROP_FRAME_MARGIN), math.ceil(height * CROP_FRAME_MARGIN)
            for position, observation in enumerate(observations, 1):
                validate_illustration_observation(observation)
                identifier = f"{frame_id}_ill{position:02d}"
                if len(identifier) > 80:
                    raise ValueError("Illustration ID exceeds 80 characters")
                path = root / f"{identifier}.png"
                temporary = root / f".{identifier}.{os.getpid()}.tmp"
                x0, y0, x1, y1 = observation["box"]
                # Model boxes can stop at the axes and clip nearby labels. Keep a
                # small, fixed native-frame margin, including for screenshot fallbacks.
                pixels = (max(0, math.floor(x0 * width) - margin_x),
                          max(0, math.floor(y0 * height) - margin_y),
                          min(width, math.ceil(x1 * width) + margin_x),
                          min(height, math.ceil(y1 * height) + margin_y))
                try:
                    image.crop(pixels).convert("RGB").save(temporary, format="PNG")
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                digest = sha256_file(path)
                result[frame_id].append({"id": identifier, **observation,
                                         "box": [pixels[0] / width, pixels[1] / height,
                                                 pixels[2] / width, pixels[3] / height],
                                         "image_sha256": digest})
                assets[identifier] = {"path": path.relative_to(job).as_posix(), "sha256": digest}
    write_json(root / "manifest.json", {"version": 1, "assets": assets})
    return result


def resolve_illustration(job: Path, evidence: dict, identifier: str) -> Path:
    """Resolve only declared PNGs whose contents agree with the evidence and manifest."""
    import pymupdf

    observation = illustration_index(evidence).get(identifier)
    if observation is None:
        raise ValueError(f"Unknown illustration ID: {identifier}")
    job = Path(job).resolve()
    root = job / "illustrations"
    try:
        if root.resolve() != root:
            raise ValueError("Illustration directory must not be a symlink")
        manifest_path = root / "manifest.json"
        if manifest_path.resolve().parent != root:
            raise ValueError("Illustration manifest escapes the managed directory")
        manifest = read_json(manifest_path)
        if (not isinstance(manifest, dict) or set(manifest) != {"version", "assets"}
                or manifest["version"] != 1 or not isinstance(manifest["assets"], dict)):
            raise ValueError("Invalid illustration manifest")
        asset = manifest["assets"].get(identifier)
        if (not isinstance(asset, dict) or set(asset) != {"path", "sha256"}
                or not isinstance(asset["path"], str)
                or asset["sha256"] != observation["image_sha256"]):
            raise ValueError("Illustration manifest does not match evidence")
        relative = Path(asset["path"])
        path = (job / relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts
                or not path.is_relative_to(root) or path.suffix != ".png"):
            raise ValueError("Illustration path escapes the managed directory")
        if sha256_file(path) != asset["sha256"]:
            raise ValueError("Illustration hash does not match evidence")
        with path.open("rb") as stream:
            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ValueError("Illustration asset is not a PNG")
        pixmap = pymupdf.Pixmap(path)
        if pixmap.width < 1 or pixmap.height < 1:
            raise ValueError("Empty illustration PNG")
        return path
    except (OSError, RuntimeError, TypeError, KeyError, pymupdf.mupdf.FzErrorBase) as error:
        raise ValueError(f"Cannot read illustration {identifier}: {error}") from error
