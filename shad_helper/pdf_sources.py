"""Bind rendered PDF destinations to the lecture blocks that produced them."""

import hashlib
from pathlib import Path

import pymupdf

from .jobs import write_json


def write_source_map(pdf: Path, manifest: dict) -> None:
    blocks = {}
    with pymupdf.open(pdf) as document:
        destinations = document.resolve_names()
        for block in manifest["source_blocks"]:
            identifier = block["id"]
            anchor = f"shad-block-{identifier}"
            start = destinations.get(anchor, {}).get("page", -1)
            end = destinations.get(f"shad-end-{identifier}", {}).get("page", -1)
            if not 0 <= start <= end < len(document):
                raise ValueError(f"PDF source anchors are missing or invalid for {identifier}")
            blocks[identifier] = {**block, "page_start": start + 1, "page_end": end + 1,
                                  "anchor": anchor}
    write_json(pdf.parent / "source-map.json", {
        "schema_version": 1, "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "evidence_fingerprint": manifest["evidence_fingerprint"], "blocks": blocks,
    })
