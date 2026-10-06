"""Keep sparse evidence available before routing or bulk model work can time out."""

import time
from collections import OrderedDict
from contextlib import closing, nullcontext
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .jobs import checkpoint, fingerprint, read_json, sha256_file, write_json
from .page_pipeline import export_selections
from .video_frames import iter_previews, read_frame
from .vision_mode import SPARSE_SELECTION_VERSION


def prepare_sparse_images(request, route, segmenter):
    from .sparse_selection import POLICY, select_sparse
    from .vision_budget import BudgetExhausted

    job, video = Path(request["job"]), Path(request["video"])
    start, end = request["start"], request["end"]
    base = {"video": request["video_sha256"], "start": start, "end": end,
            "metadata": request["metadata"], "version": SPARSE_SELECTION_VERSION,
            "policy": POLICY, "sampling": {"width": 640, "seconds": 0.5, "scene": 0.12}}
    recent, bootstraps = OrderedDict(), []
    inputs = {**base, "routing_model": request.get("primary_model", request["model"]),
              "routing_version": 1}

    def full_frame(timestamp):
        if timestamp not in recent:
            recent[timestamp] = read_frame(video, timestamp)
            if len(recent) > 4:
                recent.popitem(last=False)
        return recent[timestamp]

    def scan(classify, masks, *, overview=False):
        began = time.monotonic()
        with (
            closing(iter_previews(video, request["metadata"], start, end,
                                  width=640, sample_seconds=0.5)) as previews,
            (tqdm(total=end - start, desc="Sparse overview", unit="video-s", dynamic_ncols=True,
                  bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f} video-s [{elapsed}<{remaining}]")
             if overview else nullcontext()) as progress,
        ):
            def tracked_previews():
                for sample in previews:
                    yield sample
                    if progress is not None:
                        progress.update(max(0, sample["timestamp"] - start - progress.n))

            selections, audit = select_sparse(
                tracked_previews(), start=start, end=end, route=classify,
                segmenter=masks, read_frame=full_frame,
            )
            if progress is not None:
                progress.update(max(0, end - start - progress.n))
        return {"selections": selections, "audit": {**audit,
                "selection_seconds": time.monotonic() - began}}

    def export(selected):
        items = export_selections(job, video, request["video_sha256"], selected, prefix="sparse")
        for item, selection in zip(items, selected, strict=True):
            item["selection_role"] = selection.get("selection_role", "primary")
            if "complement_of" in selection:
                item["complement_of"] = selection["complement_of"]
        return items

    def manifest(items, audit, signature, stage):
        write_json(job / "sparse-selection.json", {
            "items": items + request.get("items", []), "audit": audit,
            "selection_fingerprint": signature, "stage": stage,
            "video": str(video), "video_sha256": request["video_sha256"],
            "start": start, "end": end,
        })

    # This first pass is cheap and complete, even if the first model load is killed.
    overview = checkpoint(job, "sparse-overview", base,
                          lambda: scan(lambda timestamp: {"kind": "other"}, None, overview=True))
    saved = job / "checkpoints/sparse-selection.json"
    reused = saved.exists() and read_json(saved).get("fingerprint") == fingerprint(inputs)
    if not reused:
        provisional = export(overview["selections"])
        for item in provisional:
            item["selection_issues"] = item.get("selection_issues", []) + [
                "Предварительная выборка: границы и полнота страницы требуют проверки."]
        manifest(provisional, overview["audit"], fingerprint(base), "provisional")

    def classify(timestamp):
        path = job / "selection" / f"sparse_bootstrap_{len(bootstraps):05d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(full_frame(timestamp)).save(path)
        try:
            result = route(timestamp, str(path.resolve()))
        except BudgetExhausted:
            result = {"kind": "other", "issues": ["Qwen budget exhausted during routing"]}
        bootstraps.append({"timestamp": timestamp, "image": str(path.resolve()),
                           "sha256": sha256_file(path), "result": result})
        return result

    # Budget changes do not invalidate completed selection or inference.
    selected = checkpoint(job, "sparse-selection", inputs,
                          lambda: {**scan(classify, segmenter), "bootstraps": bootstraps})
    if any("Qwen budget exhausted during routing" in record["result"].get("issues", [])
           for record in selected["bootstraps"]):
        saved.unlink(missing_ok=True)
    for bootstrap in selected["bootstraps"]:
        path = Path(bootstrap["image"])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(full_frame(bootstrap["timestamp"])).save(path)
    began = time.monotonic()
    items = export(selected["selections"])
    # Remove only our own obsolete provisional exports, never another mode's frames.
    retained = {Path(item["image"]) for item in items}
    for path in (job / "frames").glob("sparse_*.png"):
        if path.resolve() not in retained:
            path.unlink()
    audit = {**selected["audit"], "selected_video_frames": len(items),
             "bootstraps": selected["bootstraps"], "export_seconds": time.monotonic() - began,
             "overview_seconds": overview["audit"]["selection_seconds"], "selection_reused": reused}
    manifest(items, audit, fingerprint(inputs), "selected")
    print(f"Sparse selection: {len(items)} video images; Qwen budget "
          f"{request.get('vision_budget_minutes', (end - start) / 120):g} minutes", flush=True)
    return items, audit
