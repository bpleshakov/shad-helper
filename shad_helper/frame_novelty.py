"""Suppress repeated visual inference using visible content, not model output."""

import time
from pathlib import Path

from tqdm import tqdm

from .jobs import checkpoint, fingerprint, read_json, sha256_file

PERSON_MODEL = {
    "repository": "opencv/opencv_zoo",
    "revision": "d4938dfc9d4ec5d098bfa33e98b3f3345a236586",
    "filename": "models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx",
    "sha256": "552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24",
}
NOVELTY_SETTINGS = {
    "version": 4,
    "proxy_width": 720,
    "person_probability": 0.5,
    "person_edge_probability": 0.2,
    "pane_edge_threshold": 24,
    "pane_edge_minimum_fraction": 0.4,
    "standing_person_minimum_aspect": 1.25,
    "person_bottom_fraction": 0.98,
    "person_margin_fraction": 0.01,
    "maximum_occlusion_fraction": 0.45,
    "minimum_overlap_fraction": 0.4,
    "minimum_change_fraction": 0.0008,
    "local_change_window": 32,
    "minimum_local_change_pixels": 40,
    "scene_change_fraction": 0.14,
    "pixel_threshold": 24,
    "contrast_threshold": 12,
    "detail_threshold": 8,
    "comparison_margin_pixels": 10,
    "alignment_correlation": 0.9,
    "maximum_alignment_change": 0.03,
    "merge_window_seconds": 5.0,
    "merge_missing_fraction": 0.0004,
    "merge_local_missing_pixels": 12,
    "merge_minimum_sharpness_ratio": 0.85,
}


class _PersonSegmenter:
    """PPHumanSeg on overlapping crops, including lecturers at a video's edges."""

    def __init__(self):
        import cv2
        from huggingface_hub import hf_hub_download

        # OpenCV 5.0 misaligns this model's masks; 4.14 was checked on native frames.
        if int(cv2.__version__.split(".")[0]) >= 5:
            raise RuntimeError("Person segmentation requires OpenCV 4: run uv sync --extra local.")
        path = hf_hub_download(PERSON_MODEL["repository"], PERSON_MODEL["filename"],
                               revision=PERSON_MODEL["revision"])
        if sha256_file(Path(path)) != PERSON_MODEL["sha256"]:
            raise ValueError("Person segmentation model checksum does not match the pinned model.")
        self.net = cv2.dnn.readNetFromONNX(path)

    def __call__(self, image):
        import cv2
        import numpy as np

        height, width = image.shape[:2]
        edge = min(height, width)
        positions = sorted({*range(0, width - edge + 1, max(1, edge // 2)), width - edge})
        score = np.zeros((height, width), np.float32)
        for left in positions:
            tile = image[:, left:left + edge]
            self.net.setInput(cv2.dnn.blobFromImage(
                tile, 1 / 127.5, (192, 192), (127.5,) * 3, swapRB=True,
            ))
            probability = cv2.resize(self.net.forward()[0, 1], (edge, height))
            score[:, left:left + edge] = np.maximum(score[:, left:left + edge], probability)
        # An inset camera view can crop a lecturer at an internal pane edge.
        edges = np.max(np.abs(image[:, 1:].astype(np.int16) - image[:, :-1]), axis=2)
        edge_counts = np.pad(np.cumsum(edges > NOVELTY_SETTINGS["pane_edge_threshold"], axis=0),
                             ((1, 0), (0, 0)))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.uint8(score > NOVELTY_SETTINGS["person_edge_probability"]),
        )
        people = np.zeros((height, width), np.uint8)
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            bottom = (h >= height * 0.25 and y + h >= height * 0.85
                      and (y + h >= height * NOVELTY_SETTINGS["person_bottom_fraction"]
                           or h >= w * NOVELTY_SETTINGS["standing_person_minimum_aspect"]))
            # A portrait's own outline is not a pane boundary. Require the
            # boundary to continue outside the candidate person's vertical span.
            outside_edges = edge_counts[-1] - (edge_counts[y + h] - edge_counts[y])
            boundaries = np.flatnonzero(
                outside_edges >= height * NOVELTY_SETTINGS["pane_edge_minimum_fraction"],
            ) + 1
            near_boundary = np.any(np.minimum(abs(boundaries - x), abs(boundaries - x - w))
                                   <= width * 0.02)
            side = h >= height * 0.1 and (x <= width * 0.1 or x + w >= width * 0.9 or near_boundary)
            # Interior portraits/figures on a slide are instructional candidates.
            if area >= height * width * 0.0005 and (bottom or side):
                component = labels == label
                # Grow uncertain edges only around a confident person detection.
                if np.max(score[component]) > NOVELTY_SETTINGS["person_probability"]:
                    people[component] = 1
        margin = max(2, round(min(height, width) * NOVELTY_SETTINGS["person_margin_fraction"]))
        return cv2.dilate(people, np.ones((2 * margin + 1,) * 2, np.uint8)).astype(bool)


def _align(reference, known, gray, visible):
    import cv2
    import numpy as np

    warp = np.eye(2, 3, dtype=np.float32)
    try:
        correlation, warp = cv2.findTransformECC(
            gray, reference, warp, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-4),
            np.uint8(known & visible) * 255, 5,
        )
        limit = NOVELTY_SETTINGS["maximum_alignment_change"]
        if (correlation < NOVELTY_SETTINGS["alignment_correlation"]
                or np.max(np.abs(warp[:, :2] - np.eye(2))) > limit
                or abs(warp[0, 2]) > gray.shape[1] * limit
                or abs(warp[1, 2]) > gray.shape[0] * limit):
            warp = np.eye(2, 3, dtype=np.float32)
    except cv2.error:
        # Blank or unrelated images need a direct comparison, not a guessed warp.
        warp = np.eye(2, 3, dtype=np.float32)
    size = (gray.shape[1], gray.shape[0])
    # Small alignment shifts must not introduce black edges or unknown strips.
    # Newly entering content still has to match the replicated boundary pixels.
    aligned = cv2.warpAffine(reference, warp, size,
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
    coverage = cv2.warpAffine(np.uint8(known), warp, size,
                              flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE).astype(bool)
    return aligned, coverage, warp.tolist()


def _covers_content(previous, current):
    """Check that a later native frame preserves the earlier visible marks."""
    import cv2
    import numpy as np

    first_image, known = previous
    image, visible = current
    if (first_image.shape != image.shape
            or max((~known).mean(), (~visible).mean())
            > NOVELTY_SETTINGS["maximum_occlusion_fraction"]):
        return False
    first_gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    gray, _, warp = _align(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), visible, first_gray, known)
    size = (first_gray.shape[1], first_gray.shape[0])
    image = cv2.warpAffine(image, np.float32(warp), size,
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REPLICATE)
    # Compare in the earlier frame's coordinates. Marks shifted out of view
    # must count as missing, so visibility outside the later image stays false.
    visible = cv2.warpAffine(np.uint8(visible), np.float32(warp), size,
                             flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP).astype(bool)
    common = known & visible
    if common.mean() < NOVELTY_SETTINGS["minimum_overlap_fraction"]:
        return False
    changed = (cv2.absdiff(first_image, image).max(axis=2)
               > NOVELTY_SETTINGS["pixel_threshold"]) & common
    if changed.mean() >= NOVELTY_SETTINGS["scene_change_fraction"]:
        return False
    details = []
    for pixels, coverage in ((first_image, known), (image, visible)):
        # Normalize only against visible neighbors, without halos from people.
        weights = cv2.GaussianBlur(coverage.astype(np.float32), (0, 0), 3)
        mean = cv2.GaussianBlur(pixels.astype(np.float32) * coverage[:, :, None], (0, 0), 3)
        mean /= np.maximum(weights[:, :, None], 1e-6)
        details.append((pixels - mean) * coverage[:, :, None])
    first, second = details
    threshold = NOVELTY_SETTINGS["contrast_threshold"]
    kernel = np.ones((3, 3), np.uint8)
    missing = np.any(np.abs(first) > threshold, axis=2) & known & ~visible
    for sign in (-1, 1):
        missing |= np.any((first * sign > threshold)
                          & (first * sign > cv2.dilate(second * sign, kernel) + threshold),
                          axis=2) & common
    window = NOVELTY_SETTINGS["local_change_window"]
    local_missing = cv2.boxFilter(missing.astype(np.float32), -1, (window, window),
                                  normalize=False, borderType=cv2.BORDER_CONSTANT).max()
    old_sharpness = cv2.Laplacian(first_gray, cv2.CV_32F)[common].var()
    new_sharpness = cv2.Laplacian(gray, cv2.CV_32F)[common].var()
    return bool(missing.mean() < NOVELTY_SETTINGS["merge_missing_fraction"]
                and local_missing < NOVELTY_SETTINGS["merge_local_missing_pixels"]
                and new_sharpness >= old_sharpness * NOVELTY_SETTINGS["merge_minimum_sharpness_ratio"])


class FrameNoveltyFilter:
    """Keep a bounded map of pixels actually visible in retained native frames."""

    def __init__(self, segmenter=None):
        self.segmenter = _PersonSegmenter() if segmenter is None else segmenter
        self.reference = self.known = None
        self.best_occlusion = 1.0
        self.current = None

    def consider(self, image, reason):
        import cv2
        import numpy as np

        width = min(NOVELTY_SETTINGS["proxy_width"], image.shape[1])
        height = max(1, round(image.shape[0] * width / image.shape[1]))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        visible = ~self.segmenter(image)
        self.current = image, visible
        occlusion = float((~visible).mean())
        metrics = {"occluded_fraction": occlusion, "changed_fraction": 0.0,
                   "newly_visible_fraction": 0.0}
        if occlusion > NOVELTY_SETTINGS["maximum_occlusion_fraction"]:
            self.reference = self.known = None
            return True, {**metrics, "reason": "uncertain_mask"}
        if self.reference is None or self.reference.shape != gray.shape or reason == "scroll_overlap":
            self.reference, self.known = gray.copy(), visible.copy()
            self.best_occlusion = occlusion
            return True, {**metrics, "reason": "scroll" if reason == "scroll_overlap" else "initial"}

        aligned, known, warp = _align(self.reference, self.known, gray, visible)
        margin = NOVELTY_SETTINGS["comparison_margin_pixels"]
        neighborhood = np.ones((2 * margin + 1,) * 2, np.uint8)
        # Contrast normalization must not sample a lecturer or an unknown pixel
        # just beyond a mask boundary and turn its halo into "new writing".
        safe_visible = cv2.erode(np.uint8(visible), neighborhood).astype(bool)
        safe_known = cv2.erode(np.uint8(known), neighborhood).astype(bool)
        common = safe_known & safe_visible
        if common.mean() < NOVELTY_SETTINGS["minimum_overlap_fraction"]:
            self.reference, self.known = gray.copy(), visible.copy()
            self.best_occlusion = occlusion
            return True, {**metrics, "reason": "limited_overlap"}

        first = aligned.astype(np.float32) - cv2.GaussianBlur(aligned, (0, 0), 3)
        second = gray.astype(np.float32) - cv2.GaussianBlur(gray, (0, 0), 3)
        kernel = np.ones((3, 3), np.uint8)
        threshold = NOVELTY_SETTINGS["contrast_threshold"]
        # Bidirectional one-pixel tolerance removes registration/resampling noise
        # while detecting both added strokes and erased strokes.
        changed = ((second < cv2.erode(first, kernel) - threshold)
                   | (second > cv2.dilate(first, kernel) + threshold)
                   | (first < cv2.erode(second, kernel) - threshold)
                   | (first > cv2.dilate(second, kernel) + threshold)) & common
        revealed = ((np.abs(second) > NOVELTY_SETTINGS["detail_threshold"])
                    & ~safe_known & safe_visible)
        raw_changed = (cv2.absdiff(aligned, gray) > NOVELTY_SETTINGS["pixel_threshold"]) & common
        scene_change = raw_changed.mean() >= NOVELTY_SETTINGS["scene_change_fraction"]
        metrics.update(changed_fraction=float(changed.mean()),
                       newly_visible_fraction=float(revealed.mean()), alignment=warp)
        change = changed | revealed
        window = NOVELTY_SETTINGS["local_change_window"]
        local_change = float(cv2.boxFilter(change.astype(np.float32), -1, (window, window),
                                          normalize=False, borderType=cv2.BORDER_CONSTANT).max())
        metrics["local_change_pixels"] = local_change
        novel = (change.mean() >= NOVELTY_SETTINGS["minimum_change_fraction"]
                 or local_change >= NOVELTY_SETTINGS["minimum_local_change_pixels"])
        clearer = occlusion < 0.005 and self.best_occlusion >= 0.05
        keep = bool(novel or scene_change or clearer)
        if keep:
            if scene_change:
                self.reference, self.known = gray.copy(), visible.copy()
            else:
                self.reference = aligned.copy()
                self.reference[visible] = gray[visible]
                self.known = known | visible
            self.best_occlusion = occlusion if novel or scene_change else min(
                self.best_occlusion, occlusion,
            )
        decision = ("scene_change" if scene_change else "new_content" if novel
                    else "clearer_view" if clearer else "unchanged_content")
        return keep, {**metrics, "reason": decision}


def filter_frames(job: Path, items: list[dict]) -> tuple[list[dict], dict]:
    """Checkpoint decisions by native-image bytes, policy, model and OpenCV build."""
    import cv2

    started = time.monotonic()
    inputs = {"frames": [{**item, "sha256": sha256_file(Path(item["image"]))} for item in items],
              "model": PERSON_MODEL, "settings": NOVELTY_SETTINGS, "opencv": cv2.__version__}
    path = job / "checkpoints/video/novelty.json"
    cache_hit = path.is_file() and read_json(path).get("fingerprint") == fingerprint(inputs)

    def scan():
        filterer = FrameNoveltyFilter()
        decisions = []
        group = []
        group_start = 0.0
        retained_count = 0
        with tqdm(items, desc="Video filtering", unit="frame",
                  dynamic_ncols=True, disable=not items) as frames:
            for item in frames:
                image = cv2.imread(item["image"])
                if image is None:
                    raise ValueError(f"Cannot read selected frame: {item['image']}")
                keep, metrics = filterer.consider(image, item["reason"])
                decision = {"id": item["id"], "timestamp": item["timestamp"],
                            "retained": keep, "novelty_retained": keep, **metrics}
                if keep:
                    # Preserve the initial view and bound the whole group, so a
                    # chain of additions cannot delay a representative indefinitely.
                    if (retained_count > 1
                            and item["timestamp"] - group_start <= NOVELTY_SETTINGS["merge_window_seconds"]
                            and item["reason"] != "scroll_overlap"
                            and group[-1][1] != "scroll_overlap"
                            and all(_covers_content(entry[2], filterer.current) for entry in group)):
                        group[-1][0].update(retained=False, reason="covered_by_later_frame",
                                            represented_by=item["id"])
                    else:
                        group_start = item["timestamp"]
                        group = []
                        retained_count += 1
                    group.append((decision, item["reason"], filterer.current))
                decisions.append(decision)
        representatives = {}
        for decision in reversed(decisions):
            if "represented_by" in decision:
                decision["represented_by"] = representatives[decision["represented_by"]]
            representatives[decision["id"]] = decision.get("represented_by", decision["id"])
        return {"decisions": decisions}

    result = checkpoint(job, "video/novelty", inputs, scan)
    retained = {item["id"] for item in result["decisions"] if item["retained"]}
    selected = [item for item in items if item["id"] in retained]
    return selected, {"candidate_frames": len(items), "suppressed_frames": len(items) - len(selected),
                      "merged_frames": sum(item["reason"] == "covered_by_later_frame"
                                           for item in result["decisions"]),
                      "seconds": time.monotonic() - started, "cache_hit": cache_hit,
                      "model": dict(PERSON_MODEL), "settings": dict(NOVELTY_SETTINGS)}
