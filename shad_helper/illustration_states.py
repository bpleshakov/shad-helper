"""Track equivalent visible figure states across nearby native source crops."""

from functools import lru_cache
from pathlib import Path

from .jobs import fingerprint, sha256_file

SOURCE_STATE_SETTINGS = {
    "version": 2,
    "nearby_seconds": 180,
    "minimum_sharpness": 4,
    "background_noise": 8,
    "contrast_floor": 20,
    "detail_noise": 12,
    "maximum_detail_change": 64,
    "maximum_changed_fraction": 0.0004,
    "local_window": 16,
    "maximum_local_changed_pixels": 8,
    "maximum_color_change_fraction": 0.005,
    "maximum_candidates": 6,
    "minimum_box_overlap": 0.7,
}


def source_contexts(observations, sources, job, evidence):
    """Give a spatial track one fixed native region containing every original crop.

    Only use a frame after its pixels reproduce the original managed crop exactly.
    Refined/stitched crops and missing or changed frames keep their existing path.
    """
    media_hash = evidence.get("media", {}).get("sha256")
    if not media_hash:
        return {}
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {}
    job = Path(job).resolve()
    frames = lru_cache(maxsize=2)(lambda path: cv2.imread(str(path)))
    hashes, contexts, tracks = {}, {}, []
    for identifier in sorted(sources, key=lambda item: (observations[item]["timestamp"], item)):
        observation = observations[identifier]
        frame = job / "frames" / f"{observation['visual_id']}_{media_hash[:12]}.png"
        if (sources[identifier].resolve() != job / "illustrations" / f"{identifier}.png"
                or not frame.is_file() or frame.resolve().parent != job / "frames"):
            continue
        image = frames(frame)
        if image is None:
            continue
        height, width = image.shape[:2]
        box = [round(value * size) for value, size in
               zip(observation["box"], (width, height, width, height))]
        x0, y0, x1, y1 = box
        if not np.array_equal(cv2.imread(str(sources[identifier])), image[y0:y1, x0:x1]):
            continue
        candidates = []
        for index, track in enumerate(tracks):
            if (observation["timestamp"] - track["latest"] > SOURCE_STATE_SETTINGS["nearby_seconds"]
                    or track["kind"] != observation["source_kind"] or track["size"] != (width, height)
                    or observation["visual_id"] in track["visuals"]):
                continue
            a, b, c, d = track["anchor"]
            overlap = max(0, min(c, x1) - max(a, x0)) * max(0, min(d, y1) - max(b, y0))
            union = (c - a) * (d - b) + (x1 - x0) * (y1 - y0) - overlap
            candidates.append((overlap / union, index))
        score, index = max(candidates, default=(0, -1))
        if score < SOURCE_STATE_SETTINGS["minimum_box_overlap"]:
            index = len(tracks)
            tracks.append({"anchor": box, "region": box[:], "kind": observation["source_kind"],
                           "size": (width, height), "visuals": set()})
        track = tracks[index]
        region = track["region"]
        region[:] = [min(region[0], x0), min(region[1], y0),
                     max(region[2], x1), max(region[3], y1)]
        track["latest"] = observation["timestamp"]
        track["visuals"].add(observation["visual_id"])
        if frame not in hashes:
            hashes[frame] = sha256_file(frame)
        contexts[identifier] = {"frame": str(frame), "frame_sha256": hashes[frame],
                                "region": region, "source_box": box}
    return contexts


def _source_image(identifier, sources, contexts):
    import cv2

    context = contexts.get(identifier)
    if context is None:
        return _content_image(sources[identifier])
    image = cv2.imread(context["frame"])
    x0, y0, x1, y1 = context["region"]
    return image[y0:y1, x0:x1]


def save_state_source(identifier, sources, contexts, path):
    """Save the exact canonical source used for grouping and downstream verification."""
    import cv2

    if sha256_file(Path(contexts[identifier]["frame"])) != contexts[identifier]["frame_sha256"]:
        raise ValueError("Source-state frame changed during verification")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), _source_image(identifier, sources, contexts)):
        raise OSError(f"Cannot write source-state image: {path}")


def _content_image(path):
    import cv2
    import numpy as np

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read source-state image: {path}")
    corners = image[[0, 0, -1, -1], [0, -1, 0, -1]].astype(np.int16)
    background = np.median(corners, axis=0)
    noise = SOURCE_STATE_SETTINGS["background_noise"]
    if np.max(np.abs(corners - background)) <= noise:
        content = np.any(np.abs(image.astype(np.int16) - background) > noise, axis=2)
        rows, columns = np.flatnonzero(content.any(axis=1)), np.flatnonzero(content.any(axis=0))
        if len(rows):
            image = image[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    return image


def _prepare(image):
    import cv2
    import numpy as np

    pixels = image.astype(np.float32)
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    detail = gray - cv2.GaussianBlur(gray, (0, 0), 3)
    contrast = np.sqrt(cv2.GaussianBlur(detail ** 2, (0, 0), 3))
    detail *= 32 / np.maximum(SOURCE_STATE_SETTINGS["contrast_floor"], contrast)
    kernel = np.ones((3, 3), np.uint8)
    return {"image": image, "detail": detail,
            "low": cv2.erode(detail, kernel), "high": cv2.dilate(detail, kernel),
            "color": cv2.GaussianBlur(pixels, (0, 0), 1.2)}


def _features(image):
    import cv2
    import numpy as np

    scale = min(1.0, 1024 / max(image.shape[:2]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    points, descriptors = cv2.SIFT_create(nfeatures=600).detectAndCompute(gray, None)
    return np.float32([point.pt for point in points]).reshape(-1, 2) / scale, descriptors


def _alignment(left, right):
    """Estimate a global camera/crop transform; never deform individual strokes."""
    import cv2
    import numpy as np

    left_points, left_descriptors = left
    right_points, right_descriptors = right
    if min(len(left_points), len(right_points)) < 12:
        return None
    matches = cv2.BFMatcher().knnMatch(right_descriptors, left_descriptors, k=2)
    pairs = [first for first, second in matches if first.distance < 0.7 * second.distance]
    if len(pairs) < 12:
        return None
    source = np.float32([right_points[pair.queryIdx] for pair in pairs])
    target = np.float32([left_points[pair.trainIdx] for pair in pairs])
    matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                                ransacReprojThreshold=2, maxIters=1000)
    if matrix is None or inliers.mean() < 0.75:
        return None
    scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    if not 0.85 <= scale <= 1.2 or matrix[0, 0] <= 0 or abs(matrix[0, 1]) > 0.04:
        return None
    selected = inliers.ravel().astype(bool)
    shift = np.median(target[selected] - source[selected], axis=0)
    # Prefer lossless integer registration when the features support it.
    if np.max(np.abs(target[selected] - source[selected] - np.round(shift))) < 0.75:
        matrix = np.float32([[1, 0, round(shift[0])], [0, 1, round(shift[1])]])
    return matrix


def _changed(discrepancy, mask):
    import cv2
    import numpy as np

    changed = (discrepancy > SOURCE_STATE_SETTINGS["detail_noise"]) & mask
    window = SOURCE_STATE_SETTINGS["local_window"]
    local = cv2.boxFilter(changed.astype(np.float32), -1, (window, window),
                          normalize=False, borderType=cv2.BORDER_CONSTANT).max()
    return bool((discrepancy[mask] >= SOURCE_STATE_SETTINGS["maximum_detail_change"]).any()
                or changed.sum() >= max(1, mask.sum()) * SOURCE_STATE_SETTINGS[
                    "maximum_changed_fraction"]
                or local >= SOURCE_STATE_SETTINGS["maximum_local_changed_pixels"])


def _equivalent(left, right, matrix=None, *, check_reverse=True):
    """Check both directions at native resolution, including content outside overlap."""
    import cv2
    import numpy as np

    first, second = left["image"], right["image"]
    native_right = right
    if matrix is None:
        if first.shape != second.shape:
            return False
        common = np.ones(first.shape[:2], bool)
    else:
        size = first.shape[1], first.shape[0]
        common = cv2.warpAffine(np.ones(second.shape[:2], np.float32), matrix, size) >= 1
        reverse = cv2.warpAffine(np.ones(first.shape[:2], np.float32),
                                 cv2.invertAffineTransform(matrix),
                                 (second.shape[1], second.shape[0])) >= 1
        if min(common.mean(), reverse.mean()) < 0.6:
            return False
        # A matching overlap is insufficient if a label/panel was clipped off.
        for prepared, visible in ((left, common), (right, reverse)):
            if _changed(np.abs(prepared["detail"]), ~visible):
                return False
        right = {key: cv2.warpAffine(value, matrix, size) for key, value in right.items()}
        second = right["image"]
    discrepancy = np.maximum.reduce([
        right["detail"] - left["high"], left["low"] - right["detail"],
        left["detail"] - right["high"], right["low"] - left["detail"],
    ])
    if _changed(discrepancy, common):
        return False
    # Local contrast alone could hide changes in filled panels or figure colors.
    color_delta = left["color"] - right["color"]
    color_delta -= np.median(color_delta[common], axis=0)
    colored = (np.max(np.abs(color_delta), axis=2) > 32) & common
    local_color = cv2.boxFilter(colored.astype(np.float32), -1, (16, 16),
                                normalize=False, borderType=cv2.BORDER_CONSTANT).max()
    if (colored[common].mean() >= SOURCE_STATE_SETTINGS["maximum_color_change_fraction"]
            or local_color >= 16):
        return False
    if matrix is not None and check_reverse:
        integer_translation = (np.array_equal(matrix[:, :2], np.eye(2))
                               and np.array_equal(matrix[:, 2], np.round(matrix[:, 2])))
        if not integer_translation:
            # Interpolation can weaken new detail; inspect each image on its own pixel grid.
            return _equivalent(native_right, left, cv2.invertAffineTransform(matrix),
                               check_reverse=False)
    return True


def group_source_states(observations, sources, *, contexts=None):
    """Preserve all IDs and verify each member directly against its final representative."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {identifier: [identifier] for identifier in sources}

    contexts = contexts or {}
    pixels = lru_cache(maxsize=4)(lambda identifier: _source_image(identifier, sources, contexts))
    prepared = lru_cache(maxsize=4)(lambda identifier: _prepare(pixels(identifier)))
    features = lru_cache(maxsize=16)(lambda identifier: _features(pixels(identifier)))
    quality, hashes, thumbnails = {}, {}, {}
    for identifier, path in sources.items():
        context = contexts.get(identifier)
        hashes[identifier] = (fingerprint((context["frame_sha256"], context["region"]))
                              if context else sha256_file(path))
        gray = cv2.cvtColor(pixels(identifier), cv2.COLOR_BGR2GRAY)
        quality[identifier] = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        thumbnail = cv2.resize(gray, (32, 32)).astype(np.float32)
        thumbnails[identifier] = (thumbnail - thumbnail.mean()) / max(1, thumbnail.std())

    def equivalent(first, second):
        if hashes[first] == hashes[second]:
            return True
        if (pixels(first).shape == pixels(second).shape
                and _equivalent(prepared(first), prepared(second))):
            return True
        matrix = _alignment(features(first), features(second))
        if (matrix is not None and pixels(first).shape == pixels(second).shape
                and np.array_equal(matrix, [[1, 0, 0], [0, 1, 0]])):
            return False  # The unregistered comparison already checked these pixels.
        return matrix is not None and _equivalent(prepared(first), prepared(second), matrix)

    ordered = sorted(sources, key=lambda identifier: (observations[identifier]["timestamp"], identifier))
    groups, latest = {}, {}
    for identifier in ordered:
        observation = observations[identifier]
        representative = identifier
        if (quality[identifier] >= SOURCE_STATE_SETTINGS["minimum_sharpness"]
                and observation["status"] == "complete"
                and observation["source_kind"] != "unknown"):
            nearby = [other for other in groups
                      if observation["timestamp"] - latest[other] <= SOURCE_STATE_SETTINGS["nearby_seconds"]
                      and quality[other] >= SOURCE_STATE_SETTINGS["minimum_sharpness"]
                      and observations[other]["status"] == "complete"
                      and observations[other]["source_kind"] == observation["source_kind"]
                      and all(observations[member]["visual_id"] != observation["visual_id"]
                              for member in groups[other])]
            nearby.sort(key=lambda other: float(np.mean(
                (thumbnails[identifier] - thumbnails[other]) ** 2)))
            for other in nearby[:SOURCE_STATE_SETTINGS["maximum_candidates"]]:
                if equivalent(other, identifier):
                    representative = other
                    break
        groups.setdefault(representative, []).append(identifier)
        latest[representative] = observation["timestamp"]

    result = {}
    for representative, members in groups.items():
        clearest = max(members, key=lambda identifier: quality[identifier])
        if clearest == representative:
            result[representative] = members
            continue
        result[clearest] = []
        for identifier in members:
            # Switching to a sharper member must not introduce similarity chains.
            if identifier == clearest or equivalent(clearest, identifier):
                result[clearest].append(identifier)
            else:
                result[identifier] = [identifier]
    return result
