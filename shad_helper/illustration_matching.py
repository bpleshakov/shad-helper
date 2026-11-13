"""Local visual retrieval signals; these never authorize a figure merge."""

from functools import lru_cache

from .jobs import sha256_file

MATCHING_SETTINGS = {
    "version": 1,
    "maximum_candidates": 3,
    "feature_pixels": 800,
    "maximum_features": 600,
    "minimum_features": 30,
    "descriptor_ratio": 0.85,
    "minimum_inlier_fraction": 0.04,
    "minimum_appearance": 0.5,
    "recent_verified_sources": 3,
    "strong_match_fraction": 0.1,
    "winner_margin": 0.02,
}


class VisualMatcher:
    def __init__(self, resolve):
        self.resolve = resolve
        self.features = lru_cache(maxsize=256)(self._features)
        self.compare = lru_cache(maxsize=4096)(self._compare)
        self.examples = {}
        try:
            import cv2
            import numpy as np
        except ImportError:
            cv2 = np = None
        self.cv2, self.np = cv2, np

    def _features(self, identifier, crop_id):
        cv2, np = self.cv2, self.np
        path = self.resolve(identifier, crop_id)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        scale = min(1, MATCHING_SETTINGS["feature_pixels"] / max(image.shape))
        if min(image.shape) * min(scale, 160 / max(image.shape)) < 2:
            return None
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        points, descriptors = cv2.SIFT_create(
            nfeatures=MATCHING_SETTINGS["maximum_features"]).detectAndCompute(image, None)
        thumbnail = cv2.resize(image, None, fx=160 / max(image.shape),
                               fy=160 / max(image.shape), interpolation=cv2.INTER_AREA)
        return (np.float32([point.pt for point in points]).reshape(-1, 2), descriptors,
                thumbnail, sha256_file(path))

    def _appearance(self, left, right):
        """Allow a partial view inside a larger figure, in either direction."""
        cv2 = self.cv2
        best = 0.0
        for source, target in ((left, right), (right, left)):
            scale = min(target.shape[0] / source.shape[0], target.shape[1] / source.shape[1])
            for fraction in (1, 0.85, 0.7, 0.55, 0.4):
                if min(round(size * scale * fraction) for size in source.shape) < 12:
                    continue
                template = cv2.resize(source, None, fx=scale * fraction, fy=scale * fraction,
                                      interpolation=cv2.INTER_AREA)
                best = max(best, float(cv2.matchTemplate(
                    target, template, cv2.TM_CCOEFF_NORMED).max()))
        return best

    def _compare(self, left, right):
        cv2, np = self.cv2, self.np
        if cv2 is None:
            return None
        first, second = self.features(*left), self.features(*right)
        if first is None or second is None:
            return None
        points, descriptors, thumbnail, digest = first
        other_points, other_descriptors, other_thumbnail, other_digest = second
        if digest == other_digest:
            return {"inlier_fraction": 1.0, "appearance": 1.0, "identical": True}
        count = min(len(points), len(other_points))
        if count < MATCHING_SETTINGS["minimum_features"]:
            return None
        pairs = [first for first, second in cv2.BFMatcher().knnMatch(
            descriptors, other_descriptors, k=2)
            if first.distance < MATCHING_SETTINGS["descriptor_ratio"] * second.distance]
        fraction = 0.0
        if len(pairs) >= 4:
            source = np.float32([points[pair.queryIdx] for pair in pairs])
            target = np.float32([other_points[pair.trainIdx] for pair in pairs])
            matrix, inliers = cv2.estimateAffinePartial2D(
                source, target, method=cv2.RANSAC, ransacReprojThreshold=3, maxIters=1000)
            if matrix is not None:
                fraction = len({pair.trainIdx for pair, keep in zip(pairs, inliers.ravel())
                                if keep}) / count
        return {"inlier_fraction": fraction, "appearance": self._appearance(
            thumbnail, other_thumbnail), "identical": False}

    def score(self, current, candidate):
        source, target = [(block["illustration_id"], block.get("illustration_crop_id", ""))
                          for block in (current, candidate)]
        direct = self.compare(source, target)
        if direct is None:
            return None
        scores = [direct, *[score for example in self.examples.get(target, [])
                           if (score := self.compare(source, example)) is not None]]
        return {**direct, "inlier_fraction": max(score["inlier_fraction"] for score in scores),
                "appearance": max(score["appearance"] for score in scores),
                "verified_sources": self.examples.get(target, [])[:]}

    def remember(self, source, target):
        """Track only sources already accepted for this exact target crop."""
        source, target = [(block["illustration_id"], block.get("illustration_crop_id", ""))
                          for block in (source, target)]
        examples = self.examples.setdefault(target, [])
        if source != target and source not in examples:
            examples.append(source)
            del examples[:-MATCHING_SETTINGS["recent_verified_sources"]]


def visually_plausible(score):
    return (score is None
            or score["inlier_fraction"] >= MATCHING_SETTINGS["minimum_inlier_fraction"]
            or score["appearance"] >= MATCHING_SETTINGS["minimum_appearance"])


def candidate_limit(scores, *, broad=False):
    """Spend one comparison on a visual leader; keep ambiguity and recovery broad."""
    if not broad and len(scores) > 1 and all(score is not None for score in scores):
        first, second = (score["inlier_fraction"] for score in scores[:2])
        if (first >= MATCHING_SETTINGS["strong_match_fraction"]
                and first - second >= MATCHING_SETTINGS["winner_margin"]):
            return 1
    return MATCHING_SETTINGS["maximum_candidates"]
