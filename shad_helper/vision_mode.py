"""Keep each extraction job's visual grouping policy fixed across resumes."""

import math
from pathlib import Path

from .jobs import read_json, write_json

GROUPING_VERSION = 1
PAGE_SELECTION_VERSION = 1
SPARSE_SELECTION_VERSION = 1
VISION_MODES = ("pages", "content", "baseline", "sparse")


def _new_job_mode():
    profile = Path(".shad-helper/vision-profile.json")
    if profile.exists():
        value = read_json(profile)
        if (value.get("version") == 1 and value.get("qualified") is True
                and value.get("sparse_default") is True):
            return "sparse"
    return "pages"


def resolve_vision_mode(out: Path, requested: str | None = None, *,
                        vision_budget_minutes: float | None = None,
                        clip_seconds: float | None = None) -> dict:
    """Default new jobs to pages mode while preserving each saved or legacy policy."""
    if requested is not None and requested not in VISION_MODES:
        raise ValueError("Vision mode must be 'pages', 'content', 'baseline' or 'sparse'.")
    if vision_budget_minutes is not None and (
            isinstance(vision_budget_minutes, bool)
            or not isinstance(vision_budget_minutes, (int, float))
            or not math.isfinite(vision_budget_minutes) or vision_budget_minutes <= 0):
        raise ValueError("Vision budget must be a positive finite number of minutes.")
    out = Path(out)
    config_path = out / "extraction-config.json"
    if config_path.exists():
        try:
            config = read_json(config_path)
        except ValueError as error:
            raise ValueError("Invalid extraction-config.json; use a new output directory.") from error
        if (not isinstance(config, dict) or config.get("vision_mode") not in VISION_MODES
                or type(config.get("grouping_version")) is not int
                or config["grouping_version"] != GROUPING_VERSION):
            raise ValueError("Unsupported extraction configuration or grouping version; "
                             "use a new output directory.")
        mode = config["vision_mode"]
        if (mode == "pages" and (type(config.get("page_selection_version")) is not int
                                or config["page_selection_version"] != PAGE_SELECTION_VERSION)):
            raise ValueError("Missing or unsupported page selection version; "
                             "use a new output directory.")
        if mode == "sparse" and (
                type(config.get("sparse_selection_version")) is not int
                or config["sparse_selection_version"] != SPARSE_SELECTION_VERSION
                or isinstance(config.get("vision_budget_minutes"), bool)
                or not isinstance(config.get("vision_budget_minutes"), (int, float))
                or not math.isfinite(config["vision_budget_minutes"])
                or config["vision_budget_minutes"] <= 0):
            raise ValueError("Unsupported sparse configuration; use a new output directory.")
    else:
        legacy = any(path.is_file() for path in out.rglob("*"))
        mode = "baseline" if legacy else requested or _new_job_mode()
        config = {"vision_mode": mode, "grouping_version": GROUPING_VERSION}
        if mode == "pages":
            config["page_selection_version"] = PAGE_SELECTION_VERSION
        if mode == "sparse":
            if clip_seconds is None or not math.isfinite(clip_seconds) or clip_seconds <= 0:
                raise ValueError("Sparse mode requires a positive clip duration.")
            config.update(sparse_selection_version=SPARSE_SELECTION_VERSION,
                          vision_budget_minutes=vision_budget_minutes or clip_seconds / 120)
        if requested is not None and requested != mode:
            raise ValueError("Existing jobs keep their vision mode; use a new output directory.")
        if vision_budget_minutes is not None and mode != "sparse":
            raise ValueError("--vision-budget-minutes requires sparse mode.")
        if not legacy:
            write_json(config_path, config)
    if requested is not None and requested != mode:
        raise ValueError("Existing jobs keep their vision mode; use a new output directory.")
    if vision_budget_minutes is not None and mode != "sparse":
        raise ValueError("--vision-budget-minutes requires sparse mode.")
    if (mode == "sparse" and vision_budget_minutes is not None
            and config["vision_budget_minutes"] != vision_budget_minutes):
        config["vision_budget_minutes"] = vision_budget_minutes
        write_json(config_path, config)
    sanitized = {"vision_mode": mode, "grouping_version": GROUPING_VERSION}
    if mode == "pages":
        sanitized["page_selection_version"] = PAGE_SELECTION_VERSION
    if mode == "sparse":
        sanitized.update(sparse_selection_version=SPARSE_SELECTION_VERSION,
                         vision_budget_minutes=config["vision_budget_minutes"])
    return sanitized
