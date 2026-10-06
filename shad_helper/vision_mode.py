"""Keep each extraction job's visual grouping policy fixed across resumes."""

from pathlib import Path

from .jobs import read_json, write_json

GROUPING_VERSION = 1
PAGE_SELECTION_VERSION = 1
VISION_MODES = ("pages", "content", "baseline")


def resolve_vision_mode(out: Path, requested: str | None = None) -> dict:
    """Default new jobs to pages mode while preserving each saved or legacy policy."""
    if requested is not None and requested not in VISION_MODES:
        raise ValueError("Vision mode must be 'pages', 'content' or 'baseline'.")
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
    else:
        legacy = any(path.is_file() for path in out.rglob("*"))
        mode = "baseline" if legacy else requested or "pages"
        config = {"vision_mode": mode, "grouping_version": GROUPING_VERSION}
        if mode == "pages":
            config["page_selection_version"] = PAGE_SELECTION_VERSION
        if requested is not None and requested != mode:
            raise ValueError("Existing jobs keep their vision mode; use a new output directory.")
        if not legacy:
            write_json(config_path, config)
    if requested is not None and requested != mode:
        raise ValueError("Existing jobs keep their vision mode; use a new output directory.")
    sanitized = {"vision_mode": mode, "grouping_version": GROUPING_VERSION}
    if mode == "pages":
        sanitized["page_selection_version"] = PAGE_SELECTION_VERSION
    return sanitized
