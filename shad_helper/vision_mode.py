"""Keep each extraction job's visual grouping policy fixed across resumes."""

from pathlib import Path

from .jobs import read_json, write_json

GROUPING_VERSION = 1
VISION_MODES = ("content", "baseline")


def resolve_vision_mode(out: Path, requested: str | None = None) -> dict:
    """Default new jobs to content mode while preserving existing legacy jobs."""
    if requested is not None and requested not in VISION_MODES:
        raise ValueError("Vision mode must be 'content' or 'baseline'.")
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
    else:
        legacy = any(path.is_file() for path in out.rglob("*"))
        mode = "baseline" if legacy else requested or "content"
        config = {"vision_mode": mode, "grouping_version": GROUPING_VERSION}
        if requested is not None and requested != mode:
            raise ValueError("Existing jobs keep their vision mode; use a new output directory.")
        if not legacy:
            write_json(config_path, config)
    if requested is not None and requested != mode:
        raise ValueError("Existing jobs keep their vision mode; use a new output directory.")
    return {"vision_mode": mode, "grouping_version": GROUPING_VERSION}
