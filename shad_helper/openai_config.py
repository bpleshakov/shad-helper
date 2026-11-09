"""Read the optional API key without importing it into the process environment."""

import os
import shlex
from pathlib import Path

API_KEY_FILE = Path("config/openai.env")


def load_api_key() -> str | None:
    if "OPENAI_API_KEY" in os.environ:
        return os.environ["OPENAI_API_KEY"].strip() or None
    try:
        contents = API_KEY_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError):
        raise ValueError("Cannot read the OpenAI API-key configuration in config/openai.env.") from None
    key = None
    for line in contents.splitlines():
        name, separator, value = line.partition("=")
        if not separator or name.split() not in (["OPENAI_API_KEY"], ["export", "OPENAI_API_KEY"]):
            continue
        try:
            values = shlex.split(value, comments=True)
            if len(values) > 1:
                raise ValueError
        except ValueError:
            raise ValueError("Invalid OPENAI_API_KEY assignment in config/openai.env.") from None
        key = values[0].strip() if values else None
    return key or None
