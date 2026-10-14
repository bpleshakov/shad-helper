import pytest

from shad_helper import rendering


@pytest.fixture(autouse=True)
def chapter_latex_checker(monkeypatch):
    """Isolate model orchestration tests from TeX; expose the real checker for its tests."""
    checker = rendering.validate_chapter_latex
    monkeypatch.setattr(rendering, "validate_chapter_latex", lambda *args, **kwargs: None)
    return checker
