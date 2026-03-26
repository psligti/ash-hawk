from __future__ import annotations

from src.solution import clean_title


def test_clean_title_strips() -> None:
    assert clean_title("  the hobbit  ") == "The Hobbit"
