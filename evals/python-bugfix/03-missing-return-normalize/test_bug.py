from __future__ import annotations

from src.solution import normalize_name


def test_normalize_name_returns_string() -> None:
    assert normalize_name("  ada lovelace ") == "Ada Lovelace"
