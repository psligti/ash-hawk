from __future__ import annotations

from src.solution import parse_int


def test_parse_int_invalid_string_returns_default() -> None:
    assert parse_int("not-a-number", default=7) == 7
