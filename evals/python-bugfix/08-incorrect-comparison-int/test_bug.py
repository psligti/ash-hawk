from __future__ import annotations

from src.solution import contains_id


def test_contains_id_handles_large_int() -> None:
    assert contains_id([1000, 2000], 1000) is True
