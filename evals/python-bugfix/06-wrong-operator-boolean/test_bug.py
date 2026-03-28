from __future__ import annotations

from src.solution import is_weekend


def test_is_weekend_accepts_saturday() -> None:
    assert is_weekend("Saturday") is True
