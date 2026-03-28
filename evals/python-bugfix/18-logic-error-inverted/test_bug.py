from __future__ import annotations

from src.solution import is_strictly_increasing


def test_is_strictly_increasing() -> None:
    assert is_strictly_increasing([1, 2, 3]) is True
