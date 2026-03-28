from __future__ import annotations

from src.solution import average


def test_average_returns_float() -> None:
    assert average([2.0, 4.0, 6.0]) == 4.0
