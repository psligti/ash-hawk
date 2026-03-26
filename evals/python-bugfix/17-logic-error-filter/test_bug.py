from __future__ import annotations

from src.solution import filter_even


def test_filter_even_values() -> None:
    assert filter_even([1, 2, 3, 4, 5, 6]) == [2, 4, 6]
