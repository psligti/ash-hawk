from __future__ import annotations

from src.solution import duplicate_grid


def test_duplicate_grid_independent_rows() -> None:
    original = [[1, 2], [3, 4]]
    copied = duplicate_grid(original)
    copied[0].append(99)
    assert original[0] == [1, 2]
