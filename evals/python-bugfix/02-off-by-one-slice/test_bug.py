from __future__ import annotations

from src.solution import tail_items


def test_tail_items_exact_count() -> None:
    assert tail_items([1, 2, 3, 4, 5], 2) == [4, 5]
