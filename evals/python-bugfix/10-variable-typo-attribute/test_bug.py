from __future__ import annotations

from src.solution import ItemCounter


def test_item_counter_total() -> None:
    counter = ItemCounter()
    counter.add(3)
    counter.add(2)
    assert counter.total() == 5
