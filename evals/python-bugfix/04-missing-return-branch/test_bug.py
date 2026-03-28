from __future__ import annotations

from src.solution import discounted_total


def test_discounted_total_returns_value() -> None:
    assert discounted_total(100.0, 0.1) == 90.0
