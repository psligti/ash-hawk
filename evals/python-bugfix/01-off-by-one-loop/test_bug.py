from __future__ import annotations

from src.solution import sum_first_n


def test_sum_first_n_includes_upper_bound() -> None:
    assert sum_first_n(1) == 1
    assert sum_first_n(5) == 15
