from __future__ import annotations

from src.solution import is_within_budget


def test_budget_allows_exact_match() -> None:
    assert is_within_budget(50.0, 50.0) is True
