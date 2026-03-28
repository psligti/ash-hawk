from __future__ import annotations

from src.solution import safe_divide


def test_safe_divide_handles_zero() -> None:
    assert safe_divide(10.0, 0.0) is None
