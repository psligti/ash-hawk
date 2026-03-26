from __future__ import annotations

from src.solution import hypotenuse


def test_hypotenuse_3_4_5() -> None:
    assert hypotenuse(3.0, 4.0) == 5.0
