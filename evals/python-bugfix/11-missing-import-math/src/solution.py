from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import math


def hypotenuse(a: float, b: float) -> float:
    return math.sqrt(a * a + b * b)
