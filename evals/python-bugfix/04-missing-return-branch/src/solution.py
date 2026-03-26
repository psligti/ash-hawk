from __future__ import annotations


def discounted_total(amount: float, discount: float) -> float | None:
    if discount <= 0:
        return amount
    if discount >= 1:
        return 0.0
    total = amount * (1 - discount)
    if total < 0:
        return 0.0
