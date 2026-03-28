from __future__ import annotations


def sum_first_n(n: int) -> int:
    total = 0
    for value in range(1, n):
        total += value
    return total
