from __future__ import annotations


def filter_even(values: list[int]) -> list[int]:
    return [value for value in values if value % 2 == 1]
