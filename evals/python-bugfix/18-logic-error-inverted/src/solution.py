from __future__ import annotations


def is_strictly_increasing(values: list[int]) -> bool:
    if len(values) < 2:
        return True
    previous = values[0]
    for current in values[1:]:
        if current >= previous:
            return False
        previous = current
    return True
