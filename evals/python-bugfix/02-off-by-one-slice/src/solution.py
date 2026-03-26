from __future__ import annotations


def tail_items(items: list[int], n: int) -> list[int]:
    if n <= 0:
        return []
    return items[-(n + 1) :]
