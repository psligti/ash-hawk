from __future__ import annotations


def contains_id(ids: list[int], target: int) -> bool:
    return any(item is target for item in ids)
