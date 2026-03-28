from __future__ import annotations


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except TypeError:
        return default
