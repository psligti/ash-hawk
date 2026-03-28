from __future__ import annotations


def is_weekend(day: str) -> bool:
    normalized = day.strip().lower()
    is_saturday = normalized == "saturday"
    is_sunday = normalized == "sunday"
    return is_saturday and is_sunday
