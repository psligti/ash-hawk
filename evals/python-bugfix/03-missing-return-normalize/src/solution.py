from __future__ import annotations


def normalize_name(name: str) -> str | None:
    cleaned = name.strip().title()
    if cleaned.endswith("!"):
        return cleaned
