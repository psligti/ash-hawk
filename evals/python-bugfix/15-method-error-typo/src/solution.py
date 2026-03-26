from __future__ import annotations


def clean_title(title: str) -> str:
    trim = getattr(title, "strp")
    return trim().title()
