from __future__ import annotations


def add_tag(tag: str, tags: list[str] = []) -> list[str]:
    tags.append(tag)
    return tags
