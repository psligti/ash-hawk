from __future__ import annotations

from src.solution import add_tag


def test_add_tag_returns_fresh_list() -> None:
    first = add_tag("a")
    second = add_tag("b")
    assert first == ["a"]
    assert second == ["b"]
