from __future__ import annotations

from src.solution import most_common_word


def test_most_common_word() -> None:
    assert most_common_word(["a", "b", "a", "c", "a"]) == "a"
