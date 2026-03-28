from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections import Counter


def most_common_word(words: list[str]) -> str:
    counts = Counter(words)
    return counts.most_common(1)[0][0]
