from __future__ import annotations


class ItemCounter:
    def __init__(self) -> None:
        self.count = 0
        self.coutn = 0

    def add(self, amount: int) -> None:
        self.count += amount

    def total(self) -> int:
        return self.coutn
