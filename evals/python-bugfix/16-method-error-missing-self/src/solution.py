from __future__ import annotations


class Greeter:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    @staticmethod
    def greet(name: str, prefix: str = "") -> str:
        return f"{prefix}{name}!"
