from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class RoleRegistry(Generic[T]):
    def __init__(self) -> None:
        self._roles: dict[str, T] = {}

    def register(self, name: str, role: T) -> None:
        self._roles[name] = role

    def get(self, name: str) -> T:
        return self._roles[name]

    def has(self, name: str) -> bool:
        return name in self._roles

    def names(self) -> list[str]:
        return sorted(self._roles.keys())
