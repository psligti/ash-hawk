# type-hygiene: skip-file
from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

from ash_hawk.thin_runtime.models import MemoryScopeSpec


class ThinRuntimeMemoryManager:
    def __init__(self, scopes: list[MemoryScopeSpec]) -> None:
        self._scopes = {scope.name: scope for scope in scopes}
        self._memory: dict[str, dict[str, Any]] = {scope.name: {} for scope in scopes}

    def read_scope(self, name: str) -> dict[str, Any]:
        self._require_scope(name)
        return deepcopy(self._memory[name])

    def write_scope(self, name: str, values: dict[str, Any], *, actor: str | None = None) -> None:
        self._require_scope(name)
        self._require_writable(name, actor)
        self._memory[name].update(values)

    def append(self, name: str, key: str, value: Any, *, actor: str | None = None) -> None:
        self._require_scope(name)
        self._require_writable(name, actor)
        existing_any: Any = self._memory[name].setdefault(key, [])
        if not isinstance(existing_any, list):
            raise ValueError(f"Memory key '{key}' in scope '{name}' is not appendable")
        typed_existing = cast(list[object], existing_any)
        typed_existing.append(value)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self._memory)

    def can_write_scope(self, name: str, actor: str) -> bool:
        self._require_scope(name)
        scope = self._scopes[name]
        return not scope.writable_by or actor in scope.writable_by

    def hydrate_defaults(self) -> None:
        self.write_scope("working_memory", {"active_hypotheses": [], "phase_status": {}})
        self.write_scope(
            "session_memory",
            {
                "delegations": [],
                "retries": [],
                "validations": [],
                "traces": [],
                "transcripts": [],
                "dream_queue": [],
            },
        )
        self.write_scope("episodic_memory", {"episodes": []})
        self.write_scope("semantic_memory", {"rules": [], "boosts": [], "penalties": []})
        self.write_scope("personal_memory", {"preferences": []})
        self.write_scope("artifact_memory", {"artifacts": [], "events": [], "transcripts": []})

    def _require_scope(self, name: str) -> None:
        if name not in self._scopes:
            raise ValueError(f"Unknown memory scope: {name}")

    def _require_writable(self, name: str, actor: str | None) -> None:
        if actor is None:
            return
        scope = self._scopes[name]
        if scope.writable_by and actor not in scope.writable_by:
            raise ValueError(f"Actor '{actor}' cannot write memory scope '{name}'")
