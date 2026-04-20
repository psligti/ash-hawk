# type-hygiene: skip-file
from __future__ import annotations

import json
from typing import Any

from ash_hawk.thin_runtime.memory import ThinRuntimeMemoryManager
from ash_hawk.thin_runtime.persistence import ThinRuntimePersistence

DEFERRED_SCOPES = {"episodic_memory", "semantic_memory", "personal_memory", "artifact_memory"}


class DreamStateConsolidator:
    def __init__(
        self,
        *,
        memory: ThinRuntimeMemoryManager,
        persistence: ThinRuntimePersistence,
    ) -> None:
        self.memory = memory
        self.persistence = persistence

    def run(self) -> dict[str, Any]:
        queue = self.persistence.load_dream_queue()
        applied = 0
        for item in queue:
            scope = item.get("scope")
            if not isinstance(scope, str) or scope not in DEFERRED_SCOPES:
                continue
            key = item.get("key")
            if not isinstance(key, str):
                continue
            value = self._resolve_queue_value(item)
            if value is None:
                continue
            self.memory.append(scope, key, value)
            applied += 1

        self.memory.write_scope("session_memory", {"dream_queue": []})

        full_snapshot = self.memory.snapshot()
        self.persistence.save_memory_snapshot(full_snapshot)
        self.persistence.clear_dream_queue()
        return {
            "applied": applied,
            "remaining": 0,
            "scopes": sorted(DEFERRED_SCOPES),
        }

    def _resolve_queue_value(self, item: dict[str, Any]) -> Any | None:
        if "value" in item:
            return item["value"]
        raw_value = item.get("value_json")
        if not isinstance(raw_value, str):
            return None
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            return raw_value
