# type-hygiene: skip-file
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

from ash_hawk.thin_runtime.models import ThinRuntimeExecutionResult


class ThinRuntimePersistence:
    def __init__(self, storage_root: Path | None = None) -> None:
        self._storage_root = (storage_root or Path(".ash-hawk") / "thin_runtime").resolve()
        self._runs_root = self._storage_root / "runs"
        self._memory_root = self._storage_root / "memory"
        self._runs_root.mkdir(parents=True, exist_ok=True)
        self._memory_root.mkdir(parents=True, exist_ok=True)

    @property
    def storage_root(self) -> Path:
        return self._storage_root

    def run_dir(self, run_id: str) -> Path:
        return self._runs_root / run_id

    def execution_file(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "execution.json"

    def summary_file(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "summary.json"

    def memory_file(self) -> Path:
        return self._memory_root / "snapshot.json"

    def session_file(self) -> Path:
        return self._memory_root / "session_snapshot.json"

    def dream_queue_file(self) -> Path:
        return self._memory_root / "dream_queue.json"

    def load_memory_snapshot(self) -> dict[str, dict[str, Any]]:
        path = self.memory_file()
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        typed_raw = cast(dict[str, Any], raw)
        snapshot: dict[str, dict[str, Any]] = {}
        for key, value in typed_raw.items():
            if isinstance(value, dict):
                snapshot[key] = dict(cast(dict[str, Any], value))
        return snapshot

    def save_memory_snapshot(self, snapshot: dict[str, dict[str, Any]]) -> Path:
        path = self.memory_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(path, snapshot)
        return path

    def save_session_snapshot(self, snapshot: dict[str, dict[str, Any]]) -> Path:
        session_only = {
            "working_memory": snapshot.get("working_memory", {}),
            "session_memory": snapshot.get("session_memory", {}),
        }
        path = self.session_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(path, session_only)
        return path

    def load_dream_queue(self) -> list[dict[str, Any]]:
        path = self.dream_queue_file()
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return []
        typed_raw = cast(list[object], raw)
        return [dict(cast(dict[str, Any], item)) for item in typed_raw if isinstance(item, dict)]

    def save_dream_queue(self, queue: list[dict[str, Any]]) -> Path:
        path = self.dream_queue_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(path, queue)
        return path

    def clear_dream_queue(self) -> None:
        path = self.dream_queue_file()
        if path.exists():
            path.unlink()

    def persist_execution(self, execution: ThinRuntimeExecutionResult) -> Path:
        run_dir = self.run_dir(execution.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        execution_path = self.execution_file(execution.run_id)
        self._atomic_write_text(execution_path, execution.model_dump_json(indent=2))
        summary_path = self.summary_file(execution.run_id)
        summary = {
            "run_id": execution.run_id,
            "goal_id": execution.goal.goal_id,
            "agent": execution.agent.name,
            "selected_tool_names": execution.selected_tool_names,
            "success": execution.success,
            "error": execution.error,
            "delegations": [record.model_dump() for record in execution.delegations],
        }
        self._atomic_write_json(summary_path, summary)
        return run_dir

    def _atomic_write_json(self, path: Path, value: object) -> None:
        self._atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True))

    def _atomic_write_text(self, path: Path, text: str) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
