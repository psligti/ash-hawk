from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import pydantic as pd


class ToolTimeoutError(RuntimeError):
    pass


class ToolingRecord(pd.BaseModel):
    tool_name: str
    tool_input: Any
    normalized_input: str
    result: dict[str, Any]

    model_config = pd.ConfigDict(extra="forbid")


MALFORMED_OUTPUT: dict[str, Any] = {"malformed": True}


def _normalize_input(value: Any) -> str:
    def _to_jsonable(item: Any) -> Any:
        if isinstance(item, pd.BaseModel):
            return _to_jsonable(item.model_dump())
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, dict):
            normalized: dict[str, Any] = {}
            for key, val in item.items():
                normalized[str(key)] = _to_jsonable(val)
            return normalized
        if isinstance(item, (list, tuple)):
            return [_to_jsonable(val) for val in item]
        if isinstance(item, set):
            return sorted(_to_jsonable(val) for val in item)
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        return repr(item)

    normalized = _to_jsonable(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _scenario_id_from_root(root: Path) -> str:
    if root.is_file():
        name = root.name
        if name.endswith(".scenario.yaml"):
            return name[: -len(".scenario.yaml")]
        if name.endswith(".scenario.yml"):
            return name[: -len(".scenario.yml")]
        return root.stem
    return root.name


class ToolingHarness:
    def __init__(self, mode: Literal["mock", "record", "replay"], root: Path) -> None:
        if mode not in {"mock", "record", "replay"}:
            raise ValueError(f"Unknown tooling mode: {mode}")
        self.mode = mode
        self.root = Path(root)
        self._mocks: dict[tuple[str, str], dict[str, Any]] = {}
        self._timeout_injections: dict[str, int] = {}
        self._exception_injections: dict[str, list[Exception]] = {}
        self._malformed_injections: dict[str, int] = {}

        scenario_id = _scenario_id_from_root(self.root)
        self._trace_path = self.root / "tool_mocks" / scenario_id / "trace.jsonl"
        self._replay_records: list[ToolingRecord] = []
        self._replay_index = 0

        if self.mode == "record":
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace_path.write_text("", encoding="utf-8")
        elif self.mode == "replay":
            if not self._trace_path.exists():
                raise FileNotFoundError(f"Tooling trace not found: {self._trace_path}")
            self._replay_records = self._load_records(self._trace_path)

    def register_mock(self, tool_name: str, normalized_input: Any, result: dict[str, Any]) -> None:
        key = (tool_name, _normalize_input(normalized_input))
        self._mocks[key] = deepcopy(result)

    def inject_timeout(self, tool_name: str) -> None:
        self._timeout_injections[tool_name] = self._timeout_injections.get(tool_name, 0) + 1

    def inject_exception(self, tool_name: str, exc: Exception) -> None:
        if not isinstance(exc, Exception):
            raise TypeError("Injected exception must be an Exception instance")
        self._exception_injections.setdefault(tool_name, []).append(exc)

    def inject_malformed(self, tool_name: str) -> None:
        self._malformed_injections[tool_name] = self._malformed_injections.get(tool_name, 0) + 1

    def call(self, tool_name: str, tool_input: Any) -> dict[str, Any]:
        if self._consume_timeout(tool_name):
            raise ToolTimeoutError(f"Tool {tool_name} timed out")
        injected_exc = self._consume_exception(tool_name)
        if injected_exc is not None:
            raise injected_exc
        if self._consume_malformed(tool_name):
            return deepcopy(MALFORMED_OUTPUT)

        normalized_input = _normalize_input(tool_input)

        if self.mode == "replay":
            if self._replay_index >= len(self._replay_records):
                raise ValueError("No more recorded tool calls to replay")
            record = self._replay_records[self._replay_index]
            self._replay_index += 1
            if record.tool_name != tool_name or record.normalized_input != normalized_input:
                raise ValueError("Tool call does not match recorded replay data")
            return deepcopy(record.result)

        key = (tool_name, normalized_input)
        if key not in self._mocks:
            raise KeyError(f"No mock registered for tool {tool_name} and input {normalized_input}")

        result = deepcopy(self._mocks[key])
        if self.mode == "record":
            record = ToolingRecord(
                tool_name=tool_name,
                tool_input=tool_input,
                normalized_input=normalized_input,
                result=result,
            )
            self._append_record(self._trace_path, record)
        return result

    def close(self) -> None:
        return None

    def _consume_timeout(self, tool_name: str) -> bool:
        count = self._timeout_injections.get(tool_name, 0)
        if count <= 0:
            return False
        if count == 1:
            self._timeout_injections.pop(tool_name, None)
        else:
            self._timeout_injections[tool_name] = count - 1
        return True

    def _consume_exception(self, tool_name: str) -> Exception | None:
        queue = self._exception_injections.get(tool_name)
        if not queue:
            return None
        exc = queue.pop(0)
        if not queue:
            self._exception_injections.pop(tool_name, None)
        return exc

    def _consume_malformed(self, tool_name: str) -> bool:
        count = self._malformed_injections.get(tool_name, 0)
        if count <= 0:
            return False
        if count == 1:
            self._malformed_injections.pop(tool_name, None)
        else:
            self._malformed_injections[tool_name] = count - 1
        return True

    @staticmethod
    def _append_record(path: Path, record: ToolingRecord) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump()))
            handle.write("\n")

    @staticmethod
    def _load_records(path: Path) -> list[ToolingRecord]:
        records: list[ToolingRecord] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError("Tooling trace JSONL line must be a JSON object")
                records.append(ToolingRecord.model_validate(payload))
        return records


__all__ = ["ToolTimeoutError", "ToolingHarness"]
