from __future__ import annotations

import json
from pathlib import PurePath


def tool_event_preview(tool_name: str | None, tool_input: object, result: object) -> str | None:
    action_preview = tool_action_preview(tool_name, tool_input)
    if action_preview:
        return action_preview
    return _result_preview(result)


def tool_action_preview(tool_name: str | None, tool_input: object) -> str | None:
    normalized_tool = tool_name.strip().lower() if isinstance(tool_name, str) else ""
    if not isinstance(tool_input, dict):
        return _preview_scalar(tool_input)

    if normalized_tool in {"bash", "test"}:
        return _preview_text(tool_input.get("command")) or _preview_text(tool_input.get("args"))

    if normalized_tool == "glob":
        return _pattern_location_preview(tool_input, pattern_key="pattern")

    if normalized_tool == "grep":
        return _pattern_location_preview(
            tool_input, pattern_key="pattern"
        ) or _pattern_location_preview(tool_input, pattern_key="query")

    if normalized_tool in {"read", "write", "edit"}:
        return _path_preview(tool_input)

    if normalized_tool == "todoread":
        return "todo list"

    if normalized_tool == "todowrite":
        todos = tool_input.get("todos")
        if isinstance(todos, list):
            count = len(todos)
            return "1 todo item" if count == 1 else f"{count} todo items"
        return "todo update"

    if normalized_tool == "search_knowledge":
        return _preview_text(tool_input.get("query"))

    return _generic_input_preview(tool_input)


def _result_preview(result: object) -> str | None:
    if isinstance(result, dict):
        for key in ("output", "result", "message", "stdout", "stderr"):
            preview = _preview_scalar(result.get(key))
            if preview:
                return preview
        return _generic_input_preview(result)
    return _preview_scalar(result)


def _pattern_location_preview(tool_input: dict[object, object], *, pattern_key: str) -> str | None:
    pattern = _preview_text(tool_input.get(pattern_key))
    location = _preview_text(
        tool_input.get("path")
        or tool_input.get("dir_path")
        or tool_input.get("root")
        or tool_input.get("working_dir")
    )
    if pattern and location:
        return f"{pattern} in {_compact_path(location)}"
    if pattern:
        return pattern
    if location:
        return _compact_path(location)
    return None


def _path_preview(tool_input: dict[object, object]) -> str | None:
    for key in ("path", "filePath", "file_path", "target_file"):
        raw_value = tool_input.get(key)
        if isinstance(raw_value, str) and raw_value.strip():
            return _compact_path(raw_value)
    return None


def _generic_input_preview(tool_input: dict[object, object]) -> str | None:
    for key in ("path", "filePath", "file_path", "target_file", "pattern", "query", "command"):
        raw_value = tool_input.get(key)
        preview = _preview_scalar(raw_value)
        if preview:
            if key in {"path", "filePath", "file_path", "target_file"}:
                return _compact_path(preview)
            return preview

    compact_items: list[str] = []
    for raw_key in sorted(tool_input.keys(), key=str)[:3]:
        preview = _preview_scalar(tool_input.get(raw_key))
        if not preview:
            continue
        key = str(raw_key)
        if key in {"path", "filePath", "file_path", "target_file"}:
            preview = _compact_path(preview)
        compact_items.append(f"{key}={preview}")
    if compact_items:
        return "; ".join(compact_items)
    return None


def _preview_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    if not normalized:
        return None
    return normalized[:140]


def _preview_scalar(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _preview_text(value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        rendered_items = [item for item in (_preview_scalar(item) for item in value[:3]) if item]
        if not rendered_items:
            return None
        suffix = " ..." if len(value) > 3 else ""
        return ", ".join(rendered_items) + suffix
    if isinstance(value, dict):
        return _generic_input_preview(value)
    compact = json.dumps(value, ensure_ascii=True, default=str)
    return compact[:140] if compact else None


def _compact_path(value: str) -> str:
    if "://" in value:
        return value
    path = PurePath(value)
    parts = [part for part in path.parts if part not in {"/", ""}]
    if len(parts) <= 3:
        return "/".join(parts) if parts else value
    return "/".join(parts[-3:])


__all__ = ["tool_action_preview", "tool_event_preview"]
