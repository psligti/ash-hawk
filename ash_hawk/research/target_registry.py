from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ash_hawk.research.types import TargetSurface

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ash_hawk.auto_research.types import TargetType


JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass
class TargetEntry:
    name: str
    surface: TargetSurface
    description: str
    mutation_count: int = 0
    success_count: int = 0
    correlation_with_improvement: float = 0.0
    last_mutated: str | None = None
    status: str = "active"

    @property
    def success_rate(self) -> float:
        return self.success_count / self.mutation_count if self.mutation_count > 0 else 0.0


class TargetRegistry:
    def __init__(self, storage_path: Path | None = None, min_active: int = 3) -> None:
        self._entries: dict[str, TargetEntry] = {}
        self._storage_path = storage_path or Path(".ash-hawk/research")
        self._min_active = min_active

    @classmethod
    def load(cls, storage_path: Path, min_active: int = 3) -> TargetRegistry:
        registry = cls(storage_path=storage_path, min_active=min_active)
        targets_path = storage_path / "targets.json"
        if not targets_path.exists():
            return registry

        try:
            data_raw = json.loads(targets_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.exception("Failed to parse target registry at %s", targets_path)
            return registry

        data = _coerce_json_dict(data_raw)
        entries = _coerce_json_dict(data.get("entries"))

        for name, payload in entries.items():
            if not isinstance(payload, dict):
                continue
            surface_value = payload.get("surface")
            if not isinstance(surface_value, str):
                logger.warning("Unknown surface for target %s: %s", name, surface_value)
                continue

            try:
                surface = TargetSurface(surface_value)
            except Exception:
                logger.warning("Unknown surface for target %s: %s", name, surface_value)
                continue

            name_value = payload.get("name", name)
            entry_name = name_value if isinstance(name_value, str) else name
            description_value = payload.get("description", "")
            description = description_value if isinstance(description_value, str) else ""

            entry = TargetEntry(
                name=entry_name,
                surface=surface,
                description=description,
                mutation_count=_coerce_int(payload.get("mutation_count", 0)),
                success_count=_coerce_int(payload.get("success_count", 0)),
                correlation_with_improvement=_coerce_float(
                    payload.get("correlation_with_improvement", 0.0)
                ),
                last_mutated=_coerce_str(payload.get("last_mutated")),
                status=_coerce_str(payload.get("status")) or "active",
            )
            registry._entries[entry.name] = entry

        return registry

    async def save(self) -> None:
        storage_path = self._storage_path
        storage_path.mkdir(parents=True, exist_ok=True)
        targets_path = storage_path / "targets.json"

        payload: dict[str, JSONValue] = {
            "entries": {name: self._entry_to_dict(entry) for name, entry in self._entries.items()},
        }

        def _write_file() -> None:
            with open(targets_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

        await asyncio.to_thread(_write_file)

    def register(self, entry: TargetEntry) -> None:
        existing = self._entries.get(entry.name)
        if existing is None:
            self._entries[entry.name] = entry
            return

        if existing.surface != entry.surface:
            logger.warning(
                "Target %s surface mismatch (%s vs %s)",
                entry.name,
                existing.surface.value,
                entry.surface.value,
            )

        if entry.description:
            existing.description = entry.description

        existing.mutation_count += entry.mutation_count
        existing.success_count += entry.success_count
        existing.last_mutated = _choose_latest_timestamp(existing.last_mutated, entry.last_mutated)
        existing.correlation_with_improvement = self._recalculate_correlation(existing)

        if existing.status != "active" and entry.status == "active":
            existing.status = "active"

    def get_active_targets(self) -> list[TargetEntry]:
        active = [entry for entry in self._entries.values() if entry.status == "active"]
        return sorted(active, key=lambda entry: entry.correlation_with_improvement, reverse=True)

    def update_correlation(self, name: str, score_delta: float) -> None:
        entry = self._entries.get(name)
        if entry is None:
            logger.warning("Target %s not found in registry", name)
            return

        entry.mutation_count += 1
        if score_delta > 0:
            entry.success_count += 1
        entry.last_mutated = datetime.now(UTC).isoformat()
        entry.correlation_with_improvement = self._recalculate_correlation(entry)

    def discover_targets(self, project_root: Path) -> list[TargetEntry]:
        from ash_hawk.auto_research.target_discovery import TargetDiscovery

        discovery = TargetDiscovery(project_root)
        discovered = discovery.discover_all_targets()
        new_entries: list[TargetEntry] = []
        for target in discovered:
            surface = _map_target_surface(target.target_type)
            if target.name in self._entries:
                continue
            entry = TargetEntry(
                name=target.name,
                surface=surface,
                description=f"Auto-discovered {target.target_type.value} target",
                status="discovered",
            )
            self._entries[entry.name] = entry
            new_entries.append(entry)
        return new_entries

    def prune_low_correlation(self, threshold: float = 0.3) -> list[str]:
        active = [entry for entry in self._entries.values() if entry.status == "active"]
        if len(active) <= self._min_active:
            return []

        active_sorted = sorted(active, key=lambda entry: entry.correlation_with_improvement)
        max_prunable = max(0, len(active_sorted) - self._min_active)
        pruned: list[str] = []

        for entry in active_sorted:
            if len(pruned) >= max_prunable:
                break
            if entry.correlation_with_improvement < threshold:
                entry.status = "pruned"
                pruned.append(entry.name)

        return pruned

    def _recalculate_correlation(self, entry: TargetEntry) -> float:
        recency_weight = _compute_recency_weight(entry.last_mutated)
        return entry.success_rate * recency_weight

    @staticmethod
    def _entry_to_dict(entry: TargetEntry) -> dict[str, JSONValue]:
        return {
            "name": entry.name,
            "surface": entry.surface.value,
            "description": entry.description,
            "mutation_count": entry.mutation_count,
            "success_count": entry.success_count,
            "correlation_with_improvement": entry.correlation_with_improvement,
            "last_mutated": entry.last_mutated,
            "status": entry.status,
        }


def _map_target_surface(target_type: TargetType) -> TargetSurface:
    target_value = getattr(target_type, "value", str(target_type))
    if target_value == "agent":
        return TargetSurface.PROMPT
    if target_value == "skill":
        return TargetSurface.TOOL
    if target_value == "policy":
        return TargetSurface.POLICY
    return TargetSurface.TOOL


def _compute_recency_weight(timestamp: str | None) -> float:
    if timestamp is None:
        return 1.0

    parsed = _parse_timestamp(timestamp)
    if parsed is None:
        return 1.0

    now = datetime.now(UTC)
    age_days = max(0.0, (now - parsed).total_seconds() / 86400.0)
    decay = min(age_days / 30.0, 1.0)
    return max(0.2, 1.0 - decay)


def _parse_timestamp(timestamp: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _choose_latest_timestamp(first: str | None, second: str | None) -> str | None:
    if first is None:
        return second
    if second is None:
        return first

    first_dt = _parse_timestamp(first)
    second_dt = _parse_timestamp(second)
    if first_dt is None:
        return second
    if second_dt is None:
        return first
    return second if second_dt >= first_dt else first


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _coerce_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _coerce_json_dict(value: object) -> dict[str, JSONValue]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, JSONValue] = {}
    value_dict = cast(dict[object, object], value)
    for key, item in value_dict.items():
        if not isinstance(key, str):
            continue
        result[key] = _coerce_json_value(item)
    return result


def _coerce_json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        value_list = cast(list[object], value)
        return [_coerce_json_value(item) for item in value_list]
    if isinstance(value, dict):
        value_dict = cast(dict[object, object], value)
        return _coerce_json_dict(value_dict)
    return str(value)


__all__ = [
    "TargetEntry",
    "TargetRegistry",
]
