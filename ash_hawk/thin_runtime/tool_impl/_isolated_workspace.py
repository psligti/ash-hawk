from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ash_hawk.thin_runtime.tool_impl._native_tooling import (
    ensure_workspace_contained,
    workspace_relative_string,
)


@dataclass(frozen=True)
class IsolatedWorkspaceSnapshot:
    primary_root: Path
    isolated_root: Path
    source_scenario_path: str | None
    isolated_scenario_path: str | None
    copied_files: list[str]


def create_isolated_workspace(
    primary_root: Path,
    *,
    target_files: list[str],
    scenario_path: str | None,
    scenario_targets: list[str],
    scenario_required_files: list[str],
    agent_config: str | None,
) -> IsolatedWorkspaceSnapshot:
    resolved_root = primary_root.resolve()
    isolated_root = Path(tempfile.mkdtemp(prefix="ash-hawk-thin-runtime-"))
    copied_files: list[str] = []

    file_candidates = _workspace_relative_candidates(
        resolved_root,
        [
            *target_files,
            *scenario_targets,
            *scenario_required_files,
            *([agent_config] if agent_config else []),
        ],
    )
    for relative_path in file_candidates:
        copied = _copy_workspace_path(
            primary_root=resolved_root,
            isolated_root=isolated_root,
            relative_path=relative_path,
            create_missing=relative_path in target_files,
        )
        if copied is not None:
            copied_files.append(copied)

    isolated_scenario_path: str | None = None
    if scenario_path:
        isolated_scenario_path = _copy_scenario_path(
            primary_root=resolved_root,
            isolated_root=isolated_root,
            scenario_path=scenario_path,
        )
        if isolated_scenario_path is not None and isolated_scenario_path not in copied_files:
            copied_files.append(isolated_scenario_path)

    return IsolatedWorkspaceSnapshot(
        primary_root=resolved_root,
        isolated_root=isolated_root,
        source_scenario_path=scenario_path,
        isolated_scenario_path=(
            str(isolated_root / isolated_scenario_path) if isolated_scenario_path else None
        ),
        copied_files=sorted(set(copied_files)),
    )


def sync_isolated_changes(
    *,
    primary_root: Path,
    isolated_root: Path,
    relative_paths: list[str],
) -> list[str]:
    synced: list[str] = []
    for relative_path in _unique_paths(relative_paths):
        source = ensure_workspace_contained(isolated_root / relative_path, isolated_root)
        destination = ensure_workspace_contained(primary_root / relative_path, primary_root)
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        elif destination.exists():
            destination.unlink()
        synced.append(relative_path)
    return synced


def cleanup_isolated_workspace(isolated_root: Path | None) -> None:
    if isolated_root is None:
        return
    shutil.rmtree(isolated_root, ignore_errors=True)


def _copy_workspace_path(
    *,
    primary_root: Path,
    isolated_root: Path,
    relative_path: str,
    create_missing: bool,
) -> str | None:
    cleaned = relative_path.strip()
    if not cleaned:
        return None
    source = ensure_workspace_contained(primary_root / cleaned, primary_root)
    destination = ensure_workspace_contained(isolated_root / cleaned, isolated_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copy2(source, destination)
        return cleaned
    if create_missing:
        destination.touch()
        return cleaned
    return None


def _copy_scenario_path(
    *,
    primary_root: Path,
    isolated_root: Path,
    scenario_path: str,
) -> str | None:
    path = Path(scenario_path)
    if path.is_absolute():
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(primary_root)
        except ValueError:
            return None
    else:
        relative = Path(scenario_path)
    copied = _copy_workspace_path(
        primary_root=primary_root,
        isolated_root=isolated_root,
        relative_path=str(relative),
        create_missing=False,
    )
    return copied


def _unique_paths(paths: list[str]) -> list[str]:
    ordered: list[str] = []
    for path in paths:
        cleaned = path.strip()
        if cleaned and cleaned not in ordered:
            ordered.append(cleaned)
    return ordered


def _workspace_relative_candidates(primary_root: Path, paths: list[str]) -> list[str]:
    ordered: list[str] = []
    for path in paths:
        normalized = workspace_relative_string(path, primary_root)
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    return ordered
