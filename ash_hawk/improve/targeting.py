from __future__ import annotations

from pathlib import PurePosixPath

from ash_hawk.improve.diagnose import Diagnosis


def _path_candidates(raw_path: str) -> list[str]:
    cleaned = raw_path.strip().replace("\\", "/")
    if not cleaned:
        return []
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    pure = PurePosixPath(cleaned)
    parts = [part for part in pure.parts if part not in ("", ".")]
    if any(part == ".." for part in parts):
        return []

    candidates: list[str] = []

    def add_candidate(path_parts: list[str]) -> None:
        if not path_parts:
            return
        candidate = PurePosixPath(*path_parts).as_posix()
        name = PurePosixPath(candidate).name
        if "." not in name or candidate in candidates:
            return
        candidates.append(candidate)

    if "agent" in parts:
        last_agent = max(index for index, part in enumerate(parts) if part == "agent")
        add_candidate(parts[last_agent:])
        add_candidate(parts[last_agent + 1 :])
    else:
        add_candidate(parts)

    return candidates


def normalize_agent_relative_path(raw_path: str) -> str | None:
    candidates = _path_candidates(raw_path)
    return candidates[0] if candidates else None


def resolve_allowed_target(raw_path: str, allowed_files: set[str]) -> str | None:
    for candidate in _path_candidates(raw_path):
        if candidate in allowed_files:
            return candidate
        for allowed in allowed_files:
            if candidate.endswith(f"/{allowed}"):
                return allowed
    return None


def _allowed_directories(allowed_files: set[str]) -> set[str]:
    directories: set[str] = set()
    for path in allowed_files:
        parent = PurePosixPath(path).parent
        while parent.as_posix() not in ("", "."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _has_allowed_parent(path: str, allowed_directories: set[str]) -> bool:
    parent = PurePosixPath(path).parent
    while parent.as_posix() not in ("", "."):
        if parent.as_posix() in allowed_directories:
            return True
        parent = parent.parent
    return False


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _resolve_target_paths(
    target_files: list[str],
    anchor_files: list[str],
    allowed_files: set[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    resolved_anchors = _dedupe(
        [
            resolved
            for anchor in anchor_files
            if (resolved := resolve_allowed_target(anchor, allowed_files)) is not None
        ]
    )
    allowed_directories = _allowed_directories(allowed_files)
    resolved_targets: list[str] = []
    rejected_targets: list[str] = []
    missing_anchor_targets: list[str] = []

    for raw_target in target_files:
        resolved = resolve_allowed_target(raw_target, allowed_files)
        if resolved is not None:
            resolved_targets.append(resolved)
            continue
        architecture_candidate: str | None = None
        for candidate in _path_candidates(raw_target):
            architecture_fit = PurePosixPath(candidate).parent.as_posix() in (
                "",
                ".",
            ) or _has_allowed_parent(candidate, allowed_directories)
            if architecture_fit:
                architecture_candidate = candidate
                break
        if architecture_candidate is None:
            rejected_targets.append(raw_target)
            continue
        if resolved_anchors:
            resolved_targets.append(architecture_candidate)
            continue
        missing_anchor_targets.append(raw_target)
        continue

    return (
        _dedupe(resolved_targets),
        resolved_anchors,
        rejected_targets,
        missing_anchor_targets,
    )


def validate_diagnosis_targets(
    diagnosis: Diagnosis,
    allowed_files: set[str],
) -> tuple[Diagnosis, list[str]]:
    resolved_targets, resolved_anchors, rejected_targets, missing_anchor_targets = (
        _resolve_target_paths(
            diagnosis.target_files,
            diagnosis.anchor_files,
            allowed_files,
        )
    )

    diagnosis.target_files = _dedupe(resolved_targets)
    diagnosis.anchor_files = resolved_anchors
    if diagnosis.target_files and any(
        target not in allowed_files for target in diagnosis.target_files
    ):
        if not diagnosis.anchor_files:
            diagnosis.actionable = False
            diagnosis.degraded_reason = "diagnosis_new_file_missing_anchor"
    elif not diagnosis.target_files:
        diagnosis.actionable = False
        if missing_anchor_targets:
            diagnosis.degraded_reason = "diagnosis_new_file_missing_anchor"
        else:
            diagnosis.degraded_reason = (
                diagnosis.degraded_reason or "diagnosis_missing_target_files"
            )
    return diagnosis, rejected_targets + missing_anchor_targets


def diagnosis_targets_allowed(diagnosis: Diagnosis, allowed_files: set[str]) -> bool:
    resolved_targets, _, _, _ = _resolve_target_paths(
        diagnosis.target_files,
        diagnosis.anchor_files,
        allowed_files,
    )
    return bool(resolved_targets)


__all__ = [
    "diagnosis_targets_allowed",
    "normalize_agent_relative_path",
    "resolve_allowed_target",
    "validate_diagnosis_targets",
]
