from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml
from dawn_kestrel.skills import SkillPolicy, SkillRuntime
from dawn_kestrel.skills import SkillRegistry as DkSkillRegistry

THIN_RUNTIME_CATALOG_SOURCE = "thin_runtime"


def resolve_skill_project_root(project_root: Path | str | None = None) -> Path:
    if project_root is None:
        return Path(__file__).resolve().parents[1]
    return Path(project_root).resolve()


def build_dawn_kestrel_skill_policy(project_root: Path) -> SkillPolicy:
    trusted_roots = tuple(
        path
        for path in (
            project_root / "skills",
            project_root / ".agents" / "skills",
            project_root / ".claude" / "skills",
        )
        if path.exists()
    )
    return SkillPolicy(
        allow_project_skills=True,
        allow_user_skills=False,
        allow_unavailable_listing=False,
        trusted_roots=trusted_roots,
    )


def discover_project_skill_registry(project_root: Path | str | None = None) -> DkSkillRegistry:
    resolved_root = resolve_skill_project_root(project_root)
    return DkSkillRegistry.discover(
        project_root=resolved_root,
        policy=build_dawn_kestrel_skill_policy(resolved_root),
    )


def filter_skill_registry_by_catalog_source(
    registry: DkSkillRegistry,
    catalog_source: str,
) -> DkSkillRegistry:
    filtered = [
        parsed
        for parsed in registry.list()
        if skill_catalog_source(parsed.manifest.location) == catalog_source
    ]
    return DkSkillRegistry(filtered, policy=registry.policy)


def read_skill_frontmatter(path: Path) -> dict[str, object]:
    content = path.read_text(encoding="utf-8")
    if not content.startswith("---\n"):
        raise ValueError(f"SKILL.md missing front matter: {path}")
    _, rest = content.split("---\n", 1)
    frontmatter_text, _ = rest.split("\n---\n", 1)
    raw = yaml.safe_load(frontmatter_text) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid SKILL.md front matter: {path}")
    return dict(raw)


def skill_catalog_source(path: Path) -> str | None:
    frontmatter = read_skill_frontmatter(path)
    metadata = frontmatter.get("metadata")
    if not isinstance(metadata, dict):
        return None
    raw_source = metadata.get("catalog_source")
    return raw_source.strip() if isinstance(raw_source, str) and raw_source.strip() else None


async def prepare_skill_runtime(
    *,
    registry: DkSkillRegistry | None = None,
    project_root: Path | str | None = None,
    catalog_source: str | None = None,
    preactivate: Iterable[str] = (),
    strict_preactivate: bool = False,
) -> tuple[SkillRuntime | None, list[dict[str, object]]]:
    effective_registry = registry or discover_project_skill_registry(project_root)
    if catalog_source is not None:
        effective_registry = filter_skill_registry_by_catalog_source(
            effective_registry, catalog_source
        )
    if not effective_registry.has_skills():
        return None, []

    runtime = SkillRuntime.from_registry(effective_registry)
    preloaded_messages: list[dict[str, object]] = []
    for skill_name in _dedupe_skill_names(preactivate):
        if effective_registry.get(skill_name) is None:
            if strict_preactivate:
                raise ValueError(f"Unknown skill: {skill_name}")
            continue
        activated = await runtime.activate_skill(skill_name)
        preloaded_messages.append(
            {
                "role": "system",
                "content": activated.content,
                "metadata": {"skill_name": skill_name, "activated_skill": True},
                "_protected": True,
            }
        )
    return runtime, preloaded_messages


def _dedupe_skill_names(names: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    for name in names:
        normalized = str(name).strip()
        if not normalized or normalized in ordered:
            continue
        ordered.append(normalized)
    return ordered


__all__ = [
    "THIN_RUNTIME_CATALOG_SOURCE",
    "build_dawn_kestrel_skill_policy",
    "discover_project_skill_registry",
    "filter_skill_registry_by_catalog_source",
    "prepare_skill_runtime",
    "read_skill_frontmatter",
    "resolve_skill_project_root",
    "skill_catalog_source",
]
