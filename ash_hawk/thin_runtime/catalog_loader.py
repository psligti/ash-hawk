# type-hygiene: skip-file
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from ash_hawk.dawn_kestrel_skills import (
    THIN_RUNTIME_CATALOG_SOURCE,
    discover_project_skill_registry,
    filter_skill_registry_by_catalog_source,
    read_skill_frontmatter,
)
from ash_hawk.thin_runtime.models import AgentSpec, SkillSpec

REQUIRED_AGENT_FIELDS = {
    "id",
    "name",
    "kind",
    "version",
    "status",
    "file",
    "authority_level",
    "scope",
    "skill_names",
    "hook_names",
    "memory_read_scopes",
    "memory_write_scopes",
}

REQUIRED_SKILL_FIELDS = {
    "id",
    "name",
    "kind",
    "version",
    "status",
    "file",
    "category",
    "scope",
    "tool_names",
    "input_contexts",
    "output_contexts",
    "memory_read_scopes",
    "memory_write_scopes",
}

REQUIRED_SKILL_FRONTMATTER_FIELDS = {"name", "description", "version", "metadata"}


def load_agent_specs(base_dir: Path) -> list[AgentSpec]:
    return [
        AgentSpec.model_validate(_load_markdown_spec(path))
        for path in sorted((base_dir / "agents").glob("*.md"))
    ]


def load_skill_specs(base_dir: Path) -> list[SkillSpec]:
    registry = discover_project_skill_registry(base_dir.parents[2])
    thin_registry = filter_skill_registry_by_catalog_source(registry, THIN_RUNTIME_CATALOG_SOURCE)
    return [_load_skill_spec_from_parsed_skill(parsed) for parsed in thin_registry.list()]


def _load_skill_spec_from_parsed_skill(parsed: Any) -> SkillSpec:
    frontmatter = read_skill_frontmatter(parsed.manifest.location)
    metadata = frontmatter.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Skill metadata must be a mapping in {parsed.manifest.location}")
    payload = {
        key: value
        for key, value in metadata.items()
        if key not in {"catalog_source", "legacy_catalog_file"}
    }
    payload.update(
        {
            "name": parsed.manifest.name,
            "description": parsed.manifest.description,
            "version": parsed.manifest.version or str(payload.get("version", "1.0.0") or "1.0.0"),
            "instructions_markdown": parsed.body.strip(),
            "file": str(payload.get("file") or f"skills/{parsed.manifest.name}/SKILL.md"),
        }
    )
    missing = sorted(REQUIRED_SKILL_FIELDS - set(payload))
    if missing:
        raise ValueError(
            f"Skill metadata missing required fields in skills/{parsed.manifest.name}/SKILL.md: {missing}"
        )
    return SkillSpec.model_validate(payload)


def _load_markdown_spec(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    if not content.startswith("---\n"):
        raise ValueError(f"Markdown spec missing front matter: {path}")
    _, rest = content.split("---\n", 1)
    frontmatter_text, body = rest.split("\n---\n", 1)
    raw_frontmatter = yaml.safe_load(frontmatter_text)
    if not isinstance(raw_frontmatter, dict):
        raise ValueError(f"Invalid front matter in: {path}")
    frontmatter = cast(dict[str, Any], raw_frontmatter)
    frontmatter["instructions_markdown"] = body.strip()
    parent = path.parent.name
    if parent == "agents":
        missing = sorted(REQUIRED_AGENT_FIELDS - set(frontmatter))
        if missing:
            raise ValueError(
                f"Agent markdown missing required front matter fields in {path}: {missing}"
            )
    elif parent == "skills":
        missing = sorted(REQUIRED_SKILL_FIELDS - set(frontmatter))
        if missing:
            raise ValueError(
                f"Skill markdown missing required front matter fields in {path}: {missing}"
            )
    return frontmatter
