# type-hygiene: skip-file
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

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


def load_agent_specs(base_dir: Path) -> list[AgentSpec]:
    return [
        AgentSpec.model_validate(_load_markdown_spec(path))
        for path in sorted((base_dir / "agents").glob("*.md"))
    ]


def load_skill_specs(base_dir: Path) -> list[SkillSpec]:
    return [
        SkillSpec.model_validate(_load_markdown_spec(path))
        for path in sorted((base_dir / "skills").glob("*.md"))
    ]


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
