"""Auto-discovery of agent configuration from repository.

Discovers agent config, scenarios, and improvement targets from the repository
being improved (pytest-style).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pydantic as pd

from ash_hawk.auto_research.types import ImprovementType, RepoConfig

logger = logging.getLogger(__name__)


class AshHawkTomlConfig(pd.BaseModel):
    """Configuration from [tool.ash-hawk] in pyproject.toml."""

    agent_name: str | None = pd.Field(default=None, description="Agent being improved")
    agent_runner: str = pd.Field(default="dawn_kestrel", description="Runner type")
    scenarios_dir: str = pd.Field(default="evals", description="Default scenarios directory")
    skills_dir: str = pd.Field(default="skills", description="Skills directory")
    policies_dir: str = pd.Field(default="policies", description="Policies directory")
    tools_dir: str = pd.Field(default="tools", description="Tools directory")
    scenario_patterns: list[str] | None = pd.Field(
        default=None, description="Custom scenario patterns"
    )
    target_patterns: list[str] | None = pd.Field(default=None, description="Custom target patterns")
    scenarios: list[str] | None = pd.Field(default=None, description="Explicit scenario paths")
    improvement_targets: list[str] | None = pd.Field(
        default=None, description="Explicit target paths"
    )

    model_config = pd.ConfigDict(extra="allow")


def discover_repo_config(start_dir: Path | None = None) -> RepoConfig:
    """Discover all configuration from repository.

    Searches for pyproject.toml and auto-discovers scenarios and targets.

    Args:
        start_dir: Directory to start searching from. Uses cwd if None.

    Returns:
        RepoConfig with discovered settings.
    """
    start_dir = start_dir or Path.cwd()
    start_dir = start_dir.resolve()

    # Find pyproject.toml
    pyproject_path = _find_pyproject(start_dir)

    # Load [tool.ash-hawk] section
    config_data = _load_ash_hawk_config(pyproject_path)

    # Extract agent info
    agent_name = config_data.get("agent_name")
    agent_runner = config_data.get("agent_runner", "dawn_kestrel")

    # Discover scenarios
    scenario_patterns = config_data.get(
        "scenario_patterns",
        ["evals/**/*.yaml", "evals/**/*.scenario.yaml", "scenarios/**/*.yaml"],
    )
    scenarios = _discover_by_patterns(start_dir, scenario_patterns)

    targets = _discover_by_patterns(
        start_dir,
        config_data.get(
            "target_patterns",
            [
                "skills/**/*.md",
                "policies/**/*.md",
                "tools/**/*.md",
                ".opencode/skills/**/*.md",
                ".opencode/policies/**/*.md",
                ".opencode/tools/**/*.md",
                ".claude/SKILL.md",
            ],
        ),
    )

    # Override with explicit paths from config
    if "scenarios" in config_data:
        explicit_scenarios = [
            start_dir / s if not Path(s).is_absolute() else Path(s)
            for s in config_data["scenarios"]
        ]
        scenarios = _expand_paths(explicit_scenarios, [".yaml", ".scenario.yaml"])

    if "improvement_targets" in config_data:
        targets = [
            start_dir / t if not Path(t).is_absolute() else Path(t)
            for t in config_data["improvement_targets"]
        ]
        targets = [t for t in targets if t.exists()]

    return RepoConfig(
        agent_name=agent_name,
        agent_runner=agent_runner,
        scenarios=scenarios,
        improvement_targets=targets,
        pyproject_path=pyproject_path,
    )


def _find_pyproject(start_dir: Path) -> Path | None:
    """Find pyproject.toml by searching upward.

    Args:
        start_dir: Directory to start searching from.

    Returns:
        Path to pyproject.toml or None if not found.
    """
    current = start_dir.resolve()

    while True:
        pyproject = current / "pyproject.toml"
        if pyproject.exists():
            return pyproject

        if current.parent == current:
            return None
        current = current.parent


def _load_ash_hawk_config(pyproject_path: Path | None) -> dict[str, Any]:
    """Load [tool.ash-hawk] section from pyproject.toml.

    Args:
        pyproject_path: Path to pyproject.toml.

    Returns:
        Dictionary with config values, empty if not found.
    """
    if pyproject_path is None:
        return {}

    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            logger.warning("tomli not installed, cannot parse pyproject.toml")
            return {}

    try:
        content = pyproject_path.read_bytes()
        data = tomllib.loads(content.decode("utf-8"))
        tool_config: dict[str, Any] = data.get("tool", {}).get("ash-hawk", {})
        return tool_config
    except Exception as e:
        logger.warning(f"Failed to parse pyproject.toml: {e}")
        return {}


def _discover_by_patterns(start_dir: Path, patterns: list[str]) -> list[Path]:
    """Discover files matching glob patterns.

    Args:
        start_dir: Base directory for patterns.
        patterns: List of glob patterns.

    Returns:
        Sorted list of unique matching paths.
    """
    discovered: set[Path] = set()

    for pattern in patterns:
        for match in start_dir.glob(pattern):
            if match.is_file():
                discovered.add(match)

    return sorted(discovered)


def _expand_paths(paths: list[Path], extensions: list[str]) -> list[Path]:
    """Expand paths, including directories.

    Args:
        paths: List of paths (files or directories).
        extensions: Valid file extensions.

    Returns:
        List of file paths.
    """
    expanded: list[Path] = []

    for path in paths:
        if path.is_file():
            if any(str(path).endswith(ext.rstrip("*")) for ext in extensions):
                expanded.append(path)
        elif path.is_dir():
            for ext in extensions:
                expanded.extend(path.glob(f"**/*{ext.rstrip('*')}"))

    return sorted(set(expanded))


def generate_experiment_id(
    agent_name: str | None,
    targets: list[Path],
) -> str:
    """Generate a unique experiment ID from context.

    Format: exp-{agent}-{target_stem}-{YYMMDD}-{HHMMSS}

    Args:
        agent_name: Name of the agent being improved.
        targets: List of improvement targets.

    Returns:
        Generated experiment ID.
    """
    agent = agent_name or "unknown"

    target_stem = ""
    if targets:
        target_stem = targets[0].stem

    timestamp = datetime.now(UTC).strftime("%y%m%d-%H%M%S")

    if target_stem:
        return f"exp-{agent}-{target_stem}-{timestamp}"
    return f"exp-{agent}-{timestamp}"


_TYPE_PREFIXES: dict[ImprovementType, list[str]] = {
    ImprovementType.SKILL: ["skills/", ".opencode/skills/", ".claude/SKILL.md"],
    ImprovementType.POLICY: ["policies/", ".opencode/policies/"],
    ImprovementType.TOOL: ["tools/", ".opencode/tools/"],
    ImprovementType.AGENT: ["agents/", ".opencode/agents/"],
}


def filter_targets_by_type(
    targets: list[Path],
    target_types: list[ImprovementType],
) -> list[Path]:
    """Filter targets to only include specified types based on path prefixes."""
    if not target_types:
        return targets

    allowed_prefixes: set[str] = set()
    for t in target_types:
        allowed_prefixes.update(_TYPE_PREFIXES.get(t, []))

    filtered: list[Path] = []
    for target in targets:
        target_str = str(target)
        for prefix in allowed_prefixes:
            if prefix in target_str:
                filtered.append(target)
                break

    return filtered


__all__ = [
    "AshHawkTomlConfig",
    "discover_repo_config",
    "filter_targets_by_type",
    "generate_experiment_id",
]
