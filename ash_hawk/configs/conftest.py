"""conftest.yaml loading with inheritance.

Allows shared configuration at directory level, inherited by child
directories and suites, similar to pytest's conftest.py.

Example directory structure:
    evals/
    ├── conftest.yaml           # Root config
    ├── code-gen/
    │   ├── conftest.yaml       # Inherits from parent
    │   └── basic-suite.yaml
    └── bug-fixing/
        └── regression-suite.yaml

Inheritance rules:
1. Walk up directories from suite file to evals root
2. Collect all conftest.yaml files
3. Merge configs (child overrides parent)
4. Suite YAML has highest priority
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pydantic as pd
import yaml


class ConftestConfig(pd.BaseModel):
    """Configuration from conftest.yaml."""

    name: str | None = pd.Field(default=None, description="Config name")
    version: str | None = pd.Field(default=None, description="Config version")

    # Default policy for all suites
    policy: dict[str, Any] = pd.Field(
        default_factory=dict, description="Default tool policy settings"
    )

    # Shared fixtures
    fixtures: dict[str, str] = pd.Field(default_factory=dict, description="Shared fixture paths")

    # Default grader configuration
    default_grader: dict[str, Any] | None = pd.Field(
        default=None, description="Default grader spec to apply to all tasks"
    )

    # Agent configuration
    agent: dict[str, Any] = pd.Field(
        default_factory=dict, description="Default agent configuration"
    )

    # Tags to apply to all suites
    tags: list[str] = pd.Field(default_factory=list, description="Tags applied to all suites")

    # Metadata
    metadata: dict[str, Any] = pd.Field(default_factory=dict, description="Additional metadata")

    model_config = pd.ConfigDict(extra="allow")


class ConftestLoader:
    """Load and merge conftest.yaml files."""

    def __init__(self, search_root: Path | None = None):
        """Initialize the loader.

        Args:
            search_root: Root directory for conftest search.
                        If None, uses current working directory.
        """
        self.search_root = (Path(search_root) if search_root else Path.cwd()).resolve()

    def load_for_suite(self, suite_path: Path) -> ConftestConfig:
        """Load merged conftest configuration for a suite.

        Walks up from suite_path to search_root, loading and merging
        all conftest.yaml files found.

        Args:
            suite_path: Path to the suite YAML file.

        Returns:
            Merged ConftestConfig.
        """
        suite_path = Path(suite_path).resolve()

        # Collect conftest files from suite dir up to root
        conftest_files = []
        current = suite_path.parent

        while True:
            conftest = current / "conftest.yaml"
            if conftest.exists():
                conftest_files.append(conftest)

            if current == self.search_root or current.parent == current:
                break
            current = current.parent

        # Reverse to apply from root to leaf
        conftest_files.reverse()

        # Load and merge
        merged = ConftestConfig()
        for conftest_path in conftest_files:
            config = self._load_conftest(conftest_path)
            merged = self._merge_configs(merged, config)

        return merged

    def _load_conftest(self, path: Path) -> ConftestConfig:
        """Load a single conftest.yaml file.

        Args:
            path: Path to conftest.yaml.

        Returns:
            Loaded ConftestConfig.
        """
        content = path.read_text()
        data = yaml.safe_load(content) or {}
        return ConftestConfig(**data)

    def _merge_configs(
        self,
        base: ConftestConfig,
        override: ConftestConfig,
    ) -> ConftestConfig:
        """Merge two configurations (override takes precedence).

        Args:
            base: Base configuration.
            override: Override configuration.

        Returns:
            Merged ConftestConfig.
        """
        base_dict = base.model_dump()
        override_dict = override.model_dump(exclude_none=True)

        merged = self._deep_merge(base_dict, override_dict)
        return ConftestConfig(**merged)

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries.

        - Dicts are recursively merged
        - Lists are extended (not replaced)
        - Other values are replaced

        Args:
            base: Base dictionary.
            override: Override dictionary.

        Returns:
            Merged dictionary.
        """
        result = dict(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            elif isinstance(value, list):
                # Lists are extended, not replaced
                result[key] = result.get(key, []) + value
            else:
                result[key] = value

        return result


def apply_conftest_to_suite(
    suite_dict: dict[str, Any],
    conftest: ConftestConfig,
) -> dict[str, Any]:
    """Apply conftest configuration to a suite dictionary.

    This merges conftest settings into the suite, with suite
    values taking precedence.

    Args:
        suite_dict: Suite configuration as dictionary.
        conftest: Conftest configuration.

    Returns:
        Merged suite dictionary.
    """
    result = dict(suite_dict)

    # Apply tags
    if conftest.tags:
        existing_tags = result.get("tags", [])
        result["tags"] = list(set(existing_tags + conftest.tags))

    # Apply agent config
    if conftest.agent:
        existing_agent = result.get("agent", {})
        if isinstance(existing_agent, dict):
            merged_agent = {**conftest.agent, **existing_agent}
            result["agent"] = merged_agent

    # Apply policy
    if conftest.policy:
        existing_metadata = result.get("metadata", {})
        if isinstance(existing_metadata, dict):
            existing_metadata = dict(existing_metadata)
            existing_metadata.setdefault("policy", {})
            if isinstance(existing_metadata["policy"], dict):
                existing_metadata["policy"] = {
                    **conftest.policy,
                    **existing_metadata["policy"],
                }
            result["metadata"] = existing_metadata

    # Apply default grader to tasks without graders
    if conftest.default_grader:
        tasks = result.get("tasks", [])
        for task in tasks:
            if isinstance(task, dict) and not task.get("grader_specs"):
                task["grader_specs"] = [conftest.default_grader]

    # Apply shared fixtures
    if conftest.fixtures:
        tasks = result.get("tasks", [])
        for task in tasks:
            if isinstance(task, dict):
                existing_fixtures = task.get("fixtures", {})
                if isinstance(existing_fixtures, dict):
                    task["fixtures"] = {**conftest.fixtures, **existing_fixtures}

    return result


__all__ = [
    "ConftestConfig",
    "ConftestLoader",
    "apply_conftest_to_suite",
]
