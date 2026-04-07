# type-hygiene: skip-file
"""Fixture resolution for evaluation tasks.

This module provides fixture resolution for evaluation suites, allowing
tasks to reference files and directories relative to the suite file.

Fixture resolution supports:
- Relative paths (resolved from suite file location)
- $fixture_name variable substitution in task input
- Path validation and existence checks
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ash_hawk.types import EvalSuite, EvalTask


class FixtureError(Exception):
    """Raised when fixture resolution fails."""

    pass


class FixtureResolver:
    """Resolves fixture paths relative to a suite file.

    Fixtures are file/directory paths that tasks can reference. They are
    resolved relative to the suite file's directory, making suites portable.

    Example:
        # Given suite at /project/evals/suites/my-suite.yaml
        # with fixture: fixtures/sample.json

        resolver = FixtureResolver(suite_path, suite)
        paths = resolver.resolve(task)

        # paths["fixtures/sample.json"] -> /project/evals/suites/fixtures/sample.json
    """

    def __init__(self, suite_path: Path | str, suite: EvalSuite) -> None:
        """Initialize the fixture resolver.

        Args:
            suite_path: Path to the suite YAML file.
            suite: The EvalSuite object.
        """
        self._suite_path = Path(suite_path).resolve()
        self._suite_dir = self._suite_path.parent
        self._suite = suite

    @property
    def suite_dir(self) -> Path:
        """Get the directory containing the suite file."""
        return self._suite_dir

    def resolve_path(self, path: str) -> Path:
        """Resolve a single path relative to the suite directory.

        Args:
            path: A relative or absolute path string.

        Returns:
            Resolved absolute path.
        """
        fixture_path = Path(path)

        # If already absolute, return as-is
        if fixture_path.is_absolute():
            return fixture_path

        # Resolve relative to suite directory
        return (self._suite_dir / fixture_path).resolve()

    def resolve_task_fixtures(self, task: EvalTask) -> dict[str, Path]:
        """Resolve all fixture paths for a task.

        Args:
            task: The evaluation task.

        Returns:
            Dictionary mapping fixture keys to resolved paths.
        """
        resolved: dict[str, Path] = {}

        for fixture_name, fixture_path in task.fixtures.items():
            resolved[fixture_name] = self.resolve_path(fixture_path)

        return resolved

    def inject_fixtures(
        self,
        task: EvalTask,
        fixture_paths: dict[str, Path] | None = None,
    ) -> EvalTask:
        """Inject resolved fixture paths into task input and grader_specs.

        This replaces $fixture_name references in the task input and
        grader config with the resolved absolute paths.

        Args:
            task: The evaluation task.
            fixture_paths: Pre-resolved fixture paths (optional, computed if not provided).

        Returns:
            New EvalTask with injected fixture paths.
        """
        if fixture_paths is None:
            fixture_paths = self.resolve_task_fixtures(task)

        task_dict = task.model_dump()

        if isinstance(task.input, dict):
            task_dict["input"] = self._substitute_fixtures(task.input, fixture_paths)

        task_dict["grader_specs"] = self._substitute_fixtures(
            task_dict.get("grader_specs", []), fixture_paths
        )

        from ash_hawk.types import EvalTask

        return EvalTask(**task_dict)

    def _substitute_fixtures(
        self,
        data: Any,
        fixture_paths: dict[str, Path],
    ) -> Any:
        """Recursively substitute $fixture_name references.

        Supports both standalone references ($var) and embedded references
        ("Find code in $var").

        Args:
            data: The data structure to process.
            fixture_paths: Dictionary of fixture name -> path.

        Returns:
            Data structure with substitutions applied.
        """
        if isinstance(data, str):
            # Replace $var patterns anywhere in the string (word boundary)
            result = data
            for name, path in fixture_paths.items():
                result = re.sub(rf"\${re.escape(name)}\b", str(path), result)
            return result

        if isinstance(data, dict):
            return {
                key: self._substitute_fixtures(value, fixture_paths) for key, value in data.items()
            }

        if isinstance(data, list):
            return [self._substitute_fixtures(item, fixture_paths) for item in data]

        return data

    def validate_fixtures(self, task: EvalTask) -> list[str]:
        """Validate that all fixture paths exist.

        Args:
            task: The evaluation task.

        Returns:
            List of error messages for missing fixtures (empty if all valid).
        """
        errors: list[str] = []
        resolved = self.resolve_task_fixtures(task)

        for fixture_name, fixture_path in resolved.items():
            if not fixture_path.exists():
                errors.append(f"Fixture '{fixture_name}' not found: {fixture_path}")

        return errors

    def get_working_dir(self, task: EvalTask) -> Path:
        """Get the working directory for a task.

        If the task has a working_dir fixture, use that.
        Otherwise, use the suite directory.

        Args:
            task: The evaluation task.

        Returns:
            Working directory path.
        """
        # Check for working_dir in task fixtures
        if "working_dir" in task.fixtures:
            return self.resolve_path(task.fixtures["working_dir"])

        # Check for working_dir in task input (if dict)
        if isinstance(task.input, dict) and "working_dir" in task.input:
            wd = task.input["working_dir"]
            if isinstance(wd, str):
                if wd.startswith("$") and wd[1:] in task.fixtures:
                    return self.resolve_path(task.fixtures[wd[1:]])
                return self.resolve_path(wd)

        # Default to suite directory
        return self._suite_dir


__all__ = ["FixtureResolver", "FixtureError"]
