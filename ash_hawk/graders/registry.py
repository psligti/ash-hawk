# type-hygiene: skip-file  # entry point loading requires Any for dynamic grader resolution
"""Grader registry for managing and discovering graders.

This module provides a registry for graders that supports:
- Manual registration of grader instances
- Lookup by name
- Discovery via Python entry points
"""

from __future__ import annotations

import importlib
import logging
import sys
import tomllib
from contextlib import contextmanager
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, cast

if TYPE_CHECKING:
    from ash_hawk.graders.base import Grader

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "ash_hawk.graders"


class GraderRegistry:
    """Registry for managing grader instances.

    The registry supports:
    - Manual registration via register()
    - Lookup via get()
    - Listing all registered graders via list_graders()
    - Loading from entry points via load_from_entry_points()

    Graders are stored by name and must be unique. Attempting to register
    a grader with a name that already exists will overwrite the previous one.
    """

    def __init__(self) -> None:
        """Initialize an empty grader registry."""
        self._graders: dict[str, Grader] = {}

    def register(self, grader: Grader) -> None:
        """Register a grader instance.

        If a grader with the same name already exists, it will be replaced.

        Args:
            grader: The grader instance to register.
        """
        name = grader.name
        if name in self._graders:
            logger.warning(
                "Overwriting existing grader '%s' with %s",
                name,
                grader.__class__.__name__,
            )
        self._graders[name] = grader
        logger.debug("Registered grader: %s", name)

    def get(self, name: str) -> Grader | None:
        """Get a grader by name.

        Args:
            name: The name of the grader to retrieve.

        Returns:
            The grader instance if found, None otherwise.
        """
        return self._graders.get(name)

    def list_graders(self) -> list[str]:
        """List all registered grader names.

        Returns:
            A sorted list of registered grader names.
        """
        return sorted(self._graders.keys())

    def load_from_entry_points(self) -> None:
        """Load graders from Python entry points.

        Entry points should be defined in the 'ash_hawk.graders' group.
        Each entry point should return a Grader instance when called.

        Example pyproject.toml entry:
            [project.entry-points."ash_hawk.graders"]
            my_grader = "my_package.graders:MyGrader()"

        Graders loaded from entry points will be registered automatically.
        If a grader with the same name already exists, a warning will be logged
        but the existing grader will NOT be overwritten.
        """
        eps: Iterable[EntryPoint]
        try:
            eps = entry_points(group=ENTRY_POINT_GROUP)
        except TypeError:
            # Python < 3.10 compatibility
            eps_raw = cast(dict[str, list[EntryPoint]], entry_points())
            eps = eps_raw.get(ENTRY_POINT_GROUP, [])

        for ep in eps:
            entry_point: Any = ep
            try:
                # Entry point can be a Grader class or instance
                loaded = cast(object, entry_point.load())

                # If it's a class, instantiate it
                from ash_hawk.graders.base import Grader

                if isinstance(loaded, type) and issubclass(loaded, Grader):
                    grader = loaded()
                elif isinstance(loaded, Grader):
                    grader = loaded
                else:
                    logger.warning(
                        "Entry point '%s' must be a Grader class or instance",
                        entry_point.name,
                    )
                    continue

                # Only register if not already present
                if grader.name in self._graders:
                    logger.warning(
                        "Grader '%s' from entry point '%s' already registered, skipping",
                        grader.name,
                        entry_point.name,
                    )
                else:
                    self.register(grader)

            except Exception as e:
                logger.error(
                    "Failed to load grader from entry point '%s': %s",
                    entry_point.name,
                    e,
                )

    def load_from_project_entry_points(self, project_root: str | Path | None) -> None:
        resolved_root = _find_project_root(project_root)
        if resolved_root is None:
            return
        pyproject_path = resolved_root / "pyproject.toml"
        if not pyproject_path.is_file():
            return
        try:
            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            logger.warning("Failed to read project pyproject.toml at %s: %s", pyproject_path, exc)
            return

        entry_points_section = (
            data.get("project", {}).get("entry-points", {}).get(ENTRY_POINT_GROUP, {})
        )
        if not isinstance(entry_points_section, dict):
            return

        for entry_name, target in entry_points_section.items():
            if not isinstance(entry_name, str) or not isinstance(target, str):
                continue
            try:
                grader = _load_project_grader(target, resolved_root)
            except Exception as exc:
                logger.error(
                    "Failed to load grader from project entry point '%s' in %s: %s",
                    entry_name,
                    resolved_root,
                    exc,
                )
                continue

            if grader.name in self._graders:
                logger.warning(
                    "Grader '%s' from project entry point '%s' already registered, skipping",
                    grader.name,
                    entry_name,
                )
                continue
            self.register(grader)

    def __len__(self) -> int:
        """Return the number of registered graders."""
        return len(self._graders)

    def __contains__(self, name: str) -> bool:
        """Check if a grader with the given name is registered."""
        return name in self._graders


# Global default registry instance
_default_registry: GraderRegistry | None = None


def _register_builtin_graders(registry: GraderRegistry) -> None:
    from ash_hawk.graders.code import (
        StaticAnalysisGrader,
        StringMatchGrader,
        TestRunnerGrader,
        ToolCallGrader,
        TranscriptGrader,
    )
    from ash_hawk.graders.diff_constraints import DiffConstraintsGrader
    from ash_hawk.graders.emotional import EmotionalGrader
    from ash_hawk.graders.llm_boolean import LLMBooleanJudgeGrader
    from ash_hawk.graders.llm_boolean_specialized import create_boolean_graders
    from ash_hawk.graders.llm_judge import LLMJudgeGrader
    from ash_hawk.graders.llm_rubric import LLMRubricGrader
    from ash_hawk.graders.prompt_stack_optimizer import PromptStackOptimizerGrader
    from ash_hawk.graders.scenario_contracts import (
        CompletionHonestyGrader,
        RepoDiffGrader,
        SummaryTruthfulnessGrader,
        TodoStateGrader,
    )
    from ash_hawk.graders.structured import FormatGrader, SchemaGrader, ToolUsageGrader
    from ash_hawk.graders.trace_assertions import (
        BudgetComplianceGrader,
        EvidenceRequiredGrader,
        OrderingGrader,
        TraceContentGrader,
        TraceQualityGrader,
        TraceSchemaGrader,
        VerifyBeforeDoneGrader,
    )
    from ash_hawk.graders.validity import TranscriptValidityGrader

    registry.register(StringMatchGrader())
    registry.register(TestRunnerGrader())
    registry.register(StaticAnalysisGrader())
    registry.register(ToolCallGrader())
    registry.register(TranscriptGrader())
    registry.register(LLMJudgeGrader())
    registry.register(LLMRubricGrader())
    registry.register(LLMBooleanJudgeGrader())

    for grader in create_boolean_graders():
        registry.register(grader)

    registry.register(SchemaGrader())
    registry.register(FormatGrader())
    registry.register(ToolUsageGrader())
    registry.register(TraceSchemaGrader())
    registry.register(TraceContentGrader())
    registry.register(TraceQualityGrader())
    registry.register(BudgetComplianceGrader())
    registry.register(VerifyBeforeDoneGrader())
    registry.register(EvidenceRequiredGrader())
    registry.register(OrderingGrader())
    registry.register(DiffConstraintsGrader())
    registry.register(TodoStateGrader())
    registry.register(RepoDiffGrader())
    registry.register(CompletionHonestyGrader())
    registry.register(SummaryTruthfulnessGrader())
    registry.register(PromptStackOptimizerGrader())
    registry.register(TranscriptValidityGrader())
    registry.register(EmotionalGrader())


def get_default_registry() -> GraderRegistry:
    """Get the default global grader registry.

    Creates the registry on first access and loads entry points.

    Returns:
        The default GraderRegistry instance.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = GraderRegistry()
        _register_builtin_graders(_default_registry)
        _default_registry.load_from_entry_points()
    return _default_registry


def build_registry(project_root: str | Path | None = None) -> GraderRegistry:
    registry = GraderRegistry()
    _register_builtin_graders(registry)
    registry.load_from_entry_points()
    registry.load_from_project_entry_points(project_root)
    return registry


def _find_project_root(project_root: str | Path | None) -> Path | None:
    if project_root is None:
        return None
    candidate = Path(project_root)
    if candidate.is_file():
        candidate = candidate.parent
    for current in [candidate, *candidate.parents]:
        if (current / "pyproject.toml").is_file():
            return current
    return None


@contextmanager
def _temporary_project_import_paths(project_root: Path) -> Iterator[None]:
    inserted_paths: list[str] = []
    for candidate in (project_root, project_root / "src"):
        if not candidate.is_dir():
            continue
        path_string = str(candidate)
        if path_string in sys.path:
            continue
        sys.path.insert(0, path_string)
        inserted_paths.append(path_string)
    try:
        yield
    finally:
        for path_string in inserted_paths:
            if path_string in sys.path:
                sys.path.remove(path_string)


def _load_project_grader(target: str, project_root: Path) -> Grader:
    module_name, separator, attr_path = target.partition(":")
    if not separator:
        raise ValueError(f"Invalid grader entry point target: {target}")

    with _temporary_project_import_paths(project_root):
        module = importlib.import_module(module_name)
        loaded: object = module
        for part in attr_path.split("."):
            loaded = getattr(loaded, part)

    from ash_hawk.graders.base import Grader

    if isinstance(loaded, type) and issubclass(loaded, Grader):
        return loaded()
    if isinstance(loaded, Grader):
        return loaded
    if callable(loaded):
        candidate = loaded()
        if isinstance(candidate, Grader):
            return candidate
    raise TypeError(f"Project entry point '{target}' did not resolve to a Grader")


__all__ = [
    "GraderRegistry",
    "get_default_registry",
    "build_registry",
    "ENTRY_POINT_GROUP",
]
