"""Grader registry for managing and discovering graders.

This module provides a registry for graders that supports:
- Manual registration of grader instances
- Lookup by name
- Discovery via Python entry points
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, cast

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
        try:
            eps = entry_points(group=ENTRY_POINT_GROUP)
        except TypeError:
            # Python < 3.10 compatibility
            eps = entry_points().get(ENTRY_POINT_GROUP, [])

        for ep in eps:
            try:
                # Entry point can be a Grader class or instance
                loaded = cast(object, ep.load())

                # If it's a class, instantiate it
                from ash_hawk.graders.base import Grader

                if isinstance(loaded, type) and issubclass(loaded, Grader):
                    grader = loaded()
                elif isinstance(loaded, Grader):
                    grader = loaded
                else:
                    logger.warning(
                        "Entry point '%s' must be a Grader class or instance",
                        ep.name,
                    )
                    continue

                # Only register if not already present
                if grader.name in self._graders:
                    logger.warning(
                        "Grader '%s' from entry point '%s' already registered, skipping",
                        grader.name,
                        ep.name,
                    )
                else:
                    self.register(grader)

            except Exception as e:
                logger.error(
                    "Failed to load grader from entry point '%s': %s",
                    ep.name,
                    e,
                )

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
    from ash_hawk.graders.human import ManualReviewGrader
    from ash_hawk.graders.llm_judge import LLMJudgeGrader
    from ash_hawk.graders.structured import FormatGrader, SchemaGrader, ToolUsageGrader
    from ash_hawk.graders.trace_assertions import TraceSchemaGrader, VerifyBeforeDoneGrader

    registry.register(StringMatchGrader())
    registry.register(TestRunnerGrader())
    registry.register(StaticAnalysisGrader())
    registry.register(ToolCallGrader())
    registry.register(TranscriptGrader())
    registry.register(LLMJudgeGrader())
    registry.register(SchemaGrader())
    registry.register(FormatGrader())
    registry.register(ToolUsageGrader())
    registry.register(TraceSchemaGrader())
    registry.register(VerifyBeforeDoneGrader())
    registry.register(ManualReviewGrader())


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


__all__ = ["GraderRegistry", "get_default_registry", "ENTRY_POINT_GROUP"]
