"""Scenario adapter interface for running evaluation scenarios.

This module defines the ScenarioAdapter protocol that all scenario runners
must implement. Scenario adapters are responsible for executing specific
types of evaluation scenarios (e.g., coding tasks, conversational tasks, etc.).

Scenario adapters are discovered and loaded via the registry system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ash_hawk.scenario.models import JSONValue, ScenarioAdapterResult

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["ScenarioAdapter"]


@runtime_checkable
class ScenarioAdapter(Protocol):
    """Protocol for executable scenario adapters."""

    @property
    def name(self) -> str:
        """Unique identifier for this scenario adapter.

        This name is used to register and retrieve the adapter from the registry.

        Returns:
            The adapter's unique name (e.g., "coding", "conversational", "tool_use").
        """
        ...

    def run_scenario(
        self,
        scenario: dict[str, JSONValue],
        workdir: Path,
        tooling_harness: dict[str, object],
        budgets: dict[str, JSONValue],
    ) -> ScenarioAdapterResult:
        """Execute a scenario and return model-based results."""
        ...
