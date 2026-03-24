"""Scenario adapter interface for running evaluation scenarios.

This module defines the ScenarioAdapter protocol that all scenario runners
must implement. Scenario adapters are responsible for executing specific
types of evaluation scenarios (e.g., coding tasks, conversational tasks, etc.).

Scenario adapters are discovered and loaded via the registry system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["ScenarioAdapter"]


@runtime_checkable
class ScenarioAdapter(Protocol):
    """Protocol defining the interface for scenario adapters.

    A scenario adapter is responsible for running a specific type of evaluation
    scenario and returning the results. Each adapter must implement this protocol
    to be registered with the ScenarioAdapterRegistry.

    Example:
        >>> class CodingScenarioAdapter:
        ...     @property
        ...     def name(self) -> str:
        ...         return "coding"
        ...
    ...     def run_scenario(
    ...         self,
    ...         scenario: dict[str, Any],
    ...         workdir: Path,
    ...         tooling_harness: dict[str, Any],
    ...         budgets: dict[str, Any],
    ...     ) -> tuple[Any, list[Any], dict[str, Any], Any]:
    ...         # Run the coding scenario
    ...         final_output = "result"
    ...         trace_events = []
    ...         artifacts = {"files": {}}
    ...         outcome = None
    ...         return final_output, trace_events, artifacts, outcome
    """

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
        scenario: dict[str, Any],
        workdir: Path,
        tooling_harness: dict[str, Any],
        budgets: dict[str, Any],
    ) -> tuple[Any, ...]:
        """Execute a scenario and return results.

        Args:
            scenario: Scenario configuration and parameters.
            workdir: Working directory for scenario execution.
            tooling_harness: Tool configuration and constraints.
            budgets: Resource budgets (tokens, time, cost, etc.).

        Returns:
            A tuple containing (4-6 values):
            - final_output: The primary output from the scenario execution
            - trace_events: List of trace events recorded during execution
            - artifacts: Dictionary of artifacts produced (files, logs, etc.)
            - outcome: EvalOutcome or None with status and metadata
            - messages: (optional) List of conversation messages
            - tool_calls: (optional) List of tool calls made during execution
        """
        ...
