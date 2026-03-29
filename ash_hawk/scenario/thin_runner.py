"""Thin scenario runner using the dawn-kestrel bridge.

This runner uses the thin telemetry bridge to run real dawn-kestrel agents
instead of using the adapter registry pattern. It provides:
- Direct agent loading from .dawn-kestrel/agents/{name}
- Real-time telemetry via RuntimeHook callbacks
- Unified diff improvements instead of lesson store
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ash_hawk.bridge import (
    OutcomeData,
    RunResult,
    TelemetrySink,
    TranscriptData,
    run_real_agent,
)
from ash_hawk.scenario.models import ScenarioV1

logger = logging.getLogger(__name__)


class ScenarioTelemetrySink(TelemetrySink):
    """Telemetry sink that collects scenario execution data."""

    def __init__(self) -> None:
        self.iterations: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.policy_decisions: list[dict[str, Any]] = []

    async def on_iteration_start(self, data: dict[str, Any]) -> None:
        self.iterations.append({"type": "start", **data})

    async def on_iteration_end(self, data: dict[str, Any]) -> None:
        self.iterations.append({"type": "end", **data})

    async def on_action_decision(self, data: dict[str, Any]) -> None:
        self.policy_decisions.append(data)

    async def on_tool_result(self, data: dict[str, Any]) -> None:
        self.tool_calls.append(data)

    async def on_run_complete(self, data: dict[str, Any]) -> None:
        pass


class ThinScenarioRunner:
    """Run scenarios using the thin telemetry bridge.

    This runner:
    1. Loads the real agent from .dawn-kestrel/agents/{name}
    2. Runs via run_real_agent with telemetry capture
    3. Returns transcripts compatible with existing graders
    """

    def __init__(
        self,
        workdir: Path | None = None,
        max_iterations: int = 10,
    ) -> None:
        self.workdir = workdir or Path.cwd()
        self.max_iterations = max_iterations

    async def run_scenario(
        self,
        scenario: ScenarioV1,
        scenario_path: Path,
    ) -> RunResult:
        """Run a scenario using the thin bridge.

        Args:
            scenario: The scenario to run
            scenario_path: Path to the scenario YAML file

        Returns:
            RunResult with transcript and outcome
        """
        agent_path = self._resolve_agent_path(scenario)
        input_text = self._build_input(scenario)

        sink = ScenarioTelemetrySink()

        result = await run_real_agent(
            agent_path=agent_path,
            input=input_text,
            telemetry_sink=sink,
            max_iterations=self.max_iterations,
            workdir=self.workdir,
        )

        return result

    def _resolve_agent_path(self, scenario: ScenarioV1) -> Path:
        """Resolve the agent path from scenario config.

        Looks for agent in:
        1. scenario.sut.agent (explicit path)
        2. .dawn-kestrel/agents/{adapter_name}
        3. .opencode/agent/{adapter_name}.md
        """
        adapter_name = scenario.sut.adapter

        if hasattr(scenario.sut, "agent") and scenario.sut.agent:
            return Path(scenario.sut.agent)

        dawn_kestrel_path = self.workdir / ".dawn-kestrel" / "agents" / adapter_name
        if dawn_kestrel_path.exists():
            return dawn_kestrel_path

        opencode_path = self.workdir / ".opencode" / "agent" / f"{adapter_name}.md"
        if opencode_path.exists():
            return opencode_path.parent / adapter_name

        raise ValueError(f"Agent not found: {adapter_name}")

    def _build_input(self, scenario: ScenarioV1) -> str:
        """Build the input prompt from scenario."""
        if scenario.inputs:
            prompt = scenario.inputs.get("prompt", "")
            if isinstance(prompt, str) and prompt:
                return prompt

        return scenario.description or "Execute the scenario task"


__all__ = ["ThinScenarioRunner", "ScenarioTelemetrySink"]
