"""Thin scenario runner using the dawn-kestrel bridge.

This runner uses the thin telemetry bridge to run real dawn-kestrel agents
instead of using the adapter registry pattern. It provides:
- Direct agent loading from .dawn-kestrel/agents/{name}
- Real-time telemetry via RuntimeHook callbacks
- Integrated grading from scenario YAML grader specs
- Unified diff improvements instead of lesson store
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ash_hawk.bridge import (
    OutcomeData,
    RunResult,
    TelemetrySink,
    TranscriptData,
    run_real_agent,
)
from ash_hawk.graders.registry import get_default_registry
from ash_hawk.scenario.models import ScenarioGraderSpec, ScenarioV1
from ash_hawk.types import EvalTrial, GraderResult, GraderSpec

logger = logging.getLogger(__name__)


@dataclass
class ThinGradedResult:
    """Result from thin scenario run with grading.

    Contains both the agent run result and grader results.
    """

    run_result: RunResult
    grader_results: list[GraderResult] = field(default_factory=list)

    def all_passed(self) -> bool:
        """Return True if all graders passed."""
        return all(r.passed for r in self.grader_results)

    @property
    def aggregate_score(self) -> float:
        """Return aggregate score across all graders."""
        if not self.grader_results:
            return 0.0
        return sum(r.score for r in self.grader_results) / len(self.grader_results)


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
    4. Optionally runs graders from scenario YAML
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

    async def run_with_grading(
        self,
        scenario: ScenarioV1,
        scenario_path: Path,
    ) -> ThinGradedResult:
        """Run a scenario with integrated grading.

        This method:
        1. Runs the real dawn-kestrel agent via run_real_agent
        2. Converts transcript to EvalTranscript for grading
        3. Runs each grader from scenario.graders
        4. Returns structured result with both run and grade results

        Args:
            scenario: The scenario to run
            scenario_path: Path to the scenario YAML file

        Returns:
            ThinGradedResult with run result and grader results
        """
        result = await run_real_agent(
            agent_path=self._resolve_agent_path(scenario),
            input=self._build_input(scenario),
            telemetry_sink=ScenarioTelemetrySink(),
            max_iterations=self.max_iterations,
            workdir=self.workdir,
        )

        if not scenario.graders:
            return ThinGradedResult(run_result=result, grader_results=[])

        eval_transcript = result.transcript.to_eval_transcript()

        trial = EvalTrial(
            id=f"trial-{scenario.id}",
            task_id=scenario.id,
            input_snapshot={},
        )

        registry = get_default_registry()
        grader_results: list[GraderResult] = []

        for grader_spec in scenario.graders:
            grader = registry.get(grader_spec.grader_type)
            if grader is None:
                logger.warning(
                    "Grader '%s' not found in registry, skipping",
                    grader_spec.grader_type,
                )
                continue

            spec = GraderSpec(
                grader_type=grader_spec.grader_type,
                config=grader_spec.config,
                weight=grader_spec.weight,
                required=grader_spec.required,
                timeout_seconds=grader_spec.timeout_seconds,
            )

            try:
                grader_result = await grader.grade(trial, eval_transcript, spec)
                grader_results.append(grader_result)
            except Exception as e:
                logger.exception("Grader '%s' failed", grader_spec.grader_type)
                grader_results.append(
                    GraderResult(
                        grader_type=grader_spec.grader_type,
                        score=0.0,
                        passed=False,
                        error_message=str(e),
                    )
                )

        return ThinGradedResult(
            run_result=result,
            grader_results=grader_results,
        )

    def _resolve_agent_path(self, scenario: ScenarioV1) -> Path:
        """Resolve the agent path from scenario config.

        Looks for agent in:
        1. scenario.sut.agent (explicit path)
        2. .dawn-kestrel/agents/{adapter_name}
        3. .opencode/agent/{adapter_name}.md
        """
        adapter_name = scenario.sut.adapter
        if adapter_name in {"bolt_merlin", "bolt-merlin"}:
            dawn_root = self.workdir / ".dawn-kestrel"
            if dawn_root.exists():
                return dawn_root

        adapter_candidates = [adapter_name]
        hyphenated_name = adapter_name.replace("_", "-")
        if hyphenated_name not in adapter_candidates:
            adapter_candidates.append(hyphenated_name)

        if hasattr(scenario.sut, "agent") and scenario.sut.agent:
            return Path(scenario.sut.agent)

        for candidate in adapter_candidates:
            dawn_kestrel_path = self.workdir / ".dawn-kestrel" / "agents" / candidate
            if dawn_kestrel_path.exists():
                return dawn_kestrel_path

        for candidate in adapter_candidates:
            opencode_path = self.workdir / ".opencode" / "agent" / f"{candidate}.md"
            if opencode_path.exists():
                return opencode_path.parent / candidate

        dawn_root = self.workdir / ".dawn-kestrel"
        if dawn_root.exists():
            logger.warning("Agent not found for '%s'; falling back to %s", adapter_name, dawn_root)
            return dawn_root

        logger.warning(
            "Agent not found for '%s'; falling back to workdir %s", adapter_name, self.workdir
        )
        return self.workdir

    def _resolve_agent_name(self, scenario: ScenarioV1) -> str:
        """Resolve the agent name from scenario config."""
        run_config = scenario.sut.config
        configured_agent = run_config.get("agent")
        if isinstance(configured_agent, str) and configured_agent.strip():
            return configured_agent.strip()

        adapter_name = scenario.sut.adapter
        if adapter_name in {"bolt_merlin", "bolt-merlin"}:
            return "orchestrator"
        return adapter_name

    def _build_input(self, scenario: ScenarioV1) -> str:
        """Build the input prompt from scenario."""
        if scenario.inputs:
            prompt = scenario.inputs.get("prompt", "")
            if isinstance(prompt, str) and prompt:
                return prompt

        return scenario.description or "Execute the scenario task"


__all__ = ["ThinScenarioRunner", "ScenarioTelemetrySink", "ThinGradedResult"]
