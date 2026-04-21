# type-hygiene: skip-file
"""Thin scenario runner using the dawn-kestrel bridge.

This runner uses the thin telemetry bridge to run real dawn-kestrel agents
instead of using the adapter registry pattern. It provides:
- Direct agent loading from .dawn-kestrel/agents/{name}
- Real-time telemetry via RuntimeHook callbacks
- Integrated grading from scenario YAML grader specs
- Unified diff improvements instead of lesson store
- Provenance manifest with config hashes per run
- Structured storage at .ash-hawk/thin/{scenario_stem}/{run_id}/
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ash_hawk.context import clear_eval_context, set_eval_context
from ash_hawk.graders.registry import build_registry, get_default_registry
from ash_hawk.scenario.dawn_kestrel_bridge import run_real_agent
from ash_hawk.scenario.models import ScenarioV1
from ash_hawk.types import (
    EvalTrial,
    GraderResult,
    GraderSpec,
    RunManifest,
    RunResult,
    TelemetrySink,
    build_run_manifest,
)

logger = logging.getLogger(__name__)

STORAGE_DIR_NAME = "thin"
_DEFAULT_STORAGE_ROOT = Path(".ash-hawk") / STORAGE_DIR_NAME


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
    3. Builds provenance manifest with config hashes
    4. Returns transcripts compatible with existing graders
    5. Optionally runs graders from scenario YAML
    6. Persists artifacts to structured storage
    """

    def __init__(
        self,
        workdir: Path | None = None,
        max_iterations: int = 10,
        variant: str = "",
        storage_root: Path | None = None,
        agent_override_path: Path | None = None,
    ) -> None:
        self.workdir = workdir or Path.cwd()
        self.max_iterations = max_iterations
        self.variant = variant
        self.storage_root = storage_root or self.workdir / _DEFAULT_STORAGE_ROOT
        self.agent_override_path = agent_override_path

    def _reset_workspace(self, scenario: ScenarioV1) -> None:
        """Reset workspace files to baseline content from scenario config.

        Writes each entry in scenario.workspace to self.workdir so that
        every graded run starts from the same clean baseline.
        """
        if not scenario.workspace:
            return
        for filename, content in scenario.workspace.items():
            target = self.workdir / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    async def run_scenario(
        self,
        scenario: ScenarioV1,
        scenario_path: Path,
    ) -> RunResult:
        self._reset_workspace(scenario)
        agent_path = self._resolve_agent_path(scenario)
        input_text = self._build_input(scenario)

        sink = ScenarioTelemetrySink()

        manifest = build_run_manifest(
            run_id=None,
            scenario_path=scenario_path,
            agent_path=agent_path,
            model_name="",
            variant=self.variant,
            grader_set=[g.grader_type for g in scenario.graders] if scenario.graders else [],
        )

        set_eval_context(run_id=manifest.run_id, suite_id=scenario.id)

        result = await run_real_agent(
            agent_path=agent_path,
            input=input_text,
            telemetry_sink=sink,
            max_iterations=self.max_iterations,
            workdir=self.workdir,
            run_id=manifest.run_id,
        )
        result.manifest = manifest

        self._persist_run_artifacts(result, scenario_path)
        clear_eval_context()

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
        agent_path = self._resolve_agent_path(scenario)

        self._reset_workspace(scenario)

        grader_types = [g.grader_type for g in scenario.graders] if scenario.graders else []

        manifest = build_run_manifest(
            run_id=None,
            scenario_path=scenario_path,
            agent_path=agent_path,
            model_name="",
            variant=self.variant,
            grader_set=grader_types,
        )

        set_eval_context(run_id=manifest.run_id, suite_id=scenario.id)

        result = await run_real_agent(
            agent_path=agent_path,
            input=self._build_input(scenario),
            telemetry_sink=ScenarioTelemetrySink(),
            max_iterations=self.max_iterations,
            workdir=self.workdir,
            run_id=manifest.run_id,
        )
        result.manifest = manifest

        if not result.outcome.success:
            logger.warning(
                "Agent run failed (outcome: %s), skipping grading for scenario '%s'",
                result.outcome.error or result.outcome.message,
                scenario.id,
            )
            self._persist_run_artifacts(result, scenario_path)
            return ThinGradedResult(run_result=result, grader_results=[])

        if not scenario.graders:
            self._persist_run_artifacts(result, scenario_path)
            return ThinGradedResult(run_result=result, grader_results=[])

        eval_transcript = result.transcript.to_eval_transcript()

        trial = EvalTrial(
            id=f"trial-{scenario.id}",
            task_id=scenario.id,
            input_snapshot={
                "workdir": str(self.workdir),
                "scenario_id": scenario.id,
                "run_id": manifest.run_id,
            },
        )

        registry = build_registry(scenario_path)
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

        graded = ThinGradedResult(
            run_result=result,
            grader_results=grader_results,
        )
        self._persist_run_artifacts(result, scenario_path, grader_results=grader_results)

        clear_eval_context()
        return graded

    def _persist_run_artifacts(
        self,
        result: RunResult,
        scenario_path: Path,
        grader_results: list[GraderResult] | None = None,
    ) -> Path:
        """Write manifest.json, transcript.json, and optionally grades.json.

        Layout: ``{storage_root}/{scenario_stem}/{run_id}/``

        Args:
            result: The run result to persist.
            scenario_path: Path to scenario file (used for stem).
            grader_results: Optional grader results to persist.

        Returns:
            Path to the run directory.
        """
        scenario_stem = scenario_path.stem
        run_id = result.run_id or "unknown"
        run_dir = self.storage_root / scenario_stem / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        if result.manifest:
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text(result.manifest.model_dump_json(indent=2), encoding="utf-8")

        transcript_data: dict[str, Any] = {
            "run_id": run_id,
            "outcome": {
                "success": result.outcome.success,
                "message": result.outcome.message,
                "error": result.outcome.error,
            },
            "transcript": {
                "messages": result.transcript.messages,
                "tool_calls": result.transcript.tool_calls,
                "token_usage": result.transcript.token_usage,
                "duration_seconds": result.transcript.duration_seconds,
                "agent_response": result.transcript.agent_response,
                "error_trace": result.transcript.error_trace,
            },
        }
        (run_dir / "transcript.json").write_text(
            json.dumps(transcript_data, indent=2, default=str), encoding="utf-8"
        )

        if grader_results is not None:
            grades_data = [
                {
                    "grader_type": g.grader_type,
                    "passed": g.passed,
                    "score": g.score,
                    "error_message": g.error_message,
                    "rationale": getattr(g, "rationale", None),
                    "details": getattr(g, "details", None),
                }
                for g in grader_results
            ]
            (run_dir / "grades.json").write_text(
                json.dumps(grades_data, indent=2, default=str), encoding="utf-8"
            )

        return run_dir

    @staticmethod
    def discover_runs(
        storage_root: Path,
        scenario_stem: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find all persisted runs, optionally filtered by scenario stem.

        Args:
            storage_root: Root of the thin storage tree.
            scenario_stem: Optional scenario stem to filter by.

        Returns:
            List of dicts with ``run_id``, ``scenario_stem``, ``path``.
        """
        runs: list[dict[str, Any]] = []
        if not storage_root.is_dir():
            return runs

        stems = (
            [scenario_stem]
            if scenario_stem
            else [d.name for d in sorted(storage_root.iterdir()) if d.is_dir()]
        )

        for stem in stems:
            stem_dir = storage_root / stem
            if not stem_dir.is_dir():
                continue
            for run_dir in sorted(stem_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                manifest_file = run_dir / "manifest.json"
                if manifest_file.is_file():
                    runs.append(
                        {
                            "run_id": run_dir.name,
                            "scenario_stem": stem,
                            "path": str(run_dir),
                        }
                    )
        return runs

    @staticmethod
    def load_manifest(run_dir: Path) -> RunManifest | None:
        """Load a RunManifest from a run directory.

        Args:
            run_dir: Path to the run directory containing manifest.json.

        Returns:
            RunManifest if found, None otherwise.
        """
        manifest_file = run_dir / "manifest.json"
        if not manifest_file.is_file():
            return None
        try:
            return RunManifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to parse manifest in %s", run_dir)
            return None

    def _resolve_agent_path(self, scenario: ScenarioV1) -> Path:
        """Resolve the agent path from scenario config.

        Looks for agent in:
        1. scenario.sut.agent (explicit path)
        2. .dawn-kestrel/agents/{adapter_name}
        3. .opencode/agent/{adapter_name}.md
        """
        if self.agent_override_path is not None:
            return self.agent_override_path

        adapter_name = scenario.sut.adapter
        if adapter_name in {"bolt_merlin", "bolt-merlin"}:
            dawn_root = self.workdir / ".dawn-kestrel"
            if dawn_root.exists():
                return dawn_root

        adapter_candidates = [adapter_name]
        hyphenated_name = adapter_name.replace("_", "-")
        if hyphenated_name not in adapter_candidates:
            adapter_candidates.append(hyphenated_name)

        agent_field = getattr(scenario.sut, "agent", None)
        if isinstance(agent_field, str) and agent_field.strip():
            return Path(agent_field.strip())

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


__all__ = [
    "ThinScenarioRunner",
    "ScenarioTelemetrySink",
    "ThinGradedResult",
    "_DEFAULT_STORAGE_ROOT",
    "STORAGE_DIR_NAME",
]
