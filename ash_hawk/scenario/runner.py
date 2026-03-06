from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Literal

from ash_hawk import __version__
from ash_hawk.config import EvalConfig, get_config
from ash_hawk.execution.runner import EvalRunner
from ash_hawk.execution.trial import TrialExecutor
from ash_hawk.scenario.agent_runner import ScenarioAgentRunner
from ash_hawk.scenario.loader import discover_scenarios, load_scenario
from ash_hawk.scenario.models import ScenarioGraderSpec, ScenarioV1
from ash_hawk.scenario.registry import ScenarioAdapterRegistry, get_default_adapter_registry
from ash_hawk.storage import FileStorage
from ash_hawk.types import (
    EvalRunSummary,
    EvalSuite,
    EvalTask,
    GraderSpec,
    RunEnvelope,
    ToolSurfacePolicy,
)

PATH_KEYS = {
    "path",
    "file_path",
    "dir_path",
    "directory",
    "source",
    "destination",
    "input_path",
    "output_path",
    "workdir",
}


class ScenarioRunner:
    def __init__(
        self,
        storage_path: str | Path | None = None,
        parallelism: int | None = None,
        tooling_mode: Literal["mock", "record", "replay"] = "mock",
        adapter_registry: ScenarioAdapterRegistry | None = None,
    ) -> None:
        config = get_config()
        resolved_storage = (
            Path(storage_path) if storage_path is not None else config.storage_path_resolved()
        )
        self._storage_root = resolved_storage
        self._storage = FileStorage(base_path=resolved_storage)
        self._tooling_mode = tooling_mode
        self._adapter_registry = adapter_registry or get_default_adapter_registry()
        self._config = EvalConfig(
            parallelism=parallelism or config.parallelism,
            default_timeout_seconds=config.default_timeout_seconds,
            storage_backend=config.storage_backend,
            storage_path=config.storage_path,
            log_level=config.log_level,
            default_tool_policy=config.default_tool_policy,
        )

    async def run_paths(self, paths: list[str]) -> EvalRunSummary:
        scenario_sources = self._load_scenario_sources(paths)
        suite_id = self._suite_id_for_paths(paths, scenario_sources)

        tasks = [self._scenario_to_task(scenario, path) for scenario, path in scenario_sources]
        suite = EvalSuite(
            id=suite_id,
            name=suite_id,
            tasks=tasks,
        )

        await self._storage.save_suite(suite)

        agent_config = {
            "agent_name": "scenario-adapter",
            "provider": "scenario",
            "model": "scenario",
            "model_params": {},
            "seed": None,
            "suite_id": suite_id,
        }

        policy = self._config.default_tool_policy
        run_envelope = self._create_run_envelope(suite, agent_config, policy)
        await self._storage.save_run_envelope(suite.id, run_envelope)

        agent_config["run_id"] = run_envelope.run_id

        scenario_agent_runner = ScenarioAgentRunner(
            adapter_registry=self._adapter_registry,
            tooling_mode=self._tooling_mode,
            artifacts_root=self._storage_root,
        )
        trial_executor = TrialExecutor(
            storage=self._storage,
            policy=policy,
            agent_runner=scenario_agent_runner,
        )
        runner = EvalRunner(self._config, self._storage, trial_executor)
        return await runner.run_suite(
            suite=suite,
            agent_config=agent_config,
            run_envelope=run_envelope,
        )

    def _load_scenario_sources(self, paths: list[str]) -> list[tuple[ScenarioV1, Path]]:
        if not paths:
            raise ValueError("No scenario paths provided")

        scenario_sources: list[tuple[ScenarioV1, Path]] = []
        for raw_path in paths:
            path = Path(raw_path)
            if path.is_dir():
                for scenario_path in discover_scenarios(path):
                    scenario_sources.append((load_scenario(scenario_path), scenario_path))
            else:
                scenario_sources.append((load_scenario(path), path))

        if not scenario_sources:
            raise ValueError("No scenarios found for provided paths")

        return scenario_sources

    def _suite_id_for_paths(
        self,
        paths: list[str],
        scenario_sources: Iterable[tuple[ScenarioV1, Path]],
    ) -> str:
        if len(paths) == 1:
            root = Path(paths[0])
            if root.is_dir():
                return f"scenario-{root.name}"
            scenario = next(iter(scenario_sources))[0]
            return f"scenario-{scenario.id}"

        parent_dirs = {Path(p).resolve().parent for p in paths}
        if len(parent_dirs) == 1:
            parent = next(iter(parent_dirs))
            return f"scenario-{parent.name}"
        return f"scenario-batch-{uuid.uuid4().hex[:8]}"

    def _scenario_to_task(self, scenario: ScenarioV1, path: Path) -> EvalTask:
        resolved_inputs = self._resolve_paths(scenario.inputs, path.parent)
        scenario_resolved = scenario.model_copy(update={"inputs": resolved_inputs})
        grader_specs = [self._grader_from_spec(spec) for spec in scenario.graders]
        timeout_seconds = scenario.budgets.max_time_seconds

        return EvalTask(
            id=scenario.id,
            description=scenario.description,
            input={
                "scenario": scenario_resolved.model_dump(),
                "scenario_path": str(path.resolve()),
                "scenario_root": str(path.parent.resolve()),
            },
            grader_specs=grader_specs,
            tags=["scenario", scenario.sut.adapter],
            metadata={"scenario_path": str(path.resolve())},
            timeout_seconds=timeout_seconds,
        )

    def _grader_from_spec(self, spec: ScenarioGraderSpec) -> GraderSpec:
        return GraderSpec(
            grader_type=spec.grader_type,
            config=dict(spec.config),
            weight=spec.weight,
            required=spec.required,
            timeout_seconds=spec.timeout_seconds,
        )

    def _resolve_paths(self, value: Any, root: Path, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {k: self._resolve_paths(v, root, k) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_paths(item, root, key) for item in value]
        if isinstance(value, str) and key in PATH_KEYS:
            candidate = Path(value)
            if not candidate.is_absolute():
                return str((root / candidate).resolve())
        return value

    def _create_run_envelope(
        self,
        suite: EvalSuite,
        agent_config: dict[str, Any],
        policy: ToolSurfacePolicy,
    ) -> RunEnvelope:
        return RunEnvelope(
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            suite_id=suite.id,
            suite_hash=self._compute_hash(suite.model_dump()),
            harness_version=__version__,
            git_commit=None,
            agent_name=str(agent_config.get("agent_name", "scenario-adapter")),
            agent_version=None,
            provider=str(agent_config.get("provider", "scenario")),
            model=str(agent_config.get("model", "scenario")),
            model_params=dict(agent_config.get("model_params", {})),
            seed=agent_config.get("seed"),
            tool_policy_hash=self._compute_hash(policy.model_dump()),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            os_info=platform.platform(),
            config_snapshot={},
            created_at=datetime.now(UTC).isoformat(),
        )

    @staticmethod
    def _compute_hash(data: dict[str, Any]) -> str:
        content = json.dumps(data, sort_keys=True, default=repr)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


def run_scenarios(paths: list[str]) -> EvalRunSummary:
    return asyncio.run(run_scenarios_async(paths))


async def run_scenarios_async(
    paths: list[str],
    storage_path: str | Path | None = None,
    parallelism: int | None = None,
    tooling_mode: Literal["mock", "record", "replay"] = "mock",
    adapter_registry: ScenarioAdapterRegistry | None = None,
) -> EvalRunSummary:
    runner = ScenarioRunner(
        storage_path=storage_path,
        parallelism=parallelism,
        tooling_mode=tooling_mode,
        adapter_registry=adapter_registry,
    )
    return await runner.run_paths(paths)


__all__ = ["ScenarioRunner", "run_scenarios", "run_scenarios_async"]
