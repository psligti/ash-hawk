# type-hygiene: skip-file
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import platform
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal

from rich.console import Console

from ash_hawk import __version__
from ash_hawk.config import EvalConfig, get_config
from ash_hawk.context import clear_eval_context, set_eval_context
from ash_hawk.scenario.agent_runner import ScenarioAgentRunner
from ash_hawk.scenario.loader import expand_scenario_targets, load_scenario
from ash_hawk.scenario.models import ScenarioGraderSpec, ScenarioV1
from ash_hawk.scenario.registry import ScenarioAdapterRegistry, get_default_adapter_registry
from ash_hawk.scenario.trial import TrialExecutor
from ash_hawk.tracing import get_telemetry
from ash_hawk.types import (
    EvalOutcome,
    EvalRunSummary,
    EvalStatus,
    EvalSuite,
    EvalTask,
    EvalTrial,
    FailureMode,
    GraderSpec,
    RunEnvelope,
    SuiteMetrics,
    TokenUsage,
    ToolSurfacePolicy,
    TrialResult,
)

if TYPE_CHECKING:
    from ash_hawk.storage import StorageBackend
logger = logging.getLogger(__name__)
console = Console()


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
        tooling_mode: Literal["mock", "record", "replay"] = "record",
        adapter_registry: ScenarioAdapterRegistry | None = None,
        show_failure_patterns: bool = True,
        injector: Any | None = None,
        scenario_timeout_seconds: float | None = None,
        grader_config_overrides: dict[str, Any] | None = None,
        on_trial_progress: Callable[[int, int, int, str], Awaitable[None]] | None = None,
        on_trace_event: Callable[[dict[str, object]], None] | None = None,
        agent_path: Path | None = None,
        adapter_override: str | None = None,
    ) -> None:
        from ash_hawk.storage import FileStorage

        config = get_config()
        resolved_storage = (
            Path(storage_path) if storage_path is not None else config.storage_path_resolved()
        )
        self._storage_root = resolved_storage
        self._storage = FileStorage(base_path=resolved_storage)
        self._tooling_mode: Literal["mock", "record", "replay"] = tooling_mode
        self._adapter_registry = adapter_registry or get_default_adapter_registry()
        self._show_failure_patterns = show_failure_patterns
        self._injector = injector
        self._scenario_timeout_seconds = scenario_timeout_seconds
        self._grader_config_overrides = grader_config_overrides or {}
        self._on_trial_progress = on_trial_progress
        self._on_trace_event = on_trace_event
        self._agent_path = agent_path
        self._adapter_override = adapter_override
        resolved_parallelism = parallelism or config.parallelism
        self._config = EvalConfig(
            parallelism=resolved_parallelism,
            trial_max_workers=resolved_parallelism,
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

        agent_config: dict[str, Any] = {
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
            injector=self._injector,
            scenario_timeout_seconds=self._scenario_timeout_seconds,
            on_trace_event=self._on_trace_event,
            agent_path=self._agent_path,
            adapter_override=self._adapter_override,
        )
        trial_executor = TrialExecutor(
            storage=self._storage,
            policy=policy,
            agent_runner=scenario_agent_runner,
        )
        runner = EvalRunner(
            self._config, self._storage, trial_executor, on_trial_progress=self._on_trial_progress
        )
        summary = await runner.run_suite(
            suite=suite,
            agent_config=agent_config,
            run_envelope=run_envelope,
        )

        # Detect systematic failures and surface them
        failure_summary = self._detect_and_surface_failures(summary)
        self._raise_on_critical_failures(failure_summary)

        return summary

    def _raise_on_critical_failures(self, failure_summary: dict[str, Any]) -> None:
        empty_transcript_trials = failure_summary.get("empty_transcript_trials", [])
        if not isinstance(empty_transcript_trials, list) or not empty_transcript_trials:
            return

        sample = empty_transcript_trials[0]
        trial_id = sample.get("trial_id") if isinstance(sample, dict) else "unknown"
        task_id = sample.get("task_id") if isinstance(sample, dict) else "unknown"
        raise ValueError(
            "Detected empty transcripts; aborting scenario run. "
            f"First affected trial: trial_id={trial_id}, task_id={task_id}."
        )

    def _detect_and_surface_failures(self, summary: EvalRunSummary) -> dict[str, Any]:
        """Detect systematic failures and surface actionable insights.

        Checks for:
        1. All trials with same low score (e.g., 0.25)
        2. All transcripts with error_trace
        3. All transcripts with no observable execution data
        """

        # Collect failure patterns
        failure_patterns: list[dict[str, Any]] = []
        low_score_trials = []
        error_trace_trials = []
        empty_transcript_trials = []

        for trial in summary.trials:
            if trial.result is None:
                continue

            transcript = trial.result.transcript

            # Check for low score
            if trial.result.aggregate_score <= 0.3:
                failing_graders: list[str] = []
                for grader_result in getattr(trial.result, "grader_results", []):
                    if not getattr(grader_result, "passed", False):
                        grader_type = getattr(grader_result, "grader_type", None)
                        if grader_type:
                            failing_graders.append(str(grader_type))

                low_score_trials.append(
                    {
                        "trial_id": trial.id,
                        "task_id": trial.task_id,
                        "score": trial.result.aggregate_score,
                        "error_trace": transcript.error_trace,
                        "has_empty_tool_calls": len(transcript.tool_calls) == 0,
                        "has_empty_messages": len(transcript.messages) == 0,
                        "failing_graders": failing_graders,
                    }
                )

            # Check for error trace
            if transcript.error_trace:
                error_trace_trials.append(
                    {
                        "trial_id": trial.id,
                        "task_id": trial.task_id,
                        "error_trace": transcript.error_trace[:500]
                        if len(transcript.error_trace) > 500
                        else transcript.error_trace,
                    }
                )

            has_no_messages = len(transcript.messages) == 0
            has_no_tool_calls = len(transcript.tool_calls) == 0
            has_no_trace_events = len(transcript.trace_events) == 0
            has_no_agent_response = transcript.agent_response in (None, "")
            has_no_error_trace = transcript.error_trace in (None, "")
            if (
                has_no_messages
                and has_no_tool_calls
                and has_no_trace_events
                and has_no_agent_response
                and has_no_error_trace
            ):
                empty_transcript_trials.append(
                    {
                        "trial_id": trial.id,
                        "task_id": trial.task_id,
                    }
                )

        # Detect patterns
        if low_score_trials:
            failure_patterns.append(
                {
                    "pattern": "low_score",
                    "count": len(low_score_trials),
                    "trials": low_score_trials,
                }
            )

        if error_trace_trials:
            failure_patterns.append(
                {
                    "pattern": "error_trace",
                    "count": len(error_trace_trials),
                    "trials": error_trace_trials,
                }
            )

        if empty_transcript_trials:
            failure_patterns.append(
                {
                    "pattern": "empty_transcript",
                    "count": len(empty_transcript_trials),
                    "trials": empty_transcript_trials,
                }
            )

        # Surface actionable insights
        if failure_patterns and self._show_failure_patterns:
            console.print()
            console.rule("[yellow]Systematic Failures Detected[/yellow]")
            for pattern_info in failure_patterns:
                console.print(f"  [red]Pattern: {pattern_info['pattern']}[/red]")
                console.print(f"  [dim]Affected {pattern_info['count']} trials[/dim]")

                # Show sample errors
                if pattern_info["pattern"] == "error_trace":
                    for trial_info in pattern_info["trials"][:3]:
                        console.print(f"    [dim]Error in {trial_info['trial_id']}:[/dim]")
                        console.print(f"    [dim]{trial_info['error_trace'][:200]}[/dim]")

                if pattern_info["pattern"] == "low_score":
                    console.print()
                    console.print("  [yellow]Insights:[/yellow]")
                    for insight in self._build_low_score_insights(pattern_info["trials"]):
                        console.print(f"  [dim]• {insight}[/dim]")
                elif pattern_info["pattern"] == "empty_transcript":
                    console.print()
                    console.print("  [yellow]Insights:[/yellow]")
                    for insight in self._build_empty_transcript_insights(pattern_info["trials"]):
                        console.print(f"  [dim]• {insight}[/dim]")

        return {
            "failure_patterns": failure_patterns,
            "low_score_trials": low_score_trials,
            "error_trace_trials": error_trace_trials,
            "empty_transcript_trials": empty_transcript_trials,
        }

    def _build_low_score_insights(self, trials: list[dict[str, Any]]) -> list[str]:
        insights: list[str] = []
        if not trials:
            return insights

        scores = [float(t["score"]) for t in trials if t.get("score") is not None]
        if scores:
            insights.append(
                f"Score range {min(scores):.2f} to {max(scores):.2f}, average {sum(scores) / len(scores):.2f}"
            )

        empty_tool_calls = sum(1 for trial in trials if trial.get("has_empty_tool_calls"))
        if empty_tool_calls:
            insights.append(
                f"{empty_tool_calls}/{len(trials)} low-score trial(s) made zero tool calls"
            )

        empty_messages = sum(1 for trial in trials if trial.get("has_empty_messages"))
        if empty_messages:
            insights.append(
                f"{empty_messages}/{len(trials)} low-score trial(s) produced no conversation messages"
            )

        grader_counts: dict[str, int] = {}
        for trial in trials:
            for grader_type in trial.get("failing_graders", []):
                grader_counts[grader_type] = grader_counts.get(grader_type, 0) + 1
        if grader_counts:
            top_graders = sorted(grader_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
            grader_summary = ", ".join(f"{name} ({count})" for name, count in top_graders)
            insights.append(f"Most common failing graders: {grader_summary}")

        if not insights:
            insights.append("Low scores were detected, but no richer failure signals were recorded")

        return insights

    def _build_empty_transcript_insights(self, trials: list[dict[str, Any]]) -> list[str]:
        if not trials:
            return []

        trial_ids = ", ".join(str(trial["trial_id"]) for trial in trials[:3])
        insights = [
            "No messages, tool calls, trace events, response, or error trace were recorded",
            f"Affected trial ids: {trial_ids}",
        ]
        if len(trials) > 3:
            insights.append(f"Additional empty transcripts: {len(trials) - 3} more")
        return insights

    def _load_scenario_sources(self, paths: list[str]) -> list[tuple[ScenarioV1, Path]]:
        if not paths:
            raise ValueError("No scenario paths provided")

        scenario_sources: list[tuple[ScenarioV1, Path]] = []
        for raw_path in paths:
            for scenario_path in expand_scenario_targets(raw_path):
                scenario_sources.append((load_scenario(scenario_path), scenario_path))

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
        timeout_seconds = self._scenario_timeout_seconds
        if timeout_seconds is None:
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
        merged_config = dict(spec.config)
        merged_config.update(self._grader_config_overrides)
        return GraderSpec(
            grader_type=spec.grader_type,
            config=merged_config,
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
        if isinstance(value, str) and key == "prompt":
            return self._resolve_placeholders(value, root)
        return value

    def _resolve_placeholders(self, text: str, root: Path) -> str:
        replacements = {
            "scenario_root": str(root.resolve()),
            "scenario_path": str((root / "scenario.yaml").resolve()),
        }
        for placeholder, replacement in replacements.items():
            text = text.replace(f"{{{placeholder}}}", replacement)
        return text

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
    tooling_mode: Literal["mock", "record", "replay"] = "record",
    adapter_registry: ScenarioAdapterRegistry | None = None,
    show_failure_patterns: bool = True,
    injector: Any | None = None,
    scenario_timeout_seconds: float | None = None,
    grader_config_overrides: dict[str, Any] | None = None,
    on_trial_progress: Callable[[int, int, int, str], Awaitable[None]] | None = None,
    agent_path: Path | None = None,
    adapter_override: str | None = None,
) -> EvalRunSummary:
    runner = ScenarioRunner(
        storage_path=storage_path,
        parallelism=parallelism,
        tooling_mode=tooling_mode,
        adapter_registry=adapter_registry,
        show_failure_patterns=show_failure_patterns,
        injector=injector,
        scenario_timeout_seconds=scenario_timeout_seconds,
        grader_config_overrides=grader_config_overrides,
        on_trial_progress=on_trial_progress,
        agent_path=agent_path,
        adapter_override=adapter_override,
    )
    return await runner.run_paths(paths)


# ---------------------------------------------------------------------------
# Absorbed from execution/runner.py
# ---------------------------------------------------------------------------


class ResourceTracker:
    """Tracks resource usage during suite execution."""

    def __init__(self) -> None:
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self.total_tokens = TokenUsage()
        self.total_cost_usd = 0.0
        self.total_duration_seconds = 0.0

    def _get_lock(self) -> asyncio.Lock:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._lock is None or self._lock_loop != current_loop:
            self._lock = asyncio.Lock()
            self._lock_loop = current_loop
        assert self._lock is not None
        return self._lock

    async def add_trial_usage(
        self,
        tokens: TokenUsage,
        cost_usd: float,
        duration_seconds: float,
    ) -> None:
        async with self._get_lock():
            self.total_tokens = TokenUsage(
                input=self.total_tokens.input + tokens.input,
                output=self.total_tokens.output + tokens.output,
                reasoning=self.total_tokens.reasoning + tokens.reasoning,
                cache_read=self.total_tokens.cache_read + tokens.cache_read,
                cache_write=self.total_tokens.cache_write + tokens.cache_write,
            )
            self.total_cost_usd += cost_usd
            self.total_duration_seconds += duration_seconds


class EvalRunner:
    """Parallel suite runner with timeout, budget, and cancellation support."""

    def __init__(
        self,
        config: EvalConfig,
        storage: StorageBackend,
        trial_executor: TrialExecutor,
        post_run_hook: Any | None = None,
        on_trial_progress: Callable[[int, int, int, str], Awaitable[None]] | None = None,
    ) -> None:
        self._config = config
        self._storage = storage
        self._trial_executor = trial_executor
        self._on_trial_progress = on_trial_progress
        self._cancelled = False
        self._resource_tracker = ResourceTracker()
        self._trials: list[Any] = []
        self._trial_durations: list[float] = []

        if post_run_hook is not None:
            self._trial_executor.set_post_run_hook(post_run_hook)

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True

    async def run_suite(
        self,
        suite: EvalSuite,
        agent_config: dict[str, Any],
        run_envelope: RunEnvelope,
    ) -> EvalRunSummary:
        self._cancelled = False
        self._resource_tracker = ResourceTracker()
        self._trials = []
        self._trial_durations = []

        set_eval_context(run_id=run_envelope.run_id, suite_id=suite.id)

        start_time = time.time()

        try:
            semaphore = asyncio.Semaphore(self._config.trial_max_workers)

            async def execute_task(task: EvalTask) -> tuple[EvalTask, TrialResult | Exception]:
                async with semaphore:
                    if self._cancelled:
                        return task, Exception("Suite cancelled")
                    try:
                        result = await self._trial_executor.execute(
                            task=task,
                            agent_config=agent_config,
                            run_envelope=run_envelope,
                        )
                        return task, result
                    except Exception as e:
                        return task, e

            tasks = [execute_task(task) for task in suite.tasks]
            total_tasks = len(tasks)
            completed_count = 0
            results: list[tuple[EvalTask, TrialResult | Exception]] = []

            for coro in asyncio.as_completed(tasks):
                trial_result = await coro
                results.append(trial_result)
                completed_count += 1
                if self._on_trial_progress is not None:
                    task_obj = trial_result[0]
                    try:
                        await self._on_trial_progress(
                            completed_count,
                            total_tasks - completed_count,
                            total_tasks,
                            task_obj.id,
                        )
                    except Exception:  # nosec B110 — callback must not crash evaluation
                        pass

            for task, result in results:
                if isinstance(result, Exception):
                    trial = _create_failed_trial(task, str(result))
                    self._trials.append(trial)
                else:
                    await self._resource_tracker.add_trial_usage(
                        tokens=result.transcript.token_usage,
                        cost_usd=result.transcript.cost_usd,
                        duration_seconds=result.transcript.duration_seconds,
                    )
                    self._trial_durations.append(result.transcript.duration_seconds)

                    trial_id = str(agent_config.get("trial_id") or f"trial-{uuid.uuid4().hex[:8]}")
                    trial = EvalTrial(
                        id=trial_id,
                        task_id=task.id,
                        status=result.outcome.status,
                        attempt_number=1,
                        input_snapshot=task.input,
                        result=result,
                    )
                    self._trials.append(trial)

        except asyncio.CancelledError:
            self._cancelled = True

        finally:
            end_time = time.time()
            suite_duration = end_time - start_time
            clear_eval_context()

        metrics = _build_metrics(
            suite=suite,
            run_envelope=run_envelope,
            suite_duration=suite_duration,
            trials=self._trials,
            resource_tracker=self._resource_tracker,
            trial_durations=self._trial_durations,
        )

        summary = EvalRunSummary(
            envelope=run_envelope,
            metrics=metrics,
            trials=self._trials,
        )

        get_telemetry().emit(
            "suite.completed",
            suite_id=suite.id,
            run_id=run_envelope.run_id,
            total_tasks=metrics.total_tasks,
            completed_tasks=metrics.completed_tasks,
            passed_tasks=metrics.passed_tasks,
            pass_rate=round(metrics.pass_rate, 4),
            mean_score=round(metrics.mean_score, 4),
            duration_s=round(suite_duration, 3),
            trial_ids=[t.id for t in self._trials],
        )

        try:
            await self._storage.save_summary(
                suite_id=suite.id,
                run_id=run_envelope.run_id,
                summary=summary,
            )
        except Exception:  # nosec B110 — storage failure must not crash runner
            pass

        return summary


def _create_failed_trial(task: EvalTask, error: str) -> Any:
    trial_id = f"trial-{uuid.uuid4().hex[:8]}"
    return EvalTrial(
        id=trial_id,
        task_id=task.id,
        status=EvalStatus.ERROR,
        attempt_number=1,
        input_snapshot=task.input,
        result=TrialResult(
            trial_id=trial_id,
            outcome=EvalOutcome.failure(FailureMode.AGENT_ERROR, error),
        ),
    )


def _build_metrics(
    suite: EvalSuite,
    run_envelope: RunEnvelope,
    suite_duration: float,
    trials: list[Any],
    resource_tracker: ResourceTracker,
    trial_durations: list[float],
) -> SuiteMetrics:
    total_tasks = len(suite.tasks)
    completed_tasks = 0
    passed_tasks = 0
    failed_tasks = 0
    total_score = 0.0

    for trial in trials:
        if trial.status == EvalStatus.COMPLETED:
            completed_tasks += 1
            if trial.result and trial.result.aggregate_passed:
                passed_tasks += 1
            if trial.result:
                total_score += trial.result.aggregate_score
        elif trial.status == EvalStatus.ERROR:
            failed_tasks += 1
            completed_tasks += 1
        elif trial.status == EvalStatus.CANCELLED:
            failed_tasks += 1

    pass_rate = passed_tasks / completed_tasks if completed_tasks > 0 else 0.0
    mean_score = total_score / completed_tasks if completed_tasks > 0 else 0.0

    latency_p50 = None
    latency_p95 = None
    latency_p99 = None

    if trial_durations:
        sorted_durations = sorted(trial_durations)
        n = len(sorted_durations)
        latency_p50 = sorted_durations[int(n * 0.5)]
        latency_p95 = sorted_durations[int(n * 0.95)] if n >= 20 else sorted_durations[-1]
        latency_p99 = sorted_durations[int(n * 0.99)] if n >= 100 else sorted_durations[-1]

    return SuiteMetrics(
        suite_id=suite.id,
        run_id=run_envelope.run_id,
        total_tasks=total_tasks,
        completed_tasks=completed_tasks,
        passed_tasks=passed_tasks,
        failed_tasks=failed_tasks,
        pass_rate=pass_rate,
        mean_score=mean_score,
        total_tokens=resource_tracker.total_tokens,
        total_cost_usd=resource_tracker.total_cost_usd,
        total_duration_seconds=suite_duration,
        latency_p50_seconds=latency_p50,
        latency_p95_seconds=latency_p95,
        latency_p99_seconds=latency_p99,
        created_at=datetime.now(UTC).isoformat(),
    )


__all__ = [
    "EvalRunner",
    "ResourceTracker",
    "ScenarioRunner",
    "run_scenarios",
    "run_scenarios_async",
]
