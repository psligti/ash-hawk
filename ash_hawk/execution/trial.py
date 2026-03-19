from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from ash_hawk.events import AHEvents
from ash_hawk.execution.fixtures import FixtureResolver
from ash_hawk.graders.registry import get_default_registry
from ash_hawk.policy import PolicyEnforcer
from ash_hawk.types import (
    EvalOutcome,
    EvalStatus,
    EvalTask,
    EvalTranscript,
    EvalTrial,
    FailureMode,
    GraderResult,
    GraderSpec,
    RunEnvelope,
    ToolSurfacePolicy,
    TrialEnvelope,
    TrialResult,
)

if TYPE_CHECKING:
    from ash_hawk.graders.registry import GraderRegistry
    from ash_hawk.storage import StorageBackend


@runtime_checkable
class AgentRunner(Protocol):
    """Protocol for agent execution.

    Implementations of this protocol handle the actual agent execution
    with policy enforcement integration.
    """

    async def run(
        self,
        task: EvalTask,
        policy_enforcer: PolicyEnforcer,
        config: dict[str, object],
    ) -> tuple[EvalTranscript, EvalOutcome]: ...


class _EventPublisher(Protocol):
    async def publish(self, event_name: str, data: dict[str, object]) -> None: ...


class _FunctionRunner:
    """Adapter to wrap a callable function as an AgentRunner."""

    def __init__(
        self,
        func: Callable[
            [EvalTask, PolicyEnforcer, dict[str, object]],
            tuple[EvalTranscript, EvalOutcome],
        ],
    ) -> None:
        self._func = func

    async def run(
        self,
        task: EvalTask,
        policy_enforcer: PolicyEnforcer,
        config: dict[str, object],
    ) -> tuple[EvalTranscript, EvalOutcome]:
        result = self._func(task, policy_enforcer, config)
        if asyncio.iscoroutine(result):
            return await cast(Awaitable[tuple[EvalTranscript, EvalOutcome]], result)
        return result


def _wire_lesson_injector(runner: AgentRunner, injector: Any) -> None:
    if hasattr(runner, "set_lesson_injector"):
        cast(Any, runner).set_lesson_injector(injector)


def _wire_post_run_hook(runner: AgentRunner, hook: Any) -> None:
    if hasattr(runner, "set_post_run_hook"):
        cast(Any, runner).set_post_run_hook(hook)


class TrialExecutor:
    """Executes a single evaluation trial.

    This class handles the complete lifecycle of a trial:
    - Creates and persists the TrialEnvelope with policy snapshot
    - Executes the agent with policy enforcement
    - Captures full transcript (messages, tool calls, tokens, timing, cost)
    - Handles errors gracefully with proper failure modes
    - Supports graceful cancellation storing partial artifacts
    - Returns TrialResult with the outcome

    Example:
        >>> from ash_hawk.storage import FileStorage
        >>> from ash_hawk.types import ToolSurfacePolicy, EvalTask, RunEnvelope
        >>>
        >>> storage = FileStorage(base_path="./results")
        >>> policy = ToolSurfacePolicy(allowed_tools=["read*"])
        >>> executor = TrialExecutor(storage, policy)
        >>>
        >>> task = EvalTask(id="task-1", input="Hello, world!")
        >>> run_envelope = RunEnvelope(...)
        >>> result = await executor.execute(task, {}, run_envelope)
    """

    def __init__(
        self,
        storage: StorageBackend,
        policy: ToolSurfacePolicy,
        agent_runner: AgentRunner
        | Callable[
            [EvalTask, PolicyEnforcer, dict[str, object]], tuple[EvalTranscript, EvalOutcome]
        ],
        fixture_resolver: FixtureResolver | None = None,
        post_run_hook: Any | None = None,
    ) -> None:
        self._storage = storage
        self._policy = policy
        self._agent_runner: AgentRunner = _wrap_runner(agent_runner)
        self._fixture_resolver = fixture_resolver
        self._post_run_hook = post_run_hook

        if self._post_run_hook is not None:
            _wire_post_run_hook(self._agent_runner, self._post_run_hook)

    @property
    def policy(self) -> ToolSurfacePolicy:
        """Get the current policy being enforced."""
        return self._policy

    @property
    def fixture_resolver(self) -> FixtureResolver | None:
        return self._fixture_resolver

    @property
    def post_run_hook(self) -> Any | None:
        return self._post_run_hook

    def set_post_run_hook(self, hook: Any) -> None:
        self._post_run_hook = hook
        _wire_post_run_hook(self._agent_runner, hook)

    async def execute(
        self,
        task: EvalTask,
        agent_config: dict[str, object],
        run_envelope: RunEnvelope,
        attempt_number: int = 1,
    ) -> TrialResult:
        """Execute a single evaluation trial.

        Args:
            task: The evaluation task to execute.
            agent_config: Agent configuration (model, parameters, etc.).
            run_envelope: Parent run envelope for this trial.
            attempt_number: Attempt number (for tasks with max_attempts > 1).

        Returns:
            TrialResult containing the outcome and transcript.

        Raises:
            asyncio.CancelledError: Re-raised after storing partial results.
        """
        trial_id = f"trial-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        started_at = datetime.now(UTC).isoformat()

        resolved_task = task
        if self._fixture_resolver is not None:
            resolved_task = self._fixture_resolver.inject_fixtures(task)

        trial_envelope = TrialEnvelope(
            trial_id=trial_id,
            run_id=run_envelope.run_id,
            task_id=task.id,
            attempt_number=attempt_number,
            policy_snapshot=self._policy.model_copy(deep=True),
            created_at=datetime.now(UTC).isoformat(),
            started_at=started_at,
        )

        from ash_hawk.events import bus

        event_bus = cast(_EventPublisher, bus)

        await event_bus.publish(
            AHEvents.TRIAL_STARTED,
            {"trial_id": trial_id, "task_id": task.id, "attempt_number": attempt_number},
        )

        transcript = EvalTranscript()
        outcome: EvalOutcome | None = None
        cancelled = False
        timeout_seconds = self._policy.timeout_seconds

        try:
            timeout_seconds = (
                resolved_task.timeout_seconds
                if resolved_task.timeout_seconds is not None
                else self._policy.timeout_seconds
            )

            policy_enforcer = PolicyEnforcer(self._policy)
            runner_config = dict(agent_config)
            runner_config.setdefault("trial_id", trial_id)
            runner_config.setdefault("run_id", run_envelope.run_id)
            runner_config.setdefault("suite_id", run_envelope.suite_id)

            transcript, outcome = await asyncio.wait_for(
                self._agent_runner.run(resolved_task, policy_enforcer, runner_config),
                timeout=timeout_seconds,
            )

        except TimeoutError:
            outcome = EvalOutcome.failure(
                FailureMode.TIMEOUT,
                f"Trial timed out after {timeout_seconds}s",
            )
            transcript = EvalTranscript(
                error_trace="Trial execution timed out",
            )

        except asyncio.CancelledError:
            cancelled = True
            outcome = EvalOutcome.failure(
                FailureMode.CRASH,
                "Trial was cancelled",
            )
            transcript = EvalTranscript(
                error_trace="Trial execution was cancelled",
            )

        except Exception as e:
            outcome = EvalOutcome.failure(
                FailureMode.AGENT_ERROR,
                str(e),
            )
            transcript = EvalTranscript(
                error_trace=_format_exception(e),
            )

        finally:
            end_time = time.time()
            duration_seconds = end_time - start_time

            if transcript.duration_seconds == 0.0:
                transcript = transcript.model_copy(update={"duration_seconds": duration_seconds})

            trial_envelope = trial_envelope.model_copy(
                update={"completed_at": datetime.now(UTC).isoformat()}
            )

            trial_result = TrialResult(
                trial_id=trial_id,
                outcome=outcome or EvalOutcome.failure(FailureMode.CRASH, "Unknown error"),
                transcript=transcript,
            )

            eval_trial_for_grading = EvalTrial(
                id=trial_id,
                task_id=task.id,
                status=_outcome_to_status(trial_result.outcome),
                attempt_number=attempt_number,
                input_snapshot=resolved_task.input,
                result=trial_result,
                envelope=trial_envelope,
            )

            if trial_result.outcome.status == EvalStatus.COMPLETED:
                grader_results = await self._run_graders(
                    eval_trial_for_grading, transcript, resolved_task
                )
                aggregate_score, aggregate_passed = _aggregate_grader_results(
                    resolved_task.grader_specs,
                    grader_results,
                )
                trial_result = trial_result.model_copy(
                    update={
                        "grader_results": grader_results,
                        "aggregate_score": aggregate_score,
                        "aggregate_passed": aggregate_passed,
                    }
                )

            eval_trial = EvalTrial(
                id=trial_id,
                task_id=task.id,
                status=_outcome_to_status(trial_result.outcome),
                attempt_number=attempt_number,
                input_snapshot=resolved_task.input,
                result=trial_result,
                envelope=trial_envelope,
            )

            try:
                await self._storage.save_trial(
                    suite_id=run_envelope.suite_id,
                    run_id=run_envelope.run_id,
                    trial=eval_trial,
                    envelope=trial_envelope,
                    policy=self._policy,
                )
            except Exception:
                pass

            if trial_result.outcome.status == EvalStatus.COMPLETED:
                await event_bus.publish(
                    AHEvents.TRIAL_COMPLETED,
                    {
                        "trial_id": trial_id,
                        "task_id": task.id,
                        "outcome": trial_result.outcome.status.value,
                    },
                )
            else:
                await event_bus.publish(
                    AHEvents.TRIAL_FAILED,
                    {
                        "trial_id": trial_id,
                        "task_id": task.id,
                        "failure_mode": trial_result.outcome.failure_mode.value
                        if trial_result.outcome.failure_mode
                        else None,
                        "error_message": trial_result.outcome.error_message,
                    },
                )

        if cancelled:
            raise asyncio.CancelledError()

        return trial_result

    async def _run_graders(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        task: EvalTask,
    ) -> list[GraderResult]:
        registry = get_default_registry()
        results: list[GraderResult] = []

        for spec in task.grader_specs:
            results.append(await self._run_grader_spec(trial, transcript, spec, registry, task))

        return results

    async def _run_grader_spec(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
        registry: GraderRegistry,
        task: EvalTask,
    ) -> GraderResult:
        if spec.grader_type == "composite":
            return await self._run_composite_grader(trial, transcript, spec, registry, task)

        grader = registry.get(spec.grader_type)
        if grader is None:
            return GraderResult(
                grader_type=spec.grader_type,
                score=0.0,
                passed=False,
                error_message=f"Unknown grader type: {spec.grader_type}",
            )

        started = time.time()

        # Inject expected_output into spec config for graders that need it (e.g., llm_judge)
        enriched_config = dict(spec.config)
        if task.expected_output is not None and "expected_output" not in enriched_config:
            enriched_config["expected_output"] = task.expected_output
        enriched_spec = GraderSpec(
            grader_type=spec.grader_type,
            config=enriched_config,
            weight=spec.weight,
            required=spec.required,
            timeout_seconds=spec.timeout_seconds,
        )
        try:
            if enriched_spec.timeout_seconds is not None:
                result = await asyncio.wait_for(
                    grader.grade(trial, transcript, enriched_spec),
                    timeout=enriched_spec.timeout_seconds,
                )
            else:
                result = await grader.grade(trial, transcript, enriched_spec)
            duration = time.time() - started
            return result.model_copy(update={"execution_time_seconds": duration})
        except TimeoutError:
            return GraderResult(
                grader_type=spec.grader_type,
                score=0.0,
                passed=False,
                error_message=f"Grader timed out after {spec.timeout_seconds}s",
                execution_time_seconds=time.time() - started,
            )
        except Exception as exc:
            return GraderResult(
                grader_type=spec.grader_type,
                score=0.0,
                passed=False,
                error_message=str(exc),
                execution_time_seconds=time.time() - started,
            )

    async def _run_composite_grader(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
        registry: GraderRegistry,
        task: EvalTask,
    ) -> GraderResult:
        nested_specs_data = spec.config.get("graders")
        if not isinstance(nested_specs_data, list):
            return GraderResult(
                grader_type="composite",
                score=0.0,
                passed=False,
                error_message="Composite grader requires non-empty 'graders' list",
            )

        nested_items = cast(list[object], nested_specs_data)
        if len(nested_items) == 0:
            return GraderResult(
                grader_type="composite",
                score=0.0,
                passed=False,
                error_message="Composite grader requires non-empty 'graders' list",
            )

        nested_specs = [GraderSpec.model_validate(item) for item in nested_items]
        nested_results: list[GraderResult] = []
        for nested_spec in nested_specs:
            nested_results.append(
                await self._run_grader_spec(trial, transcript, nested_spec, registry, task)
            )

        mode_raw = spec.config.get("mode", spec.config.get("aggregation", "weighted"))
        mode = str(mode_raw)
        if mode == "weighted_average":
            mode = "weighted"
        threshold = float(spec.config.get("threshold", 0.7))

        parsed_weights: list[float] | None = None
        explicit_weights_raw = spec.config.get("weights")
        if isinstance(explicit_weights_raw, list):
            candidate_weights: list[float] = []
            valid_weights = True
            for raw_weight in explicit_weights_raw:
                if isinstance(raw_weight, (float, int, str)):
                    candidate_weights.append(float(raw_weight))
                else:
                    valid_weights = False
                    break
            if valid_weights:
                parsed_weights = candidate_weights

        if parsed_weights is not None and len(parsed_weights) == len(nested_specs):
            weights = parsed_weights
        else:
            weights = [s.weight for s in nested_specs]

        score, passed = _combine_scores(mode, threshold, weights, nested_results)

        return GraderResult(
            grader_type="composite",
            score=score,
            passed=passed,
            details={
                "mode": mode,
                "threshold": threshold,
                "grader_results": [r.model_dump() for r in nested_results],
                "weights": weights,
            },
        )


def _outcome_to_status(outcome: EvalOutcome) -> EvalStatus:
    """Convert EvalOutcome to EvalStatus."""
    return outcome.status


def _format_exception(exc: Exception) -> str:
    """Format an exception with traceback for storage."""
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return "".join(tb).strip()


def _aggregate_grader_results(
    specs: list[GraderSpec],
    results: list[GraderResult],
) -> tuple[float, bool]:
    if not specs or not results:
        return 0.0, False

    weighted_total = 0.0
    total_weight = 0.0
    required_passed = True

    for spec, result in zip(specs, results, strict=False):
        weighted_total += result.score * spec.weight
        total_weight += spec.weight
        if spec.required and not result.passed:
            required_passed = False

    score = weighted_total / total_weight if total_weight > 0 else 0.0
    passed = required_passed and score >= 0.5
    return score, passed


def _combine_scores(
    mode: str,
    threshold: float,
    weights: list[float],
    results: list[GraderResult],
) -> tuple[float, bool]:
    if not results:
        return 0.0, False

    if mode == "all_or_nothing":
        score = sum(r.score for r in results) / len(results)
        return score, all(r.passed for r in results)

    if mode == "threshold":
        score, _ = _combine_scores("weighted", threshold, weights, results)
        return score, score >= threshold

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0, False
    score = sum(r.score * w for r, w in zip(results, weights, strict=True)) / total_weight
    return score, score >= 0.5
