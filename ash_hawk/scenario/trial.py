# type-hygiene: skip-file
from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from ash_hawk.context import set_eval_context
from ash_hawk.graders.registry import build_registry
from ash_hawk.policy import PolicyEnforcer
from ash_hawk.scenario.fixtures import FixtureResolver
from ash_hawk.tracing import get_telemetry
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

logger = logging.getLogger(__name__)

_TRANSIENT_GRADER_ERROR_PATTERNS = (
    re.compile(r"provider operation failed", re.IGNORECASE),
    re.compile(r"network error", re.IGNORECASE),
    re.compile(r"connection reset", re.IGNORECASE),
    re.compile(r"temporarily unavailable", re.IGNORECASE),
    re.compile(r"api error \(code=5\d\d\)", re.IGNORECASE),
    re.compile(r"api error \(code=1234\)", re.IGNORECASE),
    re.compile(r"rate limit", re.IGNORECASE),
)
_TRANSIENT_GRADER_MAX_ATTEMPTS = 3


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


def _wrap_runner(
    runner: AgentRunner
    | Callable[[EvalTask, PolicyEnforcer, dict[str, object]], tuple[EvalTranscript, EvalOutcome]],
) -> AgentRunner:
    if runner is None:
        raise TypeError("agent_runner is required and cannot be None")
    if isinstance(runner, AgentRunner):
        return runner
    return _FunctionRunner(runner)


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
        lesson_injector: Any | None = None,
    ) -> None:
        self._storage = storage
        self._policy = policy
        self._agent_runner: AgentRunner = _wrap_runner(agent_runner)
        self._fixture_resolver = fixture_resolver
        self._post_run_hook = post_run_hook
        self._lesson_injector = lesson_injector

        if self._post_run_hook is not None:
            _wire_post_run_hook(self._agent_runner, self._post_run_hook)

        if self._lesson_injector is not None:
            _wire_lesson_injector(self._agent_runner, self._lesson_injector)

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

    @property
    def lesson_injector(self) -> Any | None:
        return self._lesson_injector

    def set_post_run_hook(self, hook: Any) -> None:
        self._post_run_hook = hook
        _wire_post_run_hook(self._agent_runner, hook)

    def set_lesson_injector(self, injector: Any) -> None:
        self._lesson_injector = injector
        _wire_lesson_injector(self._agent_runner, injector)

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

        set_eval_context(trial_id=trial_id)

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

        transcript = EvalTranscript()
        outcome: EvalOutcome | None = None
        cancelled = False
        timeout_seconds = self._policy.timeout_seconds
        logger.info(
            "Trial %s started (run=%s task=%s attempt=%s)",
            trial_id,
            run_envelope.run_id,
            task.id,
            attempt_number,
        )

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

            run_task = asyncio.create_task(
                self._agent_runner.run(resolved_task, policy_enforcer, runner_config)
            )
            logger.info(
                "Trial %s executing agent runner with timeout=%ss",
                trial_id,
                timeout_seconds,
            )
            _done, pending = await asyncio.wait({run_task}, timeout=timeout_seconds)

            if pending:
                run_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await run_task
                logger.warning(
                    "Trial %s timed out waiting for agent runner after %ss",
                    trial_id,
                    timeout_seconds,
                )
                get_telemetry().emit(
                    "trial.agent_timeout",
                    trial_id=trial_id,
                    run_id=run_envelope.run_id,
                    task_id=task.id,
                    timeout_s=timeout_seconds,
                )
                outcome = EvalOutcome.failure(
                    FailureMode.TIMEOUT,
                    f"Trial timed out after {timeout_seconds}s",
                )
                transcript = transcript.model_copy(
                    update={"error_trace": "Trial execution timed out"}
                )
            else:
                transcript, outcome = run_task.result()
                logger.info("Trial %s agent runner completed", trial_id)
                get_telemetry().emit(
                    "trial.agent_completed",
                    trial_id=trial_id,
                    run_id=run_envelope.run_id,
                    task_id=task.id,
                    status=outcome.status.value,
                    duration_s=round(time.time() - start_time, 3),
                    tool_call_count=len(transcript.tool_calls),
                    message_count=len(transcript.messages),
                )

        except asyncio.CancelledError:
            cancelled = True
            outcome = EvalOutcome.failure(
                FailureMode.CRASH,
                "Trial was cancelled",
            )
            transcript = transcript.model_copy(
                update={"error_trace": "Trial execution was cancelled"}
            )

        except Exception as e:
            outcome = EvalOutcome.failure(
                FailureMode.AGENT_ERROR,
                str(e),
            )
            transcript = transcript.model_copy(update={"error_trace": _format_exception(e)})

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

            if trial_result.outcome.status in (
                EvalStatus.COMPLETED,
                EvalStatus.ERROR,
            ):
                has_transcript_content = (
                    bool(transcript.messages) or transcript.agent_response is not None
                )
                has_tool_activity = bool(transcript.tool_calls) or bool(transcript.trace_events)
                if not has_transcript_content and not has_tool_activity:
                    logger.warning(
                        "Trial %s completed but transcript is empty "
                        "(no messages, no agent_response, no tool activity), skipping grading",
                        trial_id,
                    )
                    get_telemetry().emit(
                        "trial.grading_skipped",
                        trial_id=trial_id,
                        reason="empty_transcript",
                    )
                else:
                    logger.info(
                        "Trial %s starting grading with %d graders",
                        trial_id,
                        len(resolved_task.grader_specs),
                    )
                    grader_results = await self._run_graders(
                        eval_trial_for_grading, transcript, resolved_task
                    )
                    logger.info(
                        "Trial %s grading completed with %d results",
                        trial_id,
                        len(grader_results),
                    )
                    aggregate_score, aggregate_passed = _aggregate_grader_results(
                        resolved_task.grader_specs,
                        grader_results,
                    )
                    grader_summary = [
                        {
                            "type": r.grader_type,
                            "score": r.score,
                            "passed": r.passed,
                            "error": r.error_message,
                        }
                        for r in grader_results
                    ]
                    get_telemetry().emit(
                        "trial.grading_completed",
                        trial_id=trial_id,
                        run_id=run_envelope.run_id,
                        task_id=task.id,
                        outcome_status=trial_result.outcome.status.value,
                        aggregate_score=round(aggregate_score, 4),
                        aggregate_passed=aggregate_passed,
                        grader_count=len(grader_results),
                        graders=grader_summary,
                    )
                    trial_result = trial_result.model_copy(
                        update={
                            "grader_results": grader_results,
                            "aggregate_score": aggregate_score,
                            "aggregate_passed": aggregate_passed,
                        }
                    )
            else:
                get_telemetry().emit(
                    "trial.grading_skipped",
                    trial_id=trial_id,
                    run_id=run_envelope.run_id,
                    task_id=task.id,
                    reason=f"outcome_status={trial_result.outcome.status.value}",
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
                logger.info("Trial %s saving trial result to storage", trial_id)
                await self._storage.save_trial(
                    suite_id=run_envelope.suite_id,
                    run_id=run_envelope.run_id,
                    trial=eval_trial,
                    envelope=trial_envelope,
                    policy=self._policy,
                )
                logger.info("Trial %s storage save complete", trial_id)
            except Exception:
                logger.exception("Trial %s failed to persist storage record", trial_id)

        if cancelled:
            raise asyncio.CancelledError()

        return trial_result

    async def _run_graders(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        task: EvalTask,
    ) -> list[GraderResult]:
        registry = build_registry(_task_project_root(task, trial))

        if not task.grader_specs:
            return []

        # Run graders sequentially to avoid rate-limit cascade with shared provider
        results: list[GraderResult] = []
        for spec in task.grader_specs:
            try:
                result = await self._run_grader_spec(trial, transcript, spec, registry, task)
                results.append(result)
            except BaseException as exc:
                results.append(
                    GraderResult(
                        grader_type=spec.grader_type,
                        score=0.0,
                        passed=False,
                        error_message=f"Grader raised exception: {exc}",
                    )
                )

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
        logger.info(
            "Trial %s grader %s started (timeout=%s)",
            trial.id,
            spec.grader_type,
            spec.timeout_seconds,
        )

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
            result = await self._run_grader_with_retries(
                grader=grader,
                trial=trial,
                transcript=transcript,
                spec=enriched_spec,
                started=started,
            )
            duration = time.time() - started
            logger.info(
                "Trial %s grader %s completed in %.2fs",
                trial.id,
                spec.grader_type,
                duration,
            )
            return result.model_copy(update={"execution_time_seconds": duration})
        except TimeoutError:
            logger.warning(
                "Trial %s grader %s timed out after %ss",
                trial.id,
                spec.grader_type,
                spec.timeout_seconds,
            )
            return GraderResult(
                grader_type=spec.grader_type,
                score=0.0,
                passed=False,
                error_message=f"Grader timed out after {spec.timeout_seconds}s",
                execution_time_seconds=time.time() - started,
            )
        except Exception as exc:
            logger.exception(
                "Trial %s grader %s failed",
                trial.id,
                spec.grader_type,
            )
            return GraderResult(
                grader_type=spec.grader_type,
                score=0.0,
                passed=False,
                error_message=str(exc),
                execution_time_seconds=time.time() - started,
            )

    async def _run_grader_with_retries(
        self,
        *,
        grader: Any,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
        started: float,
    ) -> GraderResult:
        last_transient_message: str | None = None
        for attempt in range(1, _TRANSIENT_GRADER_MAX_ATTEMPTS + 1):
            try:
                if spec.timeout_seconds is not None:
                    result = await asyncio.wait_for(
                        grader.grade(trial, transcript, spec),
                        timeout=spec.timeout_seconds,
                    )
                else:
                    grader_task = asyncio.create_task(grader.grade(trial, transcript, spec))
                    heartbeat_interval_seconds = 60.0
                    while True:
                        try:
                            result = await asyncio.wait_for(
                                asyncio.shield(grader_task),
                                timeout=heartbeat_interval_seconds,
                            )
                            break
                        except TimeoutError:
                            elapsed = time.time() - started
                            logger.warning(
                                "Trial %s grader %s still running after %.1fs",
                                trial.id,
                                spec.grader_type,
                                elapsed,
                            )
            except Exception as exc:
                message = str(exc)
                if _is_transient_grader_error(message) and attempt < _TRANSIENT_GRADER_MAX_ATTEMPTS:
                    logger.warning(
                        "Retrying transient grader failure for %s on attempt %d/%d: %s",
                        spec.grader_type,
                        attempt,
                        _TRANSIENT_GRADER_MAX_ATTEMPTS,
                        message,
                    )
                    last_transient_message = message
                    continue
                if _is_transient_grader_error(message):
                    return _inconclusive_grader_result(
                        spec.grader_type,
                        message,
                        time.time() - started,
                    )
                raise

            if _is_transient_grader_error(result.error_message):
                if attempt < _TRANSIENT_GRADER_MAX_ATTEMPTS:
                    logger.warning(
                        "Retrying transient grader result for %s on attempt %d/%d: %s",
                        spec.grader_type,
                        attempt,
                        _TRANSIENT_GRADER_MAX_ATTEMPTS,
                        result.error_message,
                    )
                    last_transient_message = result.error_message
                    continue
                return _inconclusive_grader_result(
                    spec.grader_type,
                    result.error_message or last_transient_message or "Transient grader failure",
                    time.time() - started,
                )

            return cast(GraderResult, result)

        return _inconclusive_grader_result(
            spec.grader_type,
            last_transient_message or "Transient grader failure",
            time.time() - started,
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
                if isinstance(raw_weight, float | int | str):
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
        if _is_inconclusive_grader_result(result):
            continue
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


def _task_project_root(task: EvalTask, trial: EvalTrial) -> str | None:
    if isinstance(task.input, dict):
        for key in ("scenario_path", "scenario_root"):
            value = task.input.get(key)
            if isinstance(value, str) and value:
                return value
    if isinstance(trial.input_snapshot, dict):
        for key in ("scenario_path", "scenario_root", "workdir"):
            value = trial.input_snapshot.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _is_transient_grader_error(message: object) -> bool:
    if not isinstance(message, str) or not message.strip():
        return False
    return any(pattern.search(message) for pattern in _TRANSIENT_GRADER_ERROR_PATTERNS)


def _inconclusive_grader_result(
    grader_type: str,
    error_message: str,
    execution_time_seconds: float,
) -> GraderResult:
    return GraderResult(
        grader_type=grader_type,
        score=0.0,
        passed=False,
        error_message=None,
        execution_time_seconds=execution_time_seconds,
        needs_review=True,
        review_reason="transient_infrastructure_error",
        details={
            "inconclusive": True,
            "suppressed_error": error_message,
        },
    )


def _is_inconclusive_grader_result(result: GraderResult) -> bool:
    return result.needs_review and result.review_reason == "transient_infrastructure_error"
