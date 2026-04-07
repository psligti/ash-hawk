# type-hygiene: skip-file
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from ash_hawk.storage import StoredTrial
from ash_hawk.types import (
    EvalRunSummary,
    EvalTrial,
    RunArtifact,
    RunEnvelope,
    StepRecord,
    ToolCallRecord,
)

if TYPE_CHECKING:
    from ash_hawk.storage import StorageBackend


class ArtifactAdapter:
    """Converts ash-hawk storage data to RunArtifact format."""

    def __init__(self, storage: StorageBackend) -> None:
        self._storage = storage

    async def load_run_artifact(self, run_id: str) -> RunArtifact | None:
        suites = await self._storage.list_suites()
        for suite_id in suites:
            envelope = await self._storage.load_run_envelope(suite_id, run_id)
            if envelope:
                return await self._convert_envelope_to_artifact(suite_id, envelope)
        return None

    async def load_run_artifact_from_suite(self, suite_id: str, run_id: str) -> RunArtifact | None:
        envelope = await self._storage.load_run_envelope(suite_id, run_id)
        if not envelope:
            return None
        return await self._convert_envelope_to_artifact(suite_id, envelope)

    async def _convert_envelope_to_artifact(
        self, suite_id: str, envelope: RunEnvelope
    ) -> RunArtifact:
        summary = await self._storage.load_summary(suite_id, envelope.run_id)

        tool_calls: list[ToolCallRecord] = []
        steps: list[StepRecord] = []
        messages: list[dict[str, Any]] = []
        total_duration_ms = 0
        token_usage: dict[str, int] = {}
        cost_usd: float | None = None
        outcome = "success"
        error_message: str | None = None

        if summary and summary.trials:
            for trial in summary.trials:
                stored = await self._storage.load_trial(suite_id, envelope.run_id, trial.id)
                if stored:
                    tc, st, msg, dur, tok, cost, err = self._extract_trial_data(stored)
                    tool_calls.extend(tc)
                    steps.extend(st)
                    messages.extend(msg)
                    total_duration_ms += dur
                    if tok:
                        for k, v in tok.items():
                            token_usage[k] = token_usage.get(k, 0) + v
                    if cost is not None:
                        cost_usd = (cost_usd or 0.0) + cost
                    if err:
                        outcome = "failure"
                        error_message = err

        if summary and summary.metrics.pass_rate < 1.0:
            outcome = "failure"

        return RunArtifact(
            run_id=envelope.run_id,
            suite_id=suite_id,
            agent_name=envelope.agent_name,
            outcome=outcome,
            tool_calls=tool_calls,
            steps=steps,
            messages=messages,
            total_duration_ms=total_duration_ms if total_duration_ms > 0 else None,
            token_usage=token_usage,
            cost_usd=cost_usd,
            error_message=error_message,
            metadata={
                "harness_version": envelope.harness_version,
                "git_commit": envelope.git_commit,
                "provider": envelope.provider,
                "model": envelope.model,
                "suite_hash": envelope.suite_hash,
            },
            created_at=self._parse_datetime(envelope.created_at),
            completed_at=None,
        )

    def _extract_trial_data(
        self, stored: StoredTrial
    ) -> tuple[
        list[ToolCallRecord],
        list[StepRecord],
        list[dict[str, Any]],
        int,
        dict[str, int],
        float | None,
        str | None,
    ]:
        tool_calls: list[ToolCallRecord] = []
        steps: list[StepRecord] = []
        messages: list[dict[str, Any]] = []
        total_duration_ms = 0
        token_usage: dict[str, int] = {}
        cost_usd: float | None = None
        error_message: str | None = None

        if stored.trial.result:
            result = stored.trial.result
            transcript = result.transcript

            for tc in transcript.tool_calls:
                tool_name = tc.get("tool", tc.get("name", "unknown"))
                outcome_str = "failure" if tc.get("error") else "success"
                duration_ms = int(tc.get("duration_ms", tc.get("duration_seconds", 0) * 1000))

                tool_calls.append(
                    ToolCallRecord(
                        tool_name=tool_name,
                        outcome=outcome_str,
                        duration_ms=duration_ms if duration_ms > 0 else None,
                        error_message=tc.get("error"),
                        input_args=tc.get("input", tc.get("arguments", {})),
                        output=tc.get("output"),
                    )
                )

            messages.extend(transcript.messages)

            for event in transcript.trace_events:
                steps.append(
                    StepRecord(
                        step_id=event.get("id", event.get("event_id", "")),
                        step_type=event.get("type", event.get("event_type", "action")),
                        content=event.get("content", event.get("data")),
                        outcome=event.get("outcome", "pending"),
                        timestamp=self._parse_datetime(event.get("timestamp"))
                        if event.get("timestamp")
                        else None,
                    )
                )

            total_duration_ms = int(transcript.duration_seconds * 1000)

            token_usage = {
                "input": transcript.token_usage.input,
                "output": transcript.token_usage.output,
                "reasoning": transcript.token_usage.reasoning,
                "total": transcript.token_usage.input
                + transcript.token_usage.output
                + transcript.token_usage.reasoning,
            }

            cost_usd = transcript.cost_usd

            if result.outcome.failure_mode:
                error_message = result.outcome.error_message

        return (
            tool_calls,
            steps,
            messages,
            total_duration_ms,
            token_usage,
            cost_usd,
            error_message,
        )

    def _parse_datetime(self, dt_str: str | None) -> datetime | None:
        if not dt_str:
            return None
        try:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except ValueError:
            return None

    def create_artifact_from_summary(
        self, summary: EvalRunSummary, envelope: RunEnvelope
    ) -> RunArtifact:
        tool_calls: list[ToolCallRecord] = []
        total_duration_ms = 0
        failed_trials = 0

        for trial in summary.trials:
            if trial.result and trial.result.outcome.failure_mode:
                failed_trials += 1
            if trial.result and trial.result.transcript:
                total_duration_ms += int(trial.result.transcript.duration_seconds * 1000)

        outcome = "failure" if failed_trials > 0 or summary.metrics.pass_rate < 1.0 else "success"

        return RunArtifact(
            run_id=envelope.run_id,
            suite_id=envelope.suite_id,
            agent_name=envelope.agent_name,
            outcome=outcome,
            tool_calls=tool_calls,
            steps=[],
            messages=[],
            total_duration_ms=total_duration_ms if total_duration_ms > 0 else None,
            token_usage={
                "input": summary.metrics.total_tokens.input,
                "output": summary.metrics.total_tokens.output,
                "total": summary.metrics.total_tokens.input + summary.metrics.total_tokens.output,
            },
            cost_usd=summary.metrics.total_cost_usd,
            metadata={
                "pass_rate": summary.metrics.pass_rate,
                "total_tasks": summary.metrics.total_tasks,
                "failed_tasks": summary.metrics.failed_tasks,
            },
            created_at=self._parse_datetime(envelope.created_at),
        )

    def create_artifact_from_trial(
        self, stored: StoredTrial, suite_id: str, agent_name: str
    ) -> RunArtifact:
        tc, st, msg, dur, tok, cost, err = self._extract_trial_data(stored)

        outcome = "success"
        if stored.trial.result:
            if stored.trial.result.outcome.failure_mode:
                outcome = "failure"

        return RunArtifact(
            run_id=stored.envelope.run_id,
            suite_id=suite_id,
            agent_name=agent_name,
            outcome=outcome,
            tool_calls=tc,
            steps=st,
            messages=msg,
            total_duration_ms=dur if dur > 0 else None,
            token_usage=tok,
            cost_usd=cost,
            error_message=err,
            metadata={
                "trial_id": stored.trial.id,
                "task_id": stored.trial.task_id,
            },
            created_at=self._parse_datetime(stored.envelope.created_at),
            completed_at=self._parse_datetime(stored.envelope.completed_at),
        )
