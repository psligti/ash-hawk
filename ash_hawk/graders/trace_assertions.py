from __future__ import annotations

from datetime import datetime
from typing import Any

from ash_hawk.graders.base import Grader
from ash_hawk.scenario.trace import TraceEvent
from ash_hawk.types import EvalTranscript, EvalTrial, GraderResult, GraderSpec


class TraceSchemaGrader(Grader):
    _allowed_keys = {"schema_version", "event_type", "ts", "data"}

    @property
    def name(self) -> str:
        return "trace_schema"

    def _is_iso_timestamp(self, value: str) -> bool:
        if not value:
            return False
        try:
            normalized = value.replace("Z", "+00:00")
            datetime.fromisoformat(normalized)
            return True
        except ValueError:
            return False

    def _validate_event(self, event: dict[str, Any]) -> list[str]:
        errors: list[str] = []

        if "schema_version" not in event:
            errors.append("Missing schema_version")
        else:
            schema_version = event.get("schema_version")
            if not isinstance(schema_version, int):
                errors.append("schema_version must be int")
            elif schema_version != 1:
                errors.append("schema_version must be 1")

        if "event_type" not in event:
            errors.append("Missing event_type")
        else:
            event_type = event.get("event_type")
            if not isinstance(event_type, str) or not event_type.strip():
                errors.append("event_type must be non-empty string")

        if "ts" not in event:
            errors.append("Missing ts")
        else:
            ts_value = event.get("ts")
            if not isinstance(ts_value, str) or not self._is_iso_timestamp(ts_value):
                errors.append("ts must be ISO timestamp string")

        if "data" not in event:
            errors.append("Missing data")
        else:
            data = event.get("data")
            if not isinstance(data, dict):
                errors.append("data must be dict")

        extra_keys = set(event.keys()) - self._allowed_keys
        if extra_keys:
            errors.append(f"Unexpected fields: {', '.join(sorted(extra_keys))}")

        if not errors:
            try:
                TraceEvent.model_validate(event)
            except Exception as exc:
                errors.append(f"TraceEvent validation failed: {exc}")

        return errors

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        effective_transcript = transcript
        if trial.result is not None:
            effective_transcript = trial.result.transcript

        trace_events: list[object] = []
        trace_events.extend(effective_transcript.trace_events or [])
        if not trace_events:
            return GraderResult(
                grader_type=self.name,
                score=1.0,
                passed=True,
                details={"total_events": 0, "failed_events": []},
            )

        failed_events: list[dict[str, Any]] = []
        for index, event in enumerate(trace_events):
            if not isinstance(event, dict):
                failed_events.append(
                    {
                        "index": index,
                        "errors": ["Event must be a dict"],
                    }
                )
                continue

            event_payload: dict[str, Any] = {str(key): value for key, value in event.items()}
            errors = self._validate_event(event_payload)
            if errors:
                failed_events.append(
                    {
                        "index": index,
                        "errors": errors,
                    }
                )

        passed = not failed_events
        score = 1.0 if passed else 0.0

        return GraderResult(
            grader_type=self.name,
            score=score,
            passed=passed,
            details={
                "total_events": len(trace_events),
                "failed_events": failed_events,
            },
        )


class VerifyBeforeDoneGrader(Grader):
    @property
    def name(self) -> str:
        return "verify_before_done"

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        effective_transcript = transcript
        if trial.result is not None:
            effective_transcript = trial.result.transcript

        if not effective_transcript.agent_response:
            return GraderResult(
                grader_type=self.name,
                score=1.0,
                passed=True,
                details={"verified": False, "reason": "not_done"},
            )

        for event in effective_transcript.trace_events or []:
            if event.get("event_type") != "VerificationEvent":
                continue
            if event.get("data", {}).get("pass") is True:
                return GraderResult(
                    grader_type=self.name,
                    score=1.0,
                    passed=True,
                    details={"verified": True},
                )

        return GraderResult(
            grader_type=self.name,
            score=0.0,
            passed=False,
            details={"verified": False, "reason": "missing_verification"},
        )


class EvidenceRequiredGrader(Grader):
    @property
    def name(self) -> str:
        return "evidence_required"

    def _has_evidence(self, data: dict[str, Any]) -> bool:
        evidence_path = data.get("evidence_path")
        evidence_ref = data.get("evidence_ref")
        if isinstance(evidence_path, str) and evidence_path.strip():
            return True
        if isinstance(evidence_ref, str) and evidence_ref.strip():
            return True
        return False

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        effective_transcript = transcript
        if trial.result is not None:
            effective_transcript = trial.result.transcript

        missing_evidence: list[dict[str, Any]] = []
        total_todos = 0
        completed_todos = 0

        for index, event in enumerate(effective_transcript.trace_events or []):
            if not isinstance(event, dict):
                continue
            if event.get("event_type") != "TodoEvent":
                continue
            total_todos += 1
            data = event.get("data", {})
            if not isinstance(data, dict):
                continue
            if data.get("completed") is not True:
                continue
            completed_todos += 1
            if self._has_evidence(data):
                continue
            missing_evidence.append(
                {
                    "index": index,
                    "reason": "missing_evidence",
                }
            )

        passed = not missing_evidence
        score = 1.0 if passed else 0.0

        return GraderResult(
            grader_type=self.name,
            score=score,
            passed=passed,
            details={
                "total_todos": total_todos,
                "completed_todos": completed_todos,
                "missing_evidence": missing_evidence,
            },
        )


class BudgetComplianceGrader(Grader):
    @property
    def name(self) -> str:
        return "budget"

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        effective_transcript = transcript
        if trial.result is not None:
            effective_transcript = trial.result.transcript

        config = spec.config
        max_tool_calls = config.get("max_tool_calls")
        max_time_seconds = config.get("max_time_seconds")
        max_steps = config.get("max_steps")

        tool_call_count = len(effective_transcript.tool_calls or [])
        duration_seconds = effective_transcript.duration_seconds
        step_count = len(effective_transcript.trace_events or [])

        violations: list[str] = []
        if max_time_seconds is not None and duration_seconds > max_time_seconds:
            violations.append(
                f"duration_seconds {duration_seconds} exceeds max_time_seconds {max_time_seconds}"
            )
        if max_tool_calls is not None and tool_call_count > max_tool_calls:
            violations.append(
                f"tool_calls {tool_call_count} exceeds max_tool_calls {max_tool_calls}"
            )
        if max_steps is not None and step_count > max_steps:
            violations.append(f"steps {step_count} exceeds max_steps {max_steps}")

        passed = not violations
        score = 1.0 if passed else 0.0

        return GraderResult(
            grader_type=self.name,
            score=score,
            passed=passed,
            details={
                "violations": violations,
                "counts": {
                    "duration_seconds": duration_seconds,
                    "tool_calls": tool_call_count,
                    "steps": step_count,
                },
                "limits": {
                    "max_time_seconds": max_time_seconds,
                    "max_tool_calls": max_tool_calls,
                    "max_steps": max_steps,
                },
            },
        )


class OrderingGrader(Grader):
    @property
    def name(self) -> str:
        return "ordering"

    def _extract_event_type(self, event: object) -> str | None:
        if not isinstance(event, dict):
            return None
        event_map: dict[str, Any] = {str(key): value for key, value in event.items()}
        event_type = event_map.get("event_type")
        if isinstance(event_type, str) and event_type.strip():
            return event_type
        return None

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        effective_transcript = transcript
        if trial.result is not None:
            effective_transcript = trial.result.transcript

        ordering_rules = spec.config.get("ordering_rules", [])
        if not ordering_rules:
            return GraderResult(
                grader_type=self.name,
                score=1.0,
                passed=True,
                details={"total_rules": 0, "violations": []},
            )

        events = list(effective_transcript.trace_events or [])
        event_types = [self._extract_event_type(event) for event in events]

        violations: list[dict[str, Any]] = []
        for index, rule in enumerate(ordering_rules):
            if not isinstance(rule, dict):
                violations.append(
                    {
                        "index": index,
                        "error": "Ordering rule must be a mapping",
                    }
                )
                continue

            rule_map: dict[str, Any] = {str(key): value for key, value in rule.items()}
            before = rule_map.get("before")
            after = rule_map.get("after")
            errors: list[str] = []
            if not isinstance(before, str) or not before.strip():
                errors.append("before must be non-empty string")
            if not isinstance(after, str) or not after.strip():
                errors.append("after must be non-empty string")
            if errors:
                violations.append(
                    {
                        "index": index,
                        "error": "; ".join(errors),
                    }
                )
                continue

            before_index = next((i for i, value in enumerate(event_types) if value == before), None)
            after_index = next((i for i, value in enumerate(event_types) if value == after), None)
            if before_index is None:
                violations.append(
                    {
                        "index": index,
                        "before": before,
                        "after": after,
                        "error": "before event not found",
                    }
                )
                continue
            if after_index is None:
                violations.append(
                    {
                        "index": index,
                        "before": before,
                        "after": after,
                        "error": "after event not found",
                    }
                )
                continue
            if before_index > after_index:
                violations.append(
                    {
                        "index": index,
                        "before": before,
                        "after": after,
                        "error": "events out of order",
                        "before_index": before_index,
                        "after_index": after_index,
                    }
                )

        passed = not violations
        score = 1.0 if passed else 0.0

        return GraderResult(
            grader_type=self.name,
            score=score,
            passed=passed,
            details={
                "total_rules": len(ordering_rules),
                "violations": violations,
            },
        )


__all__ = [
    "TraceSchemaGrader",
    "VerifyBeforeDoneGrader",
    "EvidenceRequiredGrader",
    "BudgetComplianceGrader",
    "OrderingGrader",
]
