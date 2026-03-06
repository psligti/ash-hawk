from __future__ import annotations

from datetime import datetime
from typing import Any, cast

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
        for index, event in enumerate(cast(list[object], trace_events)):
            if not isinstance(event, dict):
                failed_events.append(
                    {
                        "index": index,
                        "errors": ["Event must be a dict"],
                    }
                )
                continue

            errors = self._validate_event(cast(dict[str, Any], event))
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
            if not isinstance(event, dict):
                continue
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


__all__ = ["TraceSchemaGrader", "VerifyBeforeDoneGrader"]
