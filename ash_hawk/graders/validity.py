"""Transcript validity grader for detecting empty/failed trials.

This grader checks if a transcript contains meaningful content and extracts
structured error signals when failures are detected. Used by the self-improvement
loop to filter out crashed/empty runs before generating improvements.
"""

from __future__ import annotations

from typing import Any

from ash_hawk.graders.base import Grader
from ash_hawk.services.error_extractor import ErrorExtractor
from ash_hawk.types import EvalTranscript, EvalTrial, GraderResult, GraderSpec


class TranscriptValidityGrader(Grader):
    """Grader that checks transcript validity and extracts error signals.

    A transcript is valid if it contains any of:
    - messages (agent conversation)
    - tool_calls (tool invocations)
    - trace_events (structured events)
    - agent_response (final output)

    When invalid, extracts structured ErrorSignal for debugging.
    Uses composition with ErrorExtractor for error classification.
    """

    def __init__(self) -> None:
        self._error_extractor = ErrorExtractor()

    @property
    def name(self) -> str:
        return "transcript_validity"

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        """Grade transcript validity.

        Pass: transcript has meaningful content (messages/tools/events/response).
        Fail: all content fields are empty (crash/timeout case).

        Error signals are extracted and included in details regardless of pass/fail.
        """
        effective_transcript = transcript
        if trial.result is not None:
            effective_transcript = trial.result.transcript

        has_messages = bool(effective_transcript.messages)
        has_tool_calls = bool(effective_transcript.tool_calls)
        has_trace_events = bool(effective_transcript.trace_events)
        has_response = effective_transcript.agent_response is not None

        is_valid = has_messages or has_tool_calls or has_trace_events or has_response

        details: dict[str, Any] = {
            "has_messages": has_messages,
            "has_tool_calls": has_tool_calls,
            "has_trace_events": has_trace_events,
            "has_response": has_response,
            "message_count": len(effective_transcript.messages),
            "tool_call_count": len(effective_transcript.tool_calls or []),
            "trace_event_count": len(effective_transcript.trace_events or []),
        }

        error_signal = self._error_extractor.extract(effective_transcript)
        if error_signal:
            details["error_signal"] = error_signal.model_dump()
            details["error_type"] = error_signal.error_type

        return GraderResult(
            grader_type=self.name,
            score=1.0 if is_valid else 0.0,
            passed=is_valid,
            details=details,
        )


__all__ = ["TranscriptValidityGrader"]
