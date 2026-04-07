# type-hygiene: skip-file
"""Transcript validity grader for detecting empty/failed trials.

This grader checks if a transcript contains meaningful content and extracts
structured error signals when failures are detected. Used by the self-improvement
loop to filter out crashed/empty runs before generating improvements.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal

import pydantic as pd

from ash_hawk.graders.base import Grader
from ash_hawk.types import EvalTranscript, EvalTrial, GraderResult, GraderSpec

MAX_STACK_TRACE_LENGTH = 500

ErrorType = Literal[
    "crash",
    "timeout",
    "rate_limit",
    "validation",
    "agent_error",
    "unknown",
]


class ErrorSignal(pd.BaseModel):
    error_type: ErrorType = pd.Field(description="Classification of the error type")
    message: str = pd.Field(description="Human-readable error message")
    stack_trace_excerpt: str | None = pd.Field(default=None)
    tool_name: str | None = pd.Field(default=None)
    failure_mode: str | None = pd.Field(default=None)
    timestamp: str | None = pd.Field(default=None)
    context: dict[str, Any] = pd.Field(default_factory=dict)

    model_config = pd.ConfigDict(extra="forbid")


class ErrorExtractor:
    RATE_LIMIT_PATTERNS = [
        r"rate.?limit",
        r"429",
        r"too many requests",
        r"request.?limit.*exceeded",
        r"api.?calls?.*limit",
        r"quota.*exceeded",
    ]
    TIMEOUT_PATTERNS = [
        r"timeout",
        r"timed.?out",
        r"deadline.*exceeded",
        r"execution.*time.*exceeded",
        r"operation.*timed",
    ]
    VALIDATION_PATTERNS = [
        r"validation.?error",
        r"invalid.*schema",
        r"pydantic.*validation",
        r"field.?required",
        r"extra.*fields?.*not.*allowed",
    ]

    def extract(self, transcript: EvalTranscript) -> ErrorSignal | None:
        error_trace = transcript.error_trace
        if not error_trace:
            error_trace = self._extract_error_from_messages(transcript.messages)
            if not error_trace:
                return None
        return self._classify_error(error_trace, transcript)

    def _extract_error_from_messages(self, messages: list[dict[str, Any]]) -> str | None:
        error_parts: list[str] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str) and self._looks_like_error(content):
                error_parts.append(content)
            tool_result = msg.get("tool_result")
            if isinstance(tool_result, str) and self._looks_like_error(tool_result):
                error_parts.append(tool_result)
        return " | ".join(error_parts) if error_parts else None

    def _looks_like_error(self, text: str) -> bool:
        error_indicators = ["error", "exception", "failed", "traceback", "stack trace"]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in error_indicators)

    def _classify_error(self, error_text: str, transcript: EvalTranscript) -> ErrorSignal:
        error_lower = error_text.lower()
        error_type: ErrorType = "unknown"
        message = self._extract_message(error_text)

        if self._matches_patterns(error_lower, self.RATE_LIMIT_PATTERNS):
            error_type = "rate_limit"
        elif self._matches_patterns(error_lower, self.TIMEOUT_PATTERNS):
            error_type = "timeout"
        elif self._matches_patterns(error_lower, self.VALIDATION_PATTERNS):
            error_type = "validation"
        elif "traceback" in error_lower or "exception" in error_lower:
            error_type = "crash"
        elif "agent" in error_lower and ("error" in error_lower or "failed" in error_lower):
            error_type = "agent_error"

        stack_trace_excerpt = self._extract_stack_trace(error_text)
        tool_name = self._extract_last_tool_name(transcript.tool_calls)
        context = self._build_context(transcript)

        return ErrorSignal(
            error_type=error_type,
            message=message,
            stack_trace_excerpt=stack_trace_excerpt,
            tool_name=tool_name,
            timestamp=datetime.now(UTC).isoformat(),
            context=context,
        )

    def _matches_patterns(self, text: str, patterns: list[str]) -> bool:
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _extract_message(self, error_text: str) -> str:
        for line in error_text.strip().split("\n"):
            line = line.strip()
            if (
                not line
                or line.startswith("Traceback")
                or line.startswith("File ")
                or line.startswith("During handling")
            ):
                continue
            return line[:200]
        for line in error_text.strip().split("\n"):
            line = line.strip()
            if line:
                return line[:200]
        return "Unknown error"

    def _extract_stack_trace(self, error_text: str) -> str | None:
        if "Traceback" not in error_text:
            return None
        tb_start = error_text.find("Traceback")
        if tb_start == -1:
            return None
        trace = error_text[tb_start:]
        if len(trace) > MAX_STACK_TRACE_LENGTH:
            trace = trace[:MAX_STACK_TRACE_LENGTH] + "..."
        return trace

    def _extract_last_tool_name(self, tool_calls: list[dict[str, Any]] | None) -> str | None:
        if not tool_calls:
            return None
        last_call = tool_calls[-1]
        return last_call.get("name") or last_call.get("tool") or last_call.get("tool_name")

    def _build_context(self, transcript: EvalTranscript) -> dict[str, Any]:
        context: dict[str, Any] = {
            "message_count": len(transcript.messages),
            "tool_call_count": len(transcript.tool_calls or []),
            "trace_event_count": len(transcript.trace_events or []),
            "duration_seconds": transcript.duration_seconds,
            "has_agent_response": transcript.agent_response is not None,
        }
        if transcript.token_usage:
            context["total_tokens"] = transcript.token_usage.total
        return context


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
