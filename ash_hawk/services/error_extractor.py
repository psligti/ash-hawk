"""Error extraction from evaluation transcripts.

This module provides structured error signal extraction from failed trial
transcripts. It parses error_trace, agent_response, and tool_calls to
identify crash causes, timeouts, rate limits, and other failure modes.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal

import pydantic as pd

from ash_hawk.types import EvalTranscript

# Maximum characters to include in stack trace excerpts
MAX_STACK_TRACE_LENGTH = 500

# Error type classification
ErrorType = Literal[
    "crash",
    "timeout",
    "rate_limit",
    "validation",
    "agent_error",
    "unknown",
]


class ErrorSignal(pd.BaseModel):
    """Structured error signal extracted from a failed transcript.

    Captures the essential information about what went wrong during a trial,
    enabling downstream systems to diagnose issues and generate targeted
    improvements.
    """

    error_type: ErrorType = pd.Field(
        description="Classification of the error type",
    )
    message: str = pd.Field(
        description="Human-readable error message",
    )
    stack_trace_excerpt: str | None = pd.Field(
        default=None,
        description="First 500 characters of stack trace if available",
    )
    tool_name: str | None = pd.Field(
        default=None,
        description="Tool that was being called when error occurred",
    )
    failure_mode: str | None = pd.Field(
        default=None,
        description="Failure mode classification from eval harness",
    )
    timestamp: str | None = pd.Field(
        default=None,
        description="ISO timestamp when error was detected",
    )
    context: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Additional context about the error",
    )

    model_config = pd.ConfigDict(extra="forbid")


class ErrorExtractor:
    """Extracts structured error signals from evaluation transcripts.

    Analyzes transcript fields (error_trace, messages, tool_calls) to
    classify and extract meaningful error information. Never fails silently -
    returns "unknown" error signal when parsing fails.
    """

    # Rate limit patterns (case-insensitive)
    RATE_LIMIT_PATTERNS = [
        r"rate.?limit",
        r"429",
        r"too many requests",
        r"request.?limit.*exceeded",
        r"api.?calls?.*limit",
        r"quota.*exceeded",
    ]

    # Timeout patterns (case-insensitive)
    TIMEOUT_PATTERNS = [
        r"timeout",
        r"timed.?out",
        r"deadline.*exceeded",
        r"execution.*time.*exceeded",
        r"operation.*timed",
    ]

    # Validation error patterns
    VALIDATION_PATTERNS = [
        r"validation.?error",
        r"invalid.*schema",
        r"pydantic.*validation",
        r"field.?required",
        r"extra.*fields?.*not.*allowed",
    ]

    def extract(self, transcript: EvalTranscript) -> ErrorSignal | None:
        """Extract error signal from a transcript.

        Args:
            transcript: The evaluation transcript to analyze.

        Returns:
            ErrorSignal if an error is detected, None if transcript appears clean.
        """
        # Check if there's any error indication
        error_trace = transcript.error_trace
        if not error_trace:
            # Check messages for error indicators
            error_trace = self._extract_error_from_messages(transcript.messages)
            if not error_trace:
                return None

        # Classify and extract
        return self._classify_error(error_trace, transcript)

    def _extract_error_from_messages(self, messages: list[dict[str, Any]]) -> str | None:
        """Extract error text from message history.

        Args:
            messages: List of message dicts from transcript.

        Returns:
            Concatenated error text if found, None otherwise.
        """
        error_parts: list[str] = []

        for msg in messages:
            # Check for error in message content
            content = msg.get("content")
            if isinstance(content, str) and self._looks_like_error(content):
                error_parts.append(content)

            # Check for error in tool results
            tool_result = msg.get("tool_result")
            if isinstance(tool_result, str) and self._looks_like_error(tool_result):
                error_parts.append(tool_result)

        return " | ".join(error_parts) if error_parts else None

    def _looks_like_error(self, text: str) -> bool:
        """Check if text looks like an error message.

        Args:
            text: Text to check.

        Returns:
            True if text appears to be an error message.
        """
        error_indicators = [
            "error",
            "exception",
            "failed",
            "traceback",
            "stack trace",
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in error_indicators)

    def _classify_error(self, error_text: str, transcript: EvalTranscript) -> ErrorSignal:
        """Classify error and extract structured signal.

        Args:
            error_text: The error text to classify.
            transcript: Full transcript for additional context.

        Returns:
            Classified ErrorSignal (never fails silently).
        """
        error_lower = error_text.lower()

        # Try to classify by pattern matching
        error_type: ErrorType = "unknown"
        message = self._extract_message(error_text)

        # Check for rate limit
        if self._matches_patterns(error_lower, self.RATE_LIMIT_PATTERNS):
            error_type = "rate_limit"

        # Check for timeout
        elif self._matches_patterns(error_lower, self.TIMEOUT_PATTERNS):
            error_type = "timeout"

        # Check for validation errors
        elif self._matches_patterns(error_lower, self.VALIDATION_PATTERNS):
            error_type = "validation"

        # Check for crash indicators (traceback, exception)
        elif "traceback" in error_lower or "exception" in error_lower:
            error_type = "crash"

        # Check for agent-specific errors
        elif "agent" in error_lower and ("error" in error_lower or "failed" in error_lower):
            error_type = "agent_error"

        # Extract stack trace excerpt
        stack_trace_excerpt = self._extract_stack_trace(error_text)

        # Extract tool name from last tool call
        tool_name = self._extract_last_tool_name(transcript.tool_calls)

        # Build context
        context = self._build_context(transcript, error_type)

        return ErrorSignal(
            error_type=error_type,
            message=message,
            stack_trace_excerpt=stack_trace_excerpt,
            tool_name=tool_name,
            timestamp=datetime.now(UTC).isoformat(),
            context=context,
        )

    def _matches_patterns(self, text: str, patterns: list[str]) -> bool:
        """Check if text matches any of the patterns.

        Args:
            text: Text to check (should be lowercase).
            patterns: List of regex patterns to match.

        Returns:
            True if any pattern matches.
        """
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _extract_message(self, error_text: str) -> str:
        """Extract a clean error message from error text.

        Args:
            error_text: Full error text.

        Returns:
            Cleaned error message (first meaningful line).
        """
        lines = error_text.strip().split("\n")

        # Skip traceback header lines
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Traceback"):
                continue
            if line.startswith("File "):
                continue
            if line.startswith("During handling"):
                continue
            # Found the actual error message
            if line:
                return line[:200]  # Truncate long messages

        # Fallback: return first non-empty line
        for line in lines:
            line = line.strip()
            if line:
                return line[:200]

        return "Unknown error"

    def _extract_stack_trace(self, error_text: str) -> str | None:
        """Extract and truncate stack trace from error text.

        Args:
            error_text: Full error text.

        Returns:
            Truncated stack trace (max 500 chars) or None.
        """
        if "Traceback" not in error_text:
            return None

        # Find traceback start
        tb_start = error_text.find("Traceback")
        if tb_start == -1:
            return None

        # Extract and truncate
        trace = error_text[tb_start:]
        if len(trace) > MAX_STACK_TRACE_LENGTH:
            trace = trace[:MAX_STACK_TRACE_LENGTH] + "..."

        return trace

    def _extract_last_tool_name(self, tool_calls: list[dict[str, Any]] | None) -> str | None:
        """Extract the name of the last tool called before error.

        Args:
            tool_calls: List of tool call dicts.

        Returns:
            Tool name or None.
        """
        if not tool_calls:
            return None

        last_call = tool_calls[-1]
        # Try common field names
        return last_call.get("name") or last_call.get("tool") or last_call.get("tool_name")

    def _build_context(self, transcript: EvalTranscript, error_type: ErrorType) -> dict[str, Any]:
        """Build additional context for the error signal.

        Args:
            transcript: Full transcript.
            error_type: Classified error type.

        Returns:
            Context dict with additional information.
        """
        context: dict[str, Any] = {
            "message_count": len(transcript.messages),
            "tool_call_count": len(transcript.tool_calls or []),
            "trace_event_count": len(transcript.trace_events or []),
            "duration_seconds": transcript.duration_seconds,
            "has_agent_response": transcript.agent_response is not None,
        }

        # Add token usage if available
        if transcript.token_usage:
            context["total_tokens"] = transcript.token_usage.total

        return context


__all__ = ["ErrorExtractor", "ErrorSignal", "ErrorType"]
