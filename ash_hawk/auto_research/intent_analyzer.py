"""Intent analyzer for extracting behavioral patterns from agent transcripts."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, cast

from ash_hawk.auto_research.types import (
    DecisionPattern,
    FailurePattern,
    IntentPatterns,
    ToolUsagePattern,
)
from ash_hawk.types import EvalTranscript

logger = logging.getLogger(__name__)

PARALLEL_EXPLORE = "parallel_explore"
SEQUENTIAL_FIX = "sequential_fix"
DELEGATE_THEN_SYNTHESIZE = "delegate_then_synthesize"
ITERATE_THEN_VERIFY = "iterate_then_verify"

FAILURE_TIMEOUT = "timeout"
FAILURE_TOOL_DENIED = "tool_denied"
FAILURE_ERROR_TRACE = "error_trace"
FAILURE_WRONG_OUTPUT = "wrong_output"

_PARALLEL_KEYWORDS = {"parallel", "concurrent", "simultaneously", "multiple", "explore"}
_SEQUENTIAL_FIX_KEYWORDS = {"fix", "retry", "again", "correct", "revert", "undo"}
_DELEGATE_KEYWORDS = {"delegate", "agent", "subagent", "task", "spawn"}
_VERIFY_KEYWORDS = {"verify", "test", "check", "assert", "confirm", "validate"}

INTENT_HYPOTHESIS_PROMPT = """You are analyzing agent behavioral patterns from evaluation transcripts.

## Tool Usage Patterns
{tool_patterns}

## Decision Patterns
{decision_patterns}

## Failure Patterns
{failure_patterns}

## Statistics
- Transcripts analyzed: {transcript_count}
- Dominant tools: {dominant_tools}

## Task
Based on the patterns above, provide a concise (2-4 sentence) hypothesis about:
1. What the agent is primarily trying to accomplish
2. Its dominant strategy (exploration vs exploitation, delegation vs direct action)
3. Key bottlenecks or failure modes limiting effectiveness

Output ONLY the hypothesis text, no markdown formatting."""


class IntentAnalyzer:
    """Extracts behavioral patterns and intent from agent transcripts.

    Analyzes tool usage, decision sequences, and failure modes to build
    a composite picture of agent behavior. Optionally uses an LLM client
    to generate a natural-language intent hypothesis.

    Args:
        llm_client: Optional LLM client with complete() or chat() method.
            If None, intent hypothesis generation is skipped.
    """

    def __init__(self, llm_client: Any | None = None) -> None:
        self._llm_client = llm_client

    async def analyze_transcripts(
        self,
        transcripts: list[EvalTranscript],
    ) -> IntentPatterns:
        """Extract intent patterns from agent transcripts.

        Args:
            transcripts: List of evaluation transcripts to analyze.

        Returns:
            IntentPatterns containing tool usage, decision, and failure patterns.
        """
        if not transcripts:
            return IntentPatterns(transcript_count=0)

        tool_patterns = self._extract_tool_patterns(transcripts)
        sequences = self._identify_sequences(transcripts)
        decision_patterns = self._cluster_decision_patterns(transcripts)
        failure_patterns = self._classify_failures(transcripts)

        for tp in tool_patterns:
            tp.common_sequences = [
                seq for seq, _count in sequences.most_common(3) if tp.tool_name in seq
            ]

        dominant_tools = [tp.tool_name for tp in sorted(tool_patterns, key=lambda p: -p.call_count)]

        patterns = IntentPatterns(
            dominant_tools=dominant_tools[:5],
            tool_usage_patterns=tool_patterns,
            decision_patterns=decision_patterns,
            failure_patterns=failure_patterns,
            transcript_count=len(transcripts),
        )

        if self._llm_client is not None:
            hypothesis = await self._generate_intent_hypothesis(patterns)
            patterns.inferred_intent = hypothesis
            patterns.confidence = 0.7 if hypothesis else 0.0

        return patterns

    def _extract_tool_patterns(
        self,
        transcripts: list[EvalTranscript],
    ) -> list[ToolUsagePattern]:
        """Extract per-tool call frequency, success/failure rates, and avg duration."""
        call_counts: Counter[str] = Counter()
        success_counts: Counter[str] = Counter()
        failure_counts: Counter[str] = Counter()
        duration_sums: dict[str, float] = {}
        duration_counts: dict[str, int] = {}

        for transcript in transcripts:
            for tc in transcript.tool_calls:
                name = tc.get("name", tc.get("tool", "unknown"))
                call_counts[name] += 1

                status = tc.get("status", tc.get("result_status", ""))
                error = tc.get("error", tc.get("error_message"))
                if error or str(status).lower() in ("error", "failed", "denied"):
                    failure_counts[name] += 1
                else:
                    success_counts[name] += 1

                duration = tc.get("duration_seconds", tc.get("duration"))
                if duration is not None:
                    try:
                        dur = float(duration)
                        duration_sums[name] = duration_sums.get(name, 0.0) + dur
                        duration_counts[name] = duration_counts.get(name, 0) + 1
                    except (ValueError, TypeError):
                        pass

        patterns: list[ToolUsagePattern] = []
        for tool_name, count in call_counts.most_common():
            avg_dur = 0.0
            if tool_name in duration_counts and duration_counts[tool_name] > 0:
                avg_dur = duration_sums[tool_name] / duration_counts[tool_name]

            patterns.append(
                ToolUsagePattern(
                    tool_name=tool_name,
                    call_count=count,
                    success_count=success_counts.get(tool_name, 0),
                    failure_count=failure_counts.get(tool_name, 0),
                    avg_duration_seconds=avg_dur,
                )
            )

        return patterns

    def _identify_sequences(
        self,
        transcripts: list[EvalTranscript],
        min_length: int = 2,
        max_length: int = 5,
    ) -> Counter[tuple[str, ...]]:
        """Extract n-gram tool sequences and count their frequency."""
        sequence_counter: Counter[tuple[str, ...]] = Counter()

        for transcript in transcripts:
            tool_names = [tc.get("name", tc.get("tool", "unknown")) for tc in transcript.tool_calls]

            if len(tool_names) < min_length:
                continue

            for n in range(min_length, min(max_length + 1, len(tool_names) + 1)):
                for i in range(len(tool_names) - n + 1):
                    seq = tuple(tool_names[i : i + n])
                    sequence_counter[seq] += 1

        return sequence_counter

    def _cluster_decision_patterns(
        self,
        transcripts: list[EvalTranscript],
    ) -> list[DecisionPattern]:
        """Classify transcripts into decision pattern types by keyword and structure."""
        pattern_hits: dict[str, list[list[str]]] = {
            PARALLEL_EXPLORE: [],
            SEQUENTIAL_FIX: [],
            DELEGATE_THEN_SYNTHESIZE: [],
            ITERATE_THEN_VERIFY: [],
        }
        pattern_success: dict[str, int] = {k: 0 for k in pattern_hits}
        pattern_total: dict[str, int] = {k: 0 for k in pattern_hits}

        for transcript in transcripts:
            msg_text = self._extract_message_text(transcript)
            tool_names = [tc.get("name", tc.get("tool", "unknown")) for tc in transcript.tool_calls]
            has_error = transcript.error_trace is not None
            is_success = not has_error

            matched = self._classify_decision_type(msg_text, tool_names)
            for pattern_type in matched:
                pattern_hits[pattern_type].append(tool_names[:10])
                pattern_total[pattern_type] += 1
                if is_success:
                    pattern_success[pattern_type] += 1

        result: list[DecisionPattern] = []
        descriptions = {
            PARALLEL_EXPLORE: "Agent explores multiple paths concurrently before deciding",
            SEQUENTIAL_FIX: "Agent applies fixes sequentially, retrying on failure",
            DELEGATE_THEN_SYNTHESIZE: "Agent delegates subtasks then synthesizes results",
            ITERATE_THEN_VERIFY: "Agent iterates on a solution then verifies correctness",
        }

        for ptype, examples in pattern_hits.items():
            freq = pattern_total[ptype]
            if freq == 0:
                continue
            result.append(
                DecisionPattern(
                    pattern_type=ptype,
                    frequency=freq,
                    success_rate=pattern_success[ptype] / freq,
                    example_sequences=examples[:3],
                    description=descriptions.get(ptype, ""),
                )
            )

        return sorted(result, key=lambda p: -p.frequency)

    def _classify_failures(
        self,
        transcripts: list[EvalTranscript],
    ) -> list[FailurePattern]:
        """Classify failures by error_trace content, tool errors, and denial status."""
        failure_data: dict[str, dict[str, Any]] = {
            FAILURE_TIMEOUT: {"count": 0, "tools": Counter(), "contexts": []},
            FAILURE_TOOL_DENIED: {"count": 0, "tools": Counter(), "contexts": []},
            FAILURE_ERROR_TRACE: {"count": 0, "tools": Counter(), "contexts": []},
            FAILURE_WRONG_OUTPUT: {"count": 0, "tools": Counter(), "contexts": []},
        }

        for transcript in transcripts:
            error = transcript.error_trace

            if error:
                ftype = self._classify_error_type(error)
                failure_data[ftype]["count"] += 1
                failure_data[ftype]["contexts"].append(error[:200])

                for tc in transcript.tool_calls:
                    name = tc.get("name", tc.get("tool", "unknown"))
                    tc_error = tc.get("error", tc.get("error_message"))
                    if tc_error:
                        failure_data[ftype]["tools"][name] += 1

            for tc in transcript.tool_calls:
                status = str(tc.get("status", tc.get("result_status", ""))).lower()
                tc_error = tc.get("error", tc.get("error_message", ""))
                name = tc.get("name", tc.get("tool", "unknown"))

                if status == "denied" or "denied" in str(tc_error).lower():
                    failure_data[FAILURE_TOOL_DENIED]["count"] += 1
                    failure_data[FAILURE_TOOL_DENIED]["tools"][name] += 1

        result: list[FailurePattern] = []
        for ftype, data in failure_data.items():
            freq = data["count"]
            if freq == 0:
                continue

            # Recovery tracking: count tool calls after an error in the same transcript
            recovery_attempts = 0
            recovery_successes = 0
            for transcript in transcripts:
                if transcript.error_trace:
                    saw_error = False
                    for tc in transcript.tool_calls:
                        tc_error = tc.get("error", tc.get("error_message"))
                        if tc_error:
                            saw_error = True
                        elif saw_error:
                            recovery_attempts += 1
                            status = str(tc.get("status", tc.get("result_status", ""))).lower()
                            if status not in ("error", "failed", "denied"):
                                recovery_successes += 1

            result.append(
                FailurePattern(
                    failure_type=ftype,
                    frequency=freq,
                    affected_tools=[t for t, _ in data["tools"].most_common(5)],
                    recovery_attempts=recovery_attempts,
                    recovery_success_rate=(
                        recovery_successes / recovery_attempts if recovery_attempts > 0 else 0.0
                    ),
                    example_contexts=data["contexts"][:3],
                )
            )

        return sorted(result, key=lambda p: -p.frequency)

    async def _generate_intent_hypothesis(
        self,
        patterns: IntentPatterns,
    ) -> str:
        """Generate a natural-language intent hypothesis via LLM."""
        if self._llm_client is None:
            return ""

        tool_text = "\n".join(
            f"- {tp.tool_name}: {tp.call_count} calls, {tp.success_rate:.0%} success rate"
            for tp in patterns.tool_usage_patterns[:10]
        )
        decision_text = "\n".join(
            f"- {dp.pattern_type}: {dp.frequency} occurrences, {dp.success_rate:.0%} success rate"
            for dp in patterns.decision_patterns
        )
        failure_text = "\n".join(
            f"- {fp.failure_type}: {fp.frequency} occurrences, "
            f"affected tools: {', '.join(fp.affected_tools[:3])}"
            for fp in patterns.failure_patterns
        )

        prompt = INTENT_HYPOTHESIS_PROMPT.format(
            tool_patterns=tool_text or "None detected",
            decision_patterns=decision_text or "None detected",
            failure_patterns=failure_text or "None detected",
            transcript_count=patterns.transcript_count,
            dominant_tools=", ".join(patterns.dominant_tools[:5]) or "None",
        )

        result = await _call_llm(self._llm_client, prompt)
        return result or ""

    @staticmethod
    def _extract_message_text(transcript: EvalTranscript) -> str:
        parts: list[str] = []
        for msg in transcript.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content.lower())
        return " ".join(parts)

    @staticmethod
    def _classify_decision_type(msg_text: str, tool_names: list[str]) -> list[str]:
        """Classify a transcript into zero or more decision pattern types."""
        matched: list[str] = []

        # parallel_explore: keywords + diverse tool usage
        if any(kw in msg_text for kw in _PARALLEL_KEYWORDS):
            matched.append(PARALLEL_EXPLORE)

        # sequential_fix: retry/fix keywords or repeated consecutive tool calls
        has_fix_keywords = any(kw in msg_text for kw in _SEQUENTIAL_FIX_KEYWORDS)
        has_repeats = _has_consecutive_repeats(tool_names)
        if has_fix_keywords or has_repeats:
            matched.append(SEQUENTIAL_FIX)

        # delegate_then_synthesize: delegation keywords
        if any(kw in msg_text for kw in _DELEGATE_KEYWORDS):
            matched.append(DELEGATE_THEN_SYNTHESIZE)

        # iterate_then_verify: verify keywords appearing after tool usage
        if any(kw in msg_text for kw in _VERIFY_KEYWORDS) and len(tool_names) >= 2:
            matched.append(ITERATE_THEN_VERIFY)

        return matched

    @staticmethod
    def _classify_error_type(error_trace: str) -> str:
        """Classify an error trace string into a failure type constant."""
        lower = error_trace.lower()
        if "timeout" in lower or "timed out" in lower:
            return FAILURE_TIMEOUT
        if "denied" in lower or "not allowed" in lower or "permission" in lower:
            return FAILURE_TOOL_DENIED
        if "wrong" in lower or "incorrect" in lower or "mismatch" in lower:
            return FAILURE_WRONG_OUTPUT
        return FAILURE_ERROR_TRACE


def _has_consecutive_repeats(tool_names: list[str]) -> bool:
    for i in range(1, len(tool_names)):
        if tool_names[i] == tool_names[i - 1]:
            return True
    return False


async def _call_llm(client: Any, prompt: str, temperature: float = 0.5) -> str | None:
    try:
        response: Any = None

        if hasattr(client, "complete"):
            response = await client.complete(
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature},
            )
        elif hasattr(client, "chat"):
            response = await client.chat(prompt)
        else:
            logger.error("LLM client has no compatible method")
            return None

        if hasattr(response, "text"):
            text = getattr(response, "text", None)
            return str(text) if text is not None else None
        if hasattr(response, "content"):
            content = getattr(response, "content", None)
            return str(content) if content is not None else None
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            resp_dict = cast(dict[str, Any], response)
            raw = resp_dict.get("content") or resp_dict.get("text")
            return str(raw) if raw is not None else None

        return None
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


__all__ = ["IntentAnalyzer"]
