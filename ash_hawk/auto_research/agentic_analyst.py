"""Agentic transcript analysis using LLM.

Analyzes failed agent transcripts to identify root causes and propose fixes.
Uses LLM for reasoning rather than heuristics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ash_hawk.auto_research.types import AnalysisResult

if TYPE_CHECKING:
    from ash_hawk.types import EvalTranscript

logger = logging.getLogger(__name__)


ANALYSIS_PROMPT_TEMPLATE = """You are an expert AI agent debugger. Analyze this FAILED agent run.

## Scenario: {scenario_name}

## Scenario Goal
{scenario_goal}

## Current Skill/Policy Content
```markdown
{current_content}
```

## Agent Transcript (what the agent actually did)
{transcript_text}

## Your Task
Analyze the transcript and identify:

1. ROOT CAUSE: What specific decision did the agent get wrong? Be precise - not generic advice.
2. MISSING GUIDANCE: What specific instruction was absent from the skill/policy that could have prevented this?
3. PROPOSED FIX: What concrete, specific line/section should be added?

Output in this EXACT format (no markdown, no code blocks):
ROOT_CAUSE: <specific wrong decision and why it was wrong>
MISSING_GUIDANCE: <specific instruction absent from config>
PROPOSED_FIX: <concrete addition/modification to make>"""


class AgenticAnalyst:
    """LLM-powered transcript analysis."""

    def __init__(self, llm_client: Any) -> None:
        self._llm_client = llm_client

    async def analyze_failure(
        self,
        transcript: EvalTranscript | None,
        scenario_name: str,
        scenario_goal: str,
        current_content: str,
    ) -> AnalysisResult | None:
        """Analyze a failed transcript using LLM.

        Args:
            transcript: The transcript from the failed run (or None if not available).
            scenario_name: Name of the scenario.
            scenario_goal: What the scenario was trying to accomplish.
            current_content: Current skill/policy content.

        Returns:
            AnalysisResult or None if analysis failed.
        """
        if self._llm_client is None:
            logger.warning("No LLM client configured for analysis")
            return None

        if transcript is None:
            logger.warning("No transcript provided, skipping analysis")
            return None

        transcript_text = self._format_transcript(transcript)
        if not transcript_text:
            logger.warning("Empty transcript, skipping analysis")
            return None

        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            scenario_name=scenario_name,
            scenario_goal=scenario_goal,
            current_content=current_content[:4000],
            transcript_text=transcript_text[:8000],
        )

        response = await self._call_llm(prompt)
        if response is None:
            return None

        return self._parse_analysis(response, scenario_name)

    def _format_transcript(self, transcript: EvalTranscript) -> str:
        """Format transcript into readable text for LLM.

        Args:
            transcript: The transcript to format.

        Returns:
            Formatted text representation.
        """
        parts: list[str] = []

        # Add messages
        for msg in transcript.messages[:20]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(f"[{role.upper()}]: {content[:500]}")

        # Add tool calls
        for tc in transcript.tool_calls[:15]:
            tool_name = tc.get("name", tc.get("tool", "unknown"))
            args = tc.get("arguments", tc.get("args", {}))
            result = tc.get("result", {})
            parts.append(f"[TOOL {tool_name}]: args={str(args)[:200]}")
            if result:
                result_str = str(result)[:300]
                parts.append(f"  -> {result_str}")

        # Add trace events
        for event in transcript.trace_events[:10]:
            event_type = event.get("event", event.get("type", "unknown"))
            msg = event.get("message", event.get("data", ""))
            if msg:
                parts.append(f"[EVENT {event_type}]: {str(msg)[:200]}")

        return "\n".join(parts)

    def _parse_analysis(self, response: str, scenario_name: str) -> AnalysisResult | None:
        """Parse LLM response into AnalysisResult.

        Args:
            response: Raw LLM response text.
            scenario_name: Name of the scenario.

        Returns:
            Parsed AnalysisResult or None.
        """
        root_cause = ""
        missing_guidance = ""
        proposed_fix = ""

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("ROOT_CAUSE:"):
                root_cause = line[11:].strip()
            elif line.startswith("MISSING_GUIDANCE:"):
                missing_guidance = line[17:].strip()
            elif line.startswith("PROPOSED_FIX:"):
                proposed_fix = line[14:].strip()

        if not root_cause:
            logger.warning("Failed to parse ROOT_CAUSE from LLM response")
            return None

        return AnalysisResult(
            scenario_name=scenario_name,
            root_cause=root_cause,
            missing_guidance=missing_guidance or "Unknown",
            proposed_fix=proposed_fix or "Unknown",
            confidence=0.0 if "unknown" in root_cause.lower() else 0.7,
            transcript_excerpt=root_cause[:200],
        )

    async def _call_llm(self, prompt: str) -> str | None:
        """Call the LLM with a prompt.

        Args:
            prompt: The prompt to send.

        Returns:
            LLM response text or None if failed.
        """
        if self._llm_client is None:
            return None

        try:
            response: Any = None
            if hasattr(self._llm_client, "complete"):
                response = await self._llm_client.complete(
                    messages=[{"role": "user", "content": prompt}],
                )
            elif hasattr(self._llm_client, "chat"):
                response = await self._llm_client.chat(prompt)
            else:
                logger.error("LLM client has no compatible method")
                return None

            # Extract text from response
            if hasattr(response, "text"):
                text_val = getattr(response, "text", None)
                return str(text_val) if text_val is not None else None
            if hasattr(response, "content"):
                content_val = getattr(response, "content", None)
                return str(content_val) if content_val is not None else None
            if isinstance(response, str):
                return response
            if isinstance(response, dict):
                content_result: Any = response.get("content")
                text_result: Any = response.get("text")
                final_result: Any = content_result or text_result
                return str(final_result) if final_result is not None else None

            return None
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None


__all__ = ["AgenticAnalyst"]
