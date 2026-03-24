"""Single LLM function for auto-research: analyze failure and generate improvement."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ash_hawk.types import EvalTranscript

logger = logging.getLogger(__name__)

ANALYZE_AND_IMPROVE_PROMPT = """You are improving an AI agent configuration file.

## Current Content
```markdown
{current_content}
```

## Recent Agent Behavior (transcript excerpt)
{transcript_text}

## Task
1. Identify what went wrong (root cause)
2. What guidance was missing from the configuration
3. Generate an IMPROVED version that addresses the issue

Output the improved content as a complete markdown file:

```markdown
---
name: "<5-10 word description of change>"
---

<the improved content - keep existing good patterns, add missing guidance>
```"""

TRANSCRIPT_FORMAT = """Error: {error_trace}

Recent messages:
{messages}

Recent tool calls:
{tool_calls}
"""


async def generate_improvement(
    llm_client: Any,
    current_content: str,
    transcripts: list[EvalTranscript],
) -> str | None:
    """Analyze failures and generate improved content.

    Args:
        llm_client: LLM client with complete() or chat() method
        current_content: Current skill/policy content
        transcripts: Failed run transcripts to analyze

    Returns:
        Improved content string, or None if generation failed
    """
    if llm_client is None:
        logger.warning("No LLM client configured")
        return None

    transcript_text = _format_transcripts(transcripts[:3])
    if not transcript_text.strip():
        logger.warning("Empty transcripts, skipping improvement")
        return None

    prompt = ANALYZE_AND_IMPROVE_PROMPT.format(
        current_content=current_content[:6000],
        transcript_text=transcript_text[:8000],
    )

    response = await _call_llm(llm_client, prompt)
    if response is None:
        return None

    return _extract_content(response)


def _format_transcripts(transcripts: list[EvalTranscript]) -> str:
    parts: list[str] = []

    for t in transcripts:
        if t.error_trace:
            parts.append(f"ERROR: {t.error_trace[:500]}")

        for msg in t.messages[:10]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(f"[{role.upper()}]: {content[:300]}")

        for tc in t.tool_calls[:5]:
            name = tc.get("name", tc.get("tool", "unknown"))
            args = tc.get("arguments", tc.get("args", {}))
            parts.append(f"[TOOL {name}]: {str(args)[:150]}")

    return "\n".join(parts)


def _extract_content(response: str) -> str | None:
    pattern = r"```markdown\s*\n(.*?)\n```"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None

    content = match.group(1)

    if content.startswith("---"):
        lines = content.split("\n")
        fm_end = 0
        fm_count = 0
        for i, line in enumerate(lines):
            if line.strip() == "---":
                fm_count += 1
                if fm_count == 2:
                    fm_end = i + 1
                    break
        if fm_end > 0:
            return "\n".join(lines[fm_end:])

    return content


async def _call_llm(client: Any, prompt: str) -> str | None:
    try:
        response: Any = None

        if hasattr(client, "complete"):
            response = await client.complete(messages=[{"role": "user", "content": prompt}])
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
            result = response.get("content") or response.get("text")
            return str(result) if result is not None else None

        return None
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


__all__ = ["generate_improvement"]
