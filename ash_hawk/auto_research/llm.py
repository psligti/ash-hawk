"""Single LLM function for auto-research: analyze failure and generate improvement."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ash_hawk.types import EvalTranscript

logger = logging.getLogger(__name__)

ANALYZE_AND_IMPROVE_PROMPT = """You are improving an AI agent skill file.

## Current Content
```markdown
{current_content}
```

## Recent Agent Behavior (transcript excerpt)
{transcript_text}

## Task
1. Identify what went wrong (root cause)
2. What behavior was missing or incorrect
3. Generate an IMPROVED skill that addresses the issue

Output the improved skill as a complete markdown file with YAML frontmatter:

```markdown
---
name: "<infer-a-descriptive-name>"
description: "<1-2 sentence description of what this skill does>"
---

## What I do

<describe the specific behaviors this skill enables>

## When to use me

<describe scenarios where this skill should be applied>

## Guidelines

<specific instructions for the agent to follow>
```

The `name` field must be:
- Lowercase alphanumeric with hyphens only (e.g., "goal-tracking", "delegation")
- Concise but descriptive of the behavior
- Match the behavior being tested in the scenarios"""


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

    content = match.group(1).strip()
    return content if content else None


def extract_skill_name(content: str) -> str | None:
    """Extract skill name from YAML frontmatter.

    Args:
        content: Skill content with frontmatter.

    Returns:
        Skill name or None if not found.
    """
    if not content.startswith("---"):
        return None

    lines = content.split("\n")
    fm_end = 0
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            fm_end = i
            break

    if fm_end == 0:
        return None

    frontmatter = "\n".join(lines[1:fm_end])
    name_match = re.search(r'^name:\s*["\']?([^"\'\n]+)["\']?', frontmatter, re.MULTILINE)
    if name_match:
        return name_match.group(1).strip().lower().replace(" ", "-").replace("_", "-")

    return None


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


__all__ = ["generate_improvement", "extract_skill_name"]
