"""Single LLM function for auto-research: analyze failure and generate improvement."""  # type-hygiene: skip-file

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ash_hawk.types import EvalTranscript

logger = logging.getLogger(__name__)

ANALYZE_AND_IMPROVE_PROMPT = """You are improving an AI agent {target_label} file based on category-level performance analysis.

## Current Content
```markdown
{current_content}
```

## Category Performance Scores (0.0-1.0 scale)
{category_scores_section}

## Recent Agent Behavior (transcript excerpt)
{transcript_text}
{error_signals_section}
{history_section}
{existing_skills_section}
## Task

1. Identify the WEAKEST category above (lowest score)
2. Analyze what behavior was missing or incorrect in that category
3. Generate an IMPROVED {target_label} that specifically addresses that weak category

IMPORTANT: Focus your improvement on the weakest category. Do NOT try to improve everything at once.

{output_format_requirements}

When output includes a YAML `name` field, the `name` must be:
- Lowercase alphanumeric with hyphens only (e.g., "goal-tracking", "delegation")
- Concise but descriptive of the behavior
- Match the behavior being tested in the scenarios
- DIFFERENT from any previously tried names listed above
- DIFFERENT from any existing skills listed above (reuse or extend existing skills instead of creating duplicates)"""

_SKILL_OUTPUT_REQUIREMENTS = """Output the improved skill as a complete markdown file with YAML frontmatter:

```markdown
---
name: "<infer-a-descriptive-name>"
description: "<1-2 sentence description targeting the weak category>"
targets_categories:
  - "<weak_category_id>"
---

## What I do

<describe specific behaviors that address the weak category>

## When to use me

<describe scenarios where this skill should be applied>

## Guidelines

<specific instructions targeting the weak category>
```"""

_AGENT_OUTPUT_REQUIREMENTS = """Output a complete AGENT.md-style markdown document.

Requirements:
- Keep behavior-focused sections (goals, decision process, and execution guidance)
- Include concrete instructions for tool usage and verification
- If you include frontmatter, keep `name` stable unless rename is essential"""

_TOOL_OUTPUT_REQUIREMENTS = """Output a complete TOOL.md-style markdown document.

Requirements:
- Define when to use the tool and when not to use it
- Add clear input/output expectations and failure handling guidance
- Include guardrails to reduce misuse in iterative cycles"""

_POLICY_OUTPUT_REQUIREMENTS = """Output a complete POLICY.md-style markdown document.

Requirements:
- State enforceable rules and explicit exceptions
- Include verification and escalation guidance
- Keep policy language unambiguous and testable"""

_DEFAULT_OUTPUT_REQUIREMENTS = """Output the improved content as a complete markdown document focused on the weakest category."""


async def generate_improvement(
    llm_client: Any,
    current_content: str,
    transcripts: list[EvalTranscript],
    failed_proposals: list[str] | None = None,
    consecutive_failures: int = 0,
    existing_skills: list[str] | None = None,
    target_type: str | None = None,
    category_scores: dict[str, float] | None = None,
    error_signals: list[dict[str, Any]] | None = None,
) -> str | None:
    """Analyze failures and generate improved content.

    Args:
        llm_client: LLM client with complete() or chat() method
        current_content: Current skill/policy content
        transcripts: Failed run transcripts to analyze
        failed_proposals: List of previously tried proposal names that failed
        consecutive_failures: Number of consecutive failures (for temperature scheduling)
        existing_skills: List of existing skill names to avoid duplicating
        target_type: Type of target being improved (skill, policy, agent, tool)
        category_scores: Per-category scores from grader (e.g., {"tool_usage": 0.5, ...})
        error_signals: Error signals from invalid transcripts for additional context

    Returns:
        Improved content string, or None if generation failed.
    """
    if llm_client is None:
        logger.warning("No LLM client configured")
        return None

    transcript_text = _format_transcripts(transcripts[:3])
    if not transcript_text.strip():
        logger.warning("Empty transcripts, skipping improvement")
        return None

    history_section = _format_history_section(failed_proposals)
    existing_skills_section = _format_existing_skills_section(existing_skills)
    category_scores_section = _format_category_scores_section(category_scores)
    error_signals_section = _format_error_signals_section(error_signals)

    target_label, output_format_requirements = _resolve_target_prompt_requirements(target_type)

    prompt = ANALYZE_AND_IMPROVE_PROMPT.format(
        current_content=current_content[:6000],
        category_scores_section=category_scores_section,
        transcript_text=transcript_text[:8000],
        history_section=history_section,
        existing_skills_section=existing_skills_section,
        error_signals_section=error_signals_section,
        target_label=target_label,
        output_format_requirements=output_format_requirements,
    )

    temperature = min(1.0, 0.3 + (0.1 * consecutive_failures))
    response = await _call_llm(llm_client, prompt, temperature)
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


def _format_history_section(failed_proposals: list[str] | None) -> str:
    if not failed_proposals:
        return ""
    names = [name for name in failed_proposals if name][-5:]
    if not names:
        return ""
    lines = [
        "\n## Previously Tried Approaches (AVOID THESE)",
        "These approaches did not improve scores. Try something DIFFERENT:",
    ]
    for name in names:
        lines.append(f"- {name}")
    lines.append("")
    return "\n".join(lines)


def _format_existing_skills_section(existing_skills: list[str] | None) -> str:
    if not existing_skills:
        return ""
    skills = [s for s in existing_skills if s][:20]
    if not skills:
        return ""
    lines = [
        "\n## Existing Skills (REUSE OR EXTEND THESE)",
        "These skills already exist. Reuse or extend them instead of creating duplicates:",
    ]
    for skill in skills:
        lines.append(f"- {skill}")
    lines.append("")
    return "\n".join(lines)


def _format_error_signals_section(error_signals: list[dict[str, Any]] | None) -> str:
    if not error_signals:
        return ""
    signals = [s for s in error_signals if s][:10]
    if not signals:
        return ""
    lines = [
        "\n## Error Signals from Failed Trials",
        "These errors occurred in invalid transcripts. Address these failure patterns:",
    ]
    for i, signal in enumerate(signals):
        error_type = signal.get("error_type", "unknown")
        message = signal.get("message", "")[:200]
        lines.append(f"{i + 1}. [{error_type}] {message}")
    lines.append("")
    return "\n".join(lines)


def _format_category_scores_section(category_scores: dict[str, float] | None) -> str:
    if not category_scores:
        return "No category scores available (scores will be used when available)."

    sorted_scores = sorted(category_scores.items(), key=lambda x: x[1])
    lines = ["| Category | Score | Status |", "|----------|-------|--------|"]

    for cat_id, score in sorted_scores:
        if score < 0.4:
            status = "WEAK - FOCUS HERE"
        elif score < 0.6:
            status = "NEEDS IMPROVEMENT"
        elif score < 0.8:
            status = "ACCEPTABLE"
        else:
            status = "GOOD"
        lines.append(f"| {cat_id} | {score:.2f} | {status} |")

    lines.append("")
    lines.append("**Weakest category needs targeted improvement.**")
    return "\n".join(lines)


def _extract_content(response: str) -> str | None:
    pattern = r"```markdown\s*\n(.*?)\n```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        content = match.group(1).strip()
        return content if content else None

    generic_fence_pattern = r"```(?:[\w-]+)?\s*\n(.*?)\n```"
    generic_match = re.search(generic_fence_pattern, response, re.DOTALL)
    if generic_match:
        content = generic_match.group(1).strip()
        return content if content else None

    stripped = response.strip()
    if stripped and ("\n" in stripped or stripped.startswith("---") or stripped.startswith("#")):
        return stripped

    return None


def _resolve_target_prompt_requirements(target_type: str | None) -> tuple[str, str]:
    normalized = (target_type or "skill").strip().lower()

    if normalized == "agent":
        return "agent", _AGENT_OUTPUT_REQUIREMENTS
    if normalized == "tool":
        return "tool", _TOOL_OUTPUT_REQUIREMENTS
    if normalized == "policy":
        return "policy", _POLICY_OUTPUT_REQUIREMENTS

    if normalized == "skill":
        return "skill", _SKILL_OUTPUT_REQUIREMENTS

    return normalized or "content", _DEFAULT_OUTPUT_REQUIREMENTS


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


def is_name_too_similar(new_name: str, previous_names: list[str], threshold: float = 0.5) -> bool:
    """Check if a new name is too similar to any previous name.

    Uses word overlap to detect near-duplicates like:
    - mandatory-task-tracking-workflow vs strict-task-tracking-workflow
    - enforced-task-initialization vs mandatory-task-initialization

    Args:
        new_name: The proposed new skill name
        previous_names: List of previously tried names
        threshold: Minimum overlap ratio to consider "too similar" (0.5 = 50%)

    Returns:
        True if the name is too similar to a previous one
    """
    if not previous_names:
        return False

    new_words = set(new_name.lower().split("-"))
    # Filter out very common words that don't distinguish names
    stop_words = {"the", "a", "an", "for", "and", "or", "to"}
    new_words = new_words - stop_words

    for prev in previous_names:
        prev_words = set(prev.lower().split("-")) - stop_words
        if not new_words or not prev_words:
            continue

        overlap = len(new_words & prev_words)
        union = len(new_words | prev_words)
        if union > 0 and overlap / union >= threshold:
            return True

    return False


async def _call_llm(client: Any, prompt: str, temperature: float = 0.7) -> str | None:
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
            result = resp_dict.get("content") or resp_dict.get("text")
            return str(result) if result is not None else None

        return None
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


call_llm = _call_llm

__all__ = [
    "generate_improvement",
    "extract_skill_name",
    "is_name_too_similar",
    "_call_llm",
    "call_llm",
]
