"""Prompt templates package for Ash-Hawk.

This package contains prompt templates for various evaluation tasks,
including LLM-as-judge rubrics for grading agent responses.

Example usage:
    from ash_hawk.prompts import load_judge_prompt

    # Load a judge prompt template
    prompt = load_judge_prompt("correctness")

    # Format with runtime variables
    formatted = prompt.format(
        task_input="Write a function",
        expected_output="A working function",
        agent_response="Here's the function...",
        transcript_context="..."
    )
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent
_JUDGE_PROMPTS_DIR = _PROMPTS_DIR / "judge"

# Cache for loaded prompts
_prompt_cache: dict[str, str] = {}


class PromptInfo(NamedTuple):
    """Information about a loaded prompt."""

    name: str
    version: str
    content: str
    content_hash: str


def _extract_template(content: str) -> str:
    """Extract the prompt template from markdown content.

    Looks for the "## Prompt Template" section and extracts the content
    within the code block that follows it.

    Args:
        content: The full markdown file content.

    Returns:
        The extracted template string.

    Raises:
        ValueError: If no valid template section is found.
    """
    template_marker = "## Prompt Template"
    marker_idx = content.find(template_marker)

    if marker_idx == -1:
        return content.strip()

    after_marker = content[marker_idx + len(template_marker) :]
    code_block_start = after_marker.find("```")
    if code_block_start == -1:
        raise ValueError(
            "Prompt file has '## Prompt Template' section but no code block. "
            "Expected a ``` code block containing the template."
        )

    after_code_start = after_marker[code_block_start + 3 :]
    newline_after_start = after_code_start.find("\n")
    if newline_after_start != -1:
        template_start = newline_after_start + 1
    else:
        template_start = 0

    remaining = after_code_start[template_start:]
    code_block_end = remaining.find("```")

    if code_block_end == -1:
        raise ValueError(
            "Prompt file has unclosed code block in '## Prompt Template' section. "
            "Expected closing ```."
        )

    return remaining[:code_block_end].strip()


def _extract_version(content: str) -> str:
    """Extract the version from markdown metadata.

    Args:
        content: The full markdown file content.

    Returns:
        The version string, or "unknown" if not found.
    """
    match = re.search(r"\*\*Version\*\*:\s*([\d.]+)", content)
    if match:
        return match.group(1)
    return "unknown"


def _compute_hash(content: str) -> str:
    """Compute SHA-256 hash of content.

    Args:
        content: The content to hash.

    Returns:
        First 16 characters of the hex digest.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def load_judge_prompt(name: str) -> PromptInfo:
    """Load a judge prompt template by name.

    Loads a markdown file from the prompts/judge directory and extracts
    the prompt template section. Supports caching for repeated loads.

    Args:
        name: Prompt name (e.g., "correctness", "relevance", "safety", "quality").

    Returns:
        PromptInfo with name, version, content, and content hash.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
        ValueError: If the prompt file doesn't contain a valid template section.

    Example:
        >>> info = load_judge_prompt("correctness")
        >>> print(info.version)
        '1.0.0'
        >>> formatted = info.content.format(task_input="...", ...)
    """
    cache_key = f"judge/{name}"
    if cache_key in _prompt_cache:
        content = _prompt_cache[cache_key]
        # Re-read to get version and hash
        prompt_path = _JUDGE_PROMPTS_DIR / f"{name}.md"
        raw_content = prompt_path.read_text(encoding="utf-8")
        version = _extract_version(raw_content)
        return PromptInfo(
            name=name,
            version=version,
            content=content,
            content_hash=_compute_hash(content),
        )

    prompt_path = _JUDGE_PROMPTS_DIR / f"{name}.md"

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Judge prompt file not found: {prompt_path}. Expected prompt at: {name}.md"
        )

    raw_content = prompt_path.read_text(encoding="utf-8")
    template = _extract_template(raw_content)
    version = _extract_version(raw_content)

    _prompt_cache[cache_key] = template

    return PromptInfo(
        name=name,
        version=version,
        content=template,
        content_hash=_compute_hash(template),
    )


def load_custom_prompt(path: str | Path) -> PromptInfo:
    """Load a custom prompt template from a file path.

    Args:
        path: Path to the prompt markdown file.

    Returns:
        PromptInfo with name, version, content, and content hash.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
        ValueError: If the prompt file doesn't contain a valid template section.

    Example:
        >>> info = load_custom_prompt("/path/to/my_prompt.md")
        >>> formatted = info.content.format(...)
    """
    prompt_path = Path(path)

    if not prompt_path.exists():
        raise FileNotFoundError(f"Custom prompt file not found: {prompt_path}")

    raw_content = prompt_path.read_text(encoding="utf-8")
    template = _extract_template(raw_content)
    version = _extract_version(raw_content)
    name = prompt_path.stem

    return PromptInfo(
        name=name,
        version=version,
        content=template,
        content_hash=_compute_hash(template),
    )


def list_judge_prompts() -> list[str]:
    """List available judge prompt names.

    Returns:
        List of judge prompt names (without .md extension).

    Example:
        >>> list_judge_prompts()
        ['correctness', 'quality', 'relevance', 'safety']
    """
    prompts: list[str] = []

    if not _JUDGE_PROMPTS_DIR.exists():
        return prompts

    for md_file in _JUDGE_PROMPTS_DIR.glob("*.md"):
        prompts.append(md_file.stem)

    return sorted(prompts)


def clear_cache() -> None:
    """Clear the prompt cache.

    Useful for testing or when prompt files have been updated on disk.
    """
    global _prompt_cache
    _prompt_cache = {}


__all__ = [
    "PromptInfo",
    "load_judge_prompt",
    "load_custom_prompt",
    "list_judge_prompts",
    "clear_cache",
    "_compute_hash",
]
