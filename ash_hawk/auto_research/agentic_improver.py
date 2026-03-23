"""LLM-powered improvement generation for auto-research."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ash_hawk.auto_research.types import (
    AnalysisResult,
    ImprovementResult,
    ImprovementType,
)

logger = logging.getLogger(__name__)

IMPROVEMENT_PROMPT = """You are improving a markdown configuration file for an AI agent.

## Target Type: {target_type}
## Target Path: {target_path}

## Current Content
```markdown
{current_content}
```

## Analysis Findings
- Root Cause: {root_cause}
- Missing Guidance: {missing_guidance}
- Proposed Fix: {proposed_fix}

## Task
Rewrite the {target_type} to address the analysis findings. Keep existing good patterns, and add the missing guidance. Make changes minimal and targeted.

Requirements:
1. Keep all existing good content
2. Add the missing guidance identified above
3. Make changes minimal and targeted
4. Output the COMPLETE new version with YAML frontmatter

Output the UPDATED_CONTENT as a complete markdown file with YAML frontmatter:

```markdown
---
name: "<A SHORT NAME DESCRIBING THIS CHANGE - e.g., 'Add explicit error handling for empty state'>"
---

<the complete improved markdown content>
```

The 'name' field should be a concise description (5-10 words) of what changed between the original and improved version.
"""


class AgenticImprover:
    """LLM-powered improvement generation (not template-based)."""

    def __init__(self, llm_client: Any) -> None:
        self._llm_client = llm_client

    async def generate_improvement(
        self,
        analysis: AnalysisResult,
        target_type: ImprovementType,
        target_path: Path,
    ) -> ImprovementResult | None:
        if self._llm_client is None:
            logger.warning("No LLM client configured for improvement")
            return None

        try:
            current_content = target_path.read_text()
        except Exception as e:
            logger.error(f"Failed to read target file: {e}")
            return None

        if not current_content.strip():
            logger.warning("Empty current content, skipping improvement")
            return None

        prompt = IMPROVEMENT_PROMPT.format(
            target_type=target_type.value,
            target_path=target_path,
            current_content=current_content[:6000],
            root_cause=analysis.root_cause,
            missing_guidance=analysis.missing_guidance,
            proposed_fix=analysis.proposed_fix,
        )

        response = await self._call_llm(prompt)
        if response is None:
            return None

        return self._parse_improvement_response(
            response, target_type, target_path, current_content, analysis
        )

    def _parse_improvement_response(
        self,
        response: str,
        target_type: ImprovementType,
        target_path: Path,
        current_content: str,
        analysis: AnalysisResult,
    ) -> ImprovementResult | None:
        updated_content = ""
        change_name = ""
        rationale = ""

        in_content = False
        in_frontmatter = False
        content_lines: list[str] = []

        for line in response.split("\n"):
            stripped = line.strip()
            if stripped == "---" and not in_frontmatter and not in_content:
                in_frontmatter = True
                continue
            elif stripped == "---" and in_frontmatter:
                in_frontmatter = False
                in_content = True
                continue
            elif in_frontmatter:
                if line.startswith("name:"):
                    change_name = line[5:].strip().strip('"').strip("'")
                elif line.startswith("rationale:"):
                    rationale = line[10:].strip().strip('"').strip("'")
            elif in_content:
                content_lines.append(line)
            elif line.startswith("UPDATED_CONTENT:"):
                in_content = True
            elif line.startswith("RATIONALE:"):
                rationale = line[10:].strip()

        if content_lines:
            updated_content = "\n".join(content_lines).strip()

        if not updated_content:
            updated_content = self._extract_markdown_block(response)

        if not updated_content:
            logger.warning("No updated content found in LLM response")
            return None

        if not change_name:
            change_name = self._extract_name_from_response(response)

        return ImprovementResult(
            target_type=target_type,
            target_path=target_path,
            original_content=current_content,
            updated_content=updated_content,
            change_name=change_name or "Unnamed improvement",
            rationale=rationale or "Generated improvement",
            analysis_ref=analysis,
        )

    def _extract_markdown_block(self, response: str) -> str:
        pattern = r"```markdown\s*\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
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
        return ""

    def _extract_name_from_response(self, response: str) -> str:
        name_match = re.search(r'name:\s*["\']?([^"\'\n]+)["\']?', response)
        if name_match:
            return name_match.group(1).strip()
        return ""

    async def _call_llm(self, prompt: str) -> str | None:
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


__all__ = ["AgenticImprover"]
