from __future__ import annotations

import pytest

from ash_hawk.auto_research import llm
from ash_hawk.types import EvalTranscript


def test_extract_content_accepts_generic_fenced_block() -> None:
    response = """```text
---
name: test
---
# Title
```"""

    extracted = llm._extract_content(response)

    assert extracted is not None
    assert "name: test" in extracted


def test_extract_content_accepts_plain_markdown() -> None:
    response = """# Agent Instructions

Use parallel search and verify outputs."""

    extracted = llm._extract_content(response)

    assert extracted == response


def test_resolve_target_prompt_requirements_for_tool() -> None:
    label, requirements = llm._resolve_target_prompt_requirements("tool")

    assert label == "tool"
    assert "TOOL.md" in requirements


@pytest.mark.asyncio
async def test_generate_improvement_builds_target_specific_prompt(monkeypatch) -> None:
    captured: dict[str, str] = {}

    async def _fake_call_llm(client, prompt: str, temperature: float = 0.7) -> str:
        captured["prompt"] = prompt
        return """```markdown
# Tool guidance

Use grep before broad scans.
```"""

    monkeypatch.setattr("ash_hawk.auto_research.llm._call_llm", _fake_call_llm)

    transcript = EvalTranscript(messages=[{"role": "user", "content": "debug this"}])
    improved = await llm.generate_improvement(
        llm_client=object(),
        current_content="# Existing",
        transcripts=[transcript],
        target_type="tool",
        category_scores={"tool_usage": 0.3},
    )

    assert improved is not None
    assert "Tool guidance" in improved
    assert "TOOL.md-style" in captured["prompt"]
