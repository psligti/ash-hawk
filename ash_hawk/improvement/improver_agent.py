"""Improver agent that proposes unified diffs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DiffProposal:
    file_path: Path
    diff: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImprovementContext:
    run_id: str
    scenario_path: Path
    transcript_path: Path
    grade_path: Path
    agent_files_dir: Path
    failed_grades: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ImproverAgent:
    model: str = "claude-3-5-sonnet-20241022"
    max_iterations: int = 3

    async def analyze_failures(
        self,
        context: ImprovementContext,
    ) -> list[DiffProposal]:
        failed_checks: list[dict[str, Any]] = []
        failed_grades = context.failed_grades

        if not failed_grades:
            return []

        proposals: list[DiffProposal] = []
        for grade in failed_grades:
            grader_name = grade.get("grader", "unknown")
            score = grade.get("score", 0.0)
            feedback = grade.get("feedback", "")

            if score < 0.7:
                proposal = await self._propose_improvement(
                    context=context,
                    grader_name=grader_name,
                    feedback=feedback,
                )
                if proposal:
                    proposals.append(proposal)

        return proposals

    async def _propose_improvement(
        self,
        context: ImprovementContext,
        grader_name: str,
        feedback: str,
    ) -> DiffProposal | None:
        target_files = self._identify_target_files(context, grader_name)

        if not target_files:
            return None

        improvements = await self._generate_improvements(
            context=context,
            target_files=target_files,
            feedback=feedback,
        )

        if not improvements:
            return None

        file_path = improvements["file_path"]
        improved_content = improvements["content"]

        original_content = self._read_file(file_path)

        from ash_hawk.improvement.differ import DiffGenerator

        generator = DiffGenerator()
        diff = generator.generate(file_path, original_content, improved_content)

        return DiffProposal(
            file_path=file_path,
            diff=diff,
            description=f"Improvements for {grader_name}",
            metadata={
                "grader": grader_name,
                "original_score": improvements.get("original_score", 0.0),
            },
        )

    def _identify_target_files(
        self,
        context: ImprovementContext,
        grader_name: str,
    ) -> list[Path]:
        agent_dir = context.agent_files_dir

        if not agent_dir.exists():
            return []

        candidates: list[Path] = [
            agent_dir / "AGENT.md",
            agent_dir / "agent.md",
            agent_dir / "skills" / f"{grader_name}.md",
            agent_dir / "tools" / f"{grader_name}.md",
            agent_dir / "policies" / f"{grader_name}.md",
            agent_dir / "policy.md",
        ]

        return [p for p in candidates if p.exists()]

    async def _generate_improvements(
        self,
        context: ImprovementContext,
        target_files: list[Path],
        feedback: str,
    ) -> dict[str, Any] | None:
        for file_path in target_files:
            original = self._read_file(file_path)

            prompt = self._build_improvement_prompt(
                file_path=file_path,
                original_content=original,
                feedback=feedback,
                context=context,
            )

            improved = await self._call_llm(prompt)

            if improved:
                return {
                    "file_path": file_path,
                    "content": improved,
                    "original_score": 0.0,
                }

        return None

    def _build_improvement_prompt(
        self,
        file_path: Path,
        original_content: str,
        feedback: str,
        context: ImprovementContext,
    ) -> str:
        return f"""You are an AI agent improvement system. Analyze the following feedback from a failed evaluation and propose improvements to the agent file.

## Context
- Agent: {context.agent_files_dir.name}
- File: {file_path.name}
- Failed feedback: {feedback}

## Original Content
```
{original_content}
```

## Task
Propose specific, actionable improvements to the content above that address the feedback. Focus on:
1. Concrete changes to instructions or prompts
2. New patterns or examples to follow
3. Edge case handling
4. Error recovery

Output ONLY the improved content, no explanations or markdown.
"""

    def _read_file(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    async def _call_llm(self, prompt: str) -> str | None:
        return None


__all__ = ["ImprovementContext", "ImproverAgent"]
