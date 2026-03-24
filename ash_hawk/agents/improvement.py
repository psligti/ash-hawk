"""Improvement agent runner for analyzing evaluations and proposing improvements.

This agent uses an LLM to decide WHERE improvements should happen based on
the control level hierarchy: Agent > Skill > Tool.

After deciding, it generates the new target content for the caller to write.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pydantic as pd

from ash_hawk.agents.dawn_kestrel import DawnKestrelAgentRunner
from ash_hawk.contracts.improvement_proposal import ImprovementProposal
from ash_hawk.improvement.decision_engine import (
    CONTROL_LEVEL_TO_LESSON_TYPE,
    CONTROL_LEVEL_TO_STRATEGY,
    ControlLevel,
    Finding,
    ReviewMetrics,
)
from ash_hawk.policy import PolicyEnforcer
from ash_hawk.prompts import load_custom_prompt
from ash_hawk.strategies import SubStrategy
from ash_hawk.types import EvalOutcome, EvalTask, EvalTranscript

logger = logging.getLogger(__name__)

_IMPROVEMENT_PROMPT_PATH = (
    Path(__file__).parent.parent / "prompts" / "improvement" / "target_selection.md"
)
_DAWN_KESTREL_DIR = Path(".dawn-kestrel")


class GeneratedTarget(pd.BaseModel):
    """New version of an improvement target."""

    target_type: str = pd.Field(description="Type: agent, skill, or tool")
    target_name: str = pd.Field(description="Name of the target")
    target_path: Path = pd.Field(description="Path where this target should be written")
    change_description: str = pd.Field(description="Brief description of the change")
    old_content: str = pd.Field(default="", description="Original content (empty if new file)")
    new_content: str = pd.Field(description="Generated new content")

    model_config = pd.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def write(self, backup: bool = True) -> Path:
        """Write the new content to target_path.

        Args:
            backup: If True, create .bak of original content.

        Returns:
            Path to the written file.
        """
        self.target_path.parent.mkdir(parents=True, exist_ok=True)

        if backup and self.target_path.exists():
            backup_path = self.target_path.with_suffix(".bak")
            backup_path.write_text(self.old_content, encoding="utf-8")

        self.target_path.write_text(self.new_content, encoding="utf-8")
        return self.target_path


class ImprovementAgentResponse(pd.BaseModel):
    """Parsed response from improvement agent containing markdown table."""

    targets: list[GeneratedTarget] = pd.Field(
        default_factory=list,
        description="List of generated improvement targets",
    )
    raw_table: str = pd.Field(default="", description="Original markdown table")

    model_config = pd.ConfigDict(extra="forbid")


class ImprovementAgentRunner(DawnKestrelAgentRunner):
    """Agent that analyzes evaluation results and generates improved targets.

    This agent uses the control level hierarchy to decide WHERE
    improvements should happen:
        - Agent (level 3): Highest control, systemic changes
        - Skill (level 2): Medium control, instruction/guidance changes
        - Tool (level 1): Lowest control, parameter/tuning changes

    The agent returns a markdown table with target improvements.

    After analysis, use the targets directly or write with:
        for target in response.targets:
            target.write()  # or
            Path(target.target_path).write_text(target.new_content)
    """

    improvement_agent_default_model = "claude-3-5-sonnet-20241022"
    improvement_agent_default_tools = ["read", "grep", "glob"]

    def __init__(
        self, provider: str = "anthropic", model: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(
            provider=provider,
            model=model or self.improvement_agent_default_model,
            **kwargs,
        )
        self._agent_type = "improvement"
        self._system_prompt: str | None = None
        self._dawn_kestrel_dir = _DAWN_KESTREL_DIR

    def set_dawn_kestrel_dir(self, path: Path | str) -> None:
        self._dawn_kestrel_dir = Path(path)

    def _load_system_prompt(self) -> str:
        if self._system_prompt is None:
            try:
                prompt_info = load_custom_prompt(_IMPROVEMENT_PROMPT_PATH)
                self._system_prompt = prompt_info.content
            except Exception as e:
                logger.warning(f"Failed to load improvement prompt: {e}")
                self._system_prompt = self._get_default_prompt()
        return self._system_prompt

    def _get_default_prompt(self) -> str:
        return """You are an improvement agent. Analyze findings and output a markdown table:

| target_name | change_description | old_content | new_content |
|-------------|-------------------|-------------|-------------|
| string | Brief description | Original or empty | Complete new content |

Rules:
- old_content: empty if file doesn't exist, otherwise current content
- new_content: entire file if old_content empty, old + changes if old_content exists
- Each row on single line, use \\n for newlines in content"""

    async def run(
        self,
        task: EvalTask,
        policy_enforcer: PolicyEnforcer,
        config: dict[str, Any],
    ) -> tuple[EvalTranscript, EvalOutcome]:
        config["agent_name"] = config.get("agent_name", "improvement")
        config["temperature"] = config.get("temperature", 0.3)

        findings = config.pop("findings", [])
        metrics = config.pop("metrics", None)
        target_agent = config.pop("target_agent", "unknown")

        prompt = self._build_analysis_prompt(findings, metrics, target_agent)

        analysis_task = EvalTask(
            id=f"improvement-{config.get('run_id', 'unknown')}",
            input={"prompt": prompt},
            timeout_seconds=120,
        )

        return await super().run(analysis_task, policy_enforcer, config)

    def _build_analysis_prompt(
        self,
        findings: list[Finding] | list[dict[str, Any]],
        metrics: ReviewMetrics | dict[str, Any] | None,
        target_agent: str,
    ) -> str:
        system_prompt = self._load_system_prompt()
        findings_text = self._format_findings(findings)
        metrics_text = self._format_metrics(metrics)

        return f"""{system_prompt}

---

## Current Analysis Request

**Target Agent:** {target_agent}

### Findings from Review

{findings_text}

### Aggregate Metrics

{metrics_text}

---

Generate the improvement table now. Remember: each row on a single line, use \\n for newlines."""

    def _format_findings(self, findings: list[Finding] | list[dict[str, Any]]) -> str:
        if not findings:
            return "No findings provided."

        lines = []
        for i, f in enumerate(findings, 1):
            if isinstance(f, Finding):
                lines.append(
                    f"**Finding {i}:** [{f.severity}] {f.category}\n"
                    f"  - Description: {f.description}\n"
                    f"  - Affected runs: {f.affected_runs}"
                )
            elif isinstance(f, dict):
                lines.append(
                    f"**Finding {i}:** [{f.get('severity', 'unknown')}] {f.get('category', 'unknown')}\n"
                    f"  - Description: {f.get('description', 'N/A')}\n"
                    f"  - Affected runs: {f.get('affected_runs', 0)}"
                )

        return "\n\n".join(lines)

    def _format_metrics(self, metrics: ReviewMetrics | dict[str, Any] | None) -> str:
        if metrics is None:
            return "No metrics provided."

        if isinstance(metrics, ReviewMetrics):
            return (
                f"- Mean Score: {metrics.mean_score:.2f}\n"
                f"- Pass Rate: {metrics.pass_rate:.1%}\n"
                f"- Tool Efficiency: {metrics.tool_efficiency:.1%}\n"
                f"- Error Rate: {metrics.error_rate:.1%}\n"
                f"- Timeout Rate: {metrics.timeout_rate:.1%}"
            )

        return "\n".join(f"- {k}: {v}" for k, v in metrics.items())

    def parse_agent_response(self, transcript: EvalTranscript) -> ImprovementAgentResponse | None:
        response_text = transcript.agent_response
        if not response_text:
            logger.warning("No agent response in transcript")
            return None

        text_content: str
        if isinstance(response_text, dict):
            text_content = json.dumps(response_text)
        else:
            text_content = response_text

        return self.parse_text_response(text_content)

    def parse_text_response(self, text: str) -> ImprovementAgentResponse:
        """Parse a text response containing a markdown table."""
        table = self._extract_markdown_table(text)
        if not table:
            logger.warning("Could not extract markdown table from response")
            return ImprovementAgentResponse(targets=[], raw_table="")

        targets = self._parse_table_to_targets(table)

        return ImprovementAgentResponse(
            targets=targets,
            raw_table=table,
        )

    def _extract_markdown_table(self, text: str) -> str | None:
        lines = text.split("\n")
        table_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                table_lines.append(line)
            elif table_lines and not stripped:
                break

        return "\n".join(table_lines) if table_lines else None

    def _parse_table_to_targets(self, table: str) -> list[GeneratedTarget]:
        targets = []
        lines = table.strip().split("\n")

        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            if re.match(r"^\|[\s\-:|]+\|$", stripped):
                continue
            if "target_name" in stripped.lower() and "change_description" in stripped.lower():
                continue

            cells = self._split_table_row(stripped)
            if len(cells) >= 4:
                target_name = cells[0].strip()
                change_description = cells[1].strip()
                old_content = cells[2].strip().replace("\\n", "\n")
                new_content = cells[3].strip().replace("\\n", "\n")

                target_type = self._infer_target_type(target_name)
                target_path = self._get_target_path(target_type, target_name)

                targets.append(
                    GeneratedTarget(
                        target_type=target_type,
                        target_name=target_name,
                        target_path=target_path,
                        change_description=change_description,
                        old_content=old_content,
                        new_content=new_content,
                    )
                )

        return targets

    def _split_table_row(self, line: str) -> list[str]:
        if not line.startswith("|"):
            line = "|" + line
        if not line.endswith("|"):
            line = line + "|"

        cells = []
        current = ""
        depth = 0

        for char in line[1:]:
            if char == "|" and depth == 0:
                cells.append(current)
                current = ""
            else:
                if char == "`":
                    depth = 1 - depth
                current += char

        return cells

    def _infer_target_type(self, target_name: str) -> str:
        target_lower = target_name.lower()
        if target_lower in ("agent", "goal_tracking", "decision_making", "system_prompt"):
            return "agent"
        if target_lower in ("bash", "read", "write", "edit", "grep", "glob", "task"):
            return "tool"
        return "skill"

    def _get_target_path(self, target_type: str, target_name: str) -> Path:
        if target_type == "agent":
            return self._dawn_kestrel_dir / "agent.md"
        elif target_type == "skill":
            return self._dawn_kestrel_dir / "skills" / target_name / "SKILL.md"
        elif target_type == "tool":
            return self._dawn_kestrel_dir / "tools" / f"{target_name}.md"
        return self._dawn_kestrel_dir / "unknown" / f"{target_name}.md"

    def get_target_path_for_name(self, target_name: str) -> Path:
        target_type = self._infer_target_type(target_name)
        return self._get_target_path(target_type, target_name)

    def read_target_content_by_name(self, target_name: str) -> str:
        target_path = self.get_target_path_for_name(target_name)
        if target_path.exists():
            return target_path.read_text(encoding="utf-8")
        return ""

    def print_changes_table(self, response: ImprovementAgentResponse) -> str:
        lines = [
            "| target_name | change_description |",
            "|-------------|-------------------|",
        ]
        for target in response.targets:
            lines.append(f"| {target.target_name} | {target.change_description} |")
        return "\n".join(lines)

    def decisions_to_proposals(
        self,
        targets: list[GeneratedTarget],
        target_agent: str,
        run_id: str,
        review_id: str | None = None,
    ) -> list[ImprovementProposal]:
        proposals = []

        for i, target in enumerate(targets):
            target_type = self._infer_target_type(target.target_name)
            control_level = ControlLevel.SKILL
            if target_type == "agent":
                control_level = ControlLevel.AGENT
            elif target_type == "tool":
                control_level = ControlLevel.TOOL

            lesson_type = CONTROL_LEVEL_TO_LESSON_TYPE.get(control_level, "skill")
            strategy = CONTROL_LEVEL_TO_STRATEGY.get(
                control_level, CONTROL_LEVEL_TO_STRATEGY[ControlLevel.SKILL]
            )

            proposal = ImprovementProposal(
                proposal_id=f"prop-{run_id}-{i + 1:03d}",
                origin_run_id=run_id,
                origin_review_id=review_id,
                target_agent=target_agent,
                proposal_type=cast(
                    Literal["policy", "skill", "tool", "harness", "eval"], lesson_type
                ),
                title=f"[{target_type.capitalize()}] {target.target_name}",
                rationale=target.change_description,
                evidence_refs=[],
                expected_benefit=f"Improve {target.target_name}",
                risk_level="medium",
                diff_payload={
                    "old_content": target.old_content[:500] if target.old_content else "",
                    "new_content": target.new_content[:500] if target.new_content else "",
                    "target_type": target_type,
                    "target_name": target.target_name,
                },
                rollback_plan=f"Revert {target_type} changes",
                created_at=datetime.now(UTC),
                strategy=strategy,
                sub_strategies=self._infer_sub_strategies(
                    target.target_name, target.change_description
                ),
                confidence=0.8,
            )
            proposals.append(proposal)

        proposals.sort(key=lambda p: p.confidence or 0, reverse=True)
        return proposals

    def _infer_sub_strategies(self, target_name: str, description: str) -> list[SubStrategy]:
        target_lower = target_name.lower()
        desc_lower = description.lower()

        if "tool" in target_lower or target_lower in (
            "bash",
            "read",
            "write",
            "edit",
            "grep",
            "glob",
        ):
            if "efficiency" in desc_lower or "timeout" in desc_lower:
                return [SubStrategy.TOOL_EFFICIENCY]
            if "error" in desc_lower:
                return [SubStrategy.ERROR_RECOVERY]
            return [SubStrategy.TOOL_SELECTION]

        if target_lower in ("agent", "goal_tracking", "decision_making"):
            if "goal" in desc_lower or "drift" in desc_lower:
                return [SubStrategy.TASK_COMPLETION]
            return [SubStrategy.EVIDENCE_QUALITY]

        if "clarity" in desc_lower:
            return [SubStrategy.INSTRUCTION_CLARITY]
        if "example" in desc_lower:
            return [SubStrategy.EXAMPLE_QUALITY]

        return [SubStrategy.INSTRUCTION_CLARITY]


__all__ = [
    "GeneratedTarget",
    "ImprovementAgentRunner",
    "ImprovementAgentResponse",
]
