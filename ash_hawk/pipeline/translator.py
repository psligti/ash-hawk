"""Translator role for converting raw competitor output into validated strategy.

The Translator sits between Competitor and Analyst in the improvement loop:
- Takes raw CompetitorOutput (findings, comparison data, replay artifacts)
- Validates and structures the output into a StrategyResult
- Maps findings to appropriate Strategy and SubStrategy values
- Ensures output conforms to lesson payload schemas
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pydantic as pd

from ash_hawk.contracts import ReviewFinding
from ash_hawk.strategies import (
    Strategy,
    SubStrategy,
    validate_strategy_pair,
)

if TYPE_CHECKING:
    from ash_hawk.pipeline.competitor import CompetitorOutput


class StrategyMapping(pd.BaseModel):
    """A validated strategy mapping derived from raw findings."""

    strategy: Strategy = pd.Field(description="Top-level improvement strategy")
    sub_strategies: list[SubStrategy] = pd.Field(
        default_factory=list,
        description="Sub-strategies this mapping addresses",
    )
    confidence: float = pd.Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this strategy mapping",
    )
    source_finding_id: str | None = pd.Field(
        default=None,
        description="ID of the finding this mapping was derived from",
    )
    rationale: str = pd.Field(
        default="",
        description="Why this strategy mapping was chosen",
    )

    model_config = pd.ConfigDict(extra="forbid")


class StructuredFinding(pd.BaseModel):
    """A finding structured with strategy context."""

    finding_id: str = pd.Field(description="Unique identifier for this finding")
    category: str = pd.Field(description="Finding category")
    severity: str = pd.Field(description="Finding severity level")
    title: str = pd.Field(description="Short title")
    description: str = pd.Field(description="Detailed description")
    strategy_mapping: StrategyMapping | None = pd.Field(
        default=None,
        description="Strategy mapping for this finding",
    )
    evidence_refs: list[str] = pd.Field(
        default_factory=list,
        description="References to supporting evidence",
    )
    recommendation: str | None = pd.Field(
        default=None,
        description="Suggested action",
    )

    model_config = pd.ConfigDict(extra="forbid")


class TranslatorOutput(pd.BaseModel):
    """Output from the Translator role."""

    translator_id: str = pd.Field(description="Unique identifier for this translation")
    structured_findings: list[StructuredFinding] = pd.Field(
        default_factory=list,
        description="Findings with strategy context",
    )
    strategy_summary: dict[Strategy, int] = pd.Field(
        default_factory=dict,
        description="Count of findings per strategy",
    )
    dominant_strategy: Strategy | None = pd.Field(
        default=None,
        description="Most frequent strategy across findings",
    )
    improvement_achieved: bool = pd.Field(
        default=False,
        description="Whether the competitor achieved improvement",
    )
    score_delta: float | None = pd.Field(
        default=None,
        description="Score delta from comparison (if available)",
    )
    lessons_applicable: bool = pd.Field(
        default=False,
        description="Whether lessons can be derived from this output",
    )
    validation_errors: list[str] = pd.Field(
        default_factory=list,
        description="Any validation errors encountered",
    )
    created_at: datetime = pd.Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this translation was created",
    )

    model_config = pd.ConfigDict(extra="forbid")


@dataclass
class TranslatorInput:
    """Input for the Translator role."""

    competitor_output: CompetitorOutput | None = None
    additional_context: dict[str, Any] = field(default_factory=dict)
    target_agent: str = ""


class TranslatorRole:
    """Translates raw competitor output into validated, structured JSON strategy.

    The Translator role:
    1. Takes CompetitorOutput (raw findings, comparison data, replay artifacts)
    2. Maps each finding to appropriate Strategy/SubStrategy values
    3. Validates strategy pairs against the hierarchy
    4. Structures findings with strategy context
    5. Produces TranslatorOutput ready for the Analyst role
    """

    # Keyword patterns for strategy inference
    STRATEGY_PATTERNS: dict[Strategy, list[str]] = {
        Strategy.POLICY_QUALITY: [
            "policy",
            "rule",
            "engagement",
            "ranking",
            "budget",
            "tool access",
            "permission",
            "allowed",
            "denied",
        ],
        Strategy.SKILL_QUALITY: [
            "instruction",
            "prompt",
            "clarity",
            "example",
            "context",
            "voice",
            "tone",
            "playbook",
            "guidance",
            "confusion",
        ],
        Strategy.TOOL_QUALITY: [
            "tool",
            "efficiency",
            "selection",
            "error recovery",
            "retry",
            "timeout",
            "failure",
            "call",
            "api",
        ],
        Strategy.HARNESS_QUALITY: [
            "grader",
            "calibration",
            "timeout tuning",
            "fixture",
            "harness",
            "eval config",
            "test setup",
        ],
        Strategy.EVAL_QUALITY: [
            "rubric",
            "precision",
            "coverage",
            "test case",
            "false positive",
            "threshold",
            "grading",
        ],
        Strategy.AGENT_BEHAVIOR: [
            "evidence",
            "completion",
            "precision",
            "safety",
            "quality",
            "behavior",
            "output",
            "result",
        ],
    }

    SUBSTRATEGY_PATTERNS: dict[SubStrategy, list[str]] = {
        SubStrategy.ENGAGEMENT_POLICY: ["engagement", "engage", "interact"],
        SubStrategy.RANKING_POLICY: ["ranking", "rank", "score", "prioritize"],
        SubStrategy.STRATEGY_BUDGET: ["budget", "limit", "constraint"],
        SubStrategy.TOOL_ACCESS: ["tool access", "permission", "allowed tool"],
        SubStrategy.INSTRUCTION_CLARITY: ["instruction", "unclear", "confusing", "clarity"],
        SubStrategy.EXAMPLE_QUALITY: ["example", "few-shot", "sample"],
        SubStrategy.CONTEXT_RELEVANCE: ["context", "relevant", "irrelevant"],
        SubStrategy.VOICE_TONE: ["voice", "tone", "style", "language"],
        SubStrategy.PLAYBOOK_ADHERENCE: ["playbook", "adherence", "follow", "protocol"],
        SubStrategy.TOOL_EFFICIENCY: ["efficiency", "efficient", "waste", "redundant"],
        SubStrategy.TOOL_SELECTION: ["selection", "choose", "pick", "wrong tool"],
        SubStrategy.ERROR_RECOVERY: [
            "error recovery",
            "recover",
            "handle error",
            "fallback",
            "timeout",
            "timed out",
        ],
        SubStrategy.RETRY_BEHAVIOR: ["retry", "retries", "attempt again"],
        SubStrategy.REPO_INSPECTION: ["inspection", "inspect", "explore", "scan"],
        SubStrategy.GRADER_CALIBRATION: ["grader", "calibration", "score alignment"],
        SubStrategy.TIMEOUT_TUNING: ["timeout", "timed out", "deadline"],
        SubStrategy.FIXTURE_DESIGN: ["fixture", "test data", "setup"],
        SubStrategy.RUBRIC_PRECISION: ["rubric", "criteria", "evaluation standard"],
        SubStrategy.TEST_COVERAGE: ["coverage", "test case", "missing test"],
        SubStrategy.FALSE_POSITIVE_RATE: ["false positive", "incorrect fail"],
        SubStrategy.EVIDENCE_QUALITY: ["evidence", "proof", "justification"],
        SubStrategy.TASK_COMPLETION: ["completion", "complete", "finish", "done"],
        SubStrategy.CHANGE_PRECISION: ["precision", "accurate", "exact"],
        SubStrategy.SAFETY_QUALITY: ["safety", "safe", "risk", "dangerous"],
        SubStrategy.ENGAGEMENT_PROXY: ["engagement proxy", "interaction metric"],
    }

    def translate(self, input_data: TranslatorInput) -> TranslatorOutput:
        """Translate competitor output into structured strategy."""
        translator_id = f"translator-{uuid4().hex[:8]}"

        if input_data.competitor_output is None:
            return TranslatorOutput(
                translator_id=translator_id,
                validation_errors=["No competitor output provided"],
            )

        competitor = input_data.competitor_output

        # Process findings into structured format
        structured_findings = self._structure_findings(competitor.findings)

        # Calculate strategy summary
        strategy_summary = self._calculate_strategy_summary(structured_findings)

        # Determine dominant strategy
        dominant_strategy = self._get_dominant_strategy(strategy_summary)

        # Determine if lessons are applicable
        lessons_applicable = (
            competitor.improvement_achieved
            or len([f for f in structured_findings if f.strategy_mapping]) > 0
        )

        # Extract score delta if available
        score_delta = None
        if competitor.comparison:
            score_delta = competitor.comparison.metrics.score_delta

        return TranslatorOutput(
            translator_id=translator_id,
            structured_findings=structured_findings,
            strategy_summary=strategy_summary,
            dominant_strategy=dominant_strategy,
            improvement_achieved=competitor.improvement_achieved,
            score_delta=score_delta,
            lessons_applicable=lessons_applicable,
        )

    def _structure_findings(
        self,
        findings: list[ReviewFinding],
    ) -> list[StructuredFinding]:
        """Convert raw findings to structured findings with strategy context."""
        structured = []

        for finding in findings:
            strategy_mapping = self._infer_strategy_mapping(finding)

            structured_finding = StructuredFinding(
                finding_id=finding.finding_id,
                category=finding.category,
                severity=finding.severity,
                title=finding.title,
                description=finding.description,
                strategy_mapping=strategy_mapping,
                evidence_refs=finding.evidence_refs,
                recommendation=finding.recommendation,
            )
            structured.append(structured_finding)

        return structured

    def _infer_strategy_mapping(self, finding: ReviewFinding) -> StrategyMapping | None:
        """Infer strategy mapping from a finding's content."""
        # Combine title and description for analysis
        text = f"{finding.title} {finding.description}".lower()

        # Find best matching strategy
        best_strategy = self._match_strategy(text)
        if best_strategy is None:
            return None

        # Find matching sub-strategies
        sub_strategies = self._match_sub_strategies(text, best_strategy)

        # Calculate confidence based on pattern matches
        confidence = self._calculate_confidence(text, best_strategy, sub_strategies)

        return StrategyMapping(
            strategy=best_strategy,
            sub_strategies=sub_strategies,
            confidence=confidence,
            source_finding_id=finding.finding_id,
            rationale=f"Inferred from finding: {finding.title}",
        )

    def _match_strategy(self, text: str) -> Strategy | None:
        """Match text to the best strategy."""
        scores: dict[Strategy, int] = {}

        for strategy, patterns in self.STRATEGY_PATTERNS.items():
            score = sum(1 for pattern in patterns if pattern in text)
            if score > 0:
                scores[strategy] = score

        if not scores:
            return None

        return max(scores.items(), key=lambda item: item[1])[0]

    def _match_sub_strategies(
        self,
        text: str,
        parent_strategy: Strategy,
    ) -> list[SubStrategy]:
        """Match text to sub-strategies within a parent strategy."""
        matches: list[SubStrategy] = []

        for sub_strategy, patterns in self.SUBSTRATEGY_PATTERNS.items():
            # Only consider sub-strategies that belong to the parent
            if not validate_strategy_pair(parent_strategy, sub_strategy):
                continue

            if any(pattern in text for pattern in patterns):
                matches.append(sub_strategy)

        return matches

    def _calculate_confidence(
        self,
        text: str,
        strategy: Strategy,
        sub_strategies: list[SubStrategy],
    ) -> float:
        """Calculate confidence score for a strategy mapping."""
        base_confidence = 0.5

        # Boost for multiple pattern matches in strategy
        strategy_patterns = self.STRATEGY_PATTERNS.get(strategy, [])
        strategy_matches = sum(1 for p in strategy_patterns if p in text)
        strategy_boost = min(0.2, strategy_matches * 0.05)

        # Boost for sub-strategy matches
        sub_boost = min(0.2, len(sub_strategies) * 0.05)

        # Boost for category alignment
        category_boost = 0.0
        if strategy == Strategy.TOOL_QUALITY and ("tool" in text or "timeout" in text):
            category_boost = 0.1
        elif strategy == Strategy.POLICY_QUALITY and ("policy" in text or "rule" in text):
            category_boost = 0.1

        return min(1.0, base_confidence + strategy_boost + sub_boost + category_boost)

    def _calculate_strategy_summary(
        self,
        findings: list[StructuredFinding],
    ) -> dict[Strategy, int]:
        """Calculate count of findings per strategy."""
        summary: dict[Strategy, int] = {}

        for finding in findings:
            if finding.strategy_mapping:
                strategy = finding.strategy_mapping.strategy
                summary[strategy] = summary.get(strategy, 0) + 1

        return summary

    def _get_dominant_strategy(
        self,
        summary: dict[Strategy, int],
    ) -> Strategy | None:
        if not summary:
            return None
        return max(summary.items(), key=lambda item: item[1])[0]

    def validate_translation(
        self,
        output: TranslatorOutput,
    ) -> tuple[bool, list[str]]:
        """Validate a translation output.

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors: list[str] = []

        # Check for at least some structured findings
        if not output.structured_findings:
            errors.append("No structured findings in translation")

        # Validate strategy mappings
        for finding in output.structured_findings:
            if finding.strategy_mapping:
                mapping = finding.strategy_mapping
                # Validate all sub-strategies belong to the parent
                for sub in mapping.sub_strategies:
                    if not validate_strategy_pair(mapping.strategy, sub):
                        errors.append(f"Invalid strategy pair: {mapping.strategy} / {sub}")

        return len(errors) == 0, errors

    def to_json_schema(self, output: TranslatorOutput) -> dict[str, Any]:
        """Convert TranslatorOutput to JSON-serializable dict."""
        return output.model_dump()


__all__ = [
    "StrategyMapping",
    "StructuredFinding",
    "TranslatorInput",
    "TranslatorOutput",
    "TranslatorRole",
]
