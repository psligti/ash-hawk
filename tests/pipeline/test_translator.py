"""Tests for TranslatorRole."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ash_hawk.contracts import ReviewFinding
from ash_hawk.pipeline.translator import (
    StrategyMapping,
    StructuredFinding,
    TranslatorInput,
    TranslatorOutput,
    TranslatorRole,
)
from ash_hawk.strategies import Strategy, SubStrategy


@pytest.fixture
def translator() -> TranslatorRole:
    return TranslatorRole()


@pytest.fixture
def tool_finding() -> ReviewFinding:
    return ReviewFinding(
        finding_id="finding-tool-001",
        category="reliability",
        severity="warning",
        title="Tool call timed out",
        description="Tool bash timed out after 5000ms during execution",
        evidence_refs=["tool_calls.1"],
        recommendation="Add timeout handling",
    )


@pytest.fixture
def policy_finding() -> ReviewFinding:
    return ReviewFinding(
        finding_id="finding-policy-001",
        category="policy",
        severity="info",
        title="Policy engagement issue",
        description="Engagement policy caused ranking issues in tool selection",
        evidence_refs=["policy.1"],
        recommendation="Adjust engagement policy",
    )


@pytest.fixture
def skill_finding() -> ReviewFinding:
    return ReviewFinding(
        finding_id="finding-skill-001",
        category="quality",
        severity="warning",
        title="Instruction clarity problem",
        description="Agent showed confusion due to unclear instructions in prompt",
        evidence_refs=["steps.1"],
        recommendation="Improve instruction clarity",
    )


class MockCompetitorOutput:
    def __init__(
        self,
        findings: list[ReviewFinding],
        improvement_achieved: bool = False,
    ) -> None:
        self.findings = findings
        self.comparison = None
        self.replay_artifact = None
        self.improvement_achieved = improvement_achieved
        self.error = None


class TestTranslatorInput:
    def test_init_defaults(self):
        input_data = TranslatorInput()
        assert input_data.competitor_output is None
        assert input_data.additional_context == {}
        assert input_data.target_agent == ""

    def test_init_with_values(self):
        competitor = MockCompetitorOutput(findings=[])
        input_data = TranslatorInput(
            competitor_output=competitor,
            additional_context={"key": "value"},
            target_agent="test-agent",
        )
        assert input_data.competitor_output == competitor
        assert input_data.additional_context == {"key": "value"}
        assert input_data.target_agent == "test-agent"


class TestStrategyMapping:
    def test_init_defaults(self):
        mapping = StrategyMapping(strategy=Strategy.TOOL_QUALITY, confidence=0.5)
        assert mapping.strategy == Strategy.TOOL_QUALITY
        assert mapping.sub_strategies == []
        assert mapping.confidence == 0.5
        assert mapping.source_finding_id is None
        assert mapping.rationale == ""

    def test_init_with_values(self):
        mapping = StrategyMapping(
            strategy=Strategy.TOOL_QUALITY,
            sub_strategies=[SubStrategy.ERROR_RECOVERY],
            confidence=0.85,
            source_finding_id="finding-001",
            rationale="Tool timeout detected",
        )
        assert mapping.strategy == Strategy.TOOL_QUALITY
        assert mapping.sub_strategies == [SubStrategy.ERROR_RECOVERY]
        assert mapping.confidence == 0.85


class TestStructuredFinding:
    def test_init_defaults(self):
        finding = StructuredFinding(
            finding_id="finding-001",
            category="reliability",
            severity="warning",
            title="Test finding",
            description="Test description",
        )
        assert finding.finding_id == "finding-001"
        assert finding.category == "reliability"
        assert finding.severity == "warning"
        assert finding.title == "Test finding"
        assert finding.description == "Test description"
        assert finding.strategy_mapping is None
        assert finding.evidence_refs == []
        assert finding.recommendation is None


class TestTranslatorOutput:
    def test_init_defaults(self):
        output = TranslatorOutput(translator_id="translator-001")
        assert output.translator_id == "translator-001"
        assert output.structured_findings == []
        assert output.strategy_summary == {}
        assert output.dominant_strategy is None
        assert output.improvement_achieved is False
        assert output.score_delta is None
        assert output.lessons_applicable is False
        assert output.validation_errors == []


class TestTranslatorRoleTranslate:
    def test_translate_null_input(self, translator: TranslatorRole):
        input_data = TranslatorInput()
        output = translator.translate(input_data)
        assert len(output.validation_errors) == 1
        assert "No competitor output" in output.validation_errors[0]

    def test_translate_empty_findings(self, translator: TranslatorRole):
        competitor = MockCompetitorOutput(findings=[])
        input_data = TranslatorInput(competitor_output=competitor)
        output = translator.translate(input_data)
        assert output.structured_findings == []
        assert output.dominant_strategy is None

    def test_translate_tool_finding(
        self,
        translator: TranslatorRole,
        tool_finding: ReviewFinding,
    ):
        competitor = MockCompetitorOutput(findings=[tool_finding])
        input_data = TranslatorInput(competitor_output=competitor)
        output = translator.translate(input_data)

        assert len(output.structured_findings) == 1
        structured = output.structured_findings[0]
        assert structured.finding_id == tool_finding.finding_id
        assert structured.strategy_mapping is not None
        assert structured.strategy_mapping.strategy == Strategy.TOOL_QUALITY

    def test_translate_policy_finding(
        self,
        translator: TranslatorRole,
        policy_finding: ReviewFinding,
    ):
        competitor = MockCompetitorOutput(findings=[policy_finding])
        input_data = TranslatorInput(competitor_output=competitor)
        output = translator.translate(input_data)

        assert len(output.structured_findings) == 1
        structured = output.structured_findings[0]
        assert structured.strategy_mapping is not None
        assert structured.strategy_mapping.strategy == Strategy.POLICY_QUALITY

    def test_translate_skill_finding(
        self,
        translator: TranslatorRole,
        skill_finding: ReviewFinding,
    ):
        competitor = MockCompetitorOutput(findings=[skill_finding])
        input_data = TranslatorInput(competitor_output=competitor)
        output = translator.translate(input_data)

        assert len(output.structured_findings) == 1
        structured = output.structured_findings[0]
        assert structured.strategy_mapping is not None
        assert structured.strategy_mapping.strategy == Strategy.SKILL_QUALITY

    def test_translate_multiple_findings(
        self,
        translator: TranslatorRole,
        tool_finding: ReviewFinding,
        policy_finding: ReviewFinding,
    ):
        competitor = MockCompetitorOutput(findings=[tool_finding, policy_finding])
        input_data = TranslatorInput(competitor_output=competitor)
        output = translator.translate(input_data)

        assert len(output.structured_findings) == 2
        assert Strategy.TOOL_QUALITY in output.strategy_summary
        assert Strategy.POLICY_QUALITY in output.strategy_summary

    def test_translate_improvement_achieved(
        self,
        translator: TranslatorRole,
        tool_finding: ReviewFinding,
    ):
        competitor = MockCompetitorOutput(
            findings=[tool_finding],
            improvement_achieved=True,
        )
        input_data = TranslatorInput(competitor_output=competitor)
        output = translator.translate(input_data)

        assert output.improvement_achieved is True
        assert output.lessons_applicable is True


class TestTranslatorRoleStrategyInference:
    def test_infer_timeout_strategy(self, translator: TranslatorRole):
        finding = ReviewFinding(
            finding_id="finding-timeout",
            category="reliability",
            severity="warning",
            title="Timeout error",
            description="Tool call timed out",
        )
        competitor = MockCompetitorOutput(findings=[finding])
        input_data = TranslatorInput(competitor_output=competitor)
        output = translator.translate(input_data)

        assert output.dominant_strategy == Strategy.TOOL_QUALITY
        structured = output.structured_findings[0]
        assert (
            SubStrategy.TIMEOUT_TUNING in structured.strategy_mapping.sub_strategies
            or SubStrategy.ERROR_RECOVERY in structured.strategy_mapping.sub_strategies
        )

    def test_infer_instruction_strategy(self, translator: TranslatorRole):
        finding = ReviewFinding(
            finding_id="finding-instruction",
            category="quality",
            severity="info",
            title="Unclear instruction",
            description="Agent was confused by unclear prompt instructions",
        )
        competitor = MockCompetitorOutput(findings=[finding])
        input_data = TranslatorInput(competitor_output=competitor)
        output = translator.translate(input_data)

        assert output.dominant_strategy == Strategy.SKILL_QUALITY

    def test_infer_grader_strategy(self, translator: TranslatorRole):
        finding = ReviewFinding(
            finding_id="finding-grader",
            category="evaluation",
            severity="info",
            title="Grader calibration issue",
            description="Grader scored incorrectly due to calibration",
        )
        competitor = MockCompetitorOutput(findings=[finding])
        input_data = TranslatorInput(competitor_output=competitor)
        output = translator.translate(input_data)

        assert output.dominant_strategy == Strategy.HARNESS_QUALITY

    def test_infer_rubric_strategy(self, translator: TranslatorRole):
        finding = ReviewFinding(
            finding_id="finding-rubric",
            category="evaluation",
            severity="info",
            title="Rubric precision problem",
            description="Rubric criteria was imprecise",
        )
        competitor = MockCompetitorOutput(findings=[finding])
        input_data = TranslatorInput(competitor_output=competitor)
        output = translator.translate(input_data)

        assert output.dominant_strategy == Strategy.EVAL_QUALITY


class TestTranslatorRoleValidation:
    def test_validate_translation_valid(
        self,
        translator: TranslatorRole,
        tool_finding: ReviewFinding,
    ):
        competitor = MockCompetitorOutput(findings=[tool_finding])
        input_data = TranslatorInput(competitor_output=competitor)
        output = translator.translate(input_data)

        is_valid, errors = translator.validate_translation(output)
        assert is_valid is True
        assert errors == []

    def test_validate_translation_empty_findings(self, translator: TranslatorRole):
        output = TranslatorOutput(translator_id="translator-001")

        is_valid, errors = translator.validate_translation(output)
        assert is_valid is False
        assert "No structured findings" in errors[0]


class TestTranslatorRoleToJson:
    def test_to_json_schema(
        self,
        translator: TranslatorRole,
        tool_finding: ReviewFinding,
    ):
        competitor = MockCompetitorOutput(findings=[tool_finding])
        input_data = TranslatorInput(competitor_output=competitor)
        output = translator.translate(input_data)

        json_dict = translator.to_json_schema(output)
        assert "translator_id" in json_dict
        assert "structured_findings" in json_dict
        assert "strategy_summary" in json_dict
        assert isinstance(json_dict["structured_findings"], list)
