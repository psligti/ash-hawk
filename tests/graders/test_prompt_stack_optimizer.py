"""Tests for PromptStackOptimizerGrader."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ash_hawk.graders.prompt_stack_optimizer import (
    DEFAULT_RUBRIC,
    REQUIRED_SUBCATEGORIES,
    CategoryEvidence,
    GrowthOpportunity,
    MetaMetrics,
    MutationTarget,
    PromptStackOptimizerConfig,
    PromptStackOptimizerGrader,
    RubricDef,
    SubcategoryEvidence,
)
from ash_hawk.types import (
    EvalTranscript,
    EvalTrial,
    GraderSpec,
    TokenUsage,
)


@pytest.fixture
def trial():
    return EvalTrial(id="trial-1", task_id="task-1")


@pytest.fixture
def spec():
    return GraderSpec(grader_type="prompt_stack_optimizer")


@pytest.fixture
def empty_transcript():
    return EvalTranscript()


@pytest.fixture
def rich_transcript():
    return EvalTranscript(
        messages=[
            {"role": "user", "content": "Fix the bug in auth.py"},
            {"role": "assistant", "content": "I'll read the file first to understand the issue."},
            {
                "role": "assistant",
                "content": "Let me fix the error. I was wrong about the root cause.",
            },
            {"role": "user", "content": "Good, now run the tests."},
            {"role": "assistant", "content": "Running tests to verify the fix."},
        ],
        tool_calls=[
            {"name": "read_file", "input": {"path": "auth.py"}, "output": "def login():\n    pass"},
            {"name": "edit_file", "input": {"path": "auth.py", "content": "..."}, "output": "OK"},
            {
                "name": "run_tests",
                "input": {"command": "pytest tests/"},
                "output": "3 passed",
            },
            {
                "name": "read_file",
                "input": {"path": "auth.py"},
                "output": "def login():\n    pass",
            },
            {
                "name": "lint",
                "input": {"path": "auth.py"},
                "output": "All checks passed",
            },
        ],
        trace_events=[
            {"type": "step", "state": "planning"},
            {"type": "step", "state": "executing"},
        ],
        token_usage=TokenUsage(
            input=5000,
            output=2000,
            reasoning=500,
            cache_read=1000,
            cache_write=200,
        ),
        duration_seconds=12.5,
    )


@pytest.fixture
def error_transcript():
    return EvalTranscript(
        messages=[
            {"role": "user", "content": "Delete the secrets file"},
            {"role": "assistant", "content": "I'll try to delete it."},
        ],
        tool_calls=[
            {
                "name": "delete_file",
                "input": {"path": "/etc/secrets"},
                "output": "permission denied: access denied",
                "is_error": True,
            },
            {
                "name": "read_file",
                "input": {"path": "config.json"},
                "output": "error: file not found",
            },
            {
                "name": "read_file",
                "input": {"path": "config.json"},
                "output": "error: file not found",
            },
        ],
        trace_events=[
            {"type": "policy_violation", "policy_violation": True},
        ],
        token_usage=TokenUsage(input=1000, output=500),
        duration_seconds=3.0,
    )


@pytest.fixture
def grader():
    return PromptStackOptimizerGrader()


@pytest.fixture
def grader_with_baseline():
    return PromptStackOptimizerGrader(
        config=PromptStackOptimizerConfig(
            baseline_scores={
                "tool_selection": 0.9,
                "verification_behavior": 0.8,
                "policy_adherence": 1.0,
            },
        )
    )


def make_mock_llm_response(scores: dict[str, dict] | None = None) -> dict:
    if scores is None:
        scores = {
            sc_id: {"score": 0.7, "evidence": ["Mock evidence"], "confidence": 0.8}
            for sc_id in REQUIRED_SUBCATEGORIES
        }
    return {"scores": scores}


class TestPromptStackOptimizerConfig:
    def test_defaults(self):
        config = PromptStackOptimizerConfig()
        assert config.pass_threshold == 0.6
        assert config.judge_max_tokens == 4096
        assert config.judge_temperature == 0.0

    def test_custom_config(self):
        config = PromptStackOptimizerConfig(
            pass_threshold=0.8,
            judge_max_tokens=2048,
            judge_temperature=0.1,
        )
        assert config.pass_threshold == 0.8
        assert config.judge_max_tokens == 2048
        assert config.judge_temperature == 0.1

    def test_invalid_threshold_too_high(self):
        with pytest.raises(Exception):
            PromptStackOptimizerConfig(pass_threshold=1.5)

    def test_invalid_threshold_too_low(self):
        with pytest.raises(Exception):
            PromptStackOptimizerConfig(pass_threshold=-0.1)

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            PromptStackOptimizerConfig(unknown_field="value")

    def test_dict_init(self):
        config = PromptStackOptimizerConfig(**{"pass_threshold": 0.75})
        assert config.pass_threshold == 0.75


class TestRubricDef:
    def test_default_rubric_structure(self):
        assert len(DEFAULT_RUBRIC.categories) == 6
        assert DEFAULT_RUBRIC.total_subcategories == 25
        assert DEFAULT_RUBRIC.version == "1.0.0"

    def test_category_weights_sum(self):
        total_weight = sum(c.weight for c in DEFAULT_RUBRIC.categories)
        assert abs(total_weight - 1.0) < 1e-9

    def test_all_categories_have_subcategories(self):
        for cat in DEFAULT_RUBRIC.categories:
            assert len(cat.subcategories) > 0
            assert cat.id
            assert cat.name


class TestRequiredSubcategories:
    def test_all_25_present(self):
        assert len(REQUIRED_SUBCATEGORIES) == 25

    def test_all_match_rubric(self):
        rubric_ids = set()
        for cat in DEFAULT_RUBRIC.categories:
            for sub in cat.subcategories:
                rubric_ids.add(sub.id)
        assert set(REQUIRED_SUBCATEGORIES) == rubric_ids


class TestMetaMetrics:
    def test_empty_transcript(self, grader, empty_transcript):
        meta = grader._extract_meta_metrics(empty_transcript)
        assert meta.total_tokens == 0
        assert meta.tool_call_count == 0
        assert meta.message_count == 0
        assert meta.error_count == 0
        assert meta.tool_success_rate == 1.0
        assert meta.reasoning_density == 0.0

    def test_rich_transcript(self, grader, rich_transcript):
        meta = grader._extract_meta_metrics(rich_transcript)
        assert meta.total_tokens == 7500
        assert meta.input_tokens == 5000
        assert meta.output_tokens == 2000
        assert meta.reasoning_tokens == 500
        assert meta.tool_call_count == 5
        assert meta.unique_tools_used == 4
        assert meta.message_count == 5
        assert meta.error_count == 0
        assert meta.tool_success_rate == 1.0
        assert meta.duration_seconds == 12.5
        assert meta.tokens_per_tool_call == 1500.0
        assert meta.reasoning_density == 0.25

    def test_error_transcript(self, grader, error_transcript):
        meta = grader._extract_meta_metrics(error_transcript)
        assert meta.error_count >= 2
        assert meta.tool_success_rate < 1.0

    def test_new_meta_metrics_present(self, grader, rich_transcript):
        meta = grader._extract_meta_metrics(rich_transcript)
        assert hasattr(meta, "prompt_efficiency")
        assert hasattr(meta, "selection_quality")
        assert hasattr(meta, "exploration")
        assert hasattr(meta, "layer_hygiene")

    def test_exploration_with_unique_tools(self, grader):
        transcript = EvalTranscript(
            tool_calls=[
                {"name": "read_file", "input": {}, "output": "ok"},
                {"name": "write_file", "input": {}, "output": "ok"},
                {"name": "run_tests", "input": {}, "output": "ok"},
            ],
            token_usage=TokenUsage(input=100, output=50),
        )
        meta = grader._extract_meta_metrics(transcript)
        assert meta.unique_tools_used == 3
        assert meta.exploration > 0

    def test_layer_hygiene_with_errors(self, grader):
        transcript = EvalTranscript(
            tool_calls=[
                {"name": "tool1", "input": {}, "output": "error: failed", "is_error": True},
                {"name": "tool2", "input": {}, "output": "ok"},
            ],
            token_usage=TokenUsage(input=100, output=50),
        )
        meta = grader._extract_meta_metrics(transcript)
        assert meta.error_count >= 1
        assert meta.layer_hygiene < 1.0


class TestScoreComputation:
    def test_compute_category_scores(self, grader):
        subcategory_results = {
            "tool_selection": SubcategoryEvidence(
                subcategory_id="tool_selection",
                subcategory_name="Tool Selection",
                score=0.8,
                confidence=0.9,
                evidence=["test"],
            ),
            "tool_call_efficiency": SubcategoryEvidence(
                subcategory_id="tool_call_efficiency",
                subcategory_name="Call Efficiency",
                score=0.9,
                confidence=0.8,
                evidence=["test"],
            ),
            "tool_error_recovery": SubcategoryEvidence(
                subcategory_id="tool_error_recovery",
                subcategory_name="Error Recovery",
                score=0.7,
                confidence=0.7,
                evidence=["test"],
            ),
            "tool_output_utilization": SubcategoryEvidence(
                subcategory_id="tool_output_utilization",
                subcategory_name="Output Utilization",
                score=0.6,
                confidence=0.6,
                evidence=["test"],
            ),
        }
        scores = grader._compute_category_scores(subcategory_results)
        tool_usage = next(s for s in scores if s.category_id == "tool_usage")
        assert 0.0 <= tool_usage.score <= 1.0

    def test_compute_overall_score(self, grader):
        category_scores = [
            CategoryEvidence(
                category_id="tool_usage",
                category_name="Tool Usage",
                score=0.8,
                weight=0.2,
                subcategory_scores=[],
            ),
            CategoryEvidence(
                category_id="reasoning",
                category_name="Reasoning",
                score=0.7,
                weight=0.2,
                subcategory_scores=[],
            ),
        ]
        overall = grader._compute_overall_score(category_scores)
        assert 0.0 <= overall <= 1.0


class TestGrowthOpportunities:
    def test_high_scores_no_opportunities(self, grader):
        high_scores = []
        for cat in DEFAULT_RUBRIC.categories:
            sub_scores = []
            for sub in cat.subcategories:
                sub_scores.append(
                    SubcategoryEvidence(
                        subcategory_id=sub.id,
                        subcategory_name=sub.name,
                        score=0.95,
                        confidence=0.9,
                        evidence=["Excellent"],
                    )
                )
            high_scores.append(
                CategoryEvidence(
                    category_id=cat.id,
                    category_name=cat.name,
                    score=0.95,
                    weight=cat.weight,
                    subcategory_scores=sub_scores,
                )
            )
        opportunities = grader._identify_growth_opportunities(high_scores)
        assert len(opportunities) == 0

    def test_low_scores_produce_opportunities(self, grader):
        low_scores = []
        for cat in DEFAULT_RUBRIC.categories:
            sub_scores = []
            for sub in cat.subcategories:
                sub_scores.append(
                    SubcategoryEvidence(
                        subcategory_id=sub.id,
                        subcategory_name=sub.name,
                        score=0.4,
                        confidence=0.7,
                        evidence=["Needs improvement"],
                    )
                )
            low_scores.append(
                CategoryEvidence(
                    category_id=cat.id,
                    category_name=cat.name,
                    score=0.4,
                    weight=cat.weight,
                    subcategory_scores=sub_scores,
                )
            )
        opportunities = grader._identify_growth_opportunities(low_scores)
        assert len(opportunities) > 0

    def test_max_opportunities_respected(self, grader):
        low_scores = []
        for cat in DEFAULT_RUBRIC.categories:
            sub_scores = []
            for sub in cat.subcategories:
                sub_scores.append(
                    SubcategoryEvidence(
                        subcategory_id=sub.id,
                        subcategory_name=sub.name,
                        score=0.3,
                        confidence=0.7,
                        evidence=["Needs work"],
                    )
                )
            low_scores.append(
                CategoryEvidence(
                    category_id=cat.id,
                    category_name=cat.name,
                    score=0.3,
                    weight=cat.weight,
                    subcategory_scores=sub_scores,
                )
            )
        opportunities = grader._identify_growth_opportunities(low_scores)
        assert len(opportunities) <= grader._config.max_growth_opportunities


class TestRegressionDetection:
    def test_no_baseline(self, grader):
        results = {
            "tool_selection": SubcategoryEvidence(
                subcategory_id="tool_selection",
                subcategory_name="Tool Selection",
                score=0.5,
                confidence=0.7,
                evidence=["test"],
            )
        }
        regressions = grader._detect_regressions(results)
        assert len(regressions) == 0

    def test_no_regressions(self, grader_with_baseline):
        results = {
            "tool_selection": SubcategoryEvidence(
                subcategory_id="tool_selection",
                subcategory_name="Tool Selection",
                score=0.95,
                confidence=0.9,
                evidence=["Excellent"],
            ),
            "verification_behavior": SubcategoryEvidence(
                subcategory_id="verification_behavior",
                subcategory_name="Verification",
                score=0.9,
                confidence=0.8,
                evidence=["Good"],
            ),
            "policy_adherence": SubcategoryEvidence(
                subcategory_id="policy_adherence",
                subcategory_name="Policy Adherence",
                score=1.0,
                confidence=0.9,
                evidence=["Perfect"],
            ),
        }
        regressions = grader_with_baseline._detect_regressions(results)
        assert len(regressions) == 0

    def test_regression_detected(self, grader_with_baseline):
        results = {
            "tool_selection": SubcategoryEvidence(
                subcategory_id="tool_selection",
                subcategory_name="Tool Selection",
                score=0.5,
                confidence=0.7,
                evidence=["Regression"],
            ),
            "verification_behavior": SubcategoryEvidence(
                subcategory_id="verification_behavior",
                subcategory_name="Verification",
                score=0.9,
                confidence=0.8,
                evidence=["Good"],
            ),
        }
        regressions = grader_with_baseline._detect_regressions(results)
        assert len(regressions) > 0


class TestMutationTargets:
    def test_empty_opportunities(self, grader):
        targets = grader._identify_mutation_targets([])
        assert len(targets) == 0

    def test_targets_from_opportunities(self, grader):
        opportunities = [
            GrowthOpportunity(
                subcategory_id="tool_selection",
                category_id="tool_usage",
                current_score=0.5,
                potential_score=0.9,
                impact=0.2,
                suggestion="Improve tool selection",
            )
        ]
        targets = grader._identify_mutation_targets(opportunities)
        assert len(targets) > 0

    def test_max_targets_respected(self, grader):
        opportunities = [
            GrowthOpportunity(
                subcategory_id=f"sub_{i}",
                category_id="cat",
                current_score=0.3,
                potential_score=0.8,
                impact=0.1,
                suggestion=f"Fix {i}",
            )
            for i in range(20)
        ]
        targets = grader._identify_mutation_targets(opportunities)
        assert len(targets) <= grader._config.max_mutation_targets


class TestGradeIntegration:
    @pytest.mark.asyncio
    async def test_grade_empty_transcript(self, grader, trial, empty_transcript, spec):
        with patch.object(grader, "_run_llm_judge") as mock_judge:
            mock_judge.return_value = {
                sc_id: SubcategoryEvidence(
                    subcategory_id=sc_id,
                    subcategory_name=sc_id,
                    score=0.7,
                    confidence=0.8,
                    evidence=["Mock"],
                )
                for sc_id in REQUIRED_SUBCATEGORIES
            }
            result = await grader.grade(trial, empty_transcript, spec)
            assert result.grader_type == "prompt_stack_optimizer"
            assert 0.0 <= result.score <= 1.0
            assert result.execution_time_seconds >= 0
            assert "rubric_evidence" in result.details
            assert "meta_metrics" in result.details
            assert "growth_opportunities" in result.details
            assert "mutation_targets" in result.details

    @pytest.mark.asyncio
    async def test_grade_rich_transcript(self, grader, trial, rich_transcript, spec):
        with patch.object(grader, "_run_llm_judge") as mock_judge:
            mock_judge.return_value = {
                sc_id: SubcategoryEvidence(
                    subcategory_id=sc_id,
                    subcategory_name=sc_id,
                    score=0.8,
                    confidence=0.9,
                    evidence=["Mock evidence"],
                )
                for sc_id in REQUIRED_SUBCATEGORIES
            }
            result = await grader.grade(trial, rich_transcript, spec)
            assert result.grader_type == "prompt_stack_optimizer"
            assert result.score > 0.0

    @pytest.mark.asyncio
    async def test_grade_with_spec_config(self, grader, trial, rich_transcript):
        spec = GraderSpec(
            grader_type="prompt_stack_optimizer",
            config={"pass_threshold": 0.9},
        )
        with patch.object(grader, "_run_llm_judge") as mock_judge:
            mock_judge.return_value = {
                sc_id: SubcategoryEvidence(
                    subcategory_id=sc_id,
                    subcategory_name=sc_id,
                    score=0.85,
                    confidence=0.8,
                    evidence=["Mock"],
                )
                for sc_id in REQUIRED_SUBCATEGORIES
            }
            result = await grader.grade(trial, rich_transcript, spec)
            assert result.details["pass_threshold"] == 0.9

    @pytest.mark.asyncio
    async def test_grade_with_regression_baseline(
        self, grader_with_baseline, trial, rich_transcript, spec
    ):
        with patch.object(grader_with_baseline, "_run_llm_judge") as mock_judge:
            scores = {
                sc_id: SubcategoryEvidence(
                    subcategory_id=sc_id,
                    subcategory_name=sc_id,
                    score=0.9,
                    confidence=0.9,
                    evidence=["Mock"],
                )
                for sc_id in REQUIRED_SUBCATEGORIES
            }
            scores["tool_selection"] = SubcategoryEvidence(
                subcategory_id="tool_selection",
                subcategory_name="Tool Selection",
                score=0.5,
                confidence=0.7,
                evidence=["Regression"],
            )
            mock_judge.return_value = scores
            result = await grader_with_baseline.grade(trial, rich_transcript, spec)
            regressions = result.details["regressions"]
            assert isinstance(regressions, list)

    @pytest.mark.asyncio
    async def test_grade_needs_review_flag(self, grader, trial, spec):
        transcript = EvalTranscript(
            tool_calls=[
                {"name": "tool", "input": {}, "output": "error", "is_error": True},
            ],
            messages=[{"role": "assistant", "content": "test"}],
            token_usage=TokenUsage(input=100, output=50),
        )
        with patch.object(grader, "_run_llm_judge") as mock_judge:
            mock_judge.return_value = {
                sc_id: SubcategoryEvidence(
                    subcategory_id=sc_id,
                    subcategory_name=sc_id,
                    score=0.7,
                    confidence=0.3,
                    evidence=["Low confidence"],
                )
                for sc_id in REQUIRED_SUBCATEGORIES
            }
            result = await grader.grade(trial, transcript, spec)
            assert result.needs_review is not None

    @pytest.mark.asyncio
    async def test_grade_llm_failure_raises(self, grader, trial, rich_transcript, spec):
        with patch.object(grader, "_run_llm_judge") as mock_judge:
            mock_judge.side_effect = ValueError("LLM failed")
            result = await grader.grade(trial, rich_transcript, spec)
            assert result.error_message is not None
            assert result.score == 0.0


class TestLLMJudgeIntegration:
    @pytest.mark.asyncio
    async def test_run_llm_judge_missing_subcategories_raises(self, grader, rich_transcript):
        incomplete_response = {
            "scores": {
                "tool_selection": {"score": 0.8, "evidence": ["test"], "confidence": 0.9},
            }
        }
        with patch.object(grader, "_get_client") as mock_client:
            mock_response = MagicMock()
            mock_response.text = json.dumps(incomplete_response)
            mock_llm = MagicMock()
            mock_llm.complete = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_llm

            with pytest.raises(ValueError, match="missing"):
                await grader._run_llm_judge(rich_transcript)

    @pytest.mark.asyncio
    async def test_run_llm_judge_invalid_json_raises(self, grader, rich_transcript):
        with patch.object(grader, "_get_client") as mock_client:
            mock_response = MagicMock()
            mock_response.text = "not valid json"
            mock_llm = MagicMock()
            mock_llm.complete = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_llm

            with pytest.raises(ValueError, match="Failed to parse"):
                await grader._run_llm_judge(rich_transcript)

    @pytest.mark.asyncio
    async def test_run_llm_judge_success(self, grader, rich_transcript):
        full_response = make_mock_llm_response()
        with patch.object(grader, "_get_client") as mock_client:
            mock_response = MagicMock()
            mock_response.text = json.dumps(full_response)
            mock_llm = MagicMock()
            mock_llm.complete = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_llm

            results = await grader._run_llm_judge(rich_transcript)
            assert len(results) == 25
            for sc_id in REQUIRED_SUBCATEGORIES:
                assert sc_id in results
                assert 0.0 <= results[sc_id].score <= 1.0


class TestGraderName:
    def test_name(self, grader):
        assert grader.name == "prompt_stack_optimizer"


class TestModelValidation:
    def test_subcategory_evidence_bounds(self):
        with pytest.raises(Exception):
            SubcategoryEvidence(
                subcategory_id="test",
                subcategory_name="Test",
                score=1.5,
                confidence=0.5,
                evidence=["test"],
            )

    def test_subcategory_evidence_valid(self):
        ev = SubcategoryEvidence(
            subcategory_id="test",
            subcategory_name="Test",
            score=0.5,
            confidence=0.5,
            evidence=["test"],
        )
        assert ev.score == 0.5

    def test_growth_opportunity_bounds(self):
        with pytest.raises(Exception):
            GrowthOpportunity(
                subcategory_id="test",
                category_id="cat",
                current_score=1.5,
                potential_score=0.9,
                impact=0.1,
                suggestion="test",
            )

    def test_meta_metrics_extra_forbidden(self):
        with pytest.raises(Exception):
            MetaMetrics(
                total_tokens=100,
                unknown_field="value",
            )
