"""Tests for PromptStackOptimizerGrader."""

from __future__ import annotations

import pytest

from ash_hawk.graders.prompt_stack_optimizer import (
    DEFAULT_RUBRIC,
    CategoryDef,
    CategoryEvidence,
    GrowthOpportunity,
    MetaMetrics,
    MutationTarget,
    PromptStackOptimizerConfig,
    PromptStackOptimizerGrader,
    RubricDef,
    RubricEvidence,
    SubcategoryDef,
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


# =========================================================================
# Config Tests
# =========================================================================


class TestPromptStackOptimizerConfig:
    def test_defaults(self):
        config = PromptStackOptimizerConfig()
        assert config.pass_threshold == 0.6
        assert config.use_llm_judge is False
        assert config.judge_model is None
        assert config.judge_temperature == 0.0
        assert config.max_growth_opportunities == 5
        assert config.max_mutation_targets == 5
        assert config.baseline_scores is None
        assert config.token_budget is None
        assert config.max_tool_calls is None
        assert config.rubric.total_subcategories == 25

    def test_custom_config(self):
        config = PromptStackOptimizerConfig(
            pass_threshold=0.8,
            use_llm_judge=True,
            judge_model="gpt-4",
            max_growth_opportunities=3,
            token_budget=10000,
            max_tool_calls=20,
        )
        assert config.pass_threshold == 0.8
        assert config.use_llm_judge is True
        assert config.judge_model == "gpt-4"
        assert config.max_growth_opportunities == 3
        assert config.token_budget == 10000

    def test_invalid_threshold_too_high(self):
        with pytest.raises(Exception):
            PromptStackOptimizerConfig(pass_threshold=1.5)

    def test_invalid_threshold_too_low(self):
        with pytest.raises(Exception):
            PromptStackOptimizerConfig(pass_threshold=-0.1)

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            PromptStackOptimizerConfig(nonexistent_field="bad")

    def test_dict_init(self):
        grader = PromptStackOptimizerGrader(config={"pass_threshold": 0.7})
        assert grader._config.pass_threshold == 0.7


# =========================================================================
# Rubric Tests
# =========================================================================


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

    def test_all_subcategories_have_scoring_mode(self):
        for cat in DEFAULT_RUBRIC.categories:
            for sub in cat.subcategories:
                assert sub.scoring_mode in ("deterministic", "llm", "hybrid")


# =========================================================================
# MetaMetrics Tests
# =========================================================================


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

    def test_cache_metrics(self, grader, rich_transcript):
        meta = grader._extract_meta_metrics(rich_transcript)
        assert meta.input_tokens == 5000

    def test_new_meta_metrics_present(self, grader, rich_transcript):
        meta = grader._extract_meta_metrics(rich_transcript)
        assert hasattr(meta, "prompt_efficiency")
        assert hasattr(meta, "selection_quality")
        assert hasattr(meta, "exploration")
        assert hasattr(meta, "layer_hygiene")
        assert 0.0 <= meta.prompt_efficiency <= 1.0
        assert 0.0 <= meta.selection_quality <= 1.0
        assert 0.0 <= meta.exploration <= 1.0
        assert 0.0 <= meta.layer_hygiene <= 1.0

    def test_exploration_with_unique_tools(self, grader):
        transcript = EvalTranscript(
            tool_calls=[{"name": f"tool_{i}", "input": {}, "output": "ok"} for i in range(5)],
            token_usage=TokenUsage(input=100, output=50),
        )
        meta = grader._extract_meta_metrics(transcript)
        assert meta.exploration == 1.0
        assert meta.unique_tools_used == 5

    def test_layer_hygiene_with_errors(self, grader):
        transcript = EvalTranscript(
            tool_calls=[
                {"name": "tool", "input": {}, "output": "ok"},
                {"name": "tool", "input": {}, "output": "error", "is_error": True},
            ],
            token_usage=TokenUsage(input=100, output=50),
        )
        meta = grader._extract_meta_metrics(transcript)
        assert meta.layer_hygiene < 1.0
        assert meta.error_count == 1


# =========================================================================
# Individual Scorer Tests
# =========================================================================


class TestToolSelectionScorer:
    def test_no_tool_calls(self, grader, empty_transcript):
        meta = grader._extract_meta_metrics(empty_transcript)
        result = grader._score_tool_selection(empty_transcript, meta)
        assert result.subcategory_id == "tool_selection"
        assert result.score == 0.5
        assert "No tool calls made" in result.evidence

    def test_high_success_rate(self, grader, rich_transcript):
        meta = grader._extract_meta_metrics(rich_transcript)
        result = grader._score_tool_selection(rich_transcript, meta)
        assert result.score >= 0.7

    def test_duplicate_tool_calls(self, grader):
        transcript = EvalTranscript(
            tool_calls=[
                {"name": "read", "input": {"path": "a.py"}, "output": "ok"},
                {"name": "read", "input": {"path": "a.py"}, "output": "ok"},
                {"name": "read", "input": {"path": "a.py"}, "output": "ok"},
                {"name": "read", "input": {"path": "a.py"}, "output": "ok"},
            ],
            token_usage=TokenUsage(input=100, output=50),
        )
        meta = grader._extract_meta_metrics(transcript)
        result = grader._score_tool_selection(transcript, meta)
        assert result.score < 0.9
        assert any("duplicate" in e.lower() for e in result.evidence)

    def test_low_success_rate(self, grader, error_transcript):
        meta = grader._extract_meta_metrics(error_transcript)
        result = grader._score_tool_selection(error_transcript, meta)
        assert result.score < 0.8


class TestToolCallEfficiencyScorer:
    def test_no_tool_calls(self, grader, empty_transcript):
        meta = grader._extract_meta_metrics(empty_transcript)
        result = grader._score_tool_call_efficiency(empty_transcript, meta)
        assert result.score == 0.7

    def test_within_budget(self, grader, rich_transcript):
        grader._config.max_tool_calls = 10
        meta = grader._extract_meta_metrics(rich_transcript)
        result = grader._score_tool_call_efficiency(rich_transcript, meta)
        assert result.score >= 0.8

    def test_exceeds_budget(self):
        grader = PromptStackOptimizerGrader(config={"max_tool_calls": 2})
        transcript = EvalTranscript(
            tool_calls=[{"name": f"tool_{i}", "input": {}, "output": "ok"} for i in range(10)],
            token_usage=TokenUsage(input=100, output=50),
        )
        meta = grader._extract_meta_metrics(transcript)
        result = grader._score_tool_call_efficiency(transcript, meta)
        assert result.score < 0.5


class TestToolErrorRecoveryScorer:
    def test_no_errors(self, grader, rich_transcript):
        meta = grader._extract_meta_metrics(rich_transcript)
        result = grader._score_tool_error_recovery(rich_transcript, meta)
        assert result.score == 1.0
        assert "No tool errors encountered" in result.evidence

    def test_errors_with_recovery(self, grader):
        transcript = EvalTranscript(
            tool_calls=[
                {"name": "build", "input": {}, "output": "error: compile failed", "is_error": True},
                {"name": "edit", "input": {}, "output": "ok"},
                {"name": "build", "input": {}, "output": "ok"},
            ],
            token_usage=TokenUsage(input=100, output=50),
        )
        meta = grader._extract_meta_metrics(transcript)
        result = grader._score_tool_error_recovery(transcript, meta)
        assert result.score >= 0.8

    def test_errors_without_recovery(self, grader):
        transcript = EvalTranscript(
            tool_calls=[
                {"name": "build", "input": {}, "output": "error: compile failed", "is_error": True},
            ],
            token_usage=TokenUsage(input=100, output=50),
        )
        meta = grader._extract_meta_metrics(transcript)
        result = grader._score_tool_error_recovery(transcript, meta)
        assert result.score == 0.0


class TestSelfCorrectionScorer:
    def test_no_errors_no_corrections(self, grader, empty_transcript):
        meta = grader._extract_meta_metrics(empty_transcript)
        result = grader._score_self_correction(empty_transcript, meta)
        assert result.score == 0.8

    def test_corrections_detected(self, grader, rich_transcript):
        meta = grader._extract_meta_metrics(rich_transcript)
        result = grader._score_self_correction(rich_transcript, meta)
        # "I was wrong" is in the rich_transcript messages
        assert result.score >= 0.8

    def test_errors_no_correction(self, grader, error_transcript):
        meta = grader._extract_meta_metrics(error_transcript)
        result = grader._score_self_correction(error_transcript, meta)
        assert result.score < 0.5


class TestInformationRetentionScorer:
    def test_no_reads(self, grader, empty_transcript):
        meta = grader._extract_meta_metrics(empty_transcript)
        result = grader._score_information_retention(empty_transcript, meta)
        assert result.score == 1.0

    def test_redundant_reads(self, grader, rich_transcript):
        meta = grader._extract_meta_metrics(rich_transcript)
        result = grader._score_information_retention(rich_transcript, meta)
        # rich_transcript has duplicate read_file calls
        assert result.score < 1.0
        assert any("re-read" in e.lower() for e in result.evidence)

    def test_no_redundant_reads(self, grader):
        transcript = EvalTranscript(
            tool_calls=[
                {"name": "read_file", "input": {"path": "a.py"}, "output": "ok"},
                {"name": "read_file", "input": {"path": "b.py"}, "output": "ok"},
            ],
            token_usage=TokenUsage(input=100, output=50),
        )
        meta = grader._extract_meta_metrics(transcript)
        result = grader._score_information_retention(transcript, meta)
        assert result.score == 1.0


class TestContextEfficiencyScorer:
    def test_within_budget(self):
        grader = PromptStackOptimizerGrader(config={"token_budget": 10000})
        transcript = EvalTranscript(
            token_usage=TokenUsage(input=3000, output=1000, cache_read=500),
        )
        meta = grader._extract_meta_metrics(transcript)
        result = grader._score_context_efficiency(transcript, meta)
        assert result.score > 0.8

    def test_exceeds_budget(self):
        grader = PromptStackOptimizerGrader(config={"token_budget": 1000})
        transcript = EvalTranscript(
            token_usage=TokenUsage(input=5000, output=2000),
        )
        meta = grader._extract_meta_metrics(transcript)
        result = grader._score_context_efficiency(transcript, meta)
        assert result.score < 0.5

    def test_no_budget(self, grader, rich_transcript):
        meta = grader._extract_meta_metrics(rich_transcript)
        result = grader._score_context_efficiency(rich_transcript, meta)
        assert result.score >= 0.5


class TestVerificationBehaviorScorer:
    def test_no_tool_calls(self, grader, empty_transcript):
        meta = grader._extract_meta_metrics(empty_transcript)
        result = grader._score_verification_behavior(empty_transcript, meta)
        assert result.score == 0.5

    def test_verification_present(self, grader, rich_transcript):
        meta = grader._extract_meta_metrics(rich_transcript)
        result = grader._score_verification_behavior(rich_transcript, meta)
        # run_tests and lint are verification tools
        assert result.score >= 0.7

    def test_no_verification(self, grader):
        transcript = EvalTranscript(
            tool_calls=[
                {"name": "edit_file", "input": {"path": "a.py"}, "output": "ok"},
                {"name": "write_file", "input": {"path": "b.py"}, "output": "ok"},
            ],
            token_usage=TokenUsage(input=100, output=50),
        )
        meta = grader._extract_meta_metrics(transcript)
        result = grader._score_verification_behavior(transcript, meta)
        assert result.score == 0.3


class TestInputTokenEfficiencyScorer:
    def test_no_tokens(self, grader, empty_transcript):
        meta = grader._extract_meta_metrics(empty_transcript)
        result = grader._score_input_token_efficiency(empty_transcript, meta)
        assert result.score == 0.5

    def test_balanced_ratio(self, grader, rich_transcript):
        meta = grader._extract_meta_metrics(rich_transcript)
        result = grader._score_input_token_efficiency(rich_transcript, meta)
        # input=5000, total=7700, ratio ~0.65 => balanced
        assert result.score >= 0.8


class TestOutputConcisenessScorer:
    def test_no_output(self, grader, empty_transcript):
        meta = grader._extract_meta_metrics(empty_transcript)
        result = grader._score_output_conciseness(empty_transcript, meta)
        assert result.score == 0.5

    def test_reasonable_output(self, grader, rich_transcript):
        meta = grader._extract_meta_metrics(rich_transcript)
        result = grader._score_output_conciseness(rich_transcript, meta)
        assert result.score >= 0.6


class TestCacheUtilizationScorer:
    def test_no_cache(self, grader, empty_transcript):
        meta = grader._extract_meta_metrics(empty_transcript)
        result = grader._score_cache_utilization(empty_transcript, meta)
        assert result.score == 0.5

    def test_cache_present(self, grader, rich_transcript):
        meta = grader._extract_meta_metrics(rich_transcript)
        result = grader._score_cache_utilization(rich_transcript, meta)
        assert result.score > 0.5


class TestReasoningTokenRatioScorer:
    def test_no_output(self, grader, empty_transcript):
        meta = grader._extract_meta_metrics(empty_transcript)
        result = grader._score_reasoning_token_ratio(empty_transcript, meta)
        assert result.score == 0.5

    def test_healthy_ratio(self, grader, rich_transcript):
        meta = grader._extract_meta_metrics(rich_transcript)
        result = grader._score_reasoning_token_ratio(rich_transcript, meta)
        # reasoning=500, output=2000, density=0.25 => healthy
        assert result.score >= 0.8


class TestPolicyAdherenceScorer:
    def test_no_violations(self, grader, rich_transcript):
        meta = grader._extract_meta_metrics(rich_transcript)
        result = grader._score_policy_adherence(rich_transcript, meta)
        assert result.score == 1.0

    def test_violations(self, grader, error_transcript):
        meta = grader._extract_meta_metrics(error_transcript)
        result = grader._score_policy_adherence(error_transcript, meta)
        assert result.score < 1.0


class TestBoundaryRespectScorer:
    def test_no_violations(self, grader, rich_transcript):
        meta = grader._extract_meta_metrics(rich_transcript)
        result = grader._score_boundary_respect(rich_transcript, meta)
        assert result.score == 1.0

    def test_boundary_violations(self, grader, error_transcript):
        meta = grader._extract_meta_metrics(error_transcript)
        result = grader._score_boundary_respect(error_transcript, meta)
        assert result.score < 1.0


class TestDataHandlingScorer:
    def test_no_sensitive_data(self, grader, rich_transcript):
        meta = grader._extract_meta_metrics(rich_transcript)
        result = grader._score_data_handling(rich_transcript, meta)
        assert result.score == 1.0

    def test_sensitive_data_exposed(self, grader):
        transcript = EvalTranscript(
            messages=[
                {"role": "assistant", "content": 'api_key: "sk-proj-abcdef1234567890abcdef"'},
            ],
            token_usage=TokenUsage(input=100, output=50),
        )
        meta = grader._extract_meta_metrics(transcript)
        result = grader._score_data_handling(transcript, meta)
        assert result.score < 1.0


# =========================================================================
# Category & Overall Score Computation
# =========================================================================


class TestScoreComputation:
    def test_compute_category_scores(self, grader, rich_transcript):
        meta = grader._extract_meta_metrics(rich_transcript)
        sub_results: dict[str, SubcategoryEvidence] = {}
        for cat in DEFAULT_RUBRIC.categories:
            for sub in cat.subcategories:
                sub_results[sub.id] = SubcategoryEvidence(
                    subcategory_id=sub.id,
                    subcategory_name=sub.name,
                    score=0.8,
                    evidence=["test"],
                )
        cat_scores = grader._compute_category_scores(sub_results)
        assert len(cat_scores) == 6
        for cs in cat_scores:
            assert abs(cs.score - 0.8) < 0.01

    def test_compute_overall_score(self, grader):
        cat_scores = [
            CategoryEvidence(category_id="a", category_name="A", score=0.8, weight=0.5),
            CategoryEvidence(category_id="b", category_name="B", score=0.6, weight=0.5),
        ]
        overall = grader._compute_overall_score(cat_scores)
        assert abs(overall - 0.7) < 0.01

    def test_overall_score_zero_weight(self, grader):
        cat_scores = [
            CategoryEvidence(category_id="a", category_name="A", score=0.5, weight=0.0),
        ]
        overall = grader._compute_overall_score(cat_scores)
        assert overall == 0.0


# =========================================================================
# Growth Opportunities
# =========================================================================


class TestGrowthOpportunities:
    def test_high_scores_no_opportunities(self, grader):
        sub_results: dict[str, SubcategoryEvidence] = {}
        for cat in DEFAULT_RUBRIC.categories:
            for sub in cat.subcategories:
                sub_results[sub.id] = SubcategoryEvidence(
                    subcategory_id=sub.id,
                    subcategory_name=sub.name,
                    score=0.95,
                    evidence=["excellent"],
                )
        cat_scores = grader._compute_category_scores(sub_results)
        opps = grader._identify_growth_opportunities(cat_scores)
        assert len(opps) == 0

    def test_low_scores_produce_opportunities(self, grader):
        sub_results: dict[str, SubcategoryEvidence] = {}
        for cat in DEFAULT_RUBRIC.categories:
            for sub in cat.subcategories:
                sub_results[sub.id] = SubcategoryEvidence(
                    subcategory_id=sub.id,
                    subcategory_name=sub.name,
                    score=0.4,
                    evidence=["needs work"],
                )
        cat_scores = grader._compute_category_scores(sub_results)
        opps = grader._identify_growth_opportunities(cat_scores)
        assert len(opps) > 0
        assert len(opps) <= grader._config.max_growth_opportunities
        # Sorted by impact descending
        for i in range(len(opps) - 1):
            assert opps[i].impact >= opps[i + 1].impact

    def test_max_opportunities_respected(self):
        grader = PromptStackOptimizerGrader(config={"max_growth_opportunities": 2})
        sub_results: dict[str, SubcategoryEvidence] = {}
        for cat in DEFAULT_RUBRIC.categories:
            for sub in cat.subcategories:
                sub_results[sub.id] = SubcategoryEvidence(
                    subcategory_id=sub.id,
                    subcategory_name=sub.name,
                    score=0.3,
                    evidence=["poor"],
                )
        cat_scores = grader._compute_category_scores(sub_results)
        opps = grader._identify_growth_opportunities(cat_scores)
        assert len(opps) <= 2


# =========================================================================
# Regression Detection
# =========================================================================


class TestRegressionDetection:
    def test_no_baseline(self, grader):
        results = grader._detect_regressions({})
        assert results == []

    def test_no_regressions(self, grader_with_baseline):
        sub_results = {
            "tool_selection": SubcategoryEvidence(
                subcategory_id="tool_selection",
                subcategory_name="Tool Selection",
                score=0.95,
                evidence=[],
            ),
        }
        regressions = grader_with_baseline._detect_regressions(sub_results)
        assert len(regressions) == 0

    def test_regression_detected(self, grader_with_baseline):
        sub_results = {
            "tool_selection": SubcategoryEvidence(
                subcategory_id="tool_selection",
                subcategory_name="Tool Selection",
                score=0.6,
                evidence=[],
            ),
            "verification_behavior": SubcategoryEvidence(
                subcategory_id="verification_behavior",
                subcategory_name="Verification",
                score=0.5,
                evidence=[],
            ),
        }
        regressions = grader_with_baseline._detect_regressions(sub_results)
        assert len(regressions) == 2
        # Sorted by delta ascending (worst first)
        assert regressions[0]["delta"] <= regressions[1]["delta"]


# =========================================================================
# Mutation Targets
# =========================================================================


class TestMutationTargets:
    def test_empty_opportunities(self, grader):
        targets = grader._identify_mutation_targets([])
        assert targets == []

    def test_targets_from_opportunities(self, grader):
        opps = [
            GrowthOpportunity(
                subcategory_id="tool_selection",
                category_id="tool_usage",
                current_score=0.5,
                potential_score=0.8,
                impact=0.03,
                suggestion="Improve tool selection",
            ),
            GrowthOpportunity(
                subcategory_id="verification_behavior",
                category_id="task_completion",
                current_score=0.3,
                potential_score=0.6,
                impact=0.02,
                suggestion="Add verification steps",
            ),
        ]
        targets = grader._identify_mutation_targets(opps)
        assert len(targets) == 2
        assert targets[0].target_type == "tool_definition"
        assert targets[0].priority == "high"
        assert targets[1].target_type == "system_prompt"

    def test_max_targets_respected(self):
        grader = PromptStackOptimizerGrader(config={"max_mutation_targets": 1})
        opps = [
            GrowthOpportunity(
                subcategory_id="tool_selection",
                category_id="tool_usage",
                current_score=0.5,
                potential_score=0.8,
                impact=0.03,
                suggestion="x",
            ),
            GrowthOpportunity(
                subcategory_id="policy_adherence",
                category_id="safety_compliance",
                current_score=0.3,
                potential_score=0.6,
                impact=0.01,
                suggestion="y",
            ),
        ]
        targets = grader._identify_mutation_targets(opps)
        assert len(targets) <= 1


# =========================================================================
# Full Grade Integration
# =========================================================================


class TestGradeIntegration:
    @pytest.mark.asyncio
    async def test_grade_empty_transcript(self, grader, trial, empty_transcript, spec):
        result = await grader.grade(trial, empty_transcript, spec)
        assert result.grader_type == "prompt_stack_optimizer"
        assert 0.0 <= result.score <= 1.0
        assert result.execution_time_seconds >= 0
        assert "rubric_evidence" in result.details
        assert "meta_metrics" in result.details
        assert "growth_opportunities" in result.details
        assert "mutation_targets" in result.details
        assert "category_summary" in result.details
        assert "pass_threshold" in result.details

    @pytest.mark.asyncio
    async def test_grade_rich_transcript(self, grader, trial, rich_transcript, spec):
        result = await grader.grade(trial, rich_transcript, spec)
        assert result.grader_type == "prompt_stack_optimizer"
        assert result.score > 0.0
        assert result.execution_time_seconds >= 0
        assert result.confidence > 0.0

        meta = result.details["meta_metrics"]
        assert meta["total_tokens"] == 7500
        assert meta["tool_call_count"] == 5

        cat_summary = result.details["category_summary"]
        assert len(cat_summary) == 6

    @pytest.mark.asyncio
    async def test_grade_error_transcript(self, grader, trial, error_transcript, spec):
        result = await grader.grade(trial, error_transcript, spec)
        assert result.grader_type == "prompt_stack_optimizer"
        assert 0.0 <= result.score <= 1.0

        meta = result.details["meta_metrics"]
        assert meta["error_count"] >= 2

    @pytest.mark.asyncio
    async def test_grade_pass_threshold(self, trial, rich_transcript, spec):
        low_threshold = PromptStackOptimizerGrader(config={"pass_threshold": 0.1})
        result = await low_threshold.grade(trial, rich_transcript, spec)
        assert result.passed is True

        high_threshold = PromptStackOptimizerGrader(config={"pass_threshold": 0.99})
        result2 = await high_threshold.grade(trial, rich_transcript, spec)
        assert result2.passed is False

    @pytest.mark.asyncio
    async def test_grade_with_spec_config(self, grader, trial, rich_transcript):
        spec = GraderSpec(
            grader_type="prompt_stack_optimizer",
            config={"pass_threshold": 0.9},
        )
        result = await grader.grade(trial, rich_transcript, spec)
        assert result.details["pass_threshold"] == 0.9

    @pytest.mark.asyncio
    async def test_grade_with_regression_baseline(
        self, grader_with_baseline, trial, rich_transcript, spec
    ):
        result = await grader_with_baseline.grade(trial, rich_transcript, spec)
        regressions = result.details["regressions"]
        assert isinstance(regressions, list)

    @pytest.mark.asyncio
    async def test_grade_llm_not_enabled_defaults(self, grader, trial, rich_transcript, spec):
        result = await grader.grade(trial, rich_transcript, spec)
        assert result.details["llm_judge_used"] is False

    @pytest.mark.asyncio
    async def test_grade_needs_review_flag(self, grader, trial, spec):
        transcript = EvalTranscript(
            tool_calls=[
                {"name": "tool", "input": {}, "output": "error", "is_error": True},
            ],
            messages=[{"role": "assistant", "content": "test"}],
            token_usage=TokenUsage(input=100, output=50),
        )
        result = await grader.grade(trial, transcript, spec)
        assert result.needs_review is not None
        assert isinstance(result.needs_review, bool)
        if result.needs_review:
            assert result.review_reason is not None

    @pytest.mark.asyncio
    async def test_grade_new_meta_metrics_in_result(self, grader, trial, rich_transcript, spec):
        result = await grader.grade(trial, rich_transcript, spec)
        meta = result.details["meta_metrics"]
        assert "prompt_efficiency" in meta
        assert "selection_quality" in meta
        assert "exploration" in meta
        assert "layer_hygiene" in meta
        assert 0.0 <= meta["prompt_efficiency"] <= 1.0
        assert 0.0 <= meta["selection_quality"] <= 1.0
        assert 0.0 <= meta["exploration"] <= 1.0
        assert 0.0 <= meta["layer_hygiene"] <= 1.0


# =========================================================================
# Edge Cases
# =========================================================================


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_transcript_with_no_messages_or_tools(self, grader, trial, spec):
        transcript = EvalTranscript(token_usage=TokenUsage(input=100, output=50))
        result = await grader.grade(trial, transcript, spec)
        assert result.grader_type == "prompt_stack_optimizer"
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_transcript_with_only_tool_errors(self, grader, trial, spec):
        transcript = EvalTranscript(
            tool_calls=[
                {"name": "run", "input": {}, "output": "error: crash", "is_error": True},
                {"name": "run", "input": {}, "output": "error: crash", "is_error": True},
            ],
            token_usage=TokenUsage(input=100, output=50),
        )
        result = await grader.grade(trial, transcript, spec)
        assert result.score < 0.7

    @pytest.mark.asyncio
    async def test_transcript_with_non_string_content(self, grader, trial, spec):
        transcript = EvalTranscript(
            messages=[
                {"role": "user", "content": ["list", "content"]},
                {"role": "assistant", "content": 42},
            ],
            token_usage=TokenUsage(input=100, output=50),
        )
        result = await grader.grade(trial, transcript, spec)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_many_tool_calls_single_tool(self, grader, trial, spec):
        transcript = EvalTranscript(
            tool_calls=[{"name": "edit", "input": {"i": i}, "output": "ok"} for i in range(20)],
            token_usage=TokenUsage(input=5000, output=2000),
        )
        result = await grader.grade(trial, transcript, spec)
        assert 0.0 <= result.score <= 1.0


# =========================================================================
# Grader Name & Registry
# =========================================================================


class TestGraderName:
    def test_name(self, grader):
        assert grader.name == "prompt_stack_optimizer"

    def test_registered_in_registry(self):
        from ash_hawk.graders.registry import GraderRegistry, _register_builtin_graders

        registry = GraderRegistry()
        _register_builtin_graders(registry)
        assert "prompt_stack_optimizer" in registry
        grader = registry.get("prompt_stack_optimizer")
        assert grader is not None
        assert grader.name == "prompt_stack_optimizer"


# =========================================================================
# Pydantic Model Validation
# =========================================================================


class TestModelValidation:
    def test_subcategory_evidence_bounds(self):
        with pytest.raises(Exception):
            SubcategoryEvidence(
                subcategory_id="test",
                subcategory_name="Test",
                score=1.5,
                evidence=[],
            )

    def test_subcategory_evidence_valid(self):
        ev = SubcategoryEvidence(
            subcategory_id="test",
            subcategory_name="Test",
            score=0.7,
            evidence=["ok"],
        )
        assert ev.score == 0.7

    def test_growth_opportunity_bounds(self):
        with pytest.raises(Exception):
            GrowthOpportunity(
                subcategory_id="test",
                category_id="cat",
                current_score=1.5,
                potential_score=0.8,
                impact=0.1,
                suggestion="x",
            )

    def test_mutation_target_invalid_type(self):
        with pytest.raises(Exception):
            MutationTarget(
                target_type="invalid_type",
                subcategory_id="test",
                description="x",
            )

    def test_meta_metrics_extra_forbidden(self):
        with pytest.raises(Exception):
            MetaMetrics(nonexistent=True)

    def test_rubric_evidence_valid(self):
        ev = RubricEvidence(
            rubric_version="1.0.0",
            overall_score=0.75,
            category_scores=[],
        )
        assert ev.overall_score == 0.75


# =========================================================================
# Deterministic Dispatch Table
# =========================================================================


class TestDeterministicDispatch:
    def test_known_subcategory(self, grader):
        scorer = grader._get_deterministic_scorer("tool_selection")
        assert scorer is not None

    def test_unknown_subcategory(self, grader):
        scorer = grader._get_deterministic_scorer("nonexistent")
        assert scorer is None

    def test_all_deterministic_subcats_have_scorers(self, grader):
        for cat in DEFAULT_RUBRIC.categories:
            for sub in cat.subcategories:
                if sub.scoring_mode == "deterministic":
                    scorer = grader._get_deterministic_scorer(sub.id)
                    assert scorer is not None, f"Missing scorer for {sub.id}"
