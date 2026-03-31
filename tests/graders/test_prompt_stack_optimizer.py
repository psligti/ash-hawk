"""Tests for PromptStackOptimizerGrader."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pydantic as pd
import pytest

from ash_hawk.graders.prompt_stack_optimizer import (
    _SUBCATEGORY_TO_CATEGORY,
    DEFAULT_RUBRIC,
    REQUIRED_SUBCATEGORIES,
    CategoryDef,
    PromptStackOptimizerConfig,
    PromptStackOptimizerGrader,
    RubricDef,
    SubcategoryDef,
)
from ash_hawk.types import (
    EvalTranscript,
    EvalTrial,
    FailureMode,
    GraderSpec,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def trial() -> EvalTrial:
    return EvalTrial(id="t1", task_id="task1")


@pytest.fixture
def transcript() -> EvalTranscript:
    return EvalTranscript(
        messages=[
            {"role": "user", "content": "Hello, write a function"},
            {"role": "assistant", "content": "Sure, here is the function:\ndef foo(): pass"},
        ],
        tool_calls=[
            {"name": "bash", "input": {"command": "ls"}, "output": "file1.py\nfile2.py"},
        ],
    )


@pytest.fixture
def spec() -> GraderSpec:
    return GraderSpec(grader_type="prompt_stack_optimizer")


@pytest.fixture
def grader() -> PromptStackOptimizerGrader:
    return PromptStackOptimizerGrader()


def _make_subcategory_results(score: float = 0.8, confidence: float = 0.9) -> dict[str, dict]:
    """Build a full subcategory_results dict with uniform scores."""
    return {
        sc_id: {
            "subcategory_id": sc_id,
            "score": score,
            "confidence": confidence,
            "evidence": ["test evidence"],
        }
        for sc_id in REQUIRED_SUBCATEGORIES
    }


# =============================================================================
# Pydantic model tests
# =============================================================================


class TestSubcategoryDef:
    """Test SubcategoryDef model."""

    def test_valid_construction(self) -> None:
        sc = SubcategoryDef(id="sc1", name="SC One", description="First subcategory")
        assert sc.id == "sc1"
        assert sc.name == "SC One"
        assert sc.weight == 1.0

    def test_custom_weight(self) -> None:
        sc = SubcategoryDef(id="sc1", name="SC", description="desc", weight=0.5)
        assert sc.weight == 0.5

    def test_weight_ge_zero(self) -> None:
        with pytest.raises(pd.ValidationError):
            SubcategoryDef(id="sc1", name="SC", description="desc", weight=-0.1)

    def test_extra_forbid(self) -> None:
        with pytest.raises(pd.ValidationError):
            SubcategoryDef(id="sc1", name="SC", description="desc", extra_field="bad")


class TestCategoryDef:
    """Test CategoryDef model."""

    def test_valid_construction(self) -> None:
        cat = CategoryDef(id="cat1", name="Cat One", description="First category")
        assert cat.id == "cat1"
        assert cat.subcategories == []

    def test_with_subcategories(self) -> None:
        sub = SubcategoryDef(id="s1", name="S1", description="d1")
        cat = CategoryDef(id="cat1", name="Cat", description="desc", subcategories=[sub])
        assert len(cat.subcategories) == 1
        assert cat.subcategories[0].id == "s1"

    def test_weight_ge_zero(self) -> None:
        with pytest.raises(pd.ValidationError):
            CategoryDef(id="c1", name="C", description="d", weight=-0.5)

    def test_extra_forbid(self) -> None:
        with pytest.raises(pd.ValidationError):
            CategoryDef(id="c1", name="C", description="d", unknown="x")


class TestRubricDef:
    """Test RubricDef model."""

    def test_default_version(self) -> None:
        r = RubricDef()
        assert r.version == "1.0.0"
        assert r.categories == []

    def test_total_subcategories_computed_field(self) -> None:
        sub1 = SubcategoryDef(id="s1", name="S1", description="d1")
        sub2 = SubcategoryDef(id="s2", name="S2", description="d2")
        cat = CategoryDef(id="c1", name="C1", description="d", subcategories=[sub1, sub2])
        r = RubricDef(categories=[cat])
        assert r.total_subcategories == 2

    def test_total_subcategories_empty(self) -> None:
        r = RubricDef()
        assert r.total_subcategories == 0

    def test_extra_forbid(self) -> None:
        with pytest.raises(pd.ValidationError):
            RubricDef(extra_field="bad")


# =============================================================================
# DEFAULT_RUBRIC structure tests
# =============================================================================


class TestDefaultRubric:
    """Test the DEFAULT_RUBRIC constant."""

    def test_has_six_categories(self) -> None:
        assert len(DEFAULT_RUBRIC.categories) == 6

    def test_category_ids(self) -> None:
        ids = [c.id for c in DEFAULT_RUBRIC.categories]
        assert ids == ["tool_usage", "reasoning", "context", "completion", "efficiency", "safety"]

    def test_total_subcategories_is_25(self) -> None:
        assert DEFAULT_RUBRIC.total_subcategories == 25

    def test_category_weights_sum_to_one(self) -> None:
        total = sum(c.weight for c in DEFAULT_RUBRIC.categories)
        assert abs(total - 1.0) < 1e-9

    def test_all_categories_have_subcategories(self) -> None:
        for cat in DEFAULT_RUBRIC.categories:
            assert len(cat.subcategories) > 0, f"Category {cat.id} has no subcategories"

    def test_version(self) -> None:
        assert DEFAULT_RUBRIC.version == "1.0.0"


# =============================================================================
# REQUIRED_SUBCATEGORIES and mapping tests
# =============================================================================


class TestRequiredSubcategories:
    """Test REQUIRED_SUBCATEGORIES and _SUBCATEGORY_TO_CATEGORY."""

    def test_contains_25_entries(self) -> None:
        assert len(REQUIRED_SUBCATEGORIES) == 25

    def test_all_ids_are_strings(self) -> None:
        for sc_id in REQUIRED_SUBCATEGORIES:
            assert isinstance(sc_id, str)

    def test_mapping_covers_all_subcategories(self) -> None:
        assert set(REQUIRED_SUBCATEGORIES) == set(_SUBCATEGORY_TO_CATEGORY.keys())

    def test_mapping_points_to_valid_categories(self) -> None:
        valid_cat_ids = {c.id for c in DEFAULT_RUBRIC.categories}
        for sc_id, cat_id in _SUBCATEGORY_TO_CATEGORY.items():
            assert cat_id in valid_cat_ids, f"{sc_id} maps to unknown category {cat_id}"

    def test_tool_usage_subcategories(self) -> None:
        tool_subs = [k for k, v in _SUBCATEGORY_TO_CATEGORY.items() if v == "tool_usage"]
        assert set(tool_subs) == {
            "tool_selection",
            "tool_call_efficiency",
            "tool_error_recovery",
            "tool_output_utilization",
        }

    def test_reasoning_subcategories(self) -> None:
        reason_subs = [k for k, v in _SUBCATEGORY_TO_CATEGORY.items() if v == "reasoning"]
        assert set(reason_subs) == {
            "step_decomposition",
            "evidence_grounding",
            "error_diagnosis",
            "self_correction",
            "reasoning_coherence",
        }


# =============================================================================
# PromptStackOptimizerConfig tests
# =============================================================================


class TestPromptStackOptimizerConfig:
    """Test PromptStackOptimizerConfig model."""

    def test_default_values(self) -> None:
        cfg = PromptStackOptimizerConfig()
        assert cfg.pass_threshold == 0.6
        assert cfg.judge_model is None
        assert cfg.judge_provider is None
        assert cfg.judge_temperature == 0.0
        assert cfg.judge_max_tokens == 4096
        assert cfg.rubric.total_subcategories == 25

    def test_custom_pass_threshold(self) -> None:
        cfg = PromptStackOptimizerConfig(pass_threshold=0.8)
        assert cfg.pass_threshold == 0.8

    def test_pass_threshold_lower_bound(self) -> None:
        cfg = PromptStackOptimizerConfig(pass_threshold=0.0)
        assert cfg.pass_threshold == 0.0

    def test_pass_threshold_upper_bound(self) -> None:
        cfg = PromptStackOptimizerConfig(pass_threshold=1.0)
        assert cfg.pass_threshold == 1.0

    def test_pass_threshold_too_low(self) -> None:
        with pytest.raises(pd.ValidationError):
            PromptStackOptimizerConfig(pass_threshold=-0.1)

    def test_pass_threshold_too_high(self) -> None:
        with pytest.raises(pd.ValidationError):
            PromptStackOptimizerConfig(pass_threshold=1.1)

    def test_extra_forbid(self) -> None:
        with pytest.raises(pd.ValidationError):
            PromptStackOptimizerConfig(unknown_field="bad")

    def test_judge_temperature_bounds(self) -> None:
        cfg = PromptStackOptimizerConfig(judge_temperature=2.0)
        assert cfg.judge_temperature == 2.0
        with pytest.raises(pd.ValidationError):
            PromptStackOptimizerConfig(judge_temperature=2.1)

    def test_judge_max_tokens_ge_one(self) -> None:
        with pytest.raises(pd.ValidationError):
            PromptStackOptimizerConfig(judge_max_tokens=0)


# =============================================================================
# Grader name property
# =============================================================================


class TestGraderName:
    """Test PromptStackOptimizerGrader.name property."""

    def test_name(self, grader: PromptStackOptimizerGrader) -> None:
        assert grader.name == "prompt_stack_optimizer"


# =============================================================================
# Constructor tests
# =============================================================================


class TestGraderInit:
    """Test PromptStackOptimizerGrader __init__."""

    def test_default_config(self) -> None:
        g = PromptStackOptimizerGrader()
        assert g._config.pass_threshold == 0.6

    def test_dict_config(self) -> None:
        g = PromptStackOptimizerGrader(config={"pass_threshold": 0.9})
        assert g._config.pass_threshold == 0.9

    def test_model_config(self) -> None:
        cfg = PromptStackOptimizerConfig(pass_threshold=0.75)
        g = PromptStackOptimizerGrader(config=cfg)
        assert g._config.pass_threshold == 0.75

    def test_none_config(self) -> None:
        g = PromptStackOptimizerGrader(config=None)
        assert g._config.pass_threshold == 0.6


# =============================================================================
# _format_transcript_for_judge tests
# =============================================================================


class TestFormatTranscriptForJudge:
    """Test _format_transcript_for_judge method."""

    def test_basic_formatting(self, grader: PromptStackOptimizerGrader) -> None:
        transcript = EvalTranscript(
            messages=[{"role": "user", "content": "hello"}],
        )
        result = grader._format_transcript_for_judge(transcript)
        assert "[user]: hello" in result

    def test_last_10_messages_only(self, grader: PromptStackOptimizerGrader) -> None:
        messages = [{"role": "user", "content": f"msg-{i}"} for i in range(15)]
        transcript = EvalTranscript(messages=messages)
        result = grader._format_transcript_for_judge(transcript)
        # Messages 0-4 should NOT be present, messages 5-14 should be
        assert "msg-0" not in result
        assert "msg-4" not in result
        assert "msg-5" in result
        assert "msg-14" in result

    def test_last_8_tool_calls_only(self, grader: PromptStackOptimizerGrader) -> None:
        tool_calls = [{"name": f"tool-{i}", "output": f"output-{i}"} for i in range(12)]
        transcript = EvalTranscript(tool_calls=tool_calls)
        result = grader._format_transcript_for_judge(transcript)
        assert "tool-0" not in result
        assert "tool-3" not in result
        assert "tool-4" in result
        assert "tool-11" in result

    def test_truncation_over_8000_chars(self, grader: PromptStackOptimizerGrader) -> None:
        long_role = "r" * 300
        messages = [{"role": long_role, "content": "x" * 600} for _ in range(10)]
        long_name = "t" * 300
        tool_calls = [{"name": long_name, "output": "y" * 600} for _ in range(8)]
        transcript = EvalTranscript(messages=messages, tool_calls=tool_calls)
        result = grader._format_transcript_for_judge(transcript)
        assert "...[truncated]" in result

    def test_message_content_truncation_at_500(self, grader: PromptStackOptimizerGrader) -> None:
        long_msg = "a" * 600
        transcript = EvalTranscript(
            messages=[{"role": "user", "content": long_msg}],
        )
        result = grader._format_transcript_for_judge(transcript)
        # Content preview should be 500 chars + "..."
        assert "..." in result

    def test_tool_output_truncation_at_200(self, grader: PromptStackOptimizerGrader) -> None:
        long_output = "b" * 300
        transcript = EvalTranscript(
            tool_calls=[{"name": "tool1", "output": long_output}],
        )
        result = grader._format_transcript_for_judge(transcript)
        assert "..." in result

    def test_empty_transcript(self, grader: PromptStackOptimizerGrader) -> None:
        transcript = EvalTranscript()
        result = grader._format_transcript_for_judge(transcript)
        assert result == "No transcript context available."

    def test_non_string_content(self, grader: PromptStackOptimizerGrader) -> None:
        transcript = EvalTranscript(
            messages=[{"role": "user", "content": {"key": "value"}}],
        )
        result = grader._format_transcript_for_judge(transcript)
        assert "[user]:" in result

    def test_missing_role_key(self, grader: PromptStackOptimizerGrader) -> None:
        transcript = EvalTranscript(
            messages=[{"content": "no role"}],
        )
        result = grader._format_transcript_for_judge(transcript)
        assert "[unknown]:" in result

    def test_tool_call_name_from_tool_key(self, grader: PromptStackOptimizerGrader) -> None:
        transcript = EvalTranscript(
            tool_calls=[{"tool": "bash_tool", "output": "ok"}],
        )
        result = grader._format_transcript_for_judge(transcript)
        assert "[tool:bash_tool]" in result


# =============================================================================
# _extract_json_object tests
# =============================================================================


class TestExtractJsonObject:
    """Test _extract_json_object method."""

    def test_raw_json(self, grader: PromptStackOptimizerGrader) -> None:
        raw = '{"score": 0.8, "evidence": "good"}'
        result = grader._extract_json_object(raw)
        assert result == '{"score": 0.8, "evidence": "good"}'

    def test_markdown_json_code_block(self, grader: PromptStackOptimizerGrader) -> None:
        raw = 'Some text\n```json\n{"score": 0.5}\n```\nMore text'
        result = grader._extract_json_object(raw)
        assert result == '{"score": 0.5}'

    def test_plain_code_block(self, grader: PromptStackOptimizerGrader) -> None:
        raw = 'Some text\n```\n{"score": 0.3}\n```\nMore text'
        result = grader._extract_json_object(raw)
        assert result == '{"score": 0.3}'

    def test_embedded_json_in_text(self, grader: PromptStackOptimizerGrader) -> None:
        raw = 'Here is the result: {"score": 0.9} and more text'
        result = grader._extract_json_object(raw)
        assert result == '{"score": 0.9}'

    def test_empty_string(self, grader: PromptStackOptimizerGrader) -> None:
        assert grader._extract_json_object("") == ""

    def test_whitespace_only(self, grader: PromptStackOptimizerGrader) -> None:
        assert grader._extract_json_object("   ") == ""

    def test_no_json(self, grader: PromptStackOptimizerGrader) -> None:
        raw = "No JSON here at all"
        result = grader._extract_json_object(raw)
        assert result == raw

    def test_nested_braces(self, grader: PromptStackOptimizerGrader) -> None:
        raw = '{"outer": {"inner": 1}}'
        result = grader._extract_json_object(raw)
        assert result == '{"outer": {"inner": 1}}'


# =============================================================================
# _parse_category_scores tests
# =============================================================================


class TestParseCategoryScores:
    """Test _parse_category_scores method."""

    def test_valid_json_string(self, grader: PromptStackOptimizerGrader) -> None:
        raw = '{"tool_selection": {"score": 0.8, "evidence": "good", "confidence": 0.9}}'
        result = grader._parse_category_scores(raw, "tool_usage")
        assert isinstance(result, dict)
        assert "tool_selection" in result
        assert result["tool_selection"]["score"] == 0.8

    def test_invalid_json(self, grader: PromptStackOptimizerGrader) -> None:
        result = grader._parse_category_scores("not json {{{", "tool_usage")
        assert result == {}

    def test_non_string_input(self, grader: PromptStackOptimizerGrader) -> None:
        result = grader._parse_category_scores(12345, "tool_usage")
        assert result == {}

    def test_none_input(self, grader: PromptStackOptimizerGrader) -> None:
        result = grader._parse_category_scores(None, "tool_usage")
        assert result == {}

    def test_empty_string(self, grader: PromptStackOptimizerGrader) -> None:
        result = grader._parse_category_scores("", "tool_usage")
        assert result == {}

    def test_json_array_returns_empty(self, grader: PromptStackOptimizerGrader) -> None:
        result = grader._parse_category_scores("[1, 2, 3]", "tool_usage")
        assert result == {}

    def test_json_in_code_block(self, grader: PromptStackOptimizerGrader) -> None:
        raw = '```json\n{"key": "value"}\n```'
        result = grader._parse_category_scores(raw, "cat1")
        assert result == {"key": "value"}


# =============================================================================
# _compute_category_scores tests
# =============================================================================


class TestComputeCategoryScores:
    """Test _compute_category_scores method."""

    def test_uniform_scores(self, grader: PromptStackOptimizerGrader) -> None:
        sub_results = _make_subcategory_results(score=0.8)
        cat_scores = grader._compute_category_scores(sub_results)
        assert len(cat_scores) == 6
        for cat in cat_scores:
            assert 0.0 <= cat["score"] <= 1.0
            assert "category_id" in cat
            assert "category_name" in cat
            assert "weight" in cat
            assert "subcategory_scores" in cat

    def test_score_bounds(self, grader: PromptStackOptimizerGrader) -> None:
        sub_results = _make_subcategory_results(score=1.0)
        cat_scores = grader._compute_category_scores(sub_results)
        for cat in cat_scores:
            assert cat["score"] <= 1.0

    def test_zero_scores(self, grader: PromptStackOptimizerGrader) -> None:
        sub_results = _make_subcategory_results(score=0.0)
        cat_scores = grader._compute_category_scores(sub_results)
        for cat in cat_scores:
            assert cat["score"] == 0.0

    def test_weighted_averaging(self) -> None:
        """Verify weighted average within a category."""
        # Build a grader with a custom rubric that has known weights
        sub1 = SubcategoryDef(id="s1", name="S1", description="d1", weight=2.0)
        sub2 = SubcategoryDef(id="s2", name="S2", description="d2", weight=1.0)
        cat = CategoryDef(
            id="cat1", name="Cat1", description="d", weight=1.0, subcategories=[sub1, sub2]
        )
        rubric = RubricDef(categories=[cat])
        cfg = PromptStackOptimizerConfig(rubric=rubric)
        g = PromptStackOptimizerGrader(config=cfg)

        sub_results = {
            "s1": {"subcategory_id": "s1", "score": 1.0, "confidence": 0.9, "evidence": ["e"]},
            "s2": {"subcategory_id": "s2", "score": 0.0, "confidence": 0.9, "evidence": ["e"]},
        }
        cat_scores = g._compute_category_scores(sub_results)
        # Weighted: (1.0 * 2.0 + 0.0 * 1.0) / (2.0 + 1.0) = 2/3
        expected = round(2.0 / 3.0, 4)
        assert cat_scores[0]["score"] == expected

    def test_missing_subcategory_results(self, grader: PromptStackOptimizerGrader) -> None:
        """When subcategory_results is empty, category scores should be 0.0."""
        cat_scores = grader._compute_category_scores({})
        for cat in cat_scores:
            assert cat["score"] == 0.0


# =============================================================================
# _compute_overall_score tests
# =============================================================================


class TestComputeOverallScore:
    """Test _compute_overall_score method."""

    def test_weighted_average(self, grader: PromptStackOptimizerGrader) -> None:
        cat_scores = [
            {"score": 1.0, "weight": 0.5},
            {"score": 0.0, "weight": 0.5},
        ]
        result = grader._compute_overall_score(cat_scores)
        assert result == 0.5

    def test_empty_list_returns_zero(self, grader: PromptStackOptimizerGrader) -> None:
        assert grader._compute_overall_score([]) == 0.0

    def test_single_category(self, grader: PromptStackOptimizerGrader) -> None:
        cat_scores = [{"score": 0.75, "weight": 1.0}]
        assert grader._compute_overall_score(cat_scores) == 0.75

    def test_unequal_weights(self, grader: PromptStackOptimizerGrader) -> None:
        cat_scores = [
            {"score": 1.0, "weight": 3.0},
            {"score": 0.0, "weight": 1.0},
        ]
        result = grader._compute_overall_score(cat_scores)
        assert result == 0.75

    def test_result_is_rounded(self, grader: PromptStackOptimizerGrader) -> None:
        cat_scores = [
            {"score": 1.0, "weight": 1.0},
            {"score": 0.0, "weight": 2.0},
        ]
        result = grader._compute_overall_score(cat_scores)
        # 1/3 rounded to 4 decimal places
        assert result == round(1.0 / 3.0, 4)


# =============================================================================
# grade() method tests (mocking _score_all_subcategories)
# =============================================================================


class TestGradeMethod:
    """Test grade() method with mocked LLM scoring."""

    @pytest.mark.asyncio
    async def test_grade_returns_grader_result(
        self,
        grader: PromptStackOptimizerGrader,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> None:
        mock_results = _make_subcategory_results(score=0.8, confidence=0.9)
        with patch.object(
            grader, "_score_all_subcategories", new_callable=AsyncMock, return_value=mock_results
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.grader_type == "prompt_stack_optimizer"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)
        assert result.execution_time_seconds is not None
        assert result.execution_time_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_grade_passes_above_threshold(
        self,
        grader: PromptStackOptimizerGrader,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> None:
        mock_results = _make_subcategory_results(score=0.9)
        with patch.object(
            grader, "_score_all_subcategories", new_callable=AsyncMock, return_value=mock_results
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score > 0.6

    @pytest.mark.asyncio
    async def test_grade_fails_below_threshold(
        self,
        grader: PromptStackOptimizerGrader,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> None:
        mock_results = _make_subcategory_results(score=0.1)
        with patch.object(
            grader, "_score_all_subcategories", new_callable=AsyncMock, return_value=mock_results
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score < 0.6

    @pytest.mark.asyncio
    async def test_grade_details_structure(
        self,
        grader: PromptStackOptimizerGrader,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> None:
        mock_results = _make_subcategory_results(score=0.7)
        with patch.object(
            grader, "_score_all_subcategories", new_callable=AsyncMock, return_value=mock_results
        ):
            result = await grader.grade(trial, transcript, spec)

        assert "category_summary" in result.details
        assert "subcategory_results" in result.details
        assert "pass_threshold" in result.details
        assert result.details["pass_threshold"] == 0.6

    @pytest.mark.asyncio
    async def test_grade_category_summary_has_all_categories(
        self,
        grader: PromptStackOptimizerGrader,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> None:
        mock_results = _make_subcategory_results(score=0.7)
        with patch.object(
            grader, "_score_all_subcategories", new_callable=AsyncMock, return_value=mock_results
        ):
            result = await grader.grade(trial, transcript, spec)

        summary = result.details["category_summary"]
        expected_ids = {"tool_usage", "reasoning", "context", "completion", "efficiency", "safety"}
        assert set(summary.keys()) == expected_ids

    @pytest.mark.asyncio
    async def test_grade_confidence_is_average(
        self,
        grader: PromptStackOptimizerGrader,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> None:
        mock_results = _make_subcategory_results(score=0.7, confidence=0.85)
        with patch.object(
            grader, "_score_all_subcategories", new_callable=AsyncMock, return_value=mock_results
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_grade_with_spec_config_override(
        self,
        grader: PromptStackOptimizerGrader,
        trial: EvalTrial,
        transcript: EvalTranscript,
    ) -> None:
        spec = GraderSpec(
            grader_type="prompt_stack_optimizer",
            config={"pass_threshold": 0.95},
        )
        mock_results = _make_subcategory_results(score=0.9)
        with patch.object(
            grader, "_score_all_subcategories", new_callable=AsyncMock, return_value=mock_results
        ):
            result = await grader.grade(trial, transcript, spec)

        # Score ~0.9 is below threshold 0.95
        assert result.passed is False
        assert result.details["pass_threshold"] == 0.95

    @pytest.mark.asyncio
    async def test_grade_with_invalid_spec_config_uses_defaults(
        self,
        grader: PromptStackOptimizerGrader,
        trial: EvalTrial,
        transcript: EvalTranscript,
    ) -> None:
        spec = GraderSpec(
            grader_type="prompt_stack_optimizer",
            config={"pass_threshold": 5.0},  # out of bounds
        )
        mock_results = _make_subcategory_results(score=0.8)
        with patch.object(
            grader, "_score_all_subcategories", new_callable=AsyncMock, return_value=mock_results
        ):
            result = await grader.grade(trial, transcript, spec)

        # Should fall back to defaults since 5.0 > 1.0 is invalid
        assert result.details["pass_threshold"] == 0.6


# =============================================================================
# grade() error handling tests
# =============================================================================


class TestGradeErrorHandling:
    """Test grade() error handling paths."""

    @pytest.mark.asyncio
    async def test_import_error(
        self,
        grader: PromptStackOptimizerGrader,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> None:
        with patch.object(
            grader,
            "_score_all_subcategories",
            new_callable=AsyncMock,
            side_effect=ImportError("No module named 'dawn_kestrel'"),
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.score == 0.0
        assert result.passed is False
        assert result.error_message is not None
        assert "Missing dependency" in result.error_message
        assert result.details["failure_mode"] == FailureMode.JUDGE_ERROR.value

    @pytest.mark.asyncio
    async def test_value_error(
        self,
        grader: PromptStackOptimizerGrader,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> None:
        with patch.object(
            grader,
            "_score_all_subcategories",
            new_callable=AsyncMock,
            side_effect=ValueError("Invalid rubric configuration"),
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.score == 0.0
        assert result.passed is False
        assert result.error_message == "Invalid rubric configuration"
        assert result.details["failure_mode"] == FailureMode.JUDGE_ERROR.value

    @pytest.mark.asyncio
    async def test_generic_exception(
        self,
        grader: PromptStackOptimizerGrader,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> None:
        with patch.object(
            grader,
            "_score_all_subcategories",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.score == 0.0
        assert result.passed is False
        assert result.error_message == "Unexpected error"
        assert result.details["failure_mode"] == FailureMode.JUDGE_ERROR.value

    @pytest.mark.asyncio
    async def test_error_includes_execution_time(
        self,
        grader: PromptStackOptimizerGrader,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> None:
        with patch.object(
            grader,
            "_score_all_subcategories",
            new_callable=AsyncMock,
            side_effect=RuntimeError("fail"),
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.execution_time_seconds is not None
        assert result.execution_time_seconds >= 0.0
