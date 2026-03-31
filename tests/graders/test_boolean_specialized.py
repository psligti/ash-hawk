"""Tests for BooleanJudgeGrader and create_boolean_graders."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ash_hawk.graders.llm_boolean_specialized import (
    _DIMENSIONS,
    BooleanJudgeGrader,
    create_boolean_graders,
)
from ash_hawk.types import EvalTranscript, EvalTrial, GraderSpec


@pytest.fixture
def trial() -> EvalTrial:
    return EvalTrial(id="t1", task_id="task1")


@pytest.fixture
def transcript() -> EvalTranscript:
    return EvalTranscript(
        agent_response="This is a well-written article about Python.",
        messages=[{"role": "user", "content": "Write an article about Python."}],
    )


@pytest.fixture
def empty_transcript() -> EvalTranscript:
    return EvalTranscript()


@pytest.fixture
def spec_aislop() -> GraderSpec:
    return GraderSpec(grader_type="boolean_judge_aislop")


def _make_mock_client(response_text: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.text = response_text
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=mock_response)
    return mock_client


class TestNameProperty:
    @pytest.mark.parametrize(
        "dimension",
        ["aislop", "voice", "soul", "reply", "engagement", "technical", "safety"],
    )
    def test_name_format(self, dimension: str) -> None:
        grader = BooleanJudgeGrader(dimension=dimension)  # type: ignore[arg-type]
        assert grader.name == f"boolean_judge_{dimension}"


class TestCreateBooleanGraders:
    def test_returns_seven_graders(self) -> None:
        graders = create_boolean_graders()
        assert len(graders) == 7

    def test_all_are_boolean_judge_grader_instances(self) -> None:
        graders = create_boolean_graders()
        for g in graders:
            assert isinstance(g, BooleanJudgeGrader)

    def test_correct_name_format(self) -> None:
        graders = create_boolean_graders()
        for g in graders:
            assert g.name.startswith("boolean_judge_")

    def test_all_dimensions_covered(self) -> None:
        graders = create_boolean_graders()
        names = {g.name for g in graders}
        expected = {
            "boolean_judge_aislop",
            "boolean_judge_voice",
            "boolean_judge_soul",
            "boolean_judge_reply",
            "boolean_judge_engagement",
            "boolean_judge_technical",
            "boolean_judge_safety",
        }
        assert names == expected


class TestDimensionsConfig:
    EXPECTED_DIMENSIONS = ["aislop", "voice", "soul", "reply", "engagement", "technical", "safety"]

    def test_all_seven_dimensions_present(self) -> None:
        assert set(_DIMENSIONS.keys()) == set(self.EXPECTED_DIMENSIONS)

    @pytest.mark.parametrize(
        "dimension",
        ["aislop", "voice", "soul", "reply", "engagement", "technical", "safety"],
    )
    def test_has_questions_key(self, dimension: str) -> None:
        assert "questions" in _DIMENSIONS[dimension]
        assert isinstance(_DIMENSIONS[dimension]["questions"], list)

    @pytest.mark.parametrize(
        "dimension",
        ["aislop", "voice", "soul", "reply", "engagement", "technical", "safety"],
    )
    def test_has_require_all_key(self, dimension: str) -> None:
        assert "require_all" in _DIMENSIONS[dimension]
        assert isinstance(_DIMENSIONS[dimension]["require_all"], bool)

    @pytest.mark.parametrize(
        "dimension",
        ["aislop", "voice", "soul", "reply", "engagement", "technical", "safety"],
    )
    def test_has_display_name_key(self, dimension: str) -> None:
        assert "display_name" in _DIMENSIONS[dimension]
        assert isinstance(_DIMENSIONS[dimension]["display_name"], str)

    @pytest.mark.parametrize(
        "dimension",
        ["aislop", "voice", "soul", "reply", "engagement", "technical", "safety"],
    )
    def test_questions_are_non_empty_strings(self, dimension: str) -> None:
        questions = _DIMENSIONS[dimension]["questions"]
        assert len(questions) == 15
        for q in questions:
            assert isinstance(q, str)
            assert len(q) > 0

    def test_require_all_values(self) -> None:
        assert _DIMENSIONS["voice"]["require_all"] is True
        assert _DIMENSIONS["soul"]["require_all"] is True
        assert _DIMENSIONS["technical"]["require_all"] is True
        assert _DIMENSIONS["aislop"]["require_all"] is False
        assert _DIMENSIONS["reply"]["require_all"] is False
        assert _DIMENSIONS["engagement"]["require_all"] is False
        assert _DIMENSIONS["safety"]["require_all"] is False


class TestExtractContent:
    def test_extracts_agent_response_string(self, trial: EvalTrial) -> None:
        grader = BooleanJudgeGrader(dimension="aislop")
        transcript = EvalTranscript(agent_response="Hello world")
        result = grader._extract_content(transcript, trial)
        assert "Hello world" in result

    def test_extracts_agent_response_dict_content(self, trial: EvalTrial) -> None:
        grader = BooleanJudgeGrader(dimension="aislop")
        transcript = EvalTranscript(agent_response={"content": "Dict content here"})
        result = grader._extract_content(transcript, trial)
        assert "Dict content here" in result

    def test_extracts_messages(self, trial: EvalTrial) -> None:
        grader = BooleanJudgeGrader(dimension="aislop")
        transcript = EvalTranscript(
            messages=[
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a language."},
            ],
        )
        result = grader._extract_content(transcript, trial)
        assert "[user]: What is Python?" in result
        assert "[assistant]: Python is a language." in result

    def test_empty_transcript_returns_empty(self, trial: EvalTrial) -> None:
        grader = BooleanJudgeGrader(dimension="aislop")
        transcript = EvalTranscript()
        result = grader._extract_content(transcript, trial)
        assert result.strip() == ""

    def test_truncation_at_max_chars(self, trial: EvalTrial) -> None:
        grader = BooleanJudgeGrader(dimension="aislop", config={"context_max_chars": 50})
        long_text = "A" * 200
        transcript = EvalTranscript(agent_response=long_text)
        result = grader._extract_content(transcript, trial)
        assert len(result) < 200
        assert result.endswith("...[truncated]")

    def test_combines_response_and_messages(self, trial: EvalTrial) -> None:
        grader = BooleanJudgeGrader(dimension="aislop")
        transcript = EvalTranscript(
            agent_response="Final answer",
            messages=[{"role": "user", "content": "Question"}],
        )
        result = grader._extract_content(transcript, trial)
        assert "Final answer" in result
        assert "[user]: Question" in result

    def test_skips_empty_message_content(self, trial: EvalTrial) -> None:
        grader = BooleanJudgeGrader(dimension="aislop")
        transcript = EvalTranscript(
            messages=[
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "Real content"},
            ],
        )
        result = grader._extract_content(transcript, trial)
        assert "[user]:" not in result
        assert "[assistant]: Real content" in result


class TestParseBooleanResponse:
    def _parse(self, response: str, num: int) -> list[bool]:
        grader = BooleanJudgeGrader(dimension="aislop")
        return grader._parse_boolean_response(response, num)

    def test_true_false_true(self) -> None:
        result = self._parse("true\nfalse\ntrue", 3)
        assert result == [True, False, True]

    def test_yes_no_1_0(self) -> None:
        result = self._parse("yes\nno\n1\n0", 4)
        assert result == [True, False, True, False]

    def test_empty_lines_ignored(self) -> None:
        result = self._parse("true\n\n\nfalse\n\ntrue", 3)
        assert result == [True, False, True]

    def test_shorter_response_padded_with_false(self) -> None:
        result = self._parse("true", 5)
        assert result == [True, False, False, False, False]

    def test_longer_response_truncated(self) -> None:
        result = self._parse("true\ntrue\ntrue\ntrue\ntrue", 3)
        assert result == [True, True, True]

    def test_case_insensitive(self) -> None:
        result = self._parse("TRUE\nFALSE\nTrue\nFalse", 4)
        assert result == [True, False, True, False]

    def test_y_and_n_shorthand(self) -> None:
        result = self._parse("y\nn", 2)
        assert result == [True, False]

    def test_all_false_on_empty(self) -> None:
        result = self._parse("", 3)
        assert result == [False, False, False]

    def test_true_prefix_match(self) -> None:
        result = self._parse("true - because X\nfalse - because Y", 2)
        assert result == [True, False]


class TestGradeRequireAll:
    @pytest.mark.asyncio
    async def test_all_true_passes(self, trial: EvalTrial, transcript: EvalTranscript) -> None:
        all_true = "\n".join(["true"] * 15)
        mock_client = _make_mock_client(all_true)
        grader = BooleanJudgeGrader(dimension="voice", client=mock_client)
        spec = GraderSpec(grader_type="boolean_judge_voice")

        with patch(
            "ash_hawk.graders.llm_boolean_specialized.LLMRequestOptions",
            create=True,
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["require_all"] is True
        assert result.details["true_count"] == 15

    @pytest.mark.asyncio
    async def test_one_false_fails(self, trial: EvalTrial, transcript: EvalTranscript) -> None:
        response = "\n".join(["true"] * 14 + ["false"])
        mock_client = _make_mock_client(response)
        grader = BooleanJudgeGrader(dimension="soul", client=mock_client)
        spec = GraderSpec(grader_type="boolean_judge_soul")

        with patch(
            "ash_hawk.graders.llm_boolean_specialized.LLMRequestOptions",
            create=True,
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == pytest.approx(14 / 15)
        assert result.details["true_count"] == 14

    @pytest.mark.asyncio
    async def test_all_false_fails(self, trial: EvalTrial, transcript: EvalTranscript) -> None:
        all_false = "\n".join(["false"] * 15)
        mock_client = _make_mock_client(all_false)
        grader = BooleanJudgeGrader(dimension="technical", client=mock_client)
        spec = GraderSpec(grader_type="boolean_judge_technical")

        with patch(
            "ash_hawk.graders.llm_boolean_specialized.LLMRequestOptions",
            create=True,
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0


class TestGradeRequireAny:
    @pytest.mark.asyncio
    async def test_any_true_passes(self, trial: EvalTrial, transcript: EvalTranscript) -> None:
        response = "\n".join(["false"] * 14 + ["true"])
        mock_client = _make_mock_client(response)
        grader = BooleanJudgeGrader(dimension="aislop", client=mock_client)
        spec = GraderSpec(grader_type="boolean_judge_aislop")

        with patch(
            "ash_hawk.graders.llm_boolean_specialized.LLMRequestOptions",
            create=True,
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_all_false_fails(self, trial: EvalTrial, transcript: EvalTranscript) -> None:
        all_false = "\n".join(["false"] * 15)
        mock_client = _make_mock_client(all_false)
        grader = BooleanJudgeGrader(dimension="reply", client=mock_client)
        spec = GraderSpec(grader_type="boolean_judge_reply")

        with patch(
            "ash_hawk.graders.llm_boolean_specialized.LLMRequestOptions",
            create=True,
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_all_true_passes(self, trial: EvalTrial, transcript: EvalTranscript) -> None:
        all_true = "\n".join(["true"] * 15)
        mock_client = _make_mock_client(all_true)
        grader = BooleanJudgeGrader(dimension="engagement", client=mock_client)
        spec = GraderSpec(grader_type="boolean_judge_engagement")

        with patch(
            "ash_hawk.graders.llm_boolean_specialized.LLMRequestOptions",
            create=True,
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_safety_any_true_passes(
        self, trial: EvalTrial, transcript: EvalTranscript
    ) -> None:
        response = "true\n" + "\n".join(["false"] * 14)
        mock_client = _make_mock_client(response)
        grader = BooleanJudgeGrader(dimension="safety", client=mock_client)
        spec = GraderSpec(grader_type="boolean_judge_safety")

        with patch(
            "ash_hawk.graders.llm_boolean_specialized.LLMRequestOptions",
            create=True,
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0


class TestGradeEmptyContent:
    @pytest.mark.asyncio
    async def test_empty_transcript_returns_error(
        self, trial: EvalTrial, empty_transcript: EvalTranscript
    ) -> None:
        mock_client = _make_mock_client("true")
        grader = BooleanJudgeGrader(dimension="aislop", client=mock_client)
        spec = GraderSpec(grader_type="boolean_judge_aislop")

        result = await grader.grade(trial, empty_transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        assert result.error_message is not None
        assert "No content found" in result.error_message


class TestGradeUnknownDimension:
    @pytest.mark.asyncio
    async def test_unknown_dimension_returns_error(self, trial: EvalTrial) -> None:
        grader = BooleanJudgeGrader.__new__(BooleanJudgeGrader)
        grader._dimension = "nonexistent"
        grader._config = {}
        grader._client = None

        transcript = EvalTranscript(agent_response="Some content")
        spec = GraderSpec(grader_type="boolean_judge_nonexistent")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        assert result.error_message is not None
        assert "Unknown dimension" in result.error_message
        assert "nonexistent" in result.error_message


class TestGradeImportError:
    @pytest.mark.asyncio
    async def test_import_error_returns_error_result(
        self, trial: EvalTrial, transcript: EvalTranscript
    ) -> None:
        grader = BooleanJudgeGrader(dimension="aislop")
        spec = GraderSpec(grader_type="boolean_judge_aislop")

        with patch.object(
            grader,
            "_get_client",
            side_effect=ImportError("No module named 'dawn_kestrel'"),
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        assert result.error_message is not None
        assert "dawn-kestrel not installed" in result.error_message


class TestGradeEmptyLLMResponse:
    @pytest.mark.asyncio
    async def test_empty_response_text(self, trial: EvalTrial, transcript: EvalTranscript) -> None:
        mock_response = MagicMock()
        mock_response.text = ""
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        grader = BooleanJudgeGrader(dimension="aislop", client=mock_client)
        spec = GraderSpec(grader_type="boolean_judge_aislop")

        with patch(
            "ash_hawk.graders.llm_boolean_specialized.LLMRequestOptions",
            create=True,
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.error_message is not None
        assert "Empty response" in result.error_message

    @pytest.mark.asyncio
    async def test_none_response(self, trial: EvalTrial, transcript: EvalTranscript) -> None:
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=None)

        grader = BooleanJudgeGrader(dimension="aislop", client=mock_client)
        spec = GraderSpec(grader_type="boolean_judge_aislop")

        with patch(
            "ash_hawk.graders.llm_boolean_specialized.LLMRequestOptions",
            create=True,
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.error_message is not None
        assert "Empty response" in result.error_message


class TestCustomQuestionsViaSpec:
    @pytest.mark.asyncio
    async def test_custom_questions_used(
        self, trial: EvalTrial, transcript: EvalTranscript
    ) -> None:
        mock_client = _make_mock_client("true")
        grader = BooleanJudgeGrader(dimension="aislop", client=mock_client)
        spec = GraderSpec(
            grader_type="boolean_judge_aislop",
            config={"questions": ["Custom question?"], "require_all": True},
        )

        with patch(
            "ash_hawk.graders.llm_boolean_specialized.LLMRequestOptions",
            create=True,
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["questions"] == ["Custom question?"]
        assert result.details["total_count"] == 1

    @pytest.mark.asyncio
    async def test_custom_require_all_override(
        self, trial: EvalTrial, transcript: EvalTranscript
    ) -> None:
        mock_client = _make_mock_client("true\nfalse")
        grader = BooleanJudgeGrader(dimension="aislop", client=mock_client)
        spec = GraderSpec(
            grader_type="boolean_judge_aislop",
            config={
                "questions": ["Q1?", "Q2?"],
                "require_all": True,
            },
        )

        with patch(
            "ash_hawk.graders.llm_boolean_specialized.LLMRequestOptions",
            create=True,
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.details["require_all"] is True


class TestGradeDetails:
    @pytest.mark.asyncio
    async def test_details_structure(self, trial: EvalTrial, transcript: EvalTranscript) -> None:
        response_text = "\n".join(["true", "false"] * 7 + ["true"])
        mock_client = _make_mock_client(response_text)
        grader = BooleanJudgeGrader(dimension="aislop", client=mock_client)
        spec = GraderSpec(grader_type="boolean_judge_aislop")

        with patch(
            "ash_hawk.graders.llm_boolean_specialized.LLMRequestOptions",
            create=True,
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.details["dimension"] == "aislop"
        assert "questions" in result.details
        assert "answers" in result.details
        assert "true_count" in result.details
        assert "total_count" in result.details
        assert "require_all" in result.details
        assert "raw_response" in result.details
        assert result.details["total_count"] == 15
        assert result.details["true_count"] == 8
        assert len(result.details["answers"]) == 15


class TestGradeGenericException:
    @pytest.mark.asyncio
    async def test_generic_exception_returns_error_result(
        self, trial: EvalTrial, transcript: EvalTranscript
    ) -> None:
        grader = BooleanJudgeGrader(dimension="aislop")
        spec = GraderSpec(grader_type="boolean_judge_aislop")

        with patch.object(
            grader,
            "_get_client",
            side_effect=RuntimeError("Unexpected failure"),
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        assert result.error_message is not None
        assert "Unexpected failure" in result.error_message
