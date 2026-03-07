"""Tests for rubric_guard module."""

from __future__ import annotations

import pytest

from ash_hawk.graders.rubric_guard import (
    _is_rubric_based_llm_judge,
    enforce_rubric_based_evaluation,
    task_has_rubric_based_evaluation,
    validate_suite_has_rubric_based_evaluation,
    validate_task_has_rubric,
)


class TestIsRubricBasedLlmJudge:
    """Tests for _is_rubric_based_llm_judge function."""

    def test_rubric_in_config(self) -> None:
        grader = {"grader_type": "llm_judge", "config": {"rubric": "security_review"}}
        assert _is_rubric_based_llm_judge(grader) is True

    def test_custom_prompt_in_config(self) -> None:
        grader = {
            "grader_type": "llm_judge",
            "config": {"custom_prompt": "Evaluate this..."},
        }
        assert _is_rubric_based_llm_judge(grader) is True

    def test_custom_prompt_path_in_config(self) -> None:
        grader = {
            "grader_type": "llm_judge",
            "config": {"custom_prompt_path": "/path/to/prompt.md"},
        }
        assert _is_rubric_based_llm_judge(grader) is True

    def test_non_llm_judge(self) -> None:
        grader = {"grader_type": "string_match", "config": {"rubric": "test"}}
        assert _is_rubric_based_llm_judge(grader) is False

    def test_llm_judge_without_rubric(self) -> None:
        grader = {"grader_type": "llm_judge", "config": {}}
        assert _is_rubric_based_llm_judge(grader) is False

    def test_empty_rubric(self) -> None:
        grader = {"grader_type": "llm_judge", "config": {"rubric": ""}}
        assert _is_rubric_based_llm_judge(grader) is False

    def test_whitespace_only_rubric(self) -> None:
        grader = {"grader_type": "llm_judge", "config": {"rubric": "   "}}
        assert _is_rubric_based_llm_judge(grader) is False

    def test_object_with_attrs(self) -> None:
        class Grader:
            grader_type = "llm_judge"
            config = {"rubric": "security_review"}

        assert _is_rubric_based_llm_judge(Grader()) is True


class TestTaskHasRubricBasedEvaluation:
    """Tests for task_has_rubric_based_evaluation function."""

    def test_has_rubric_grader(self) -> None:
        task = {"grader_specs": [{"grader_type": "llm_judge", "config": {"rubric": "security"}}]}
        assert task_has_rubric_based_evaluation(task) is True

    def test_no_rubric_grader(self) -> None:
        task = {
            "grader_specs": [{"grader_type": "string_match", "config": {"contains": ["error"]}}]
        }
        assert task_has_rubric_based_evaluation(task) is False

    def test_empty_grader_specs(self) -> None:
        task = {"grader_specs": []}
        assert task_has_rubric_based_evaluation(task) is False

    def test_missing_grader_specs(self) -> None:
        task: dict = {}
        assert task_has_rubric_based_evaluation(task) is False

    def test_object_with_grader_specs(self) -> None:
        class Task:
            grader_specs = [{"grader_type": "llm_judge", "config": {"rubric": "quality"}}]

        assert task_has_rubric_based_evaluation(Task()) is True


class TestValidateTaskHasRubric:
    """Tests for validate_task_has_rubric function."""

    def test_valid_task_passes(self) -> None:
        task = {
            "id": "test-001",
            "grader_specs": [{"grader_type": "llm_judge", "config": {"rubric": "security"}}],
        }
        validate_task_has_rubric(task)  # Should not raise

    def test_invalid_task_raises(self) -> None:
        task = {
            "id": "test-001",
            "grader_specs": [{"grader_type": "string_match", "config": {}}],
        }
        with pytest.raises(ValueError, match="test-001"):
            validate_task_has_rubric(task)

    def test_custom_task_id(self) -> None:
        task = {"grader_specs": []}
        with pytest.raises(ValueError, match="custom-id"):
            validate_task_has_rubric(task, task_id="custom-id")


class TestValidateSuiteHasRubricBasedEvaluation:
    """Tests for validate_suite_has_rubric_based_evaluation function."""

    def test_all_tasks_have_rubric(self) -> None:
        suite = {
            "id": "test-suite",
            "tasks": [
                {
                    "id": "task-1",
                    "grader_specs": [{"grader_type": "llm_judge", "config": {"rubric": "r1"}}],
                },
                {
                    "id": "task-2",
                    "grader_specs": [{"grader_type": "llm_judge", "config": {"rubric": "r2"}}],
                },
            ],
        }
        missing = validate_suite_has_rubric_based_evaluation(suite)
        assert missing == []

    def test_some_tasks_missing_rubric(self) -> None:
        suite = {
            "id": "test-suite",
            "tasks": [
                {
                    "id": "task-1",
                    "grader_specs": [{"grader_type": "llm_judge", "config": {"rubric": "r1"}}],
                },
                {
                    "id": "task-2",
                    "grader_specs": [{"grader_type": "string_match", "config": {}}],
                },
            ],
        }
        missing = validate_suite_has_rubric_based_evaluation(suite)
        assert missing == ["task-2"]

    def test_empty_tasks_raises(self) -> None:
        suite = {"id": "test-suite", "tasks": []}
        with pytest.raises(ValueError, match="has no tasks"):
            validate_suite_has_rubric_based_evaluation(suite)

    def test_missing_tasks_raises(self) -> None:
        suite = {"id": "test-suite"}
        with pytest.raises(ValueError, match="has no tasks"):
            validate_suite_has_rubric_based_evaluation(suite)


class TestEnforceRubricBasedEvaluation:
    """Tests for enforce_rubric_based_evaluation function."""

    def test_valid_suite_passes(self) -> None:
        suite = {
            "id": "test-suite",
            "tasks": [
                {
                    "id": "task-1",
                    "grader_specs": [{"grader_type": "llm_judge", "config": {"rubric": "r1"}}],
                }
            ],
        }
        enforce_rubric_based_evaluation(suite)  # Should not raise

    def test_invalid_suite_raises(self) -> None:
        suite = {
            "id": "test-suite",
            "tasks": [
                {
                    "id": "task-1",
                    "grader_specs": [{"grader_type": "string_match", "config": {}}],
                }
            ],
        }
        with pytest.raises(ValueError, match="task-1"):
            enforce_rubric_based_evaluation(suite)
