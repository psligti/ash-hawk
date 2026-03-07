"""Rubric guard validator for enforcing rubric-based evaluation.

This module provides utilities to validate that evaluation tasks
have proper rubric-based LLM judge graders configured.

Extracted from iron-rook eval infrastructure.
"""

from __future__ import annotations

from typing import Any


def _is_rubric_based_llm_judge(grader_spec: Any) -> bool:
    """Check if a grader spec is a rubric-based LLM judge.

    A rubric-based LLM judge must have:
    - grader_type == "llm_judge"
    - One of: rubric, custom_prompt, or custom_prompt_path configured

    Args:
        grader_spec: Grader specification object or dict

    Returns:
        True if this is a rubric-based LLM judge
    """
    # Handle both object and dict
    if isinstance(grader_spec, dict):
        grader_type = grader_spec.get("grader_type")
        config = grader_spec.get("config", {})
    else:
        grader_type = getattr(grader_spec, "grader_type", None)
        config = getattr(grader_spec, "config", None)

    if grader_type != "llm_judge":
        return False

    if not isinstance(config, dict):
        return False

    rubric = config.get("rubric")
    custom_prompt = config.get("custom_prompt")
    custom_prompt_path = config.get("custom_prompt_path")

    if isinstance(rubric, str) and rubric.strip():
        return True
    if isinstance(custom_prompt, str) and custom_prompt.strip():
        return True
    if isinstance(custom_prompt_path, str) and custom_prompt_path.strip():
        return True

    return False


def task_has_rubric_based_evaluation(task: Any) -> bool:
    """Check if a task has at least one rubric-based LLM judge grader.

    Args:
        task: EvalTask object or dict with grader_specs

    Returns:
        True if the task has rubric-based evaluation
    """
    if isinstance(task, dict):
        grader_specs = task.get("grader_specs", [])
    else:
        grader_specs = getattr(task, "grader_specs", None)

    if not isinstance(grader_specs, list):
        return False

    return any(_is_rubric_based_llm_judge(spec) for spec in grader_specs)


def validate_task_has_rubric(task: Any, task_id: str | None = None) -> None:
    """Validate that a task has rubric-based evaluation.

    Args:
        task: EvalTask object or dict
        task_id: Optional task ID for error messages

    Raises:
        ValueError: If the task lacks rubric-based evaluation
    """
    if task_id is None:
        if isinstance(task, dict):
            task_id = task.get("id", "unknown-task")
        else:
            task_id = getattr(task, "id", "unknown-task")

    if not task_has_rubric_based_evaluation(task):
        raise ValueError(
            f"Task '{task_id}' lacks rubric-based LLM judge evaluation. "
            "Every eval task should have at least one grader with grader_type='llm_judge' "
            "and a configured rubric, custom_prompt, or custom_prompt_path."
        )


def validate_suite_has_rubric_based_evaluation(suite: Any) -> list[str]:
    """Validate that all tasks in a suite have rubric-based evaluation.

    Args:
        suite: EvalSuite object or dict with tasks

    Returns:
        List of task IDs that lack rubric-based evaluation (empty if all valid)

    Raises:
        ValueError: If the suite has no tasks
    """
    if isinstance(suite, dict):
        suite_id = suite.get("id", "unknown-suite")
        tasks = suite.get("tasks", [])
    else:
        suite_id = getattr(suite, "id", "unknown-suite")
        tasks = getattr(suite, "tasks", None)

    if not isinstance(tasks, list) or not tasks:
        raise ValueError(
            f"Eval suite '{suite_id}' has no tasks; cannot enforce rubric-based evaluation."
        )

    missing: list[str] = []
    for task in tasks:
        if isinstance(task, dict):
            task_id = task.get("id", "unknown-task")
        else:
            task_id = getattr(task, "id", "unknown-task")

        if not task_has_rubric_based_evaluation(task):
            missing.append(task_id)

    return missing


def enforce_rubric_based_evaluation(suite: Any) -> None:
    """Enforce rubric-based evaluation for all tasks in a suite.

    Args:
        suite: EvalSuite object or dict with tasks

    Raises:
        ValueError: If any task lacks rubric-based evaluation
    """
    if isinstance(suite, dict):
        suite_id = suite.get("id", "unknown-suite")
    else:
        suite_id = getattr(suite, "id", "unknown-suite")

    missing = validate_suite_has_rubric_based_evaluation(suite)

    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise ValueError(
            "Rubric-based evaluation is required for every eval task. "
            f"Suite '{suite_id}' has tasks without rubric-based llm_judge graders: {missing_csv}"
        )


__all__ = [
    "task_has_rubric_based_evaluation",
    "validate_task_has_rubric",
    "validate_suite_has_rubric_based_evaluation",
    "enforce_rubric_based_evaluation",
]
