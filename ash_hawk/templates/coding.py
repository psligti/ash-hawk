"""Coding evaluation template (SWE-bench-inspired).

Provides task format for coding agent benchmarks with:
- Issue description + test file + expected behavior
- Graders: test runner (sandboxed), static analysis, code review
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import pydantic as pd

from ash_hawk.templates import (
    EvalTemplate,
    EvalTemplateConfig,
    TemplateValidationError,
)
from ash_hawk.types import EvalTask, GraderSpec, ToolSurfacePolicy

if TYPE_CHECKING:
    pass


class CodingTaskConfig(pd.BaseModel):
    """Configuration for a coding evaluation task."""

    issue_description: str = pd.Field(description="The issue/bug description to fix")
    codebase_path: str | None = pd.Field(
        default=None,
        description="Path to codebase (relative or absolute)",
    )
    test_files: list[str] = pd.Field(
        default_factory=list,
        description="Test files to run",
    )
    expected_behavior: str | None = pd.Field(
        default=None,
        description="Description of expected behavior after fix",
    )
    hints: list[str] = pd.Field(
        default_factory=list,
        description="Optional hints for the agent",
    )
    difficulty: str = pd.Field(
        default="medium",
        description="Task difficulty: easy, medium, hard",
    )
    language: str = pd.Field(
        default="python",
        description="Programming language",
    )
    fixtures: dict[str, str] = pd.Field(
        default_factory=dict,
        description="Fixture files (name -> path)",
    )

    model_config = pd.ConfigDict(extra="forbid")


class CodingEvalConfig(EvalTemplateConfig):
    """Configuration for coding evaluation template."""

    language: str = pd.Field(default="python", description="Default language")
    test_framework: str = pd.Field(
        default="pytest",
        description="Test framework to use",
    )
    run_static_analysis: bool = pd.Field(
        default=True,
        description="Whether to run static analysis",
    )
    run_code_review: bool = pd.Field(
        default=True,
        description="Whether to run LLM code review",
    )
    sandbox_workspace: str | None = pd.Field(
        default=None,
        description="Workspace root for sandboxed execution",
    )
    pytest_args: list[str] = pd.Field(
        default_factory=lambda: ["-v", "--tb=short"],
        description="Default pytest arguments",
    )

    model_config = pd.ConfigDict(extra="allow")


class CodingEvalTemplate(EvalTemplate):
    """SWE-bench-inspired coding evaluation template.

    Supports task format: issue description + test file + expected behavior
    Includes graders: test runner (sandboxed), static analysis, code review
    """

    template_type: ClassVar[str] = "coding"
    config_class: ClassVar[type[EvalTemplateConfig]] = CodingEvalConfig

    def __init__(self, config: CodingEvalConfig | dict[str, Any]) -> None:
        super().__init__(config)
        self._coding_config: CodingEvalConfig = (
            config if isinstance(config, CodingEvalConfig) else CodingEvalConfig(**config)
        )

    def validate_task(self, task_data: dict[str, Any]) -> EvalTask:
        if "issue_description" not in task_data and "input" not in task_data:
            raise TemplateValidationError(
                "Coding task must have 'issue_description' or 'input' field"
            )

        raw_input = task_data.get("input")
        input_dict: dict[str, Any] = raw_input if isinstance(raw_input, dict) else {}

        issue_desc: str
        if "issue_description" in task_data:
            val = task_data["issue_description"]
            issue_desc = str(val) if val is not None else ""
        elif isinstance(raw_input, str):
            issue_desc = raw_input
        elif "issue_description" in input_dict:
            val = input_dict["issue_description"]
            issue_desc = str(val) if val is not None else ""
        else:
            issue_desc = ""

        codebase_path_val = task_data.get("codebase_path") or input_dict.get("codebase_path")
        codebase_path: str | None = str(codebase_path_val) if codebase_path_val else None

        test_files_val = task_data.get("test_files", [])
        test_files: list[str] = [
            str(f) for f in (test_files_val if isinstance(test_files_val, list) else [])
        ]

        hints_val = task_data.get("hints", [])
        hints: list[str] = [str(h) for h in (hints_val if isinstance(hints_val, list) else [])]

        fixtures_val = task_data.get("fixtures", {})
        fixtures: dict[str, str] = {
            str(k): str(v)
            for k, v in (fixtures_val if isinstance(fixtures_val, dict) else {}).items()
        }

        difficulty_val = task_data.get("difficulty", "medium")
        difficulty = str(difficulty_val) if difficulty_val else "medium"

        language_val = task_data.get("language", self._coding_config.language)
        language = str(language_val) if language_val else self._coding_config.language

        expected_behavior_val = task_data.get("expected_behavior")
        expected_behavior: str | None = (
            str(expected_behavior_val) if expected_behavior_val else None
        )

        task_config = CodingTaskConfig(
            issue_description=issue_desc,
            codebase_path=codebase_path,
            test_files=test_files,
            expected_behavior=expected_behavior,
            hints=hints,
            difficulty=difficulty,
            language=language,
            fixtures=fixtures,
        )

        input_payload = {
            "issue_description": task_config.issue_description,
            "codebase_path": task_config.codebase_path,
            "test_files": task_config.test_files,
            "expected_behavior": task_config.expected_behavior,
            "hints": task_config.hints,
            "language": task_config.language,
        }

        grader_specs = self._build_grader_specs(task_config, task_data)

        return EvalTask(
            id=task_data.get("id", f"coding-{len(self._tasks) + 1}"),
            description=task_data.get("description", task_config.issue_description[:200]),
            input=input_payload,
            expected_output=task_config.expected_behavior,
            grader_specs=grader_specs,
            tags=list(set(["coding", task_config.language, task_config.difficulty])),
            metadata={
                "difficulty": task_config.difficulty,
                "language": task_config.language,
                "test_files": task_config.test_files,
                "hints": task_config.hints,
            },
            fixtures=task_config.fixtures,
            timeout_seconds=task_data.get("timeout_seconds", self._config.default_timeout_seconds),
        )

    def _build_grader_specs(
        self,
        task_config: CodingTaskConfig,
        task_data: dict[str, Any],
    ) -> list[GraderSpec]:
        """Build grader specs for a coding task."""
        specs: list[GraderSpec] = []

        if task_config.test_files:
            test_path = (
                task_config.test_files[0]
                if len(task_config.test_files) == 1
                else " ".join(task_config.test_files)
            )
            specs.append(
                GraderSpec(
                    grader_type="test_runner",
                    config={
                        "test_path": test_path,
                        "pytest_args": task_data.get(
                            "pytest_args",
                            self._coding_config.pytest_args,
                        ),
                        "pass_threshold": task_data.get("pass_threshold", 1.0),
                        "partial_credit": task_data.get("partial_credit", True),
                        "workspace_root": self._coding_config.sandbox_workspace,
                    },
                    weight=0.5,
                    required=True,
                )
            )

        if self._coding_config.run_static_analysis and task_config.codebase_path:
            specs.append(
                GraderSpec(
                    grader_type="static_analysis",
                    config={
                        "tools": ["ruff", "mypy"],
                        "target_path": task_config.codebase_path,
                        "fail_on_issues": task_data.get("fail_on_static_issues", False),
                        "max_issues": task_data.get("max_static_issues", 5),
                        "partial_credit": True,
                        "workspace_root": self._coding_config.sandbox_workspace,
                    },
                    weight=0.25,
                    required=False,
                )
            )

        if self._coding_config.run_code_review:
            specs.append(
                GraderSpec(
                    grader_type="llm_judge",
                    config={
                        "rubric": "correctness",
                        "pass_threshold": 0.7,
                    },
                    weight=0.25,
                    required=False,
                )
            )

        custom_graders = task_data.get("grader_specs", [])
        for cg in custom_graders:
            if isinstance(cg, GraderSpec):
                specs.append(cg)
            else:
                specs.append(GraderSpec(**cg))

        return specs

    def get_default_graders(self) -> list[GraderSpec]:
        """Get default grader specs for coding tasks."""
        return [
            GraderSpec(
                grader_type="test_runner",
                config={
                    "pass_threshold": 1.0,
                    "partial_credit": True,
                },
                weight=0.6,
                required=True,
            ),
            GraderSpec(
                grader_type="static_analysis",
                config={
                    "tools": ["ruff"],
                    "max_issues": 0,
                    "partial_credit": True,
                },
                weight=0.2,
                required=False,
            ),
            GraderSpec(
                grader_type="llm_judge",
                config={
                    "rubric": "correctness",
                    "pass_threshold": 0.7,
                },
                weight=0.2,
                required=False,
            ),
        ]

    def get_default_policy(self) -> ToolSurfacePolicy:
        """Get default tool policy for coding tasks."""
        return ToolSurfacePolicy(
            allowed_tools=["read", "write", "edit", "bash", "glob", "grep"],
            allowed_roots=["/tmp/workspace", "."],
            network_allowed=False,
            max_tool_calls=100,
            timeout_seconds=600.0,
            max_file_size_bytes=10 * 1024 * 1024,
        )

    @classmethod
    def get_example_tasks(cls) -> list[dict[str, Any]]:
        """Get example coding tasks."""
        return [
            {
                "id": "fix-off-by-one",
                "description": "Fix off-by-one error in array iteration",
                "issue_description": (
                    "The function `find_first_negative` in `utils.py` has an off-by-one error. "
                    "It should iterate through all elements of the array, but currently skips "
                    "the last element. The function returns the index of the first negative "
                    "number, or -1 if none found."
                ),
                "codebase_path": "fixtures/off_by_one/",
                "test_files": ["tests/test_find_negative.py"],
                "expected_behavior": (
                    "The function should correctly iterate through all elements and return "
                    "the index of the first negative number."
                ),
                "hints": [
                    "Check the loop boundary condition",
                    "Python range() is exclusive of end",
                ],
                "difficulty": "easy",
                "language": "python",
            },
            {
                "id": "fix-auth-bypass",
                "description": "Fix authentication bypass vulnerability",
                "issue_description": (
                    "The authentication middleware in `auth.py` has a vulnerability where "
                    "requests with empty Authorization headers are incorrectly treated as "
                    "authenticated. This allows unauthorized access to protected endpoints. "
                    "The middleware should reject requests with missing or empty auth headers."
                ),
                "codebase_path": "fixtures/auth_bypass/",
                "test_files": ["tests/test_auth.py", "tests/test_protected.py"],
                "expected_behavior": (
                    "Requests without valid authentication should be rejected with 401 status. "
                    "Empty Authorization headers should be treated as unauthenticated."
                ),
                "hints": ["Check how empty strings are evaluated in the auth check"],
                "difficulty": "medium",
                "language": "python",
            },
            {
                "id": "implement-caching",
                "description": "Implement memoization for expensive computations",
                "issue_description": (
                    "The `FibonacciCalculator` class in `math_utils.py` recalculates "
                    "fibonacci numbers on every call, causing exponential time complexity. "
                    "Implement memoization to cache previously computed values. The cache "
                    "should have a maximum size of 1000 entries and use LRU eviction."
                ),
                "codebase_path": "fixtures/fibonacci/",
                "test_files": ["tests/test_fibonacci.py", "tests/test_performance.py"],
                "expected_behavior": (
                    "The calculator should cache results and return cached values on "
                    "subsequent calls. Performance test requires < 1ms for cached values."
                ),
                "hints": [
                    "Use functools.lru_cache or implement custom caching",
                    "Consider thread-safety requirements",
                ],
                "difficulty": "hard",
                "language": "python",
            },
        ]


__all__ = [
    "CodingEvalTemplate",
    "CodingEvalConfig",
    "CodingTaskConfig",
]
