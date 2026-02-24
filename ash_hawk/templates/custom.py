"""Custom task framework for user-defined evaluations.

Provides:
- YAML task definition format
- Custom grader configuration via GraderSpec
- Task validation and schema checking
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pydantic as pd
import yaml

from ash_hawk.templates import (
    EvalTemplate,
    EvalTemplateConfig,
    TemplateLoadError,
    TemplateValidationError,
)
from ash_hawk.types import EvalTask, GraderSpec, ToolSurfacePolicy

if TYPE_CHECKING:
    pass


class CustomTaskSchema(pd.BaseModel):
    """Schema definition for a custom task."""

    id: str = pd.Field(description="Unique task identifier")
    description: str = pd.Field(default="", description="Task description")
    input: str | dict[str, Any] = pd.Field(
        description="Task input - string prompt or structured data",
    )
    expected_output: str | dict[str, Any] | None = pd.Field(
        default=None,
        description="Expected output for validation",
    )
    grader_specs: list[dict[str, Any]] = pd.Field(
        default_factory=list,
        description="Grader configurations",
    )
    tags: list[str] = pd.Field(
        default_factory=list,
        description="Tags for categorization",
    )
    metadata: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Additional metadata",
    )
    fixtures: dict[str, str] = pd.Field(
        default_factory=dict,
        description="Fixture file paths",
    )
    timeout_seconds: float | None = pd.Field(
        default=None,
        description="Task timeout",
    )
    max_attempts: int = pd.Field(
        default=1,
        ge=1,
        description="Maximum attempts",
    )

    model_config = pd.ConfigDict(extra="allow")


class CustomEvalConfig(EvalTemplateConfig):
    """Configuration for custom evaluation template."""

    task_schema: CustomTaskSchema | None = pd.Field(
        default=None,
        description="Default task schema",
    )
    default_grader_type: str = pd.Field(
        default="llm_judge",
        description="Default grader type when not specified",
    )
    default_grader_config: dict[str, Any] = pd.Field(
        default_factory=lambda: {"rubric": "correctness", "pass_threshold": 0.7},
        description="Default grader configuration",
    )
    validate_inputs: bool = pd.Field(
        default=True,
        description="Whether to validate input schema",
    )
    strict_validation: bool = pd.Field(
        default=False,
        description="Fail on unknown fields",
    )

    model_config = pd.ConfigDict(extra="allow")


class CustomTaskBuilder(EvalTemplate):
    """Custom task framework for user-defined evaluations.

    Supports YAML task definition format with custom grader configuration
    via GraderSpec. Provides validation and schema checking.

    YAML Format:
        template_type: custom
        name: my-eval-suite
        description: My custom evaluation
        tasks:
          - id: task-1
            input: "Write a function to..."
            expected_output: "A working function"
            grader_specs:
              - grader_type: test_runner
                config:
                  test_path: tests/test_my_func.py
                weight: 0.6
    """

    template_type: ClassVar[str] = "custom"
    config_class: ClassVar[type[EvalTemplateConfig]] = CustomEvalConfig

    def __init__(self, config: CustomEvalConfig | dict[str, Any]) -> None:
        super().__init__(config)
        self._custom_config: CustomEvalConfig = (
            config if isinstance(config, CustomEvalConfig) else CustomEvalConfig(**config)
        )
        self._task_schemas: dict[str, CustomTaskSchema] = {}

    def validate_task(self, task_data: dict[str, Any]) -> EvalTask:
        """Validate and create a custom task from raw data."""
        if "id" not in task_data:
            raise TemplateValidationError("Custom task must have 'id' field")

        if "input" not in task_data:
            raise TemplateValidationError("Custom task must have 'input' field")

        try:
            if self._custom_config.strict_validation:
                schema = CustomTaskSchema(**task_data)
            else:
                schema = CustomTaskSchema.model_construct(**task_data)
        except pd.ValidationError as e:
            raise TemplateValidationError(f"Invalid task schema: {e}") from e

        self._task_schemas[schema.id] = schema

        grader_specs = self._build_grader_specs(schema)

        return EvalTask(
            id=schema.id,
            description=schema.description or self._extract_description(schema.input),
            input=schema.input,
            expected_output=schema.expected_output,
            grader_specs=grader_specs,
            tags=schema.tags,
            metadata=schema.metadata,
            fixtures=schema.fixtures,
            timeout_seconds=schema.timeout_seconds or self._config.default_timeout_seconds,
            max_attempts=schema.max_attempts,
        )

    def _extract_description(self, input_data: str | dict[str, Any]) -> str:
        """Extract a description from input data."""
        if isinstance(input_data, str):
            return input_data[:200] if len(input_data) > 200 else input_data
        if isinstance(input_data, dict):
            if "description" in input_data:
                return str(input_data["description"])[:200]
            if "question" in input_data:
                return str(input_data["question"])[:200]
            if "prompt" in input_data:
                return str(input_data["prompt"])[:200]
        return "Custom task"

    def _build_grader_specs(self, schema: CustomTaskSchema) -> list[GraderSpec]:
        """Build grader specs from task schema."""
        if schema.grader_specs:
            specs: list[GraderSpec] = []
            for gs in schema.grader_specs:
                if isinstance(gs, GraderSpec):
                    specs.append(gs)
                else:
                    specs.append(GraderSpec(**gs))
            return specs

        default_config = self._custom_config.default_grader_config.copy()
        default_config["grader_type"] = self._custom_config.default_grader_type

        return [
            GraderSpec(
                grader_type=self._custom_config.default_grader_type,
                config=self._custom_config.default_grader_config,
                weight=1.0,
                required=True,
            ),
        ]

    def get_default_graders(self) -> list[GraderSpec]:
        """Get default grader specs for custom tasks."""
        return [
            GraderSpec(
                grader_type=self._custom_config.default_grader_type,
                config=self._custom_config.default_grader_config,
                weight=1.0,
                required=True,
            ),
        ]

    def get_default_policy(self) -> ToolSurfacePolicy:
        """Get default tool policy for custom tasks."""
        return ToolSurfacePolicy(
            allowed_tools=["read", "write", "bash", "grep", "glob"],
            network_allowed=False,
            max_tool_calls=50,
            timeout_seconds=300.0,
        )

    @classmethod
    def get_example_tasks(cls) -> list[dict[str, Any]]:
        """Get example custom tasks."""
        return [
            {
                "id": "simple-qa",
                "description": "Simple question answering task",
                "input": "What is the capital of France?",
                "expected_output": "Paris",
                "grader_specs": [
                    {
                        "grader_type": "string_match",
                        "config": {"expected": "Paris", "mode": "exact"},
                        "weight": 1.0,
                    }
                ],
                "tags": ["qa", "geography"],
            },
            {
                "id": "code-generation",
                "description": "Generate a Python function",
                "input": {
                    "prompt": "Write a function that returns the sum of two numbers",
                    "language": "python",
                },
                "grader_specs": [
                    {
                        "grader_type": "test_runner",
                        "config": {
                            "test_path": "tests/test_sum.py",
                            "pass_threshold": 1.0,
                        },
                        "weight": 0.6,
                    },
                    {
                        "grader_type": "static_analysis",
                        "config": {
                            "tools": ["ruff"],
                            "max_issues": 0,
                        },
                        "weight": 0.4,
                    },
                ],
                "tags": ["code", "python"],
            },
            {
                "id": "open-ended",
                "description": "Open-ended creative task",
                "input": "Write a haiku about programming",
                "grader_specs": [
                    {
                        "grader_type": "llm_judge",
                        "config": {
                            "rubric": "quality",
                            "pass_threshold": 0.6,
                        },
                        "weight": 1.0,
                    }
                ],
                "tags": ["creative", "writing"],
            },
        ]

    @classmethod
    def load_tasks_from_yaml(cls, path: str | Path) -> list[dict[str, Any]]:
        """Load tasks from a YAML file.

        Args:
            path: Path to YAML file containing task definitions.

        Returns:
            List of task data dictionaries.

        Raises:
            TemplateLoadError: If file cannot be loaded.
        """
        path = Path(path)
        if not path.exists():
            raise TemplateLoadError(f"Task file not found: {path}")

        try:
            content = path.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise TemplateLoadError(f"Failed to parse YAML: {e}") from e

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "tasks" in data:
                return data["tasks"]
            return [data]
        raise TemplateLoadError(f"Unexpected YAML structure: {type(data)}")

    @classmethod
    def validate_yaml_schema(cls, path: str | Path) -> list[str]:
        """Validate YAML file against custom task schema.

        Args:
            path: Path to YAML file.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: list[str] = []
        tasks = cls.load_tasks_from_yaml(path)

        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                errors.append(f"Task {i}: must be a dictionary")
                continue

            if "id" not in task:
                errors.append(f"Task {i}: missing 'id' field")

            if "input" not in task:
                errors.append(f"Task {i}: missing 'input' field")

            if "grader_specs" in task:
                for j, gs in enumerate(task["grader_specs"]):
                    if not isinstance(gs, dict):
                        errors.append(f"Task {i}, grader {j}: must be a dictionary")
                        continue
                    if "grader_type" not in gs:
                        errors.append(f"Task {i}, grader {j}: missing 'grader_type'")

        return errors


__all__ = [
    "CustomTaskBuilder",
    "CustomEvalConfig",
    "CustomTaskSchema",
]
