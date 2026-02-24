"""Evaluation templates for Ash-Hawk.

This package provides template implementations for different types of
evaluation tasks:
- CodingEvalTemplate: SWE-bench-inspired coding benchmarks
- ConversationalEvalTemplate: τ-Bench-inspired multi-turn dialogue
- ResearchEvalTemplate: BrowseComp-inspired information gathering
- CustomTaskBuilder: YAML-based custom task definitions

All templates extend the EvalTemplate base class and provide:
- Task definition schemas
- Default grader configurations
- Example tasks for each type
- Validation and loading utilities

Example usage:
    from ash_hawk.templates import CodingEvalTemplate

    # Load a coding eval suite from YAML
    template = CodingEvalTemplate.from_yaml("examples/coding/fix-bug.yaml")

    # Create a suite programmatically
    suite = template.create_suite()

    # Run the evaluation
    result = await runner.run(suite)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pydantic as pd
import yaml

from ash_hawk.types import EvalSuite, EvalTask, GraderSpec, ToolSurfacePolicy

if TYPE_CHECKING:
    pass


class TemplateError(Exception):
    """Base exception for template-related errors."""

    pass


class TemplateValidationError(TemplateError):
    """Raised when template validation fails."""

    pass


class TemplateLoadError(TemplateError):
    """Raised when template loading fails."""

    pass


class EvalTemplateConfig(pd.BaseModel):
    """Base configuration for eval templates.

    This defines common configuration options shared by all template types.
    Each specific template extends this with type-specific options.
    """

    name: str = pd.Field(description="Template/suite name")
    description: str = pd.Field(default="", description="Template description")
    version: str = pd.Field(default="1.0.0", description="Template version")
    tags: list[str] = pd.Field(default_factory=list, description="Tags for categorization")
    metadata: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Additional metadata",
    )

    # Policy defaults
    default_policy: ToolSurfacePolicy | None = pd.Field(
        default=None,
        description="Default tool surface policy for tasks",
    )

    # Timing defaults
    default_timeout_seconds: float = pd.Field(
        default=300.0,
        ge=1.0,
        description="Default timeout for tasks",
    )
    max_attempts: int = pd.Field(
        default=1,
        ge=1,
        description="Default max attempts per task",
    )

    model_config = pd.ConfigDict(extra="allow")


class EvalTemplate(ABC):
    """Abstract base class for evaluation templates.

    Templates define structured formats for creating evaluation tasks
    with appropriate graders, policies, and configurations for different
    agent types and use cases.

    Each template type provides:
    - A configuration schema (extends EvalTemplateConfig)
    - Task definition loading and validation
    - Default grader configurations
    - Example tasks for documentation/testing
    """

    # Template type identifier (set by subclasses)
    template_type: ClassVar[str] = "base"

    # Configuration class (override in subclasses)
    config_class: ClassVar[type[EvalTemplateConfig]] = EvalTemplateConfig

    def __init__(self, config: EvalTemplateConfig | dict[str, Any]) -> None:
        """Initialize the template with configuration.

        Args:
            config: Template configuration (dict or config object).
        """
        if isinstance(config, dict):
            self._config = self.config_class(**config)
        else:
            self._config = config

        self._tasks: list[EvalTask] = []
        self._grader_specs: list[GraderSpec] = []

    @property
    def config(self) -> EvalTemplateConfig:
        """Get the template configuration."""
        return self._config

    @property
    def name(self) -> str:
        """Get the template name."""
        return self._config.name

    @property
    def description(self) -> str:
        """Get the template description."""
        return self._config.description

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalTemplate:
        """Load a template from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Instantiated template.

        Raises:
            TemplateLoadError: If the file cannot be loaded.
            TemplateValidationError: If the content is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise TemplateLoadError(f"Template file not found: {path}")

        try:
            content = path.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise TemplateLoadError(f"Failed to parse YAML: {e}") from e

        if not isinstance(data, dict):
            raise TemplateValidationError("Template must be a YAML mapping")

        # Get template type from data
        template_type = data.get("template_type", cls.template_type)

        # Use appropriate subclass based on type
        # This allows from_yaml to be called on base class
        return cls._create_from_data(data, template_type)

    @classmethod
    def _create_from_data(
        cls,
        data: dict[str, Any],
        template_type: str,
    ) -> EvalTemplate:
        """Create appropriate template instance based on type.

        Args:
            data: Parsed YAML data.
            template_type: Type identifier for the template.

        Returns:
            Appropriate template subclass instance.
        """
        # Import here to avoid circular imports
        from ash_hawk.templates.coding import CodingEvalTemplate
        from ash_hawk.templates.conversational import ConversationalEvalTemplate
        from ash_hawk.templates.custom import CustomTaskBuilder
        from ash_hawk.templates.research import ResearchEvalTemplate

        type_map: dict[str, type[EvalTemplate]] = {
            "coding": CodingEvalTemplate,
            "conversational": ConversationalEvalTemplate,
            "research": ResearchEvalTemplate,
            "custom": CustomTaskBuilder,
        }

        template_class = type_map.get(template_type)
        if template_class is None:
            raise TemplateValidationError(f"Unknown template type: {template_type}")

        return template_class(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalTemplate:
        """Create a template from a dictionary.

        Args:
            data: Template configuration as dict.

        Returns:
            Instantiated template.

        Raises:
            TemplateValidationError: If the data is invalid.
        """
        template_type = data.get("template_type", cls.template_type)
        return cls._create_from_data(data, template_type)

    @abstractmethod
    def validate_task(self, task_data: dict[str, Any]) -> EvalTask:
        """Validate and create a task from raw data.

        Args:
            task_data: Raw task data dictionary.

        Returns:
            Validated EvalTask instance.

        Raises:
            TemplateValidationError: If task data is invalid.
        """
        ...

    def add_task(self, task_data: dict[str, Any] | EvalTask) -> None:
        """Add a task to the template.

        Args:
            task_data: Task data dict or EvalTask instance.

        Raises:
            TemplateValidationError: If task data is invalid.
        """
        if isinstance(task_data, EvalTask):
            self._tasks.append(task_data)
        else:
            task = self.validate_task(task_data)
            self._tasks.append(task)

    def add_grader_spec(self, spec: GraderSpec | dict[str, Any]) -> None:
        """Add a grader specification to the template.

        Args:
            spec: GraderSpec instance or config dict.
        """
        if isinstance(spec, GraderSpec):
            self._grader_specs.append(spec)
        else:
            self._grader_specs.append(GraderSpec(**spec))

    @abstractmethod
    def get_default_graders(self) -> list[GraderSpec]:
        """Get default grader specifications for this template type.

        Returns:
            List of default GraderSpec instances.
        """
        ...

    @abstractmethod
    def get_default_policy(self) -> ToolSurfacePolicy:
        """Get default tool surface policy for this template type.

        Returns:
            Default ToolSurfacePolicy instance.
        """
        ...

    def create_suite(self, suite_id: str | None = None) -> EvalSuite:
        """Create an EvalSuite from this template.

        Args:
            suite_id: Optional suite ID (defaults to template name).

        Returns:
            EvalSuite with all tasks and configurations.
        """
        tasks = list(self._tasks)

        # Apply default graders to tasks without graders
        default_graders = self.get_default_graders()
        for i, task in enumerate(tasks):
            if not task.grader_specs:
                # Create a copy with default graders
                task_dict = task.model_dump()
                task_dict["grader_specs"] = [g.model_dump() for g in default_graders]
                tasks[i] = EvalTask(**task_dict)

        # Apply default policy to tasks without timeout
        for i, task in enumerate(tasks):
            if task.timeout_seconds is None:
                task_dict = task.model_dump()
                task_dict["timeout_seconds"] = self._config.default_timeout_seconds
                tasks[i] = EvalTask(**task_dict)

        return EvalSuite(
            id=suite_id or self._config.name,
            name=self._config.name,
            description=self._config.description,
            tasks=tasks,
            tags=self._config.tags,
            metadata=self._config.metadata,
            version=self._config.version,
        )

    @classmethod
    @abstractmethod
    def get_example_tasks(cls) -> list[dict[str, Any]]:
        """Get example task definitions for this template type.

        Returns:
            List of example task data dictionaries.
        """
        ...

    @classmethod
    def create_example_suite(cls, name: str = "example") -> EvalSuite:
        """Create an example suite with sample tasks.

        Args:
            name: Name for the example suite.

        Returns:
            EvalSuite with example tasks.
        """
        template = cls({"name": name, "description": f"Example {cls.template_type} suite"})
        for task_data in cls.get_example_tasks():
            template.add_task(task_data)
        return template.create_suite()


__all__ = [
    "EvalTemplate",
    "EvalTemplateConfig",
    "TemplateError",
    "TemplateValidationError",
    "TemplateLoadError",
]
