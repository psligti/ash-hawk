"""Tests for evaluation templates."""

from __future__ import annotations

from pathlib import Path

import pytest

from ash_hawk.templates import (
    EvalTemplate,
    EvalTemplateConfig,
    TemplateLoadError,
    TemplateValidationError,
)
from ash_hawk.templates.coding import CodingEvalConfig, CodingEvalTemplate, CodingTaskConfig
from ash_hawk.templates.conversational import (
    ConversationalEvalConfig,
    ConversationalEvalTemplate,
    DialogueScenarioConfig,
    UserPersonaConfig,
)
from ash_hawk.templates.custom import CustomEvalConfig, CustomTaskBuilder, CustomTaskSchema
from ash_hawk.templates.research import (
    GroundednessConfig,
    NeedleConfig,
    ResearchEvalConfig,
    ResearchEvalTemplate,
    SourceDocument,
)
from ash_hawk.types import EvalSuite, EvalTask, GraderSpec, ToolSurfacePolicy


class TestEvalTemplateConfig:
    """Tests for EvalTemplateConfig."""

    def test_create_basic_config(self) -> None:
        config = EvalTemplateConfig(
            name="test-template",
            description="A test template",
        )
        assert config.name == "test-template"
        assert config.description == "A test template"
        assert config.version == "1.0.0"
        assert config.default_timeout_seconds == 300.0

    def test_config_with_policy(self) -> None:
        policy = ToolSurfacePolicy(
            allowed_tools=["read", "write"],
            timeout_seconds=600.0,
        )
        config = EvalTemplateConfig(
            name="test",
            default_policy=policy,
        )
        assert config.default_policy is not None
        assert config.default_policy.allowed_tools == ["read", "write"]


class TestCodingEvalTemplate:
    """Tests for CodingEvalTemplate."""

    def test_create_template(self) -> None:
        template = CodingEvalTemplate({"name": "test-coding"})
        assert template.name == "test-coding"
        assert template.template_type == "coding"

    def test_validate_task(self) -> None:
        template = CodingEvalTemplate({"name": "test"})
        task_data = {
            "id": "fix-bug-1",
            "issue_description": "Fix the bug in utils.py",
            "test_files": ["tests/test_utils.py"],
            "difficulty": "easy",
        }
        task = template.validate_task(task_data)
        assert task.id == "fix-bug-1"
        assert "coding" in task.tags
        assert len(task.grader_specs) > 0

    def test_validate_task_missing_field(self) -> None:
        template = CodingEvalTemplate({"name": "test"})
        with pytest.raises(TemplateValidationError):
            template.validate_task({"id": "task-1"})

    def test_get_default_graders(self) -> None:
        template = CodingEvalTemplate({"name": "test"})
        graders = template.get_default_graders()
        assert len(graders) >= 2
        grader_types = [g.grader_type for g in graders]
        assert "test_runner" in grader_types

    def test_get_default_policy(self) -> None:
        template = CodingEvalTemplate({"name": "test"})
        policy = template.get_default_policy()
        assert policy is not None
        assert "read" in policy.allowed_tools
        assert policy.network_allowed is False

    def test_get_example_tasks(self) -> None:
        tasks = CodingEvalTemplate.get_example_tasks()
        assert len(tasks) == 3
        assert all("issue_description" in t for t in tasks)

    def test_create_example_suite(self) -> None:
        suite = CodingEvalTemplate.create_example_suite("test-suite")
        assert isinstance(suite, EvalSuite)
        assert suite.name == "test-suite"
        assert len(suite.tasks) == 3


class TestConversationalEvalTemplate:
    """Tests for ConversationalEvalTemplate."""

    def test_create_template(self) -> None:
        template = ConversationalEvalTemplate({"name": "test-conv"})
        assert template.name == "test-conv"
        assert template.template_type == "conversational"

    def test_validate_task(self) -> None:
        template = ConversationalEvalTemplate({"name": "test"})
        task_data = {
            "id": "support-1",
            "scenario": {
                "scenario_type": "support",
                "context": "Customer needs help",
                "user_persona": {
                    "name": "test_user",
                    "role": "customer",
                },
            },
        }
        task = template.validate_task(task_data)
        assert task.id == "support-1"
        assert "conversational" in task.tags

    def test_user_persona_config(self) -> None:
        persona = UserPersonaConfig(
            name="frustrated_customer",
            role="customer",
            personality_traits=["frustrated"],
            knowledge_level="novice",
        )
        assert persona.name == "frustrated_customer"
        assert "frustrated" in persona.personality_traits

    def test_dialogue_scenario_config(self) -> None:
        scenario = DialogueScenarioConfig(
            scenario_type="support",
            context="Customer issue",
            max_turns=10,
        )
        assert scenario.scenario_type == "support"
        assert scenario.max_turns == 10

    def test_get_example_tasks(self) -> None:
        tasks = ConversationalEvalTemplate.get_example_tasks()
        assert len(tasks) == 3
        assert all("scenario" in t for t in tasks)


class TestResearchEvalTemplate:
    """Tests for ResearchEvalTemplate."""

    def test_create_template(self) -> None:
        template = ResearchEvalTemplate({"name": "test-research"})
        assert template.name == "test-research"
        assert template.template_type == "research"

    def test_validate_task(self) -> None:
        template = ResearchEvalTemplate({"name": "test"})
        task_data = {
            "id": "find-info-1",
            "question": "What is the error code?",
            "sources": [
                {
                    "id": "log-1",
                    "title": "error.log",
                    "content": "ERROR AUTH_TIMEOUT_8472: timeout",
                    "contains_answer": True,
                },
            ],
            "expected_output": "AUTH_TIMEOUT_8472",
        }
        task = template.validate_task(task_data)
        assert task.id == "find-info-1"
        assert "research" in task.tags

    def test_source_document(self) -> None:
        source = SourceDocument(
            id="doc-1",
            title="Test Document",
            content="Some content here",
            contains_answer=True,
        )
        assert source.id == "doc-1"
        assert source.contains_answer is True

    def test_needle_config(self) -> None:
        needle = NeedleConfig(
            needle="Find this info",
            expected_answer="The answer",
        )
        assert needle.needle == "Find this info"
        assert needle.expected_answer == "The answer"

    def test_groundedness_config(self) -> None:
        config = GroundednessConfig(
            require_source_citations=True,
            min_sources_required=2,
        )
        assert config.require_source_citations is True
        assert config.min_sources_required == 2

    def test_get_example_tasks(self) -> None:
        tasks = ResearchEvalTemplate.get_example_tasks()
        assert len(tasks) == 3
        assert all("sources" in t for t in tasks)


class TestCustomTaskBuilder:
    """Tests for CustomTaskBuilder."""

    def test_create_builder(self) -> None:
        builder = CustomTaskBuilder({"name": "test-custom"})
        assert builder.name == "test-custom"
        assert builder.template_type == "custom"

    def test_validate_task(self) -> None:
        builder = CustomTaskBuilder({"name": "test"})
        task_data = {
            "id": "custom-1",
            "input": "What is 2 + 2?",
            "expected_output": "4",
        }
        task = builder.validate_task(task_data)
        assert task.id == "custom-1"
        assert task.input == "What is 2 + 2?"

    def test_validate_task_with_graders(self) -> None:
        builder = CustomTaskBuilder({"name": "test"})
        task_data = {
            "id": "custom-2",
            "input": "Write a function",
            "grader_specs": [
                {
                    "grader_type": "test_runner",
                    "config": {"test_path": "tests/test_func.py"},
                    "weight": 1.0,
                }
            ],
        }
        task = builder.validate_task(task_data)
        assert len(task.grader_specs) == 1
        assert task.grader_specs[0].grader_type == "test_runner"

    def test_validate_task_missing_id(self) -> None:
        builder = CustomTaskBuilder({"name": "test"})
        with pytest.raises(TemplateValidationError):
            builder.validate_task({"input": "test"})

    def test_validate_task_missing_input(self) -> None:
        builder = CustomTaskBuilder({"name": "test"})
        with pytest.raises(TemplateValidationError):
            builder.validate_task({"id": "task-1"})

    def test_get_example_tasks(self) -> None:
        tasks = CustomTaskBuilder.get_example_tasks()
        assert len(tasks) == 3
        assert all("id" in t and "input" in t for t in tasks)

    def test_load_tasks_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
- id: task-1
  input: "Question 1"
  expected_output: "Answer 1"
- id: task-2
  input: "Question 2"
  expected_output: "Answer 2"
"""
        yaml_file = tmp_path / "tasks.yaml"
        yaml_file.write_text(yaml_content)

        tasks = CustomTaskBuilder.load_tasks_from_yaml(yaml_file)
        assert len(tasks) == 2
        assert tasks[0]["id"] == "task-1"

    def test_validate_yaml_schema(self, tmp_path: Path) -> None:
        yaml_content = """
- id: task-1
  input: "Question 1"
  grader_specs:
    - grader_type: test_runner
      config: {}
"""
        yaml_file = tmp_path / "valid.yaml"
        yaml_file.write_text(yaml_content)

        errors = CustomTaskBuilder.validate_yaml_schema(yaml_file)
        assert len(errors) == 0

    def test_validate_yaml_schema_errors(self, tmp_path: Path) -> None:
        yaml_content = """
- input: "Missing id"
- id: "missing-input"
"""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text(yaml_content)

        errors = CustomTaskBuilder.validate_yaml_schema(yaml_file)
        assert len(errors) >= 2


class TestEvalTemplate:
    """Tests for EvalTemplate base class."""

    def test_from_yaml_coding(self, tmp_path: Path) -> None:
        yaml_content = """
template_type: coding
name: test-suite
description: Test suite
tasks:
  - id: task-1
    issue_description: Fix this bug
    test_files: [tests/test_x.py]
"""
        yaml_file = tmp_path / "coding.yaml"
        yaml_file.write_text(yaml_content)

        template = EvalTemplate.from_yaml(yaml_file)
        assert isinstance(template, CodingEvalTemplate)
        assert template.name == "test-suite"

    def test_from_yaml_custom(self, tmp_path: Path) -> None:
        yaml_content = """
template_type: custom
name: custom-suite
tasks:
  - id: task-1
    input: What is 2+2?
"""
        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text(yaml_content)

        template = EvalTemplate.from_yaml(yaml_file)
        assert isinstance(template, CustomTaskBuilder)

    def test_from_yaml_file_not_found(self) -> None:
        with pytest.raises(TemplateLoadError):
            EvalTemplate.from_yaml("/nonexistent/file.yaml")

    def test_from_dict(self) -> None:
        data = {
            "template_type": "coding",
            "name": "test",
            "tasks": [
                {
                    "id": "t1",
                    "issue_description": "Bug",
                    "test_files": ["tests/"],
                }
            ],
        }
        template = EvalTemplate.from_dict(data)
        assert isinstance(template, CodingEvalTemplate)

    def test_add_task(self) -> None:
        template = CodingEvalTemplate({"name": "test"})
        template.add_task(
            {
                "id": "t1",
                "issue_description": "Bug",
                "test_files": [],
            }
        )
        assert len(template._tasks) == 1

    def test_add_grader_spec(self) -> None:
        template = CodingEvalTemplate({"name": "test"})
        template.add_grader_spec(
            {
                "grader_type": "test_runner",
                "config": {},
            }
        )
        assert len(template._grader_specs) == 1

    def test_create_suite(self) -> None:
        template = CodingEvalTemplate({"name": "test-suite"})
        template.add_task(
            {
                "id": "t1",
                "issue_description": "Bug",
                "test_files": ["tests/"],
            }
        )
        suite = template.create_suite()
        assert isinstance(suite, EvalSuite)
        assert suite.name == "test-suite"
        assert len(suite.tasks) == 1


class TestTemplateIntegration:
    """Integration tests for templates."""

    def test_coding_suite_creation(self) -> None:
        template = CodingEvalTemplate(
            {
                "name": "integration-test",
                "language": "python",
                "run_static_analysis": True,
            }
        )
        for task_data in CodingEvalTemplate.get_example_tasks():
            template.add_task(task_data)

        suite = template.create_suite("integration-suite")
        assert len(suite.tasks) == 3
        for task in suite.tasks:
            assert len(task.grader_specs) > 0

    def test_research_suite_with_needle(self) -> None:
        template = ResearchEvalTemplate(
            {
                "name": "needle-test",
                "groundedness_check": True,
            }
        )
        task_data = {
            "id": "needle-1",
            "question": "Find the API key",
            "sources": [
                {
                    "id": "s1",
                    "title": "config",
                    "content": "api_key: secret123",
                    "contains_answer": True,
                }
            ],
            "needle_config": {
                "needle": "API key",
                "expected_answer": "secret123",
            },
        }
        template.add_task(task_data)
        suite = template.create_suite()
        assert len(suite.tasks) == 1
        assert "needle-in-haystack" in suite.tasks[0].tags
