from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from ash_hawk.thin_runtime import build_default_catalog
from ash_hawk.thin_runtime.catalog_loader import REQUIRED_AGENT_FIELDS, REQUIRED_SKILL_FIELDS
from ash_hawk.thin_runtime.tool_types import ToolSchemaSpec


def test_default_catalog_contains_agentic_runtime_objects() -> None:
    catalog = build_default_catalog()

    agent_names = {agent.name for agent in catalog.agents}
    skill_names = {skill.name for skill in catalog.skills}
    tool_names = {tool.name for tool in catalog.tools}
    hook_names = {hook.name for hook in catalog.hooks}
    memory_names = {scope.name for scope in catalog.memory_scopes}
    context_names = {field.name for field in catalog.context_fields}

    assert agent_names == {
        "coordinator",
        "improver",
        "researcher",
        "executor",
        "verifier",
        "reviewer",
        "memory_manager",
    }
    assert "phase1-review" in skill_names
    assert "improvement-loop" in skill_names
    assert "signal-driven-workspace" in skill_names
    assert "hypothesis-ranking" in skill_names
    assert "call_llm_structured" in tool_names
    assert "run_integrity_validation" in tool_names
    assert "commit_workspace_changes" in tool_names
    assert "before_run" in hook_names
    assert "on_policy_decision" in hook_names
    assert memory_names == {
        "working_memory",
        "session_memory",
        "episodic_memory",
        "semantic_memory",
        "personal_memory",
        "artifact_memory",
    }
    assert context_names == {
        "goal_context",
        "runtime_context",
        "workspace_context",
        "evaluation_context",
        "failure_context",
        "memory_context",
        "tool_context",
        "audit_context",
    }
    coordinator = next(agent for agent in catalog.agents if agent.name == "coordinator")
    process_control = next(skill for skill in catalog.skills if skill.name == "process-control")
    assert coordinator.instructions_markdown
    assert process_control.instructions_markdown
    assert "process-control" in coordinator.skill_names
    assert process_control.tool_names == []
    improver = next(agent for agent in catalog.agents if agent.name == "improver")
    assert improver.default_goal_description
    assert improver.iteration_budget_mode == "loop"
    assert improver.iteration_completion_tools == ["run_eval_repeated"]
    assert "improvement-loop" in improver.skill_names
    assert "signal-driven-workspace" in improver.skill_names

    signal_workspace = next(
        skill for skill in catalog.skills if skill.name == "signal-driven-workspace"
    )
    assert signal_workspace.tool_names == [
        "load_workspace_state",
        "detect_agent_config",
        "scope_workspace",
        "read",
        "grep",
        "diff_workspace_changes",
        "mutate_agent_files",
    ]
    improvement_loop = next(skill for skill in catalog.skills if skill.name == "improvement-loop")
    assert improvement_loop.tool_names == [
        "run_baseline_eval",
        "run_eval_repeated",
        "mutate_agent_files",
        "diff_workspace_changes",
        "call_llm_structured",
    ]


def test_catalog_cross_references_and_tool_entrypoints_are_valid() -> None:
    catalog = build_default_catalog()

    skill_names = {skill.name for skill in catalog.skills}
    tool_names = {tool.name for tool in catalog.tools}
    hook_names = {hook.name for hook in catalog.hooks}
    memory_names = {scope.name: scope for scope in catalog.memory_scopes}
    context_names = {field.name for field in catalog.context_fields}

    for agent in catalog.agents:
        assert set(agent.skill_names).issubset(skill_names)
        assert set(agent.hook_names).issubset(hook_names)
        assert set(agent.memory_read_scopes).issubset(memory_names)
        assert set(agent.memory_write_scopes).issubset(memory_names)
        for scope_name in agent.memory_write_scopes:
            assert agent.name in memory_names[scope_name].writable_by

    for skill in catalog.skills:
        assert set(skill.tool_names).issubset(tool_names)
        assert set(skill.input_contexts).issubset(context_names)
        assert set(skill.output_contexts).issubset(context_names)
        assert set(skill.memory_read_scopes).issubset(memory_names)
        assert set(skill.memory_write_scopes).issubset(memory_names)

    for tool in catalog.tools:
        assert tool.id.startswith("tool.")
        assert tool.python_file.endswith(f"{tool.name}.py")
        assert tool.entrypoint.endswith(tool.name)
        assert tool.callable == "run"
        assert Path("/Users/parkersligting/develop/pt/ash-hawk").joinpath(tool.python_file).exists()
        module = importlib.import_module(tool.entrypoint)
        assert callable(getattr(module, tool.callable))
        assert tool.summary
        assert tool.goal
        assert tool.when_to_use
        assert tool.when_not_to_use
        assert isinstance(tool.inputs, ToolSchemaSpec)
        assert isinstance(tool.outputs, ToolSchemaSpec)
        assert tool.completion_criteria
        assert tool.escalation_rules
        tool_source = (
            Path("/Users/parkersligting/develop/pt/ash-hawk")
            .joinpath(tool.python_file)
            .read_text(encoding="utf-8")
        )
        assert "COMMAND = ToolCommand(" in tool_source
        assert "input_schema={" not in tool_source
        assert "output_schema={" not in tool_source
        assert "tuple[bool, dict[str, object], str, list[str]]" not in tool_source

    referenced_tools = {tool_name for skill in catalog.skills for tool_name in skill.tool_names}
    assert tool_names.issubset(referenced_tools)


def test_raw_markdown_catalog_files_include_required_front_matter_fields() -> None:
    root = Path("/Users/parkersligting/develop/pt/ash-hawk/ash_hawk/thin_runtime/catalog")

    for path in sorted((root / "agents").glob("*.md")):
        raw = _read_frontmatter(path)
        missing = sorted(REQUIRED_AGENT_FIELDS - set(raw))
        assert not missing, f"{path} missing agent fields: {missing}"
        body = _read_body(path)
        assert body
        assert "# Identity" in body
        assert "# Decision Policy" in body
        assert "# Completion Rule" in body

    for path in sorted((root / "skills").glob("*.md")):
        raw = _read_frontmatter(path)
        missing = sorted(REQUIRED_SKILL_FIELDS - set(raw))
        assert not missing, f"{path} missing skill fields: {missing}"
        body = _read_body(path)
        assert body
        assert "# Purpose" in body
        assert "# Procedure" in body
        assert "# Output Contract" in body


def _read_frontmatter(path: Path) -> dict[str, object]:
    content = path.read_text(encoding="utf-8")
    _, rest = content.split("---\n", 1)
    frontmatter_text, _ = rest.split("\n---\n", 1)
    raw = yaml.safe_load(frontmatter_text)
    assert isinstance(raw, dict)
    return raw


def _read_body(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    _, rest = content.split("---\n", 1)
    _, body = rest.split("\n---\n", 1)
    return body.strip()
