from __future__ import annotations

import pytest

from ash_hawk.thin_runtime import create_default_harness
from ash_hawk.thin_runtime.models import ContextSnapshot

pytestmark = pytest.mark.unit


def test_improver_bootstrap_phase_requires_workspace_load_first() -> None:
    harness = create_default_harness(console_output=False)
    context = ContextSnapshot(runtime={"active_agent": "improver"})

    assert harness.runner.phase_allows_tool("load_workspace_state", context) is True
    assert harness.runner.phase_allows_tool("detect_agent_config", context) is False
    assert harness.runner.phase_allows_tool("run_baseline_eval", context) is False


def test_improver_requires_baseline_once_agent_config_is_known() -> None:
    harness = create_default_harness(console_output=False)
    context = ContextSnapshot(
        runtime={"active_agent": "improver"},
        workspace={
            "agent_config": "/repo/.dawn-kestrel/agent_config.yaml",
            "source_root": "/repo",
            "package_name": "bolt_merlin",
        },
        audit={"tool_results": [{"tool": "load_workspace_state"}]},
    )

    assert harness.runner.phase_allows_tool("run_baseline_eval", context) is True
    assert harness.runner.phase_allows_tool("detect_agent_config", context) is False
    assert harness.runner.phase_allows_tool("read", context) is True


def test_improver_post_baseline_allows_diagnosis_and_blocks_repeated_bootstrap_tools() -> None:
    harness = create_default_harness(console_output=False)
    context = ContextSnapshot(
        runtime={"active_agent": "improver"},
        workspace={
            "agent_config": "/repo/.dawn-kestrel/agent_config.yaml",
            "source_root": "/repo",
            "package_name": "bolt_merlin",
        },
        evaluation={"baseline_summary": {"score": 0.71, "status": "completed"}},
        failure={"failure_family": "needs_improvement"},
        audit={"tool_results": [{"tool": "load_workspace_state"}]},
    )

    assert harness.runner.phase_allows_tool("call_llm_structured", context) is True
    assert harness.runner.phase_allows_tool("load_workspace_state", context) is False
    assert harness.runner.phase_allows_tool("detect_agent_config", context) is False
    assert harness.runner.phase_allows_tool("run_baseline_eval", context) is False


def test_executor_phase_requires_isolated_workspace_before_mutation() -> None:
    harness = create_default_harness(console_output=False)
    context = ContextSnapshot(
        runtime={"active_agent": "executor"},
        workspace={"allowed_target_files": ["agent.md"]},
    )

    assert harness.runner.phase_allows_tool("prepare_isolated_workspace", context) is True
    assert harness.runner.phase_allows_tool("mutate_agent_files", context) is False

    context.workspace["isolated_workspace_path"] = "/tmp/candidate"

    assert harness.runner.phase_allows_tool("prepare_isolated_workspace", context) is False
    assert harness.runner.phase_allows_tool("mutate_agent_files", context) is True
