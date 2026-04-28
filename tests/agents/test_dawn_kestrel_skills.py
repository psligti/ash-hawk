from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ash_hawk.agents import DawnKestrelAgentRunner
from ash_hawk.policy import PolicyEnforcer
from ash_hawk.types import EvalTask, ToolSurfacePolicy


@pytest.mark.asyncio
async def test_runner_passes_preactivated_dk_skill_runtime_to_run_agent() -> None:
    runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")
    task = EvalTask(id="task-skill", description="Test task", input="Fix the bug")
    policy_enforcer = PolicyEnforcer(
        ToolSurfacePolicy(allowed_tools=["read*"], timeout_seconds=60.0)
    )

    class MockResult:
        response = MagicMock(text="done")
        error = None
        iterations = 1
        total_usage = {}
        session = None

    captured: dict[str, object] = {}

    async def _capture_run_agent(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return MockResult()

    mock_agent_module = SimpleNamespace(run_agent=AsyncMock(side_effect=_capture_run_agent))
    mock_types_module = SimpleNamespace(LoopConfig=MagicMock(return_value=MagicMock()))

    def _import_module(name: str) -> Any:
        if name == "dawn_kestrel.agent.loop":
            return mock_agent_module
        if name == "dawn_kestrel.agent.types":
            return mock_types_module
        if name == "dawn_kestrel.tools.registry":
            return SimpleNamespace(ToolRegistry=MagicMock)
        if name == "dawn_kestrel.tools.framework":
            return SimpleNamespace(ToolContext=MagicMock, ToolResult=MagicMock)
        if name == "dawn_kestrel.base.config":
            return MagicMock()
        raise AssertionError(f"Unexpected module import: {name}")

    base_registry = MagicMock()
    mock_registry = MagicMock()
    mock_registry.tools = {}
    mock_registry.get_all = AsyncMock(return_value={})
    skill_runtime = MagicMock()
    preloaded_messages = [
        {
            "role": "system",
            "content": '<skill_content name="python-bugfix">body</skill_content>',
            "_protected": True,
        }
    ]

    with patch.object(runner, "_get_client", return_value=MagicMock()):
        with patch.object(runner, "_create_base_registry", return_value=base_registry):
            with patch.object(runner, "_register_mcp_tools", AsyncMock(return_value=[])):
                with patch.object(runner, "_create_filtered_registry", return_value=mock_registry):
                    with patch(
                        "ash_hawk.agents.dawn_kestrel.prepare_skill_runtime",
                        AsyncMock(return_value=(skill_runtime, preloaded_messages)),
                    ):
                        with patch(
                            "ash_hawk.agents.dawn_kestrel.importlib.import_module",
                            side_effect=_import_module,
                        ):
                            await runner.run(
                                task=task,
                                policy_enforcer=policy_enforcer,
                                config={"skill_name": "python-bugfix"},
                            )

    assert captured["skills"] is skill_runtime
    assert captured["messages"][0] == preloaded_messages[0]
    assert captured["messages"][1] == {"role": "user", "content": "Fix the bug"}
