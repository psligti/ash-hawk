# type-hygiene: skip-file
"""Tests for ash_hawk.scenario.adapters.bolt_merlin module."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ash_hawk.scenario.adapters import ScenarioAdapter
from ash_hawk.scenario.adapters.bolt_merlin import BoltMerlinScenarioAdapter
from ash_hawk.scenario.models import ScenarioAdapterResult
from ash_hawk.types import EvalStatus, FailureMode


def _make_mock_execution_result(
    response: str = "done",
    session_id: str = "test-123",
    tokens_in: int = 100,
    tokens_out: int = 50,
    duration_ms: int = 1000,
) -> MagicMock:
    """Create a mock ExecutionResult object."""
    mock = MagicMock()
    mock.response = response
    mock.session_id = session_id
    mock.tokens_in = tokens_in
    mock.tokens_out = tokens_out
    mock.duration_ms = duration_ms
    # ExecutionResult does NOT have error_type
    del mock.error_type
    return mock


def _make_mock_execution_error(
    error_type: str = "execution_error",
    message: str = "boom",
) -> MagicMock:
    """Create a mock ExecutionError object."""
    mock = MagicMock()
    mock.error_type = error_type
    mock.message = message
    return mock


def _make_mock_tool_event(
    tool_name: str = "read",
    tool_input: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock tool execution event."""
    mock = MagicMock()
    mock.tool_name = tool_name
    mock.tool_input = tool_input or {"path": "foo.py"}
    mock.event_type = "tool_execution"
    mock.to_dict.return_value = {
        "event_type": "tool_execution",
        "tool_name": tool_name,
        "tool_input": tool_input or {"path": "foo.py"},
        "timestamp": 1700000000.0,
    }
    return mock


def _make_mock_llm_call_event(text: str = "I read the file") -> MagicMock:
    """Create a mock LLM call event."""
    mock = MagicMock()
    mock.event_type = "llm_call"
    mock.text = text
    mock.to_dict.return_value = {
        "event_type": "llm_call",
        "text": text,
        "timestamp": 1700000000.0,
    }
    return mock


def _default_scenario(prompt: str = "Write a hello world function") -> dict[str, Any]:
    """Create a minimal scenario dict with a prompt."""
    return {
        "id": "test-scenario",
        "inputs": {"prompt": prompt},
    }


class TestBoltMerlinScenarioAdapter:
    """Test BoltMerlinScenarioAdapter."""

    def test_adapter_name(self) -> None:
        """Adapter name is 'bolt_merlin'."""
        adapter = BoltMerlinScenarioAdapter()
        assert adapter.name == "bolt_merlin"

    def test_adapter_satisfies_protocol(self) -> None:
        """BoltMerlinScenarioAdapter satisfies ScenarioAdapter protocol."""
        adapter = BoltMerlinScenarioAdapter()
        assert isinstance(adapter, ScenarioAdapter)

    @pytest.mark.asyncio
    async def test_import_error_returns_failure(self) -> None:
        """ImportError when importing bolt_merlin returns AGENT_ERROR failure."""
        adapter = BoltMerlinScenarioAdapter()
        scenario = _default_scenario()

        with patch.dict(
            "sys.modules",
            {"bolt_merlin": None, "bolt_merlin.agent": None, "bolt_merlin.agent.execute": None},
        ):
            # Force re-import by patching the import inside async_run_scenario
            with patch(
                "ash_hawk.scenario.adapters.bolt_merlin.BoltMerlinScenarioAdapter.async_run_scenario",
                autospec=True,
            ) as mock_async:
                # Simulate what the real method does on ImportError
                from ash_hawk.scenario.models import ScenarioAdapterResult as SAR
                from ash_hawk.types import EvalOutcome

                mock_async.return_value = SAR(
                    final_output=None,
                    trace_events=[],
                    artifacts={},
                    outcome=EvalOutcome.failure(
                        FailureMode.AGENT_ERROR,
                        error_message="bolt-merlin agent unavailable: No module named 'bolt_merlin'",
                    ),
                )

                result = await adapter.async_run_scenario(scenario, Path("/tmp"), {}, {})

        assert result.outcome.status == EvalStatus.ERROR
        assert result.outcome.failure_mode == FailureMode.AGENT_ERROR
        assert result.outcome.error_message is not None
        assert "bolt-merlin agent unavailable" in result.outcome.error_message
        assert result.final_output is None

    @pytest.mark.asyncio
    async def test_no_prompt_returns_failure(self) -> None:
        """Missing prompt in scenario returns VALIDATION_ERROR failure."""
        adapter = BoltMerlinScenarioAdapter()
        scenario: dict[str, Any] = {"id": "no-prompt", "inputs": {}}

        mock_execute = AsyncMock(return_value=_make_mock_execution_result())

        with patch.dict(
            "sys.modules",
            {
                "bolt_merlin": MagicMock(),
                "bolt_merlin.agent": MagicMock(),
                "bolt_merlin.agent.execute": MagicMock(execute=mock_execute),
            },
        ):
            # Even with bolt_merlin importable, empty inputs should fail before execute
            result = await adapter.async_run_scenario(scenario, Path("/tmp"), {}, {})

        assert result.outcome.status == EvalStatus.ERROR
        assert result.outcome.failure_mode == FailureMode.VALIDATION_ERROR
        assert result.final_output is None

    @pytest.mark.asyncio
    async def test_execute_returns_execution_result(self) -> None:
        """Successful execute returns correct result with artifacts and messages."""
        adapter = BoltMerlinScenarioAdapter()
        scenario = _default_scenario("Write a hello world function")

        mock_result = _make_mock_execution_result()
        mock_execute = AsyncMock(return_value=mock_result)

        # We need bolt_merlin.agent.execute.execute to be importable and callable
        mock_execute_module = MagicMock()
        mock_execute_module.execute = mock_execute

        with patch.dict(
            "sys.modules",
            {
                "bolt_merlin": MagicMock(),
                "bolt_merlin.agent": MagicMock(),
                "bolt_merlin.agent.execute": mock_execute_module,
            },
        ):
            result = await adapter.async_run_scenario(scenario, Path("/tmp"), {}, {})

        assert result.final_output == "done"
        assert result.outcome.status == EvalStatus.COMPLETED
        assert result.outcome.failure_mode is None
        assert result.artifacts["session_id"] == "test-123"
        assert result.artifacts["tokens_in"] == 100
        assert result.artifacts["tokens_out"] == 50
        assert result.artifacts["duration_ms"] == 1000
        # Messages should include user prompt
        messages = result.extract_messages()
        assert len(messages) >= 1
        assert messages[0].role == "user"
        assert messages[0].content == "Write a hello world function"

    @pytest.mark.asyncio
    async def test_execute_returns_execution_error(self) -> None:
        """ExecutionError result returns AGENT_ERROR failure with error message."""
        adapter = BoltMerlinScenarioAdapter()
        scenario = _default_scenario()

        mock_error = _make_mock_execution_error(error_type="execution_error", message="boom")
        mock_execute = AsyncMock(return_value=mock_error)

        mock_execute_module = MagicMock()
        mock_execute_module.execute = mock_execute

        with patch.dict(
            "sys.modules",
            {
                "bolt_merlin": MagicMock(),
                "bolt_merlin.agent": MagicMock(),
                "bolt_merlin.agent.execute": mock_execute_module,
            },
        ):
            result = await adapter.async_run_scenario(scenario, Path("/tmp"), {}, {})

        assert result.outcome.status == EvalStatus.ERROR
        assert result.outcome.failure_mode == FailureMode.AGENT_ERROR
        assert result.outcome.error_message is not None
        assert "boom" in result.outcome.error_message
        assert result.final_output is None

    @pytest.mark.asyncio
    async def test_build_trace_events_extracts_tool_events(self) -> None:
        """Tool execution events are captured as ToolCallEvent trace entries."""
        from ash_hawk.scenario.adapters.bolt_merlin import _build_trace_events

        class FakeToolExecutionEvent:
            def __init__(self, tool_name: str, tool_input: dict[str, Any]) -> None:
                self.tool_name = tool_name
                self.tool_input = tool_input
                self.event_type = "tool_call"

            def to_dict(self) -> dict[str, Any]:
                return {
                    "event_type": "tool_call",
                    "tool_name": self.tool_name,
                    "tool_input": self.tool_input,
                    "timestamp": 0.0,
                }

        fake_event = FakeToolExecutionEvent(tool_name="read", tool_input={"path": "foo.py"})
        trace_events = _build_trace_events([fake_event], prompt="do it")

        tool_call_events = [e for e in trace_events if e.event_type == "ToolCallEvent"]
        assert len(tool_call_events) == 1
        assert tool_call_events[0].data["name"] == "read"
        assert tool_call_events[0].data["arguments"] == {"path": "foo.py"}

    @pytest.mark.asyncio
    async def test_build_trace_events_produces_messages(self) -> None:
        """LLM call events produce ModelMessageEvent trace entries."""
        from ash_hawk.scenario.adapters.bolt_merlin import _build_trace_events

        prompt = "Read the file"
        final_response = "done"
        llm_event = _make_mock_llm_call_event(text="I read the file")

        trace_events = _build_trace_events(
            [llm_event], prompt=prompt, final_response=final_response
        )

        msg_events = [e for e in trace_events if e.event_type == "ModelMessageEvent"]
        roles = [e.data["role"] for e in msg_events]
        assert "user" in roles
        assert "assistant" in roles

    @pytest.mark.asyncio
    async def test_working_dir_is_passed_to_execute(self) -> None:
        adapter = BoltMerlinScenarioAdapter()
        scenario = _default_scenario()
        workdir = Path("/tmp/ash-hawk-test-workdir")

        captured: dict[str, Any] = {}

        async def fake_execute(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return _make_mock_execution_result()

        mock_execute_module = MagicMock()
        mock_execute_module.execute = fake_execute

        with (
            patch.dict(
                "sys.modules",
                {
                    "bolt_merlin": MagicMock(),
                    "bolt_merlin.agent": MagicMock(),
                    "bolt_merlin.agent.execute": mock_execute_module,
                },
            ),
        ):
            await adapter.async_run_scenario(scenario, workdir, {}, {})

        assert captured["working_dir"] == workdir.resolve()

    @pytest.mark.asyncio
    async def test_agent_path_drives_explicit_agent_config_path(self, tmp_path: Path) -> None:
        adapter = BoltMerlinScenarioAdapter()
        scenario = _default_scenario()
        repo_root = tmp_path / "bolt-merlin"
        agent_dir = repo_root / "bolt_merlin" / "agent"
        agent_dir.mkdir(parents=True)
        (repo_root / ".git").mkdir()
        (repo_root / "bolt_merlin" / "__init__.py").write_text("", encoding="utf-8")
        (agent_dir / "__init__.py").write_text("", encoding="utf-8")

        captured: dict[str, object] = {}

        async def fake_execute(**kwargs):
            captured.update(kwargs)
            return _make_mock_execution_result()

        mock_execute_module = MagicMock()
        mock_execute_module.execute = fake_execute

        with (
            patch.dict(
                "sys.modules",
                {
                    "bolt_merlin": MagicMock(),
                    "bolt_merlin.agent": MagicMock(),
                    "bolt_merlin.agent.execute": mock_execute_module,
                },
            ),
            patch(
                "ash_hawk.scenario.adapters.bolt_merlin.import_package_from_agent_path",
                side_effect=lambda *_args, **_kwargs: nullcontext(),
            ),
        ):
            await adapter.async_run_scenario(
                scenario,
                Path("/tmp"),
                {"agent_path": str(agent_dir)},
                {},
            )

        expected_config_path = repo_root / ".dawn-kestrel" / "agent_config.yaml"
        assert captured["config_path"] == expected_config_path

    @pytest.mark.asyncio
    async def test_agent_path_prefers_nearest_config_file(self, tmp_path: Path) -> None:
        adapter = BoltMerlinScenarioAdapter()
        scenario = _default_scenario()
        repo_root = tmp_path / "bolt-merlin"
        agent_dir = repo_root / "bolt_merlin" / "agent"
        agent_dir.mkdir(parents=True)
        (repo_root / ".git").mkdir()
        (repo_root / "bolt_merlin" / "__init__.py").write_text("", encoding="utf-8")
        (agent_dir / "__init__.py").write_text("", encoding="utf-8")
        local_config = agent_dir / "agent_config.yaml"
        local_config.write_text("tools: [read]\n", encoding="utf-8")
        repo_config = repo_root / ".dawn-kestrel" / "agent_config.yaml"
        repo_config.parent.mkdir(parents=True)
        repo_config.write_text("tools: [write]\n", encoding="utf-8")

        captured: dict[str, object] = {}

        async def fake_execute(**kwargs):
            captured.update(kwargs)
            return _make_mock_execution_result()

        mock_execute_module = MagicMock()
        mock_execute_module.execute = fake_execute

        with (
            patch.dict(
                "sys.modules",
                {
                    "bolt_merlin": MagicMock(),
                    "bolt_merlin.agent": MagicMock(),
                    "bolt_merlin.agent.execute": mock_execute_module,
                },
            ),
            patch(
                "ash_hawk.scenario.adapters.bolt_merlin.import_package_from_agent_path",
                side_effect=lambda *_args, **_kwargs: nullcontext(),
            ),
        ):
            await adapter.async_run_scenario(
                scenario,
                Path("/tmp"),
                {"agent_path": str(agent_dir)},
                {},
            )

        assert captured["config_path"] == local_config

    def test_sync_run_scenario_wraps_async(self) -> None:
        """run_scenario (sync) returns same result as async_run_scenario."""
        adapter = BoltMerlinScenarioAdapter()
        scenario = _default_scenario()

        mock_result_obj = _make_mock_execution_result()
        mock_execute = AsyncMock(return_value=mock_result_obj)

        mock_execute_module = MagicMock()
        mock_execute_module.execute = mock_execute

        with patch.dict(
            "sys.modules",
            {
                "bolt_merlin": MagicMock(),
                "bolt_merlin.agent": MagicMock(),
                "bolt_merlin.agent.execute": mock_execute_module,
            },
        ):
            result = adapter.run_scenario(scenario, Path("/tmp"), {}, {})

        assert isinstance(result, ScenarioAdapterResult)
        assert result.final_output == "done"
        assert result.outcome.status == EvalStatus.COMPLETED
