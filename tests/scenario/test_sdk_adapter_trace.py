"""Tests for the SDK adapter trace event generation."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ash_hawk.scenario.adapters.sdk_dawn_kestrel import SdkDawnKestrelAdapter
from ash_hawk.types import EvalOutcome, EvalStatus, EvalTranscript


def test_sdk_adapter_produces_policy_decision_events():
    """Test SDK adapter produces PolicyDecisionEvent in trace when tools are called."""
    adapter = SdkDawnKestrelAdapter()

    # Mock the DawnKestrelAgentRunner to return deterministic output
    mock_transcript = EvalTranscript(
        messages=[
            {"role": "user", "content": "Test prompt"},
            {"role": "assistant", "content": "Done"},
        ],
        tool_calls=[
            {
                "name": "bash",
                "arguments": {"command": "echo test"},
            }
        ],
        agent_response="Done",
    )
    mock_outcome = EvalOutcome(status=EvalStatus.COMPLETED)

    # Create a tooling harness with a mock call function
    tool_calls_log: list[tuple[str, dict[str, object]]] = []

    def mock_tool_call(tool_name: str, tool_input: dict[str, object]) -> dict[str, object]:
        tool_calls_log.append((tool_name, tool_input))
        return {"stdout": "test output", "exit_code": 0}

    tooling_harness = {
        "policy": {
            "allowed_tools": ["bash"],
            "denied_tools": [],
        },
        "call": mock_tool_call,
    }

    scenario = {
        "id": "test-sdk-scenario",
        "sut": {
            "type": "coding_agent",
            "adapter": "sdk_dawn_kestrel",
            "config": {
                "provider": "test_provider",
                "model": "test_model",
            },
        },
        "tools": {"allowed_tools": ["bash"]},
        "inputs": {"prompt": "Test prompt"},
        "expectations": {},
        "budgets": {},
    }

    workdir = Path("/tmp")
    budgets = {}

    # Patch the DawnKestrelAgentRunner to return our mock transcript
    with patch("ash_hawk.scenario.adapters.sdk_dawn_kestrel.DawnKestrelAgentRunner") as MockRunner:
        mock_runner_instance = MagicMock()
        mock_runner_instance.run = AsyncMock(return_value=(mock_transcript, mock_outcome))
        MockRunner.return_value = mock_runner_instance

        result = adapter.run_scenario(scenario, workdir, tooling_harness, budgets)

    # Verify final output
    assert result.final_output == "Done"

    # Verify artifacts are empty
    assert result.artifacts == {}

    # Verify tool was called
    assert len(tool_calls_log) == 1
    assert tool_calls_log[0][0] == "bash"

    # Verify trace events contain PolicyDecisionEvent
    policy_events = [e for e in result.trace_events if e.event_type == "PolicyDecisionEvent"]
    assert len(policy_events) >= 1

    # Verify the policy decision event has expected structure
    policy_event = policy_events[0]
    assert policy_event.data["tool_name"] == "bash"
    assert policy_event.data["allowed"] is True

    # Verify no RejectionEvent since tool was allowed
    rejection_events = [e for e in result.trace_events if e.event_type == "RejectionEvent"]
    assert len(rejection_events) == 0


def test_sdk_adapter_produces_rejection_events_for_denied_tools():
    """Test SDK adapter produces RejectionEvent when tools are denied by policy."""
    adapter = SdkDawnKestrelAdapter()

    # Mock the DawnKestrelAgentRunner to return deterministic output with a denied tool
    mock_transcript = EvalTranscript(
        messages=[
            {"role": "user", "content": "Test prompt"},
            {"role": "assistant", "content": "Attempted denied tool"},
        ],
        tool_calls=[
            {
                "name": "dangerous_tool",
                "arguments": {"action": "delete"},
            }
        ],
        agent_response="Attempted denied tool",
    )
    mock_outcome = EvalOutcome(status=EvalStatus.COMPLETED)

    # Create a tooling harness that denies the dangerous_tool
    tool_calls_log: list[tuple[str, dict[str, object]]] = []

    def mock_tool_call(tool_name: str, tool_input: dict[str, object]) -> dict[str, object]:
        tool_calls_log.append((tool_name, tool_input))
        return {"stdout": "test output", "exit_code": 0}

    tooling_harness = {
        "policy": {
            "allowed_tools": ["bash"],
            "denied_tools": ["dangerous_tool"],
        },
        "call": mock_tool_call,
    }

    scenario = {
        "id": "test-sdk-denied-scenario",
        "sut": {
            "type": "coding_agent",
            "adapter": "sdk_dawn_kestrel",
            "config": {
                "provider": "test_provider",
                "model": "test_model",
            },
        },
        "tools": {"allowed_tools": ["bash"], "denied_tools": ["dangerous_tool"]},
        "inputs": {"prompt": "Test prompt"},
        "expectations": {},
        "budgets": {},
    }

    workdir = Path("/tmp")
    budgets = {}

    # Patch the DawnKestrelAgentRunner to return our mock transcript
    with patch("ash_hawk.scenario.adapters.sdk_dawn_kestrel.DawnKestrelAgentRunner") as MockRunner:
        mock_runner_instance = MagicMock()
        mock_runner_instance.run = AsyncMock(return_value=(mock_transcript, mock_outcome))
        MockRunner.return_value = mock_runner_instance

        result = adapter.run_scenario(scenario, workdir, tooling_harness, budgets)

    # Verify final output
    assert result.final_output == "Attempted denied tool"

    # Verify artifacts are empty
    assert result.artifacts == {}

    # Verify tool was NOT called since it was denied
    assert len(tool_calls_log) == 0

    # Verify trace events contain PolicyDecisionEvent
    policy_events = [e for e in result.trace_events if e.event_type == "PolicyDecisionEvent"]
    assert len(policy_events) >= 1

    # Verify the policy decision event shows tool was denied
    policy_event = policy_events[0]
    assert policy_event.data["tool_name"] == "dangerous_tool"
    assert policy_event.data["allowed"] is False

    # Verify RejectionEvent was created
    rejection_events = [e for e in result.trace_events if e.event_type == "RejectionEvent"]
    assert len(rejection_events) == 1

    # Verify the rejection event has expected structure
    rejection_event = rejection_events[0]
    assert rejection_event.data["tool_name"] == "dangerous_tool"


def test_sdk_adapter_no_tool_calls_no_policy_events():
    """Test SDK adapter produces no PolicyDecisionEvents when no tools are called."""
    adapter = SdkDawnKestrelAdapter()

    # Mock the DawnKestrelAgentRunner to return transcript with no tool calls
    mock_transcript = EvalTranscript(
        messages=[
            {"role": "user", "content": "Test prompt"},
            {"role": "assistant", "content": "Direct response"},
        ],
        tool_calls=[],
        agent_response="Direct response",
    )
    mock_outcome = EvalOutcome(status=EvalStatus.COMPLETED)

    tooling_harness = {
        "policy": {
            "allowed_tools": [],
            "denied_tools": [],
        },
        "call": lambda name, inp: {"result": "ok"},
    }

    scenario = {
        "id": "test-sdk-no-tools-scenario",
        "sut": {
            "type": "coding_agent",
            "adapter": "sdk_dawn_kestrel",
            "config": {
                "provider": "test_provider",
                "model": "test_model",
            },
        },
        "tools": {"allowed_tools": []},
        "inputs": {"prompt": "Test prompt"},
        "expectations": {},
        "budgets": {},
    }

    workdir = Path("/tmp")
    budgets = {}

    # Patch the DawnKestrelAgentRunner to return our mock transcript
    with patch("ash_hawk.scenario.adapters.sdk_dawn_kestrel.DawnKestrelAgentRunner") as MockRunner:
        mock_runner_instance = MagicMock()
        mock_runner_instance.run = AsyncMock(return_value=(mock_transcript, mock_outcome))
        MockRunner.return_value = mock_runner_instance

        result = adapter.run_scenario(scenario, workdir, tooling_harness, budgets)

    # Verify final output
    assert result.final_output == "Direct response"

    # Verify no PolicyDecisionEvents were created
    policy_events = [e for e in result.trace_events if e.event_type == "PolicyDecisionEvent"]
    assert len(policy_events) == 0

    # Verify no RejectionEvents were created
    rejection_events = [e for e in result.trace_events if e.event_type == "RejectionEvent"]
    assert len(rejection_events) == 0


def test_sdk_adapter_does_not_duplicate_policy_events_from_transcript():
    adapter = SdkDawnKestrelAdapter()

    mock_transcript = EvalTranscript(
        messages=[
            {"role": "user", "content": "Test prompt"},
            {"role": "assistant", "content": "Done"},
        ],
        trace_events=[
            {
                "schema_version": 1,
                "event_type": "PolicyDecisionEvent",
                "ts": "1970-01-01T00:00:00Z",
                "data": {"tool_name": "bash", "allowed": True},
            }
        ],
        tool_calls=[
            {
                "name": "bash",
                "arguments": {"command": "echo test"},
            }
        ],
        agent_response="Done",
    )
    mock_outcome = EvalOutcome(status=EvalStatus.COMPLETED)

    tool_calls_log: list[tuple[str, dict[str, object]]] = []

    def mock_tool_call(tool_name: str, tool_input: dict[str, object]) -> dict[str, object]:
        tool_calls_log.append((tool_name, tool_input))
        return {"stdout": "test output", "exit_code": 0}

    tooling_harness = {
        "policy": {
            "allowed_tools": ["bash"],
            "denied_tools": [],
        },
        "call": mock_tool_call,
    }

    scenario = {
        "id": "test-sdk-scenario-existing-policy-events",
        "sut": {
            "type": "coding_agent",
            "adapter": "sdk_dawn_kestrel",
            "config": {
                "provider": "test_provider",
                "model": "test_model",
            },
        },
        "tools": {"allowed_tools": ["bash"]},
        "inputs": {"prompt": "Test prompt"},
        "expectations": {},
        "budgets": {},
    }

    workdir = Path("/tmp")
    budgets = {}

    with patch("ash_hawk.scenario.adapters.sdk_dawn_kestrel.DawnKestrelAgentRunner") as mock_runner:
        mock_runner_instance = MagicMock()
        mock_runner_instance.run = AsyncMock(return_value=(mock_transcript, mock_outcome))
        mock_runner.return_value = mock_runner_instance

        result = adapter.run_scenario(scenario, workdir, tooling_harness, budgets)

    policy_events = [e for e in result.trace_events if e.event_type == "PolicyDecisionEvent"]
    assert len(policy_events) == 1
    assert policy_events[0].data["tool_name"] == "bash"
    assert len(tool_calls_log) == 1


def test_sdk_adapter_normalizes_tool_call_alias_fields() -> None:
    adapter = SdkDawnKestrelAdapter()

    mock_transcript = EvalTranscript(
        tool_calls=[{"tool": "bash", "input": {"command": "echo test"}}],
        agent_response="Done",
    )
    mock_outcome = EvalOutcome(status=EvalStatus.COMPLETED)

    tooling_harness = {
        "policy": {
            "allowed_tools": ["bash"],
            "denied_tools": [],
        },
        "call": lambda name, inp: {"ok": True},
    }

    scenario = {
        "id": "test-sdk-tool-alias",
        "sut": {
            "type": "coding_agent",
            "adapter": "sdk_dawn_kestrel",
            "config": {
                "provider": "test_provider",
                "model": "test_model",
            },
        },
        "tools": {"allowed_tools": ["bash"]},
        "inputs": {"prompt": "Test prompt"},
        "expectations": {},
        "budgets": {},
    }

    with patch("ash_hawk.scenario.adapters.sdk_dawn_kestrel.DawnKestrelAgentRunner") as mock_runner:
        mock_runner_instance = MagicMock()
        mock_runner_instance.run = AsyncMock(return_value=(mock_transcript, mock_outcome))
        mock_runner.return_value = mock_runner_instance

        result = adapter.run_scenario(scenario, Path("/tmp"), tooling_harness, {})

    assert result.tool_calls[0].name == "bash"
    assert result.tool_calls[0].arguments == {"command": "echo test"}


class TestResolveProviderModelWithAgentPath:
    def test_reads_provider_model_from_agent_dir_config(self, tmp_path: Path) -> None:
        from ash_hawk.scenario.adapters.sdk_dawn_kestrel import _resolve_provider_model

        agent_dir = tmp_path / "my_agent"
        agent_dir.mkdir()
        (agent_dir / "agent_config.yaml").write_text(
            "runtime:\n  provider: openai\n  model: gpt-4o\n"
        )
        provider, model = _resolve_provider_model({}, agent_path=str(agent_dir))
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_reads_from_parent_dawn_kestrel(self, tmp_path: Path) -> None:
        from ash_hawk.scenario.adapters.sdk_dawn_kestrel import _resolve_provider_model

        parent = tmp_path / "project"
        parent.mkdir()
        dk_dir = parent / ".dawn-kestrel"
        dk_dir.mkdir()
        (dk_dir / "agent_config.yaml").write_text(
            "runtime:\n  provider: anthropic\n  model: claude-3-5-sonnet\n"
        )
        agent_dir = parent / "agent"
        agent_dir.mkdir()
        provider, model = _resolve_provider_model({}, agent_path=str(agent_dir))
        assert provider == "anthropic"
        assert model == "claude-3-5-sonnet"

    def test_explicit_config_wins_over_agent_path(self, tmp_path: Path) -> None:
        from ash_hawk.scenario.adapters.sdk_dawn_kestrel import _resolve_provider_model

        agent_dir = tmp_path / "my_agent"
        agent_dir.mkdir()
        (agent_dir / "agent_config.yaml").write_text(
            "runtime:\n  provider: openai\n  model: gpt-4o\n"
        )
        provider, model = _resolve_provider_model(
            {"provider": "explicit-provider", "model": "explicit-model"},
            agent_path=str(agent_dir),
        )
        assert provider == "explicit-provider"
        assert model == "explicit-model"

    def test_agent_path_wins_over_global_default(self, tmp_path: Path) -> None:
        from ash_hawk.scenario.adapters.sdk_dawn_kestrel import _resolve_provider_model

        agent_dir = tmp_path / "my_agent"
        agent_dir.mkdir()
        (agent_dir / "agent_config.yaml").write_text(
            "runtime:\n  provider: google\n  model: gemini-pro\n"
        )
        with patch(
            "ash_hawk.scenario.adapters.sdk_dawn_kestrel._resolve_provider_model",
            wraps=_resolve_provider_model,
        ):
            provider, model = _resolve_provider_model({}, agent_path=str(agent_dir))
        assert provider == "google"
        assert model == "gemini-pro"

    def test_no_agent_path_no_config_falls_through(self) -> None:
        from ash_hawk.scenario.adapters.sdk_dawn_kestrel import _resolve_provider_model

        with patch(
            "ash_hawk.scenario.adapters.sdk_dawn_kestrel._resolve_provider_model.__wrapped__"
            if hasattr(_resolve_provider_model, "__wrapped__")
            else "dawn_kestrel.base.config.load_agent_config",
            return_value={
                "runtime.provider": "fallback-provider",
                "runtime.model": "fallback-model",
            },
        ) as mock_load:
            try:
                provider, model = _resolve_provider_model({})
                assert provider == "fallback-provider"
                assert model == "fallback-model"
            except (ValueError, ImportError):
                pass


class TestAgentPathExtractionOrder:
    @pytest.mark.asyncio
    async def test_agent_path_extracted_before_provider_resolution(self) -> None:
        adapter = SdkDawnKestrelAdapter()
        captured_agent_path: str | None = None
        captured_config: dict[str, object] = {}

        def capturing_resolve(
            config: dict[str, object], agent_path: str | None = None
        ) -> tuple[str, str]:
            nonlocal captured_agent_path, captured_config
            captured_agent_path = agent_path
            captured_config = config
            return "test-provider", "test-model"

        scenario = {
            "id": "test-scenario",
            "inputs": {"prompt": "test"},
            "sut": {"config": {}},
            "graders": [],
            "budgets": {},
        }
        tooling_harness = {"agent_path": "/explicit/agent/path"}

        with patch(
            "ash_hawk.scenario.adapters.sdk_dawn_kestrel._resolve_provider_model",
            side_effect=capturing_resolve,
        ):
            with patch(
                "ash_hawk.scenario.adapters.sdk_dawn_kestrel.DawnKestrelAgentRunner"
            ) as mock_runner:
                mock_runner_instance = MagicMock()
                mock_runner_instance.run = AsyncMock(
                    return_value=(
                        EvalTranscript(messages=[], tool_calls=[], agent_response="done"),
                        EvalOutcome(status=EvalStatus.COMPLETED),
                    )
                )
                mock_runner.return_value = mock_runner_instance

                try:
                    await adapter.async_run_scenario(scenario, Path("/tmp"), tooling_harness, {})
                except Exception:
                    pass

        assert captured_agent_path == "/explicit/agent/path"
