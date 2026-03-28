"""Tests for DawnKestrelAgentRunner."""

import time
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ash_hawk.agents import DawnKestrelAgentRunner
from ash_hawk.policy import PolicyEnforcer
from ash_hawk.types import (
    EvalOutcome,
    EvalStatus,
    EvalTask,
    EvalTranscript,
    FailureMode,
    TokenUsage,
    ToolSurfacePolicy,
)


@pytest.fixture
def sample_policy():
    return ToolSurfacePolicy(
        allowed_tools=["read*", "write*"],
        denied_tools=["*bash*"],
        timeout_seconds=60.0,
    )


@pytest.fixture
def sample_task():
    return EvalTask(
        id="task-001",
        description="Test task",
        input="What is 2+2?",
        expected_output="4",
    )


@pytest.fixture
def sample_task_with_dict_input():
    return EvalTask(
        id="task-002",
        description="Task with dict input",
        input={"prompt": "Hello world", "context": "test"},
    )


@pytest.fixture
def policy_enforcer(sample_policy):
    return PolicyEnforcer(sample_policy)


class TestDawnKestrelAgentRunnerInit:
    def test_init_with_required_params(self):
        runner = DawnKestrelAgentRunner(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
        )
        assert runner._provider == "anthropic"
        assert runner._model == "claude-3-5-sonnet-20241022"
        assert runner._kwargs == {}
        assert runner._client is None

    def test_init_with_kwargs(self):
        runner = DawnKestrelAgentRunner(
            provider="zai",
            model="glm-4",
            temperature=0.7,
            max_tokens=1000,
        )
        assert runner._provider == "zai"
        assert runner._model == "glm-4"
        assert runner._kwargs == {"temperature": 0.7, "max_tokens": 1000}

    def test_init_with_mcp_servers(self):
        runner = DawnKestrelAgentRunner(
            provider="anthropic",
            model="claude-3-5-sonnet",
            mcp_servers=[
                {
                    "name": "note-lark",
                    "command": "note-lark-mcp-stdio",
                    "args": ["--stdio"],
                    "env": {"NOTE_LARK_PROJECT": "ash-hawk"},
                }
            ],
            temperature=0.2,
        )

        assert len(runner._mcp_servers) == 1
        assert runner._mcp_servers[0].name == "note-lark"
        assert runner._mcp_servers[0].command == "note-lark-mcp-stdio"
        assert runner._mcp_servers[0].args == ["--stdio"]
        assert runner._mcp_servers[0].env == {"NOTE_LARK_PROJECT": "ash-hawk"}
        assert runner._kwargs == {"temperature": 0.2}


class TestDawnKestrelAgentRunnerProtocol:
    def test_implements_agent_runner_protocol(self):
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")
        assert hasattr(runner, "run")
        assert callable(runner.run)

    def test_run_is_async(self):
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")
        import asyncio

        assert asyncio.iscoroutinefunction(runner.run)


class TestDawnKestrelAgentRunnerClientConfig:
    def test_get_client_uses_env_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")
        monkeypatch.setenv("ASH_HAWK_LLM_TIMEOUT_SECONDS", "420")

        mock_llm_client_type = MagicMock(return_value=object())
        mock_settings = MagicMock()
        mock_settings.get_api_key_for_provider.return_value = None
        mock_settings_module = MagicMock()
        mock_settings_module.get_settings.return_value = mock_settings
        mock_llm_module = MagicMock()
        mock_llm_module.LLMClient = mock_llm_client_type

        def _import_module(name: str) -> Any:
            if name == "dawn_kestrel.core.settings":
                return mock_settings_module
            if name == "dawn_kestrel.llm.client":
                return mock_llm_module
            raise AssertionError(f"Unexpected module import: {name}")

        with patch(
            "ash_hawk.agents.dawn_kestrel.importlib.import_module", side_effect=_import_module
        ):
            runner._get_client()

        called_kwargs = mock_llm_client_type.call_args.kwargs
        assert called_kwargs["timeout_seconds"] == 420.0

    def test_get_client_prefers_runner_timeout_over_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = DawnKestrelAgentRunner(
            provider="anthropic",
            model="claude-3-5-sonnet",
            timeout_seconds=600,
        )
        monkeypatch.setenv("ASH_HAWK_LLM_TIMEOUT_SECONDS", "420")

        mock_llm_client_type = MagicMock(return_value=object())
        mock_settings = MagicMock()
        mock_settings.get_api_key_for_provider.return_value = None
        mock_settings_module = MagicMock()
        mock_settings_module.get_settings.return_value = mock_settings
        mock_llm_module = MagicMock()
        mock_llm_module.LLMClient = mock_llm_client_type

        def _import_module(name: str) -> Any:
            if name == "dawn_kestrel.core.settings":
                return mock_settings_module
            if name == "dawn_kestrel.llm.client":
                return mock_llm_module
            raise AssertionError(f"Unexpected module import: {name}")

        with patch(
            "ash_hawk.agents.dawn_kestrel.importlib.import_module", side_effect=_import_module
        ):
            runner._get_client()

        called_kwargs = mock_llm_client_type.call_args.kwargs
        assert called_kwargs["timeout_seconds"] == 600.0


class TestDawnKestrelAgentRunnerExtraction:
    def test_extract_token_usage_from_object(self):
        runner = DawnKestrelAgentRunner(provider="test", model="test")

        class MockUsage:
            input = 100
            output = 50
            reasoning = 25
            cache_read = 10
            cache_write = 5

        class MockResponse:
            usage = MockUsage()

        usage = runner._extract_token_usage(MockResponse())
        assert usage.input == 100
        assert usage.output == 50
        assert usage.reasoning == 25
        assert usage.cache_read == 10
        assert usage.cache_write == 5

    def test_extract_token_usage_from_dict(self):
        runner = DawnKestrelAgentRunner(provider="test", model="test")

        mock_response = {
            "usage": {
                "input": 200,
                "output": 100,
            }
        }

        usage = runner._extract_token_usage(mock_response)
        assert usage.input == 200
        assert usage.output == 100

    def test_extract_token_usage_empty(self):
        runner = DawnKestrelAgentRunner(provider="test", model="test")

        class MockResponse:
            usage = None

        usage = runner._extract_token_usage(MockResponse())
        assert usage.input == 0
        assert usage.output == 0

    def test_extract_cost_from_object(self):
        runner = DawnKestrelAgentRunner(provider="test", model="test")

        class MockResponse:
            cost = Decimal("0.05")

        cost = runner._extract_cost(MockResponse())
        assert cost == 0.05

    def test_extract_cost_from_dict(self):
        runner = DawnKestrelAgentRunner(provider="test", model="test")

        mock_response = {"cost": 0.03}
        cost = runner._extract_cost(mock_response)
        assert cost == 0.03

    def test_extract_cost_none(self):
        runner = DawnKestrelAgentRunner(provider="test", model="test")

        class MockResponse:
            pass

        cost = runner._extract_cost(MockResponse())
        assert cost == 0.0

    def test_extract_agent_response_from_text(self):
        runner = DawnKestrelAgentRunner(provider="test", model="test")

        class MockResponse:
            text = "Hello world"

        response = runner._extract_agent_response(MockResponse())
        assert response == "Hello world"

    def test_extract_agent_response_from_dict(self):
        runner = DawnKestrelAgentRunner(provider="test", model="test")

        mock_response = {"text": "Test response"}
        response = runner._extract_agent_response(mock_response)
        assert response == "Test response"


class TestDawnKestrelAgentRunnerRun:
    @pytest.mark.asyncio
    async def test_run_returns_correct_types(self, sample_task, policy_enforcer):
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")

        class MockUsage:
            input = 10
            output = 5

        class MockResponse:
            text = "Test response"
            messages = []
            tool_calls = []
            usage = MockUsage()
            cost = Decimal("0.001")

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=MockResponse())

        mock_registry = MagicMock()
        mock_registry.tools = {}
        mock_registry.get_all = AsyncMock(return_value={})

        with patch.object(runner, "_get_client", return_value=mock_client):
            with patch.object(runner, "_create_filtered_registry", return_value=mock_registry):
                transcript, outcome = await runner.run(
                    task=sample_task,
                    policy_enforcer=policy_enforcer,
                    config={},
                )

        assert isinstance(transcript, EvalTranscript)
        assert isinstance(outcome, EvalOutcome)
        assert outcome.status == EvalStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_captures_token_usage(self, sample_task, policy_enforcer):
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")

        class MockUsage:
            input = 150
            output = 75

        class MockResponse:
            text = "Test"
            messages = []
            tool_calls = []
            usage = MockUsage()
            cost = Decimal("0.01")

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=MockResponse())

        mock_registry = MagicMock()
        mock_registry.tools = {}
        mock_registry.get_all = AsyncMock(return_value={})

        with patch.object(runner, "_get_client", return_value=mock_client):
            with patch.object(runner, "_create_filtered_registry", return_value=mock_registry):
                transcript, outcome = await runner.run(
                    task=sample_task,
                    policy_enforcer=policy_enforcer,
                    config={},
                )

        assert transcript.token_usage.input == 150
        assert transcript.token_usage.output == 75
        assert transcript.cost_usd == 0.01

    @pytest.mark.asyncio
    async def test_run_handles_dict_input(self, sample_task_with_dict_input, policy_enforcer):
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")

        class MockResponse:
            text = "Response"
            messages = []
            tool_calls = []
            usage = None
            cost = 0.0

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=MockResponse())

        mock_registry = MagicMock()
        mock_registry.tools = {}
        mock_registry.get_all = AsyncMock(return_value={})

        with patch.object(runner, "_get_client", return_value=mock_client):
            with patch.object(runner, "_create_filtered_registry", return_value=mock_registry):
                transcript, outcome = await runner.run(
                    task=sample_task_with_dict_input,
                    policy_enforcer=policy_enforcer,
                    config={},
                )

        assert outcome.status == EvalStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_handles_import_error(self, sample_task, policy_enforcer):
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")

        def raise_import_error():
            raise ImportError("dawn-kestrel not installed")

        with patch.object(runner, "_get_client", side_effect=raise_import_error):
            transcript, outcome = await runner.run(
                task=sample_task,
                policy_enforcer=policy_enforcer,
                config={},
            )

        assert outcome.status == EvalStatus.ERROR
        assert outcome.failure_mode == FailureMode.AGENT_ERROR
        assert "dawn-kestrel not installed" in outcome.error_message

    @pytest.mark.asyncio
    async def test_run_handles_general_exception(self, sample_task, policy_enforcer):
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")

        def raise_error():
            raise RuntimeError("API error")

        with patch.object(runner, "_get_client", side_effect=raise_error):
            transcript, outcome = await runner.run(
                task=sample_task,
                policy_enforcer=policy_enforcer,
                config={},
            )

        assert outcome.status == EvalStatus.ERROR
        assert outcome.failure_mode == FailureMode.AGENT_ERROR
        assert "API error" in outcome.error_message
        assert transcript.error_trace is not None

    @pytest.mark.asyncio
    async def test_run_records_duration(self, sample_task, policy_enforcer):
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")

        class MockResponse:
            text = "Test"
            messages = []
            tool_calls = []
            usage = None
            cost = 0.0

        mock_client = MagicMock()

        async def slow_complete(*args, **kwargs):
            time.sleep(0.01)
            return MockResponse()

        mock_client.complete = slow_complete

        mock_registry = MagicMock()
        mock_registry.tools = {}
        mock_registry.get_all = AsyncMock(return_value={})

        with patch.object(runner, "_get_client", return_value=mock_client):
            with patch.object(runner, "_create_filtered_registry", return_value=mock_registry):
                transcript, outcome = await runner.run(
                    task=sample_task,
                    policy_enforcer=policy_enforcer,
                    config={},
                )

        assert transcript.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_run_uses_complete_method(self, sample_task, policy_enforcer):
        runner = DawnKestrelAgentRunner(
            provider="anthropic",
            model="claude-3-5-sonnet",
            temperature=0.5,
        )

        class MockResponse:
            text = "Test"
            messages = []
            tool_calls = []
            usage = None
            cost = 0.0

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=MockResponse())

        mock_registry = MagicMock()
        mock_registry.tools = {}
        mock_registry.get_all = AsyncMock(return_value={})

        with patch.object(runner, "_get_client", return_value=mock_client):
            with patch.object(runner, "_create_filtered_registry", return_value=mock_registry):
                await runner.run(
                    task=sample_task,
                    policy_enforcer=policy_enforcer,
                    config={"max_tokens": 500},
                )

        mock_client.complete.assert_called_once()
        call_args = mock_client.complete.call_args
        assert call_args is not None


class TestDawnKestrelAgentRunnerPolicyIntegration:
    def test_create_filtered_registry_uses_tool_registry(self, policy_enforcer):
        runner = DawnKestrelAgentRunner(provider="test", model="test")

        mock_registry = MagicMock()
        mock_registry.tools = {
            "read_file": MagicMock(),
            "read_dir": MagicMock(),
            "write_file": MagicMock(),
            "execute_bash": MagicMock(),
        }

        filtered = runner._create_filtered_registry(policy_enforcer, mock_registry)

        assert isinstance(filtered.tools, dict)

    def test_create_filtered_registry_supports_allowlist_alias(self, policy_enforcer):
        runner = DawnKestrelAgentRunner(provider="test", model="test")
        mock_registry = MagicMock()
        mock_registry.tools = {}

        captured: dict[str, object] = {}

        class LegacyToolPermissionFilter:
            def __init__(self, tool_registry=None, allowlist=None, denylist=None):
                captured["tool_registry"] = tool_registry
                captured["allowlist"] = allowlist
                captured["denylist"] = denylist

            def get_filtered_registry(self):
                return captured["tool_registry"]

        with patch(
            "dawn_kestrel.tools.permission_filter.ToolPermissionFilter",
            LegacyToolPermissionFilter,
        ):
            runner._create_filtered_registry(policy_enforcer, mock_registry)

        assert captured["allowlist"] == ["read*", "write*"]
        assert captured["denylist"] == ["*bash*"]

    def test_create_filtered_registry_supports_permissions_only(self, policy_enforcer):
        runner = DawnKestrelAgentRunner(provider="test", model="test")
        mock_registry = MagicMock()
        mock_registry.tools = {}

        captured: dict[str, object] = {}

        class PermissionsOnlyToolPermissionFilter:
            def __init__(self, permissions=None, tool_registry=None):
                captured["permissions"] = permissions
                captured["tool_registry"] = tool_registry

            def get_filtered_registry(self):
                return captured["tool_registry"]

        with patch(
            "dawn_kestrel.tools.permission_filter.ToolPermissionFilter",
            PermissionsOnlyToolPermissionFilter,
        ):
            runner._create_filtered_registry(policy_enforcer, mock_registry)

        permissions = captured["permissions"]
        assert isinstance(permissions, list)
        assert {
            "permission": "read*",
            "pattern": "*",
            "action": "allow",
        } in permissions
        assert {
            "permission": "*bash*",
            "pattern": "*",
            "action": "deny",
        } in permissions

    @pytest.mark.asyncio
    async def test_run_passes_policy_to_registry_creation(self, sample_task, sample_policy):
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")
        enforcer = PolicyEnforcer(sample_policy)

        class MockResponse:
            text = "Test"
            messages = []
            tool_calls = []
            usage = None
            cost = 0.0

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=MockResponse())

        captured_enforcer = None

        def capture_enforcer(
            pe,
            base_registry=None,
            allowed_tools_override=None,
            denied_tools_override=None,
        ):
            nonlocal captured_enforcer
            captured_enforcer = pe
            mock_result = MagicMock()
            mock_result.tools = {}
            mock_result.get_all = AsyncMock(return_value={})
            return mock_result

        with patch.object(runner, "_get_client", return_value=mock_client):
            with patch.object(runner, "_create_filtered_registry", side_effect=capture_enforcer):
                await runner.run(
                    task=sample_task,
                    policy_enforcer=enforcer,
                    config={},
                )

        assert captured_enforcer is enforcer

    @pytest.mark.asyncio
    async def test_run_ignores_empty_policy_snapshot_allowed_tools_override(self, sample_policy):
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")
        enforcer = PolicyEnforcer(sample_policy)
        task = EvalTask(
            id="task-003",
            description="Task with empty policy snapshot override",
            input={
                "prompt": "Do the thing",
                "policy_snapshot": {
                    "allowed_tools": [],
                    "denied_tools": [],
                },
            },
        )

        class MockResponse:
            text = "Test"
            messages = []
            tool_calls = []
            usage = None
            cost = 0.0

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=MockResponse())

        captured_allowed: list[str] | None = None
        captured_denied: list[str] | None = None

        def capture_overrides(
            _policy_enforcer,
            base_registry=None,
            allowed_tools_override=None,
            denied_tools_override=None,
        ):
            del base_registry
            nonlocal captured_allowed, captured_denied
            captured_allowed = allowed_tools_override
            captured_denied = denied_tools_override
            mock_result = MagicMock()
            mock_result.tools = {}
            mock_result.get_all = AsyncMock(return_value={})
            return mock_result

        with patch.object(runner, "_get_client", return_value=mock_client):
            with patch.object(runner, "_create_filtered_registry", side_effect=capture_overrides):
                await runner.run(task=task, policy_enforcer=enforcer, config={})

        assert captured_allowed is None
        assert captured_denied is None


class TestDawnKestrelAgentRunnerMCPIntegration:
    @pytest.mark.asyncio
    async def test_run_registers_and_executes_mcp_tools(self, sample_task):
        class MockResponseToolCall:
            text = 'note-lark_notes_search(query="prefs")'
            messages = []
            tool_calls = []
            usage = None
            cost = 0.0

        class MockResponseFinal:
            text = "Done"
            messages = []
            tool_calls = []
            usage = None
            cost = 0.0

        class FakeMcpClient:
            instances: list["FakeMcpClient"] = []

            def __init__(self, config):
                self.config = config
                self.calls = []
                self.started = False
                self.closed = False
                FakeMcpClient.instances.append(self)

            async def start(self):
                self.started = True

            async def list_tools(self):
                return [
                    {
                        "name": "notes_search",
                        "description": "Search notes",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                            },
                            "required": ["query"],
                        },
                    }
                ]

            async def call_tool(self, tool_name, arguments):
                self.calls.append((tool_name, arguments))
                query = arguments.get("query", "")
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"MCP tool {tool_name} called with {query}",
                        }
                    ]
                }

            async def close(self):
                self.closed = True

        runner = DawnKestrelAgentRunner(
            provider="anthropic",
            model="claude-3-5-sonnet",
            mcp_servers=[
                {
                    "name": "note-lark",
                    "command": "note-lark-mcp-stdio",
                }
            ],
        )

        policy = ToolSurfacePolicy(
            allowed_tools=["note-lark_*"],
            timeout_seconds=60.0,
        )
        enforcer = PolicyEnforcer(policy)

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(side_effect=[MockResponseToolCall(), MockResponseFinal()])

        with patch.object(runner, "_get_client", return_value=mock_client):
            with patch("ash_hawk.agents.dawn_kestrel._McpStdioClient", FakeMcpClient):
                with patch.object(
                    runner,
                    "_create_filtered_registry",
                    side_effect=lambda policy_enforcer, base_registry=None, **kwargs: base_registry,
                ):
                    transcript, outcome = await runner.run(
                        task=sample_task,
                        policy_enforcer=enforcer,
                        config={},
                    )

        assert outcome.status == EvalStatus.COMPLETED
        assert len(transcript.tool_calls) == 1
        assert transcript.tool_calls[0]["tool"] == "note-lark_notes_search"
        assert "MCP tool notes_search called with prefs" in transcript.tool_calls[0]["output"]

        assert len(FakeMcpClient.instances) == 1
        client_instance = FakeMcpClient.instances[0]
        assert client_instance.started is True
        assert client_instance.closed is True
        assert client_instance.calls == [("notes_search", {"query": "prefs"})]
