# type-hygiene: skip-file
"""Tests for DawnKestrelAgentRunner."""

import time
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ash_hawk.agents import DawnKestrelAgentRunner
from ash_hawk.agents import dawn_kestrel as dawn_kestrel_module


def _make_import_mock(
    agent_loop: Any = None,
    agent_types: Any = None,
) -> Any:
    """Create a mock importlib.import_module for run() tests."""
    mock_agent_loop = agent_loop or MagicMock(
        run_agent=AsyncMock(
            return_value=MagicMock(
                response=MagicMock(text="Test"),
                error=None,
                iterations=1,
                total_usage={},
                session=None,
            )
        )
    )
    mock_agent_types = agent_types or MagicMock(LoopConfig=MagicMock(return_value=MagicMock()))

    def _import_module(name: str) -> Any:
        if name == "dawn_kestrel.agent.loop":
            return mock_agent_loop
        if name == "dawn_kestrel.agent.types":
            return mock_agent_types
        if name == "dawn_kestrel.tools.registry":
            return MagicMock(ToolRegistry=MagicMock)
        if name == "dawn_kestrel.provider.llm_client":
            return MagicMock(LLMClient=MagicMock)
        if name == "dawn_kestrel.base.config":
            return MagicMock()
        if name == "dawn_kestrel.tools.framework":
            return MagicMock()
        raise AssertionError(f"Unexpected module import: {name}")

    return _import_module


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
        mock_llm_module = MagicMock()
        mock_llm_module.LLMClient = mock_llm_client_type

        def _import_module(name: str) -> Any:
            if name == "dawn_kestrel.provider.llm_client":
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
        mock_llm_module = MagicMock()
        mock_llm_module.LLMClient = mock_llm_client_type

        def _import_module(name: str) -> Any:
            if name == "dawn_kestrel.provider.llm_client":
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

        class MockResult:
            response = MagicMock(text="Test response")
            error = None
            iterations = 1
            total_usage = {"input": 10, "output": 5, "reasoning": 0}
            session = None

        async def mock_run_agent(*args: Any, **kwargs: Any) -> Any:
            return MockResult()

        mock_agent_module = MagicMock()
        mock_agent_module.run_agent = mock_run_agent
        mock_types_module = MagicMock()
        mock_types_module.LoopConfig = MagicMock(return_value=MagicMock())

        def _import_module(name: str) -> Any:
            if name == "dawn_kestrel.agent.loop":
                return mock_agent_module
            if name == "dawn_kestrel.agent.types":
                return mock_types_module
            if name == "dawn_kestrel.tools.registry":
                return MagicMock(ToolRegistry=MagicMock)
            if name == "dawn_kestrel.provider.llm_client":
                return MagicMock(LLMClient=MagicMock)
            if name == "dawn_kestrel.base.config":
                return MagicMock()
            if name == "dawn_kestrel.tools.framework":
                return MagicMock()
            raise AssertionError(f"Unexpected module import: {name}")

        mock_registry = MagicMock()
        mock_registry.tools = {}
        mock_registry.get_all = AsyncMock(return_value={})

        with patch.object(runner, "_get_client", return_value=MagicMock()):
            with patch.object(runner, "_create_filtered_registry", return_value=mock_registry):
                with patch(
                    "ash_hawk.agents.dawn_kestrel.importlib.import_module",
                    side_effect=_import_module,
                ):
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

        class MockResult:
            response = MagicMock(text="Test")
            error = None
            iterations = 1
            total_usage = {"input": 150, "output": 75, "reasoning": 0}
            session = None

        async def mock_run_agent(*args: Any, **kwargs: Any) -> Any:
            return MockResult()

        mock_agent_module = MagicMock()
        mock_agent_module.run_agent = mock_run_agent
        mock_types_module = MagicMock()
        mock_types_module.LoopConfig = MagicMock(return_value=MagicMock())

        def _import_module(name: str) -> Any:
            if name == "dawn_kestrel.agent.loop":
                return mock_agent_module
            if name == "dawn_kestrel.agent.types":
                return mock_types_module
            if name == "dawn_kestrel.tools.registry":
                return MagicMock(ToolRegistry=MagicMock)
            if name == "dawn_kestrel.provider.llm_client":
                return MagicMock(LLMClient=MagicMock)
            if name == "dawn_kestrel.base.config":
                return MagicMock()
            if name == "dawn_kestrel.tools.framework":
                return MagicMock()
            raise AssertionError(f"Unexpected module import: {name}")

        mock_registry = MagicMock()
        mock_registry.tools = {}
        mock_registry.get_all = AsyncMock(return_value={})

        with patch.object(runner, "_get_client", return_value=MagicMock()):
            with patch.object(runner, "_create_filtered_registry", return_value=mock_registry):
                with patch(
                    "ash_hawk.agents.dawn_kestrel.importlib.import_module",
                    side_effect=_import_module,
                ):
                    transcript, outcome = await runner.run(
                        task=sample_task,
                        policy_enforcer=policy_enforcer,
                        config={},
                    )

        assert transcript.token_usage.input == 150
        assert transcript.token_usage.output == 75

    @pytest.mark.asyncio
    async def test_run_handles_dict_input(self, sample_task_with_dict_input, policy_enforcer):
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")

        class MockResult:
            response = MagicMock(text="Response")
            error = None
            iterations = 1
            total_usage = {}
            session = None

        async def mock_run_agent(*args: Any, **kwargs: Any) -> Any:
            return MockResult()

        mock_agent_module = MagicMock()
        mock_agent_module.run_agent = mock_run_agent
        mock_types_module = MagicMock()
        mock_types_module.LoopConfig = MagicMock(return_value=MagicMock())

        def _import_module(name: str) -> Any:
            if name == "dawn_kestrel.agent.loop":
                return mock_agent_module
            if name == "dawn_kestrel.agent.types":
                return mock_types_module
            if name == "dawn_kestrel.tools.registry":
                return MagicMock(ToolRegistry=MagicMock)
            if name == "dawn_kestrel.provider.llm_client":
                return MagicMock(LLMClient=MagicMock)
            if name == "dawn_kestrel.base.config":
                return MagicMock()
            if name == "dawn_kestrel.tools.framework":
                return MagicMock()
            raise AssertionError(f"Unexpected module import: {name}")

        mock_registry = MagicMock()
        mock_registry.tools = {}
        mock_registry.get_all = AsyncMock(return_value={})

        with patch.object(runner, "_get_client", return_value=MagicMock()):
            with patch.object(runner, "_create_filtered_registry", return_value=mock_registry):
                with patch(
                    "ash_hawk.agents.dawn_kestrel.importlib.import_module",
                    side_effect=_import_module,
                ):
                    transcript, outcome = await runner.run(
                        task=sample_task_with_dict_input,
                        policy_enforcer=policy_enforcer,
                        config={},
                    )

        assert outcome.status == EvalStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_handles_import_error(self, sample_task, policy_enforcer):
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")

        with patch.object(runner, "_get_client", return_value=MagicMock()):
            with patch.object(runner, "_create_filtered_registry", return_value=MagicMock()):
                with patch(
                    "ash_hawk.agents.dawn_kestrel.importlib.import_module",
                    side_effect=ImportError("dawn-kestrel not installed"),
                ):
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

        async def raise_on_run(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("API error")

        mock_agent_module = MagicMock()
        mock_agent_module.run_agent = raise_on_run
        mock_types_module = MagicMock()
        mock_types_module.LoopConfig = MagicMock(return_value=MagicMock())

        def _import_module(name: str) -> Any:
            if name == "dawn_kestrel.agent.loop":
                return mock_agent_module
            if name == "dawn_kestrel.agent.types":
                return mock_types_module
            if name == "dawn_kestrel.tools.registry":
                return MagicMock(ToolRegistry=MagicMock)
            if name == "dawn_kestrel.provider.llm_client":
                return MagicMock(LLMClient=MagicMock)
            if name == "dawn_kestrel.base.config":
                return MagicMock()
            if name == "dawn_kestrel.tools.framework":
                return MagicMock()
            raise AssertionError(f"Unexpected module import: {name}")

        mock_registry = MagicMock()
        mock_registry.tools = {}
        mock_registry.get_all = AsyncMock(return_value={})

        with patch.object(runner, "_get_client", return_value=MagicMock()):
            with patch.object(runner, "_create_filtered_registry", return_value=mock_registry):
                with patch(
                    "ash_hawk.agents.dawn_kestrel.importlib.import_module",
                    side_effect=_import_module,
                ):
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

        class MockResult:
            response = MagicMock(text="Test")
            error = None
            iterations = 1
            total_usage = {}
            session = None

        async def slow_run_agent(*args: Any, **kwargs: Any) -> Any:
            time.sleep(0.01)
            return MockResult()

        mock_agent_module = MagicMock()
        mock_agent_module.run_agent = slow_run_agent
        mock_types_module = MagicMock()
        mock_types_module.LoopConfig = MagicMock(return_value=MagicMock())

        def _import_module(name: str) -> Any:
            if name == "dawn_kestrel.agent.loop":
                return mock_agent_module
            if name == "dawn_kestrel.agent.types":
                return mock_types_module
            if name == "dawn_kestrel.tools.registry":
                return MagicMock(ToolRegistry=MagicMock)
            if name == "dawn_kestrel.provider.llm_client":
                return MagicMock(LLMClient=MagicMock)
            if name == "dawn_kestrel.base.config":
                return MagicMock()
            if name == "dawn_kestrel.tools.framework":
                return MagicMock()
            raise AssertionError(f"Unexpected module import: {name}")

        mock_registry = MagicMock()
        mock_registry.tools = {}
        mock_registry.get_all = AsyncMock(return_value={})

        with patch.object(runner, "_get_client", return_value=MagicMock()):
            with patch.object(runner, "_create_filtered_registry", return_value=mock_registry):
                with patch(
                    "ash_hawk.agents.dawn_kestrel.importlib.import_module",
                    side_effect=_import_module,
                ):
                    transcript, outcome = await runner.run(
                        task=sample_task,
                        policy_enforcer=policy_enforcer,
                        config={},
                    )

        assert transcript.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_run_calls_run_agent(self, sample_task, policy_enforcer):
        runner = DawnKestrelAgentRunner(
            provider="anthropic",
            model="claude-3-5-sonnet",
            temperature=0.5,
        )

        class MockResult:
            response = MagicMock(text="Test")
            error = None
            iterations = 1
            total_usage = {}
            session = None

        mock_run_agent = AsyncMock(return_value=MockResult())
        mock_agent_module = MagicMock()
        mock_agent_module.run_agent = mock_run_agent
        mock_types_module = MagicMock()
        mock_types_module.LoopConfig = MagicMock(return_value=MagicMock())

        def _import_module(name: str) -> Any:
            if name == "dawn_kestrel.agent.loop":
                return mock_agent_module
            if name == "dawn_kestrel.agent.types":
                return mock_types_module
            if name == "dawn_kestrel.tools.registry":
                return MagicMock(ToolRegistry=MagicMock)
            if name == "dawn_kestrel.provider.llm_client":
                return MagicMock(LLMClient=MagicMock)
            if name == "dawn_kestrel.base.config":
                return MagicMock()
            if name == "dawn_kestrel.tools.framework":
                return MagicMock()
            raise AssertionError(f"Unexpected module import: {name}")

        mock_registry = MagicMock()
        mock_registry.tools = {}
        mock_registry.get_all = AsyncMock(return_value={})

        with patch.object(runner, "_get_client", return_value=MagicMock()):
            with patch.object(runner, "_create_filtered_registry", return_value=mock_registry):
                with patch(
                    "ash_hawk.agents.dawn_kestrel.importlib.import_module",
                    side_effect=_import_module,
                ):
                    await runner.run(
                        task=sample_task,
                        policy_enforcer=policy_enforcer,
                        config={"max_iterations": 5},
                    )

        mock_run_agent.assert_called_once()
        call_kwargs = mock_run_agent.call_args.kwargs
        assert call_kwargs.get("config") is not None or call_kwargs.get("tools") is not None


class TestDawnKestrelAgentRunnerPolicyIntegration:
    def test_create_filtered_registry_returns_base_registry(self, policy_enforcer):
        runner = DawnKestrelAgentRunner(provider="test", model="test")

        mock_registry = MagicMock()
        mock_registry.tools = {
            "read_file": MagicMock(),
            "read_dir": MagicMock(),
            "write_file": MagicMock(),
            "execute_bash": MagicMock(),
        }

        filtered = runner._create_filtered_registry(
            policy_enforcer,
            mock_registry,
            use_policy_filters=True,
        )

        assert filtered is mock_registry

    def test_create_filtered_registry_creates_default_if_none(self, policy_enforcer):
        runner = DawnKestrelAgentRunner(provider="test", model="test")

        filtered = runner._create_filtered_registry(
            policy_enforcer,
            base_registry=None,
        )

        assert hasattr(filtered, "tool_metadata")

    @pytest.mark.asyncio
    async def test_run_passes_policy_to_registry_creation(self, sample_task, sample_policy):
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")
        enforcer = PolicyEnforcer(sample_policy)

        class MockResult:
            response = MagicMock(text="Test")
            error = None
            iterations = 1
            total_usage = {}
            session = None

        async def mock_run_agent(*args: Any, **kwargs: Any) -> Any:
            return MockResult()

        mock_agent_module = MagicMock()
        mock_agent_module.run_agent = mock_run_agent
        mock_types_module = MagicMock()
        mock_types_module.LoopConfig = MagicMock(return_value=MagicMock())

        def _import_module(name: str) -> Any:
            if name == "dawn_kestrel.agent.loop":
                return mock_agent_module
            if name == "dawn_kestrel.agent.types":
                return mock_types_module
            if name == "dawn_kestrel.tools.registry":
                return MagicMock(ToolRegistry=MagicMock)
            if name == "dawn_kestrel.provider.llm_client":
                return MagicMock(LLMClient=MagicMock)
            if name == "dawn_kestrel.base.config":
                return MagicMock()
            if name == "dawn_kestrel.tools.framework":
                return MagicMock()
            raise AssertionError(f"Unexpected module import: {name}")

        captured_enforcer = None
        captured_use_policy_filters = None

        def capture_enforcer(
            pe,
            base_registry=None,
            allowed_tools_override=None,
            denied_tools_override=None,
            use_policy_filters=False,
        ):
            nonlocal captured_enforcer
            nonlocal captured_use_policy_filters
            del allowed_tools_override, denied_tools_override
            captured_enforcer = pe
            captured_use_policy_filters = use_policy_filters
            mock_result = MagicMock()
            mock_result.tools = {}
            mock_result.get_all = AsyncMock(return_value={})
            return mock_result

        with patch.object(runner, "_get_client", return_value=MagicMock()):
            with patch.object(runner, "_create_filtered_registry", side_effect=capture_enforcer):
                with patch(
                    "ash_hawk.agents.dawn_kestrel.importlib.import_module",
                    side_effect=_import_module,
                ):
                    await runner.run(
                        task=sample_task,
                        policy_enforcer=enforcer,
                        config={},
                    )

        assert captured_enforcer is enforcer
        assert captured_use_policy_filters is False

    @pytest.mark.asyncio
    async def test_run_ignores_policy_snapshot_tool_overrides(self, sample_policy):
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")
        enforcer = PolicyEnforcer(sample_policy)
        task = EvalTask(
            id="task-003",
            description="Task with policy snapshot override",
            input={
                "prompt": "Do the thing",
                "policy_snapshot": {
                    "allowed_tools": ["read"],
                    "denied_tools": ["bash"],
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
            use_policy_filters=False,
        ):
            del base_registry
            nonlocal captured_allowed, captured_denied
            nonlocal captured_use_policy_filters
            captured_allowed = allowed_tools_override
            captured_denied = denied_tools_override
            captured_use_policy_filters = use_policy_filters
            mock_result = MagicMock()
            mock_result.tools = {}
            mock_result.get_all = AsyncMock(return_value={})
            return mock_result

        captured_use_policy_filters: bool | None = None

        with patch.object(runner, "_get_client", return_value=mock_client):
            with patch.object(runner, "_create_filtered_registry", side_effect=capture_overrides):
                await runner.run(task=task, policy_enforcer=enforcer, config={})

        assert captured_allowed is None
        assert captured_denied is None
        assert captured_use_policy_filters is False

    @pytest.mark.asyncio
    async def test_run_ignores_enforce_tool_policy_config(self, sample_task, sample_policy):
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

        captured_use_policy_filters: bool | None = None

        def capture_filter_flag(
            _policy_enforcer,
            base_registry=None,
            allowed_tools_override=None,
            denied_tools_override=None,
            use_policy_filters=False,
        ):
            del base_registry, allowed_tools_override, denied_tools_override
            nonlocal captured_use_policy_filters
            captured_use_policy_filters = use_policy_filters
            mock_result = MagicMock()
            mock_result.tools = {}
            mock_result.get_all = AsyncMock(return_value={})
            return mock_result

        with patch.object(runner, "_get_client", return_value=mock_client):
            with patch.object(runner, "_create_filtered_registry", side_effect=capture_filter_flag):
                await runner.run(
                    task=sample_task,
                    policy_enforcer=enforcer,
                    config={"enforce_tool_policy": True},
                )

        assert captured_use_policy_filters is False

    def test_ensure_eval_command_allowlist_is_noop_without_security_module(self) -> None:
        with patch.object(
            dawn_kestrel_module.importlib,
            "import_module",
            return_value=MagicMock(),
        ):
            dawn_kestrel_module._ensure_eval_command_allowlist()

    def test_ensure_eval_command_allowlist_handles_import_error(self) -> None:
        with patch.object(
            dawn_kestrel_module.importlib,
            "import_module",
            side_effect=ImportError("no module"),
        ):
            dawn_kestrel_module._ensure_eval_command_allowlist()


class TestDawnKestrelAgentRunnerMCPIntegration:
    @pytest.mark.asyncio
    async def test_run_registers_mcp_tools_in_registry(self, sample_task):
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

        class FakeMcpClient:
            instances: list["FakeMcpClient"] = []

            def __init__(self, config: Any) -> None:
                self.config = config
                self.started = False
                self.closed = False
                FakeMcpClient.instances.append(self)

            async def start(self) -> None:
                self.started = True

            async def list_tools(self) -> list[dict[str, Any]]:
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

            async def close(self) -> None:
                self.closed = True

        class MockResult:
            response = MagicMock(text="Done")
            error = None
            iterations = 1
            total_usage = {}
            session = None

        captured_registry: Any = None

        async def mock_run_agent(*args: Any, **kwargs: Any) -> Any:
            nonlocal captured_registry
            if "tools" in kwargs:
                captured_registry = kwargs["tools"]
            return MockResult()

        mock_agent_module = MagicMock()
        mock_agent_module.run_agent = mock_run_agent
        mock_types_module = MagicMock()
        mock_types_module.LoopConfig = MagicMock(return_value=MagicMock())

        def _import_module(name: str) -> Any:
            if name == "dawn_kestrel.agent.loop":
                return mock_agent_module
            if name == "dawn_kestrel.agent.types":
                return mock_types_module
            if name == "dawn_kestrel.tools.registry":
                return MagicMock(ToolRegistry=MagicMock)
            if name == "dawn_kestrel.provider.llm_client":
                return MagicMock(LLMClient=MagicMock)
            if name == "dawn_kestrel.base.config":
                return MagicMock()
            if name == "dawn_kestrel.tools.framework":
                return MagicMock()
            raise AssertionError(f"Unexpected module import: {name}")

        with patch.object(runner, "_get_client", return_value=MagicMock()):
            with patch("ash_hawk.agents.dawn_kestrel._McpStdioClient", FakeMcpClient):
                with patch(
                    "ash_hawk.agents.dawn_kestrel.importlib.import_module",
                    side_effect=_import_module,
                ):
                    transcript, outcome = await runner.run(
                        task=sample_task,
                        policy_enforcer=enforcer,
                        config={},
                    )

        assert outcome.status == EvalStatus.COMPLETED
        assert len(FakeMcpClient.instances) == 1
        assert FakeMcpClient.instances[0].started is True
        assert FakeMcpClient.instances[0].closed is True


class TestTextToolCallParsingSmoke:
    def test_extract_text_tool_calls_skips_logging_noise(self) -> None:
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")

        text = (
            'logging.info("db connected")\n'
            'todo_update(item="task-1", status="completed")\n'
            'read(filePath="auth.py")\n'
        )

        calls = runner._extract_text_tool_calls(text)

        assert all(call["tool"] != "info" for call in calls)
        assert any(call["tool"] == "todo_update" for call in calls)
        assert any(
            call["tool"] == "read" and call["input"].get("filePath") == "auth.py" for call in calls
        )

    def test_coerce_todo_alias_calls_create_and_update(self) -> None:
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")
        trial_id = "trial-smoke"

        created = runner._coerce_todo_alias_call(
            trial_id=trial_id,
            alias_name="todo_create",
            alias_input={"tasks": ["task a", "task b"]},
        )
        assert "todos" in created
        assert len(created["todos"]) == 2
        assert created["todos"][0]["state"] == "pending"

        updated = runner._coerce_todo_alias_call(
            trial_id=trial_id,
            alias_name="todo_update",
            alias_input={"item": "task a", "status": "in_progress"},
        )
        assert updated["todos"][0]["state"] == "in_progress"

    def test_extract_text_tool_calls_ignores_unknown_underscore_names(self) -> None:
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")

        text = 'validate_password("hunter2")\nread(filePath="auth.py")\n'

        calls = runner._extract_text_tool_calls(text)

        assert all(call["tool"] != "validate_password" for call in calls)
        assert any(call["tool"] == "read" for call in calls)

    @pytest.mark.asyncio
    async def test_execute_tool_calls_uses_todo_alias_only_with_todowrite(self) -> None:
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")

        class FakeToolContext:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeResult:
            def __init__(self) -> None:
                self.output = "ok"
                self.title = "done"
                self.metadata = {}

        class FakeTool:
            def __init__(self) -> None:
                self.last_input: dict[str, Any] | None = None

            async def execute(self, args: dict[str, Any], ctx: Any) -> FakeResult:
                del ctx
                self.last_input = args
                return FakeResult()

        class FakeRegistry:
            def __init__(self, tools: dict[str, Any]) -> None:
                self._tools = tools

            def get(self, name: str) -> Any | None:
                return self._tools.get(name)

        todo_update_tool = FakeTool()
        registry = FakeRegistry({"todo_update": todo_update_tool})

        fake_tools_module = MagicMock()
        fake_tools_module.ToolContext = FakeToolContext

        with patch.object(
            dawn_kestrel_module.importlib,
            "import_module",
            return_value=fake_tools_module,
        ):
            executed = await runner._execute_tool_calls(
                tool_calls=[
                    {
                        "tool": "todo_update",
                        "input": {"item": "task a", "status": "completed"},
                    }
                ],
                filtered_registry=registry,
                config={"workdir": "."},
                trial_id="trial-1",
            )

        assert len(executed) == 1
        assert "error" not in executed[0]
        assert todo_update_tool.last_input == {"item": "task a", "status": "completed"}
