"""Tests for DawnKestrelAgentRunner."""

import time
from decimal import Decimal
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
    ToolSurfacePolicy,
    TokenUsage,
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


class TestDawnKestrelAgentRunnerProtocol:
    def test_implements_agent_runner_protocol(self):
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")
        assert hasattr(runner, "run")
        assert callable(runner.run)

    def test_run_is_async(self):
        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-3-5-sonnet")
        import asyncio

        assert asyncio.iscoroutinefunction(runner.run)


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

        def capture_enforcer(pe, base=None):
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
