"""Tests for ash_hawk.scenario.trial module (absorbed from execution/)."""

import asyncio

import pytest

from ash_hawk.policy import PolicyEnforcer
from ash_hawk.scenario import AgentRunner, TrialExecutor
from ash_hawk.storage import StorageBackend, StoredTrial
from ash_hawk.types import (
    EvalOutcome,
    EvalStatus,
    EvalTask,
    EvalTranscript,
    EvalTrial,
    FailureMode,
    GraderSpec,
    RunEnvelope,
    TokenUsage,
    ToolSurfacePolicy,
    TrialEnvelope,
    TrialResult,
)


@pytest.fixture
def mock_storage():
    """Create a mock storage backend."""

    class MockStorage:
        def __init__(self):
            self.saved_trials = []
            self.suites = {}
            self.run_envelopes = {}

        async def save_suite(self, suite):
            self.suites[suite.id] = suite

        async def load_suite(self, suite_id):
            return self.suites.get(suite_id)

        async def save_run_envelope(self, suite_id, envelope):
            self.run_envelopes[(suite_id, envelope.run_id)] = envelope

        async def load_run_envelope(self, suite_id, run_id):
            return self.run_envelopes.get((suite_id, run_id))

        async def save_trial(self, suite_id, run_id, trial, envelope, policy):
            self.saved_trials.append(StoredTrial(trial=trial, envelope=envelope, policy=policy))

        async def load_trial(self, suite_id, run_id, trial_id):
            for stored in self.saved_trials:
                if stored.trial.id == trial_id:
                    return stored
            return None

        async def list_runs(self, suite_id):
            return list(run_id for (sid, run_id) in self.run_envelopes.keys() if sid == suite_id)

        async def list_suites(self):
            return list(self.suites.keys())

        async def save_summary(self, suite_id, run_id, summary):
            pass

        async def load_summary(self, suite_id, run_id):
            return None

    return MockStorage()


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
def sample_run_envelope():
    return RunEnvelope(
        run_id="run-001",
        suite_id="suite-001",
        suite_hash="abc123",
        harness_version="0.1.0",
        agent_name="test-agent",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        tool_policy_hash="policy-hash-123",
        python_version="3.11.0",
        os_info="macos",
        created_at="2024-01-01T00:00:00Z",
    )


@pytest.fixture
def mock_agent_runner():
    async def runner(
        task: EvalTask, policy_enforcer: PolicyEnforcer, config: dict[str, object]
    ) -> tuple[EvalTranscript, EvalOutcome]:
        return EvalTranscript(agent_response="Mock response"), EvalOutcome.success()

    return runner


@pytest.fixture
def executor(mock_storage, sample_policy, mock_agent_runner):
    return TrialExecutor(mock_storage, sample_policy, agent_runner=mock_agent_runner)


class TestTrialExecutorInit:
    """Test TrialExecutor initialization."""

    def test_init_with_required_params(self, mock_storage, sample_policy, mock_agent_runner):
        executor = TrialExecutor(mock_storage, sample_policy, agent_runner=mock_agent_runner)
        assert executor.policy == sample_policy

    def test_init_with_custom_runner(self, mock_storage, sample_policy):
        async def custom_runner(
            task: EvalTask, enforcer: PolicyEnforcer, config: dict
        ) -> tuple[EvalTranscript, EvalOutcome]:
            return EvalTranscript(), EvalOutcome.success()

        executor = TrialExecutor(mock_storage, sample_policy, agent_runner=custom_runner)
        assert executor._agent_runner is not None

    def test_requires_explicit_runner(self, mock_storage, sample_policy):
        with pytest.raises(TypeError):
            TrialExecutor(mock_storage, sample_policy, agent_runner=None)  # type: ignore[arg-type]


class TestTrialExecutorExecute:
    """Test TrialExecutor.execute method."""

    @pytest.mark.asyncio
    async def test_successful_execution(
        self, executor, sample_task, sample_run_envelope, mock_storage
    ):
        result = await executor.execute(
            task=sample_task,
            agent_config={"model": "claude-3-5-sonnet-20241022"},
            run_envelope=sample_run_envelope,
        )

        assert isinstance(result, TrialResult)
        assert result.outcome.status == EvalStatus.COMPLETED
        assert result.outcome.failure_mode is None
        assert result.transcript is not None
        assert len(mock_storage.saved_trials) == 1

    @pytest.mark.asyncio
    async def test_trial_result_has_valid_trial_id(
        self, executor, sample_task, sample_run_envelope
    ):
        result = await executor.execute(
            task=sample_task,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        assert result.trial_id.startswith("trial-")
        assert len(result.trial_id) == len("trial-") + 8

    @pytest.mark.asyncio
    async def test_trial_envelope_created(
        self, executor, sample_task, sample_run_envelope, mock_storage
    ):
        await executor.execute(
            task=sample_task,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        stored = mock_storage.saved_trials[0]
        assert isinstance(stored.envelope, TrialEnvelope)
        assert stored.envelope.run_id == sample_run_envelope.run_id
        assert stored.envelope.task_id == sample_task.id
        assert stored.envelope.started_at is not None
        assert stored.envelope.completed_at is not None

    @pytest.mark.asyncio
    async def test_policy_snapshot_in_envelope(
        self, executor, sample_task, sample_run_envelope, mock_storage, sample_policy
    ):
        await executor.execute(
            task=sample_task,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        stored = mock_storage.saved_trials[0]
        assert stored.envelope.policy_snapshot == sample_policy

    @pytest.mark.asyncio
    async def test_attempt_number_tracked(
        self, executor, sample_task, sample_run_envelope, mock_storage
    ):
        await executor.execute(
            task=sample_task,
            agent_config={},
            run_envelope=sample_run_envelope,
            attempt_number=3,
        )

        stored = mock_storage.saved_trials[0]
        assert stored.envelope.attempt_number == 3
        assert stored.trial.attempt_number == 3

    @pytest.mark.asyncio
    async def test_custom_agent_runner_used(
        self, mock_storage, sample_policy, sample_task, sample_run_envelope
    ):
        custom_transcript = EvalTranscript(
            messages=[{"role": "user", "content": "test"}],
            agent_response="Custom response",
            token_usage=TokenUsage(input=100, output=50),
        )
        custom_outcome = EvalOutcome.success()

        async def custom_runner(
            task: EvalTask, enforcer: PolicyEnforcer, config: dict
        ) -> tuple[EvalTranscript, EvalOutcome]:
            return custom_transcript, custom_outcome

        executor = TrialExecutor(mock_storage, sample_policy, agent_runner=custom_runner)
        result = await executor.execute(
            task=sample_task,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        assert result.transcript.agent_response == "Custom response"


class TestTrialExecutorTimeout:
    """Test TrialExecutor timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_creates_failure_outcome(
        self, mock_storage, sample_task, sample_run_envelope
    ):
        slow_policy = ToolSurfacePolicy(timeout_seconds=0.05)

        async def slow_runner(
            task: EvalTask, enforcer: PolicyEnforcer, config: dict
        ) -> tuple[EvalTranscript, EvalOutcome]:
            await asyncio.sleep(1)
            return EvalTranscript(), EvalOutcome.success()

        executor = TrialExecutor(mock_storage, slow_policy, agent_runner=slow_runner)

        result = await executor.execute(
            task=sample_task,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        assert result.outcome.status == EvalStatus.ERROR
        assert result.outcome.failure_mode == FailureMode.TIMEOUT
        assert result.outcome.error_message is not None
        assert "timed out" in result.outcome.error_message.lower()

    @pytest.mark.asyncio
    async def test_task_timeout_override(self, mock_storage, sample_run_envelope):
        policy = ToolSurfacePolicy(timeout_seconds=60.0)
        task_with_timeout = EvalTask(
            id="task-timeout",
            input="test",
            timeout_seconds=0.05,
        )

        async def slow_runner(
            task: EvalTask, enforcer: PolicyEnforcer, config: dict
        ) -> tuple[EvalTranscript, EvalOutcome]:
            await asyncio.sleep(1)
            return EvalTranscript(), EvalOutcome.success()

        executor = TrialExecutor(mock_storage, policy, agent_runner=slow_runner)

        result = await executor.execute(
            task=task_with_timeout,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        assert result.outcome.failure_mode == FailureMode.TIMEOUT
        assert result.outcome.error_message is not None
        assert "0.05" in result.outcome.error_message


class TestTrialExecutorCancellation:
    """Test TrialExecutor cancellation handling."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Cancellation test has event loop timing issues")
    async def test_cancellation_stores_partial_results(
        self, mock_storage, sample_policy, sample_task, sample_run_envelope
    ):
        started = False

        async def cancellable_runner(
            task: EvalTask, enforcer: PolicyEnforcer, config: dict
        ) -> tuple[EvalTranscript, EvalOutcome]:
            nonlocal started
            started = True
            await asyncio.sleep(10)
            return EvalTranscript(), EvalOutcome.success()

        executor = TrialExecutor(mock_storage, sample_policy, agent_runner=cancellable_runner)

        execute_task = asyncio.create_task(
            executor.execute(
                task=sample_task,
                agent_config={},
                run_envelope=sample_run_envelope,
            )
        )

        while not started:
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.01)
        execute_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await execute_task

        assert len(mock_storage.saved_trials) == 1
        stored = mock_storage.saved_trials[0]
        assert stored.trial.result.outcome.failure_mode == FailureMode.CRASH
        assert stored.trial.result.outcome.error_message is not None
        assert "cancelled" in stored.trial.result.outcome.error_message.lower()


class TestTrialExecutorExceptions:
    """Test TrialExecutor exception handling."""

    @pytest.mark.asyncio
    async def test_agent_exception_creates_failure_outcome(
        self, mock_storage, sample_policy, sample_task, sample_run_envelope
    ):
        async def failing_runner(
            task: EvalTask, enforcer: PolicyEnforcer, config: dict
        ) -> tuple[EvalTranscript, EvalOutcome]:
            raise ValueError("Agent crashed")

        executor = TrialExecutor(mock_storage, sample_policy, agent_runner=failing_runner)

        result = await executor.execute(
            task=sample_task,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        assert result.outcome.status == EvalStatus.ERROR
        assert result.outcome.failure_mode == FailureMode.AGENT_ERROR
        assert result.outcome.error_message is not None
        assert "Agent crashed" in result.outcome.error_message
        assert result.transcript.error_trace is not None

    @pytest.mark.asyncio
    async def test_exception_traceback_captured(
        self, mock_storage, sample_policy, sample_task, sample_run_envelope
    ):
        async def failing_runner(
            task: EvalTask, enforcer: PolicyEnforcer, config: dict
        ) -> tuple[EvalTranscript, EvalOutcome]:
            raise RuntimeError("Detailed error")

        executor = TrialExecutor(mock_storage, sample_policy, agent_runner=failing_runner)

        result = await executor.execute(
            task=sample_task,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        assert result.transcript.error_trace is not None
        assert "RuntimeError" in result.transcript.error_trace
        assert "Detailed error" in result.transcript.error_trace


class TestTrialExecutorPolicyIntegration:
    """Test TrialExecutor policy enforcement integration."""

    @pytest.mark.asyncio
    async def test_policy_enforcer_passed_to_runner(
        self, mock_storage, sample_policy, sample_task, sample_run_envelope
    ):
        received_enforcer = None

        async def capturing_runner(
            task: EvalTask, policy_enforcer: PolicyEnforcer, config: dict
        ) -> tuple[EvalTranscript, EvalOutcome]:
            nonlocal received_enforcer
            received_enforcer = policy_enforcer
            return EvalTranscript(), EvalOutcome.success()

        executor = TrialExecutor(mock_storage, sample_policy, agent_runner=capturing_runner)

        await executor.execute(
            task=sample_task,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        assert received_enforcer is not None
        assert isinstance(received_enforcer, PolicyEnforcer)
        assert received_enforcer.policy == sample_policy


class TestAgentRunnerProtocol:
    """Test AgentRunner protocol compliance."""

    def test_function_runner_implements_protocol(self):
        from ash_hawk.scenario.trial import _FunctionRunner

        async def sample(
            task: EvalTask, enforcer: PolicyEnforcer, config: dict[str, object]
        ) -> tuple[EvalTranscript, EvalOutcome]:
            return EvalTranscript(), EvalOutcome.success()

        runner = _FunctionRunner(sample)
        assert hasattr(runner, "run")
        assert callable(runner.run)

    @pytest.mark.asyncio
    async def test_function_runner_returns_correct_types(self):
        from ash_hawk.scenario.trial import _FunctionRunner

        async def sample(
            task: EvalTask, enforcer: PolicyEnforcer, config: dict[str, object]
        ) -> tuple[EvalTranscript, EvalOutcome]:
            return EvalTranscript(agent_response="ok"), EvalOutcome.success()

        runner = _FunctionRunner(sample)
        policy = ToolSurfacePolicy()
        task = EvalTask(id="test", input="test")
        enforcer = PolicyEnforcer(policy)

        transcript, outcome = await runner.run(task, enforcer, {})

        assert isinstance(transcript, EvalTranscript)
        assert isinstance(outcome, EvalOutcome)
        assert outcome.status == EvalStatus.COMPLETED


class TestTrialResultFields:
    """Test TrialResult has all required fields."""

    @pytest.mark.asyncio
    async def test_result_has_transcript_with_timing(
        self, executor, sample_task, sample_run_envelope
    ):
        result = await executor.execute(
            task=sample_task,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        assert result.transcript.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_stored_trial_has_complete_data(
        self, executor, sample_task, sample_run_envelope, mock_storage
    ):
        await executor.execute(
            task=sample_task,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        stored = mock_storage.saved_trials[0]
        assert stored.trial.id is not None
        assert stored.trial.task_id == sample_task.id
        assert stored.trial.status in [EvalStatus.COMPLETED, EvalStatus.ERROR]
        assert stored.trial.input_snapshot == sample_task.input
        assert stored.trial.result is not None
        assert stored.trial.envelope is not None
        assert stored.policy == executor.policy


class TestTrialExecutorGraders:
    @pytest.mark.asyncio
    async def test_runs_string_match_grader(self, mock_storage, sample_policy, sample_run_envelope):
        task = EvalTask(
            id="task-grade-1",
            input="What is 2 + 2?",
            grader_specs=[
                GraderSpec(
                    grader_type="string_match",
                    config={"expected": "4", "mode": "exact"},
                    weight=1.0,
                    required=True,
                )
            ],
        )

        async def custom_runner(
            task: EvalTask, enforcer: PolicyEnforcer, config: dict
        ) -> tuple[EvalTranscript, EvalOutcome]:
            return EvalTranscript(agent_response="4"), EvalOutcome.success()

        executor = TrialExecutor(mock_storage, sample_policy, agent_runner=custom_runner)
        result = await executor.execute(
            task=task, agent_config={}, run_envelope=sample_run_envelope
        )

        assert len(result.grader_results) == 1
        assert result.grader_results[0].grader_type == "string_match"
        assert result.grader_results[0].passed is True
        assert result.aggregate_score == 1.0
        assert result.aggregate_passed is True

    @pytest.mark.asyncio
    async def test_required_grader_failure_vetoes_aggregate_pass(
        self, mock_storage, sample_policy, sample_run_envelope
    ):
        task = EvalTask(
            id="task-grade-2",
            input="test",
            grader_specs=[
                GraderSpec(
                    grader_type="string_match",
                    config={"expected": "A", "mode": "exact"},
                    weight=1.0,
                    required=True,
                ),
                GraderSpec(
                    grader_type="string_match",
                    config={"expected": "B", "mode": "exact"},
                    weight=1.0,
                    required=False,
                ),
            ],
        )

        async def custom_runner(
            task: EvalTask, enforcer: PolicyEnforcer, config: dict
        ) -> tuple[EvalTranscript, EvalOutcome]:
            return EvalTranscript(agent_response="B"), EvalOutcome.success()

        executor = TrialExecutor(mock_storage, sample_policy, agent_runner=custom_runner)
        result = await executor.execute(
            task=task, agent_config={}, run_envelope=sample_run_envelope
        )

        assert len(result.grader_results) == 2
        assert result.aggregate_score == 0.5
        assert result.aggregate_passed is False

    @pytest.mark.asyncio
    async def test_unknown_grader_type_returns_error_result(
        self, mock_storage, sample_policy, sample_run_envelope
    ):
        task = EvalTask(
            id="task-grade-3",
            input="test",
            grader_specs=[
                GraderSpec(grader_type="unknown_grader", config={}, weight=1.0, required=False)
            ],
        )

        async def custom_runner(
            task: EvalTask, enforcer: PolicyEnforcer, config: dict
        ) -> tuple[EvalTranscript, EvalOutcome]:
            return EvalTranscript(agent_response="anything"), EvalOutcome.success()

        executor = TrialExecutor(mock_storage, sample_policy, agent_runner=custom_runner)
        result = await executor.execute(
            task=task, agent_config={}, run_envelope=sample_run_envelope
        )

        assert len(result.grader_results) == 1
        assert result.grader_results[0].passed is False
        assert result.grader_results[0].error_message is not None
        assert "Unknown grader type" in result.grader_results[0].error_message

    @pytest.mark.asyncio
    async def test_composite_grader_executes_nested_specs(
        self, mock_storage, sample_policy, sample_run_envelope
    ):
        task = EvalTask(
            id="task-grade-4",
            input="test",
            grader_specs=[
                GraderSpec(
                    grader_type="composite",
                    config={
                        "mode": "weighted",
                        "weights": [0.7, 0.3],
                        "graders": [
                            {
                                "grader_type": "string_match",
                                "config": {"expected": "hello", "mode": "exact"},
                                "weight": 1.0,
                                "required": False,
                            },
                            {
                                "grader_type": "string_match",
                                "config": {"expected": "bye", "mode": "exact"},
                                "weight": 1.0,
                                "required": False,
                            },
                        ],
                    },
                    weight=1.0,
                    required=False,
                )
            ],
        )

        async def custom_runner(
            task: EvalTask, enforcer: PolicyEnforcer, config: dict
        ) -> tuple[EvalTranscript, EvalOutcome]:
            return EvalTranscript(agent_response="hello"), EvalOutcome.success()

        executor = TrialExecutor(mock_storage, sample_policy, agent_runner=custom_runner)
        result = await executor.execute(
            task=task, agent_config={}, run_envelope=sample_run_envelope
        )

        assert len(result.grader_results) == 1
        assert result.grader_results[0].grader_type == "composite"
        assert result.grader_results[0].passed is True
        assert result.aggregate_passed is True
        assert result.aggregate_score == pytest.approx(0.7)
