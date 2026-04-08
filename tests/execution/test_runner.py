"""Tests for ash_hawk.scenario.runner module (absorbed from execution/)."""

import asyncio

import pytest

from ash_hawk.config import EvalConfig
from ash_hawk.policy import PolicyEnforcer
from ash_hawk.scenario import EvalRunner, TrialExecutor
from ash_hawk.storage import StoredTrial
from ash_hawk.types import (
    EvalOutcome,
    EvalStatus,
    EvalSuite,
    EvalTask,
    EvalTranscript,
    RunEnvelope,
    SuiteMetrics,
    TokenUsage,
    ToolSurfacePolicy,
)


@pytest.fixture
def mock_storage():
    class MockStorage:
        def __init__(self):
            self.saved_trials = []
            self.suites = {}
            self.run_envelopes = {}
            self.summaries = {}

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
            self.summaries[(suite_id, run_id)] = summary

        async def load_summary(self, suite_id, run_id):
            return self.summaries.get((suite_id, run_id))

    return MockStorage()


@pytest.fixture
def sample_policy():
    return ToolSurfacePolicy(
        allowed_tools=["read*", "write*"],
        timeout_seconds=60.0,
    )


@pytest.fixture
def sample_config():
    return EvalConfig(parallelism=2)


@pytest.fixture
def sample_tasks():
    return [
        EvalTask(id="task-001", description="Task 1", input="What is 2+2?"),
        EvalTask(id="task-002", description="Task 2", input="What is 3+3?"),
        EvalTask(id="task-003", description="Task 3", input="What is 4+4?"),
    ]


@pytest.fixture
def sample_suite(sample_tasks):
    return EvalSuite(
        id="suite-001",
        name="Test Suite",
        description="A test suite for runner tests",
        tasks=sample_tasks,
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
        del task
        del policy_enforcer
        del config
        return (
            EvalTranscript(
                agent_response="Mock response",
                token_usage=TokenUsage(input=10, output=5),
                cost_usd=0.001,
            ),
            EvalOutcome.success(),
        )

    return runner


@pytest.fixture
def runner(sample_config, mock_storage, sample_policy, mock_agent_runner):
    trial_executor = TrialExecutor(mock_storage, sample_policy, agent_runner=mock_agent_runner)
    return EvalRunner(sample_config, mock_storage, trial_executor)


class TestEvalRunnerInit:
    def test_init_with_required_params(
        self, sample_config, mock_storage, sample_policy, mock_agent_runner
    ):
        trial_executor = TrialExecutor(mock_storage, sample_policy, agent_runner=mock_agent_runner)
        runner = EvalRunner(sample_config, mock_storage, trial_executor)
        assert runner._config == sample_config
        assert runner._storage == mock_storage
        assert runner._trial_executor == trial_executor

    def test_initial_state_not_cancelled(self, runner):
        assert not runner.is_cancelled


class TestEvalRunnerCancel:
    def test_cancel_sets_flag(self, runner):
        assert not runner.is_cancelled
        runner.cancel()
        assert runner.is_cancelled


class TestEvalRunnerRunSuite:
    @pytest.mark.asyncio
    async def test_run_suite_returns_summary(self, runner, sample_suite, sample_run_envelope):
        summary = await runner.run_suite(
            suite=sample_suite,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        assert summary is not None
        assert summary.envelope == sample_run_envelope
        assert isinstance(summary.metrics, SuiteMetrics)

    @pytest.mark.asyncio
    async def test_run_suite_creates_trials_for_all_tasks(
        self, runner, sample_suite, sample_run_envelope
    ):
        summary = await runner.run_suite(
            suite=sample_suite,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        assert len(summary.trials) == len(sample_suite.tasks)

    @pytest.mark.asyncio
    async def test_run_suite_trials_have_correct_task_ids(
        self, runner, sample_suite, sample_run_envelope
    ):
        summary = await runner.run_suite(
            suite=sample_suite,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        task_ids = {t.task_id for t in summary.trials}
        expected_ids = {t.id for t in sample_suite.tasks}
        assert task_ids == expected_ids

    @pytest.mark.asyncio
    async def test_run_suite_tracks_resource_usage(self, runner, sample_suite, sample_run_envelope):
        summary = await runner.run_suite(
            suite=sample_suite,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        assert summary.metrics.total_tokens.total > 0
        assert summary.metrics.total_duration_seconds > 0

    @pytest.mark.asyncio
    async def test_run_suite_stores_summary(
        self, runner, sample_suite, sample_run_envelope, mock_storage
    ):
        await runner.run_suite(
            suite=sample_suite,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        stored = await mock_storage.load_summary(
            sample_suite.id,
            sample_run_envelope.run_id,
        )
        assert stored is not None


class TestEvalRunnerCancellation:
    @pytest.mark.asyncio
    async def test_cancellation_flag_set_when_requested(
        self, mock_storage, sample_policy, sample_run_envelope
    ):
        config = EvalConfig(parallelism=1)

        started = asyncio.Event()

        async def slow_runner(task, enforcer, config_dict):
            started.set()
            await asyncio.sleep(10)
            return EvalTranscript(), EvalOutcome.success()

        trial_executor = TrialExecutor(mock_storage, sample_policy, agent_runner=slow_runner)
        runner = EvalRunner(config, mock_storage, trial_executor)

        tasks = [
            EvalTask(id="task-001", input="test"),
            EvalTask(id="task-002", input="test"),
        ]
        suite = EvalSuite(id="suite-001", name="Test", tasks=tasks)

        run_task = asyncio.create_task(
            runner.run_suite(
                suite=suite,
                agent_config={},
                run_envelope=sample_run_envelope,
            )
        )

        await started.wait()
        runner.cancel()
        run_task.cancel()

        try:
            await run_task
        except asyncio.CancelledError:
            pass

        assert runner.is_cancelled

    @pytest.mark.asyncio
    async def test_cancelled_flag_set_on_cancellation(
        self, runner, sample_suite, sample_run_envelope
    ):
        assert not runner.is_cancelled

        run_task = asyncio.create_task(
            runner.run_suite(
                suite=sample_suite,
                agent_config={},
                run_envelope=sample_run_envelope,
            )
        )

        await asyncio.sleep(0.05)
        runner.cancel()
        run_task.cancel()

        try:
            await run_task
        except asyncio.CancelledError:
            pass


class TestEvalRunnerMetrics:
    @pytest.mark.asyncio
    async def test_metrics_total_tasks_correct(self, runner, sample_suite, sample_run_envelope):
        summary = await runner.run_suite(
            suite=sample_suite,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        assert summary.metrics.total_tasks == len(sample_suite.tasks)

    @pytest.mark.asyncio
    async def test_metrics_completed_tasks_increments(
        self, runner, sample_suite, sample_run_envelope
    ):
        summary = await runner.run_suite(
            suite=sample_suite,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        assert summary.metrics.completed_tasks == len(sample_suite.tasks)

    @pytest.mark.asyncio
    async def test_metrics_pass_rate_calculated(self, runner, sample_suite, sample_run_envelope):
        summary = await runner.run_suite(
            suite=sample_suite,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        assert 0.0 <= summary.metrics.pass_rate <= 1.0

    @pytest.mark.asyncio
    async def test_metrics_latency_percentiles_empty_for_no_trials(
        self, mock_storage, sample_policy, sample_run_envelope, mock_agent_runner
    ):
        config = EvalConfig(parallelism=2)
        trial_executor = TrialExecutor(mock_storage, sample_policy, agent_runner=mock_agent_runner)
        runner = EvalRunner(config, mock_storage, trial_executor)

        empty_suite = EvalSuite(id="empty-suite", name="Empty", tasks=[])

        summary = await runner.run_suite(
            suite=empty_suite,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        assert summary.metrics.latency_p50_seconds is None
        assert summary.metrics.latency_p95_seconds is None
        assert summary.metrics.latency_p99_seconds is None

    @pytest.mark.asyncio
    async def test_metrics_latency_percentiles_populated_with_trials(
        self, runner, sample_suite, sample_run_envelope
    ):
        summary = await runner.run_suite(
            suite=sample_suite,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        assert summary.metrics.latency_p50_seconds is not None


class TestEvalRunnerExceptionHandling:
    @pytest.mark.asyncio
    async def test_exception_in_task_creates_failed_trial(
        self, mock_storage, sample_policy, sample_run_envelope
    ):
        async def failing_runner(task, enforcer, config_dict):
            raise ValueError("Test error")

        config = EvalConfig(parallelism=2)
        trial_executor = TrialExecutor(mock_storage, sample_policy, agent_runner=failing_runner)
        runner = EvalRunner(config, mock_storage, trial_executor)

        task = EvalTask(id="task-001", input="test")
        suite = EvalSuite(id="suite-001", name="Test", tasks=[task])

        summary = await runner.run_suite(
            suite=suite,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        assert len(summary.trials) == 1
        assert summary.trials[0].status == EvalStatus.ERROR


class TestEvalRunnerEmptySuite:
    @pytest.mark.asyncio
    async def test_empty_suite_returns_empty_summary(self, runner, sample_run_envelope):
        empty_suite = EvalSuite(id="empty-suite", name="Empty", tasks=[])

        summary = await runner.run_suite(
            suite=empty_suite,
            agent_config={},
            run_envelope=sample_run_envelope,
        )

        assert len(summary.trials) == 0
        assert summary.metrics.total_tasks == 0
        assert summary.metrics.completed_tasks == 0
        assert summary.metrics.pass_rate == 0.0


class TestResourceTracker:
    def test_init_zero_values(self):
        from ash_hawk.scenario.runner import ResourceTracker

        tracker = ResourceTracker()
        assert tracker.total_tokens.input == 0
        assert tracker.total_cost_usd == 0.0
        assert tracker.total_duration_seconds == 0.0

    @pytest.mark.asyncio
    async def test_add_trial_usage_accumulates(self):
        from ash_hawk.scenario.runner import ResourceTracker

        tracker = ResourceTracker()

        await tracker.add_trial_usage(
            tokens=TokenUsage(input=100, output=50),
            cost_usd=0.01,
            duration_seconds=1.5,
        )

        await tracker.add_trial_usage(
            tokens=TokenUsage(input=200, output=100),
            cost_usd=0.02,
            duration_seconds=2.5,
        )

        assert tracker.total_tokens.input == 300
        assert tracker.total_tokens.output == 150
        assert tracker.total_cost_usd == 0.03
        assert tracker.total_duration_seconds == 4.0

    @pytest.mark.asyncio
    async def test_concurrent_add_trial_usage_is_thread_safe(self):
        from ash_hawk.scenario.runner import ResourceTracker

        tracker = ResourceTracker()

        async def add_usage():
            for _ in range(100):
                await tracker.add_trial_usage(
                    tokens=TokenUsage(input=1, output=1),
                    cost_usd=0.001,
                    duration_seconds=0.1,
                )

        await asyncio.gather(*[add_usage() for _ in range(10)])

        assert tracker.total_tokens.input == 1000
        assert tracker.total_tokens.output == 1000
