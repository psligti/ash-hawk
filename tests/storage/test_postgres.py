import os

import pytest

from ash_hawk.storage import StoredTrial
from ash_hawk.types import (
    EvalOutcome,
    EvalRunSummary,
    EvalStatus,
    EvalSuite,
    EvalTask,
    EvalTranscript,
    EvalTrial,
    GraderResult,
    RunEnvelope,
    SuiteMetrics,
    TokenUsage,
    ToolSurfacePolicy,
    TrialEnvelope,
    TrialResult,
)

POSTGRES_URL = os.environ.get("ASH_HAWK_TEST_POSTGRES_URL")
HAS_POSTGRES = POSTGRES_URL is not None

postgres_skip = pytest.mark.skipif(
    not HAS_POSTGRES,
    reason="PostgreSQL not available. Set ASH_HAWK_TEST_POSTGRES_URL environment variable.",
)


@pytest.fixture
async def storage():
    from ash_hawk.storage.postgres import PostgresStorage

    assert POSTGRES_URL is not None
    async with PostgresStorage(POSTGRES_URL) as s:
        yield s


@pytest.fixture
def sample_suite():
    return EvalSuite(
        id="suite-001",
        name="Test Suite",
        description="A test suite",
        tasks=[
            EvalTask(
                id="task-001",
                input="What is 2+2?",
                expected_output="4",
            ),
            EvalTask(
                id="task-002",
                input="What is 3+3?",
                expected_output="6",
            ),
        ],
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
def sample_policy():
    return ToolSurfacePolicy(
        allowed_tools=["read", "write"],
        denied_tools=["delete"],
        timeout_seconds=60.0,
    )


@pytest.fixture
def sample_trial_envelope(sample_policy):
    return TrialEnvelope(
        trial_id="trial-001",
        run_id="run-001",
        task_id="task-001",
        policy_snapshot=sample_policy,
        created_at="2024-01-01T00:00:00Z",
    )


@pytest.fixture
def sample_trial():
    return EvalTrial(
        id="trial-001",
        task_id="task-001",
        status=EvalStatus.COMPLETED,
        result=TrialResult(
            trial_id="trial-001",
            outcome=EvalOutcome.success(),
            grader_results=[
                GraderResult(
                    grader_type="string_match",
                    score=1.0,
                    passed=True,
                )
            ],
            aggregate_score=1.0,
            aggregate_passed=True,
        ),
    )


@pytest.fixture
def sample_summary(sample_run_envelope):
    return EvalRunSummary(
        envelope=sample_run_envelope,
        metrics=SuiteMetrics(
            suite_id="suite-001",
            run_id="run-001",
            total_tasks=2,
            completed_tasks=2,
            passed_tasks=2,
            pass_rate=1.0,
            mean_score=1.0,
            total_tokens=TokenUsage(input=100, output=50),
            created_at="2024-01-01T00:01:00Z",
        ),
    )


@postgres_skip
class TestPostgresStorageSuite:
    async def test_save_and_load_suite(self, storage, sample_suite):
        await storage.save_suite(sample_suite)
        loaded = await storage.load_suite("suite-001")
        assert loaded is not None
        assert loaded.id == "suite-001"
        assert loaded.name == "Test Suite"
        assert len(loaded.tasks) == 2

    async def test_load_nonexistent_suite(self, storage):
        loaded = await storage.load_suite("nonexistent-suite-id")
        assert loaded is None

    async def test_list_suites(self, storage, sample_suite):
        await storage.save_suite(sample_suite)
        suite2 = EvalSuite(
            id="suite-002",
            name="Another Suite",
        )
        await storage.save_suite(suite2)
        suites = await storage.list_suites()
        assert "suite-001" in suites
        assert "suite-002" in suites

    async def test_list_suites_empty(self, storage):
        suites = await storage.list_suites()
        assert isinstance(suites, list)


@postgres_skip
class TestPostgresStorageRunEnvelope:
    async def test_save_and_load_envelope(self, storage, sample_run_envelope):
        await storage.save_run_envelope("suite-001", sample_run_envelope)
        loaded = await storage.load_run_envelope("suite-001", "run-001")
        assert loaded is not None
        assert loaded.run_id == "run-001"
        assert loaded.suite_id == "suite-001"
        assert loaded.model == "claude-3-5-sonnet-20241022"

    async def test_load_nonexistent_envelope(self, storage):
        loaded = await storage.load_run_envelope("suite-001", "nonexistent")
        assert loaded is None

    async def test_list_runs(self, storage, sample_run_envelope):
        await storage.save_run_envelope("suite-001", sample_run_envelope)
        envelope2 = RunEnvelope(
            run_id="run-002",
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
        await storage.save_run_envelope("suite-001", envelope2)
        runs = await storage.list_runs("suite-001")
        assert "run-001" in runs
        assert "run-002" in runs

    async def test_list_runs_empty(self, storage):
        runs = await storage.list_runs("nonexistent-suite")
        assert runs == []


@postgres_skip
class TestPostgresStorageTrial:
    async def test_save_and_load_trial(
        self, storage, sample_trial, sample_trial_envelope, sample_policy
    ):
        await storage.save_trial(
            "suite-001",
            "run-001",
            sample_trial,
            sample_trial_envelope,
            sample_policy,
        )
        loaded = await storage.load_trial("suite-001", "run-001", "trial-001")
        assert loaded is not None
        assert loaded.trial.id == "trial-001"
        assert loaded.trial.status == EvalStatus.COMPLETED
        assert loaded.envelope.run_id == "run-001"
        assert loaded.policy.allowed_tools == ["read", "write"]

    async def test_load_nonexistent_trial(self, storage):
        loaded = await storage.load_trial("suite-001", "run-001", "nonexistent")
        assert loaded is None

    async def test_trial_with_complex_result(self, storage, sample_policy):
        trial = EvalTrial(
            id="trial-complex",
            task_id="task-001",
            status=EvalStatus.COMPLETED,
            result=TrialResult(
                trial_id="trial-complex",
                outcome=EvalOutcome.success(),
                transcript=EvalTranscript(
                    messages=[{"role": "user", "content": "Hello"}],
                    tool_calls=[],
                    token_usage=TokenUsage(input=100, output=50),
                    cost_usd=0.001,
                    duration_seconds=1.5,
                ),
                grader_results=[
                    GraderResult(
                        grader_type="llm_judge",
                        grader_id="judge-001",
                        score=0.95,
                        passed=True,
                        details={"reasoning": "Good answer"},
                    )
                ],
                aggregate_score=0.95,
                aggregate_passed=True,
            ),
        )
        envelope = TrialEnvelope(
            trial_id="trial-complex",
            run_id="run-001",
            task_id="task-001",
            policy_snapshot=sample_policy,
            created_at="2024-01-01T00:00:00Z",
        )
        await storage.save_trial("suite-001", "run-001", trial, envelope, sample_policy)
        loaded = await storage.load_trial("suite-001", "run-001", "trial-complex")
        assert loaded is not None
        assert loaded.trial.result.aggregate_score == 0.95


@postgres_skip
class TestPostgresStorageSummary:
    async def test_save_and_load_summary(self, storage, sample_summary):
        await storage.save_summary("suite-001", "run-001", sample_summary)
        loaded = await storage.load_summary("suite-001", "run-001")
        assert loaded is not None
        assert loaded.envelope.run_id == "run-001"
        assert loaded.metrics.total_tasks == 2
        assert loaded.metrics.pass_rate == 1.0

    async def test_load_nonexistent_summary(self, storage):
        loaded = await storage.load_summary("suite-001", "nonexistent")
        assert loaded is None


@postgres_skip
class TestPostgresStorageConnectionPooling:
    async def test_connection_pool_initialization(self):
        from ash_hawk.storage.postgres import PostgresStorage

        assert POSTGRES_URL is not None
        async with PostgresStorage(POSTGRES_URL, min_pool_size=2, max_pool_size=5) as storage:
            assert storage._pool is not None

    async def test_context_manager(self, storage, sample_suite):
        await storage.save_suite(sample_suite)
        loaded = await storage.load_suite("suite-001")
        assert loaded is not None


@postgres_skip
class TestPostgresStorageUpsert:
    async def test_upsert_suite(self, storage, sample_suite):
        await storage.save_suite(sample_suite)
        updated_suite = EvalSuite(
            id="suite-001",
            name="Updated Suite",
            description="Updated description",
        )
        await storage.save_suite(updated_suite)
        loaded = await storage.load_suite("suite-001")
        assert loaded is not None
        assert loaded.name == "Updated Suite"
        assert loaded.description == "Updated description"

    async def test_upsert_run_envelope(self, storage, sample_run_envelope):
        await storage.save_run_envelope("suite-001", sample_run_envelope)
        updated_envelope = RunEnvelope(
            run_id="run-001",
            suite_id="suite-001",
            suite_hash="updated-hash",
            harness_version="0.2.0",
            agent_name="updated-agent",
            provider="openai",
            model="gpt-4",
            tool_policy_hash="updated-policy-hash",
            python_version="3.12.0",
            os_info="linux",
            created_at="2024-01-02T00:00:00Z",
        )
        await storage.save_run_envelope("suite-001", updated_envelope)
        loaded = await storage.load_run_envelope("suite-001", "run-001")
        assert loaded is not None
        assert loaded.model == "gpt-4"
        assert loaded.provider == "openai"
