import uuid
from contextlib import contextmanager
from typing import AsyncGenerator, Generator

import pytest
from moto.moto_server.threaded_moto_server import ThreadedMotoServer

from ash_hawk.storage.s3 import S3Config, S3Storage
from ash_hawk.types import (
    EvalOutcome,
    EvalRunSummary,
    EvalStatus,
    EvalSuite,
    EvalTask,
    EvalTrial,
    GraderResult,
    RunEnvelope,
    SuiteMetrics,
    TokenUsage,
    ToolSurfacePolicy,
    TrialEnvelope,
    TrialResult,
)


@contextmanager
def _s3_server_ctx(port: int = 5555) -> Generator[str, None, None]:
    server = ThreadedMotoServer(port=port, verbose=False)
    server.start()
    try:
        yield f"http://localhost:{port}"
    finally:
        server.stop()


@pytest.fixture
def s3_server() -> Generator[str, None, None]:
    with _s3_server_ctx() as url:
        yield url


@pytest.fixture
async def storage(s3_server: str) -> AsyncGenerator[S3Storage, None]:
    bucket_name = f"test-bucket-{uuid.uuid4().hex[:8]}"
    config = S3Config(
        bucket=bucket_name,
        endpoint_url=s3_server,
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    storage = S3Storage(config)

    client = await storage._get_client()
    await client.create_bucket(Bucket=bucket_name)

    yield storage

    await storage.close()


@pytest.fixture
def sample_suite() -> EvalSuite:
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
def sample_run_envelope() -> RunEnvelope:
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
def sample_policy() -> ToolSurfacePolicy:
    return ToolSurfacePolicy(
        allowed_tools=["read", "write"],
        denied_tools=["delete"],
        timeout_seconds=60.0,
    )


@pytest.fixture
def sample_trial_envelope(sample_policy: ToolSurfacePolicy) -> TrialEnvelope:
    return TrialEnvelope(
        trial_id="trial-001",
        run_id="run-001",
        task_id="task-001",
        policy_snapshot=sample_policy,
        created_at="2024-01-01T00:00:00Z",
    )


@pytest.fixture
def sample_trial() -> EvalTrial:
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
def sample_summary(sample_run_envelope: RunEnvelope) -> EvalRunSummary:
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


class TestS3StorageSuite:
    async def test_save_and_load_suite(self, storage: S3Storage, sample_suite: EvalSuite):
        await storage.save_suite(sample_suite)
        loaded = await storage.load_suite("suite-001")
        assert loaded is not None
        assert loaded.id == "suite-001"
        assert loaded.name == "Test Suite"
        assert len(loaded.tasks) == 2

    async def test_load_nonexistent_suite(self, storage: S3Storage):
        loaded = await storage.load_suite("nonexistent")
        assert loaded is None

    async def test_list_suites(self, storage: S3Storage, sample_suite: EvalSuite):
        await storage.save_suite(sample_suite)
        suite2 = EvalSuite(
            id="suite-002",
            name="Another Suite",
        )
        await storage.save_suite(suite2)
        suites = await storage.list_suites()
        assert suites == ["suite-001", "suite-002"]

    async def test_list_suites_empty(self, storage: S3Storage):
        suites = await storage.list_suites()
        assert suites == []


class TestS3StorageRunEnvelope:
    async def test_save_and_load_envelope(
        self, storage: S3Storage, sample_run_envelope: RunEnvelope
    ):
        await storage.save_run_envelope("suite-001", sample_run_envelope)
        loaded = await storage.load_run_envelope("suite-001", "run-001")
        assert loaded is not None
        assert loaded.run_id == "run-001"
        assert loaded.suite_id == "suite-001"
        assert loaded.model == "claude-3-5-sonnet-20241022"

    async def test_load_nonexistent_envelope(self, storage: S3Storage):
        loaded = await storage.load_run_envelope("suite-001", "nonexistent")
        assert loaded is None

    async def test_list_runs(self, storage: S3Storage, sample_run_envelope: RunEnvelope):
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
        assert runs == ["run-001", "run-002"]

    async def test_list_runs_empty(self, storage: S3Storage):
        runs = await storage.list_runs("nonexistent-suite")
        assert runs == []


class TestS3StorageTrial:
    async def test_save_and_load_trial(
        self,
        storage: S3Storage,
        sample_trial: EvalTrial,
        sample_trial_envelope: TrialEnvelope,
        sample_policy: ToolSurfacePolicy,
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

    async def test_load_nonexistent_trial(self, storage: S3Storage):
        loaded = await storage.load_trial("suite-001", "run-001", "nonexistent")
        assert loaded is None

    async def test_trial_with_complex_result(
        self, storage: S3Storage, sample_policy: ToolSurfacePolicy
    ):
        trial = EvalTrial(
            id="trial-complex",
            task_id="task-001",
            status=EvalStatus.COMPLETED,
            result=TrialResult(
                trial_id="trial-complex",
                outcome=EvalOutcome.success(),
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


class TestS3StorageSummary:
    async def test_save_and_load_summary(self, storage: S3Storage, sample_summary: EvalRunSummary):
        await storage.save_summary("suite-001", "run-001", sample_summary)
        loaded = await storage.load_summary("suite-001", "run-001")
        assert loaded is not None
        assert loaded.envelope.run_id == "run-001"
        assert loaded.metrics.total_tasks == 2
        assert loaded.metrics.pass_rate == 1.0

    async def test_load_nonexistent_summary(self, storage: S3Storage):
        loaded = await storage.load_summary("suite-001", "nonexistent")
        assert loaded is None


class TestS3StorageWithPrefix:
    async def test_with_prefix(self, s3_server: str, sample_suite: EvalSuite):
        bucket_name = f"test-bucket-prefix-{uuid.uuid4().hex[:8]}"
        config = S3Config(
            bucket=bucket_name,
            prefix="my-prefix",
            endpoint_url=s3_server,
            region_name="us-east-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
        storage = S3Storage(config)

        client = await storage._get_client()
        await client.create_bucket(Bucket=bucket_name)

        await storage.save_suite(sample_suite)
        loaded = await storage.load_suite("suite-001")
        assert loaded is not None
        assert loaded.id == "suite-001"

        key = storage._suite_key("suite-001")
        assert key.startswith("my-prefix/")

        await storage.close()


class TestS3Config:
    def test_config_defaults(self):
        config = S3Config(bucket="my-bucket")
        assert config.bucket == "my-bucket"
        assert config.prefix == ""
        assert config.endpoint_url is None
        assert config.region_name == "us-east-1"

    def test_config_custom_endpoint(self):
        config = S3Config(
            bucket="my-bucket",
            endpoint_url="http://localhost:9000",
            region_name="custom-region",
        )
        assert config.endpoint_url == "http://localhost:9000"
        assert config.region_name == "custom-region"
