"""Tests for ash_hawk.types module."""

from datetime import UTC, datetime, timezone

import pytest

from ash_hawk.types import (
    CalibrationSample,
    EvalAgentConfig,
    EvalOutcome,
    EvalRunSummary,
    EvalStatus,
    EvalSuite,
    EvalTask,
    EvalTranscript,
    EvalTrial,
    FailureMode,
    GraderResult,
    GraderSpec,
    RunEnvelope,
    StorageBackend,
    SuiteMetrics,
    TokenUsage,
    ToolPermission,
    ToolSurfacePolicy,
    TrialEnvelope,
    TrialResult,
)


class TestEnums:
    """Test enum definitions."""

    def test_eval_status_values(self):
        """Test EvalStatus enum values."""
        assert EvalStatus.PENDING == "pending"
        assert EvalStatus.RUNNING == "running"
        assert EvalStatus.COMPLETED == "completed"
        assert EvalStatus.ERROR == "error"
        assert EvalStatus.CANCELLED == "cancelled"

    def test_failure_mode_values(self):
        """Test FailureMode enum values."""
        assert FailureMode.TIMEOUT == "timeout"
        assert FailureMode.TOOL_DENIED == "tool_denied"
        assert FailureMode.CRASH == "crash"
        assert FailureMode.JUDGE_ERROR == "judge_error"
        assert FailureMode.POLICY_VIOLATION == "policy_violation"
        assert FailureMode.AGENT_ERROR == "agent_error"
        assert FailureMode.VALIDATION_ERROR == "validation_error"
        assert FailureMode.RESOURCE_EXCEEDED == "resource_exceeded"

    def test_tool_permission_values(self):
        """Test ToolPermission enum values."""
        assert ToolPermission.ALLOW == "allow"
        assert ToolPermission.ASK == "ask"
        assert ToolPermission.DENY == "deny"

    def test_enum_membership(self):
        """Test enum membership checks."""
        assert FailureMode.TIMEOUT in list(FailureMode)
        assert EvalStatus.COMPLETED in list(EvalStatus)


class TestTokenUsage:
    """Test TokenUsage model."""

    def test_default_values(self):
        """Test default token usage values."""
        usage = TokenUsage()
        assert usage.input == 0
        assert usage.output == 0
        assert usage.reasoning == 0
        assert usage.cache_read == 0
        assert usage.cache_write == 0
        assert usage.total == 0

    def test_total_computation(self):
        """Test total token computation."""
        usage = TokenUsage(input=100, output=50, reasoning=25)
        assert usage.total == 175

    def test_total_excludes_cache(self):
        """Test that cache tokens are excluded from total."""
        usage = TokenUsage(input=100, output=50, reasoning=25, cache_read=500, cache_write=200)
        assert usage.total == 175  # Only input + output + reasoning

    def test_extra_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(Exception):  # ValidationError
            TokenUsage(input=100, extra_field="not_allowed")


class TestToolSurfacePolicy:
    """Test ToolSurfacePolicy model."""

    def test_default_values(self):
        """Test default policy values."""
        policy = ToolSurfacePolicy()
        assert policy.allowed_tools == []
        assert policy.denied_tools == []
        assert policy.default_permission == ToolPermission.ASK
        assert policy.network_allowed is False
        assert policy.timeout_seconds == 300.0

    def test_is_tool_allowed_denylist(self):
        """Test denylist takes precedence."""
        policy = ToolSurfacePolicy(
            allowed_tools=["read", "write"],
            denied_tools=["write"],
        )
        assert policy.is_tool_allowed("read") == ToolPermission.ALLOW
        assert policy.is_tool_allowed("write") == ToolPermission.DENY

    def test_is_tool_allowed_glob_patterns(self):
        """Test glob pattern matching."""
        policy = ToolSurfacePolicy(
            allowed_tools=["read*", "grep"],
            denied_tools=["*bash*"],
        )
        assert policy.is_tool_allowed("read") == ToolPermission.ALLOW
        assert policy.is_tool_allowed("read_file") == ToolPermission.ALLOW
        assert policy.is_tool_allowed("run_bash_cmd") == ToolPermission.DENY

    def test_is_tool_allowed_default(self):
        """Test fallback to default permission."""
        policy = ToolSurfacePolicy(
            default_permission=ToolPermission.DENY,
        )
        assert policy.is_tool_allowed("unknown_tool") == ToolPermission.DENY

    def test_extra_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(Exception):  # ValidationError
            ToolSurfacePolicy(allowed_tools=["read"], custom_field="not_allowed")


class TestGraderSpec:
    """Test GraderSpec model."""

    def test_basic_grader_spec(self):
        """Test basic grader specification."""
        spec = GraderSpec(grader_type="string_match")
        assert spec.grader_type == "string_match"
        assert spec.config == {}
        assert spec.weight == 1.0
        assert spec.required is False

    def test_grader_spec_with_config(self):
        """Test grader spec with configuration."""
        spec = GraderSpec(
            grader_type="test_runner",
            config={"tests": ["test_auth.py"], "timeout": 30},
            weight=2.0,
            required=True,
        )
        assert spec.grader_type == "test_runner"
        assert spec.config["tests"] == ["test_auth.py"]
        assert spec.weight == 2.0
        assert spec.required is True

    def test_extra_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(Exception):  # ValidationError
            GraderSpec(grader_type="test", unknown_field="not_allowed")


class TestGraderResult:
    """Test GraderResult model."""

    def test_passing_result(self):
        """Test a passing grader result."""
        result = GraderResult(
            grader_type="string_match",
            score=1.0,
            passed=True,
        )
        assert result.score == 1.0
        assert result.passed is True
        assert result.error_message is None

    def test_failing_result(self):
        """Test a failing grader result."""
        result = GraderResult(
            grader_type="test_runner",
            score=0.0,
            passed=False,
            details={"failed_tests": ["test_login"]},
        )
        assert result.score == 0.0
        assert result.passed is False
        assert "failed_tests" in result.details

    def test_partial_score(self):
        """Test partial score result."""
        result = GraderResult(
            grader_type="multi_check",
            score=0.75,
            passed=False,  # May require threshold > 0.75
        )
        assert result.score == 0.75


class TestEvalTask:
    """Test EvalTask model."""

    def test_simple_string_task(self):
        """Test task with simple string input."""
        task = EvalTask(
            id="task-1",
            input="What is 2+2?",
        )
        assert task.id == "task-1"
        assert task.input == "What is 2+2?"
        assert task.description == ""
        assert task.expected_output is None
        assert task.grader_specs == []

    def test_structured_input_task(self):
        """Test task with structured payload input (coding task)."""
        task = EvalTask(
            id="fix-bug-1",
            description="Fix the auth bypass",
            input={"issue_file": "issues/auth.md", "codebase": "/tmp/repo"},
            expected_output={"tests_pass": True},
            grader_specs=[
                GraderSpec(grader_type="test_runner", config={"tests": ["test_auth.py"]})
            ],
            tags=["coding", "security"],
        )
        assert task.id == "fix-bug-1"
        assert isinstance(task.input, dict)
        assert task.input["issue_file"] == "issues/auth.md"
        assert len(task.grader_specs) == 1
        assert "coding" in task.tags

    def test_conversational_task(self):
        """Test conversational task type."""
        task = EvalTask(
            id="chat-1",
            input="Help me understand recursion",
            expected_output="A clear explanation of recursion",
            tags=["conversational", "explanation"],
            max_attempts=1,
        )
        assert task.id == "chat-1"
        assert "conversational" in task.tags

    def test_research_task(self):
        """Test research task type."""
        task = EvalTask(
            id="research-1",
            input={
                "query": "Find all uses of deprecated API",
                "scope": "/project/src",
            },
            fixtures={"codebase": "/fixtures/project"},
            tags=["research", "code-analysis"],
        )
        assert task.id == "research-1"
        assert "research" in task.tags
        assert "codebase" in task.fixtures

    def test_extra_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(Exception):  # ValidationError
            EvalTask(id="test", input="test", custom_field="not_allowed")


class TestEvalSuite:
    """Test EvalSuite model."""

    def test_empty_suite(self):
        """Test empty evaluation suite."""
        suite = EvalSuite(id="suite-1", name="Test Suite")
        assert suite.id == "suite-1"
        assert suite.name == "Test Suite"
        assert suite.tasks == []
        assert suite.task_count == 0

    def test_suite_with_tasks(self):
        """Test suite with multiple tasks."""
        suite = EvalSuite(
            id="suite-1",
            name="Coding Benchmarks",
            tasks=[
                EvalTask(id="t1", input="Task 1"),
                EvalTask(id="t2", input="Task 2"),
                EvalTask(id="t3", input="Task 3"),
            ],
        )
        assert suite.task_count == 3

    def test_suite_with_agent_defaults(self):
        suite = EvalSuite(
            id="suite-agent",
            name="Suite With Agent",
            agent=EvalAgentConfig(
                name="build",
                provider="zai-coding-plan",
                model="glm-4.7",
            ),
        )
        assert suite.agent is not None
        assert suite.agent.name == "build"
        assert suite.agent.provider == "zai-coding-plan"
        assert suite.agent.model == "glm-4.7"

    def test_suite_with_agent_class_and_location(self):
        suite = EvalSuite(
            id="suite-agent-class",
            name="Suite With Agent Class",
            agent=EvalAgentConfig.model_validate(
                {
                    "class": "my_package.runner:CustomRunner",
                    "location": "./evals/custom_runner.py",
                }
            ),
        )
        assert suite.agent is not None
        assert suite.agent.class_name == "my_package.runner:CustomRunner"
        assert suite.agent.location == "./evals/custom_runner.py"

    def test_extra_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(Exception):  # ValidationError
            EvalSuite(id="test", name="Test", custom_field="not_allowed")


class TestEvalTranscript:
    """Test EvalTranscript model."""

    def test_default_transcript(self):
        """Test default transcript values."""
        transcript = EvalTranscript()
        assert transcript.messages == []
        assert transcript.tool_calls == []
        assert transcript.cost_usd == 0.0
        assert transcript.duration_seconds == 0.0

    def test_transcript_with_data(self):
        """Test transcript with execution data."""
        transcript = EvalTranscript(
            messages=[{"role": "user", "content": "Hello"}],
            tool_calls=[{"tool": "read", "input": {"path": "/tmp/file"}}],
            token_usage=TokenUsage(input=100, output=50),
            cost_usd=0.0025,
            duration_seconds=5.3,
        )
        assert len(transcript.messages) == 1
        assert len(transcript.tool_calls) == 1
        assert transcript.token_usage.total == 150

    def test_transcript_with_error(self):
        """Test transcript with error trace."""
        transcript = EvalTranscript(
            error_trace="Traceback (most recent call last):\n  ...",
        )
        assert transcript.error_trace is not None


class TestEvalOutcome:
    """Test EvalOutcome model."""

    def test_success_factory(self):
        """Test success factory method."""
        outcome = EvalOutcome.success()
        assert outcome.status == EvalStatus.COMPLETED
        assert outcome.failure_mode is None
        assert outcome.completed_at is not None

    def test_failure_factory(self):
        """Test failure factory method."""
        outcome = EvalOutcome.failure(
            failure_mode=FailureMode.TIMEOUT,
            error_message="Trial exceeded 300s timeout",
        )
        assert outcome.status == EvalStatus.ERROR
        assert outcome.failure_mode == FailureMode.TIMEOUT
        assert outcome.error_message == "Trial exceeded 300s timeout"

    def test_manual_outcome(self):
        """Test manually created outcome."""
        outcome = EvalOutcome(
            status=EvalStatus.CANCELLED,
            failure_mode=FailureMode.POLICY_VIOLATION,
        )
        assert outcome.status == EvalStatus.CANCELLED


class TestTrialResult:
    """Test TrialResult model."""

    def test_minimal_result(self):
        """Test minimal trial result."""
        result = TrialResult(
            trial_id="trial-1",
            outcome=EvalOutcome.success(),
        )
        assert result.trial_id == "trial-1"
        assert result.aggregate_score == 0.0
        assert result.aggregate_passed is False

    def test_result_with_graders(self):
        """Test result with multiple grader results."""
        result = TrialResult(
            trial_id="trial-1",
            outcome=EvalOutcome.success(),
            grader_results=[
                GraderResult(grader_type="string_match", score=1.0, passed=True),
                GraderResult(grader_type="test_runner", score=1.0, passed=True),
            ],
            aggregate_score=1.0,
            aggregate_passed=True,
        )
        assert len(result.grader_results) == 2
        assert result.aggregate_passed is True


class TestEvalTrial:
    """Test EvalTrial model."""

    def test_pending_trial(self):
        """Test pending trial."""
        trial = EvalTrial(id="trial-1", task_id="task-1")
        assert trial.id == "trial-1"
        assert trial.task_id == "task-1"
        assert trial.status == EvalStatus.PENDING
        assert trial.result is None

    def test_completed_trial(self):
        """Test completed trial with result."""
        trial = EvalTrial(
            id="trial-1",
            task_id="task-1",
            status=EvalStatus.COMPLETED,
            result=TrialResult(
                trial_id="trial-1",
                outcome=EvalOutcome.success(),
                aggregate_passed=True,
            ),
        )
        assert trial.status == EvalStatus.COMPLETED
        assert trial.result is not None
        assert trial.result.aggregate_passed is True


class TestRunEnvelope:
    """Test RunEnvelope model."""

    def test_minimal_envelope(self):
        """Test minimal run envelope."""
        envelope = RunEnvelope(
            run_id="run-1",
            suite_id="suite-1",
            suite_hash="abc123",
            harness_version="0.1.0",
            agent_name="test-agent",
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            tool_policy_hash="def456",
            python_version="3.11.0",
            os_info="macOS-14.0",
            created_at=datetime.now(UTC).isoformat(),
        )
        assert envelope.run_id == "run-1"
        assert envelope.provider == "anthropic"
        assert envelope.model == "claude-3-5-sonnet-20241022"

    def test_envelope_with_all_fields(self):
        """Test envelope with all optional fields."""
        envelope = RunEnvelope(
            run_id="run-1",
            suite_id="suite-1",
            suite_hash="abc123",
            harness_version="0.1.0",
            git_commit="xyz789",
            agent_name="test-agent",
            agent_version="1.0.0",
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            model_params={"temperature": 0.7, "max_tokens": 4096},
            seed=42,
            tool_policy_hash="def456",
            python_version="3.11.0",
            os_info="macOS-14.0",
            config_snapshot={"parallelism": 4},
            created_at=datetime.now(UTC).isoformat(),
        )
        assert envelope.git_commit == "xyz789"
        assert envelope.seed == 42
        assert envelope.model_params["temperature"] == 0.7


class TestTrialEnvelope:
    """Test TrialEnvelope model."""

    def test_trial_envelope(self):
        """Test trial envelope creation."""
        envelope = TrialEnvelope(
            trial_id="trial-1",
            run_id="run-1",
            task_id="task-1",
            policy_snapshot=ToolSurfacePolicy(allowed_tools=["read"]),
            created_at=datetime.now(UTC).isoformat(),
        )
        assert envelope.trial_id == "trial-1"
        assert envelope.attempt_number == 1
        assert isinstance(envelope.policy_snapshot, ToolSurfacePolicy)


class TestSuiteMetrics:
    """Test SuiteMetrics model."""

    def test_minimal_metrics(self):
        """Test minimal suite metrics."""
        metrics = SuiteMetrics(
            suite_id="suite-1",
            run_id="run-1",
            total_tasks=10,
            created_at=datetime.now(UTC).isoformat(),
        )
        assert metrics.total_tasks == 10
        assert metrics.pass_rate == 0.0

    def test_metrics_with_results(self):
        """Test metrics with computed results."""
        metrics = SuiteMetrics(
            suite_id="suite-1",
            run_id="run-1",
            total_tasks=10,
            completed_tasks=10,
            passed_tasks=7,
            failed_tasks=3,
            pass_rate=0.7,
            mean_score=0.75,
            latency_p50_seconds=5.2,
            latency_p95_seconds=12.5,
            pass_at_k={1: 0.7, 3: 0.9, 5: 0.95},
            created_at=datetime.now(UTC).isoformat(),
        )
        assert metrics.pass_rate == 0.7
        assert metrics.pass_at_k[1] == 0.7


class TestEvalRunSummary:
    """Test EvalRunSummary model."""

    def test_run_summary(self):
        """Test complete run summary."""
        envelope = RunEnvelope(
            run_id="run-1",
            suite_id="suite-1",
            suite_hash="abc123",
            harness_version="0.1.0",
            agent_name="test-agent",
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            tool_policy_hash="def456",
            python_version="3.11.0",
            os_info="macOS-14.0",
            created_at=datetime.now(UTC).isoformat(),
        )
        metrics = SuiteMetrics(
            suite_id="suite-1",
            run_id="run-1",
            total_tasks=5,
            created_at=datetime.now(UTC).isoformat(),
        )
        summary = EvalRunSummary(envelope=envelope, metrics=metrics)
        assert summary.envelope.run_id == "run-1"
        assert summary.metrics.total_tasks == 5


class TestStorageBackend:
    """Test StorageBackend type alias."""

    def test_storage_backend_values(self):
        """Test valid storage backend values."""
        backends: list[StorageBackend] = ["file", "sqlite", "postgres", "s3"]
        assert len(backends) == 4


class TestExtraForbidden:
    """Test that all models enforce extra='forbid'."""

    def test_token_usage_extra_forbidden(self):
        with pytest.raises(Exception):
            TokenUsage(input=100, extra="not_allowed")

    def test_tool_surface_policy_extra_forbidden(self):
        with pytest.raises(Exception):
            ToolSurfacePolicy(allowed_tools=["read"], extra="not_allowed")

    def test_grader_spec_extra_forbidden(self):
        with pytest.raises(Exception):
            GraderSpec(grader_type="test", extra="not_allowed")

    def test_grader_result_extra_forbidden(self):
        with pytest.raises(Exception):
            GraderResult(grader_type="test", score=1.0, passed=True, extra="not_allowed")

    def test_eval_task_extra_forbidden(self):
        with pytest.raises(Exception):
            EvalTask(id="test", input="test", extra="not_allowed")

    def test_eval_suite_extra_forbidden(self):
        with pytest.raises(Exception):
            EvalSuite(id="test", name="Test", extra="not_allowed")

    def test_eval_transcript_extra_forbidden(self):
        with pytest.raises(Exception):
            EvalTranscript(extra="not_allowed")

    def test_eval_outcome_extra_forbidden(self):
        with pytest.raises(Exception):
            EvalOutcome(status=EvalStatus.COMPLETED, extra="not_allowed")

    def test_trial_result_extra_forbidden(self):
        with pytest.raises(Exception):
            TrialResult(trial_id="test", outcome=EvalOutcome.success(), extra="not_allowed")

    def test_eval_trial_extra_forbidden(self):
        with pytest.raises(Exception):
            EvalTrial(id="test", task_id="test", extra="not_allowed")

    def test_run_envelope_extra_forbidden(self):
        with pytest.raises(Exception):
            RunEnvelope(
                run_id="test",
                suite_id="test",
                suite_hash="test",
                harness_version="test",
                agent_name="test",
                provider="test",
                model="test",
                tool_policy_hash="test",
                python_version="test",
                os_info="test",
                created_at="test",
                extra="not_allowed",
            )

    def test_trial_envelope_extra_forbidden(self):
        with pytest.raises(Exception):
            TrialEnvelope(
                trial_id="test",
                run_id="test",
                task_id="test",
                policy_snapshot=ToolSurfacePolicy(),
                created_at="test",
                extra="not_allowed",
            )

    def test_suite_metrics_extra_forbidden(self):
        with pytest.raises(Exception):
            SuiteMetrics(
                suite_id="test",
                run_id="test",
                total_tasks=0,
                created_at="test",
                extra="not_allowed",
            )

    def test_eval_run_summary_extra_forbidden(self):
        with pytest.raises(Exception):
            EvalRunSummary(
                envelope=RunEnvelope(
                    run_id="test",
                    suite_id="test",
                    suite_hash="test",
                    harness_version="test",
                    agent_name="test",
                    provider="test",
                    model="test",
                    tool_policy_hash="test",
                    python_version="test",
                    os_info="test",
                    created_at="test",
                ),
                metrics=SuiteMetrics(
                    suite_id="test",
                    run_id="test",
                    total_tasks=0,
                    created_at="test",
                ),
                extra="not_allowed",
            )


class TestGraderResultConfidenceFields:
    """Test GraderResult confidence and review fields."""

    def test_confidence_field_exists(self):
        """Test that confidence field exists with default None."""
        result = GraderResult(
            grader_type="llm_judge",
            score=0.8,
            passed=True,
        )
        assert result.confidence is None

    def test_confidence_field_with_value(self):
        """Test confidence field accepts valid values."""
        result = GraderResult(
            grader_type="llm_judge",
            score=0.8,
            passed=True,
            confidence=0.9,
        )
        assert result.confidence == 0.9

    def test_confidence_must_be_at_least_zero(self):
        """Test that confidence below 0 is rejected."""
        with pytest.raises(Exception):  # ValidationError
            GraderResult(
                grader_type="llm_judge",
                score=0.8,
                passed=True,
                confidence=-0.1,
            )

    def test_confidence_must_be_at_most_one(self):
        """Test that confidence above 1 is rejected."""
        with pytest.raises(Exception):  # ValidationError
            GraderResult(
                grader_type="llm_judge",
                score=0.8,
                passed=True,
                confidence=1.5,
            )

    def test_confidence_boundaries(self):
        """Test that boundary values 0 and 1 are valid."""
        result_zero = GraderResult(
            grader_type="llm_judge",
            score=0.8,
            passed=True,
            confidence=0.0,
        )
        result_one = GraderResult(
            grader_type="llm_judge",
            score=0.8,
            passed=True,
            confidence=1.0,
        )
        assert result_zero.confidence == 0.0
        assert result_one.confidence == 1.0

    def test_needs_review_field_exists(self):
        """Test that needs_review field exists with default False."""
        result = GraderResult(
            grader_type="llm_judge",
            score=0.8,
            passed=True,
        )
        assert result.needs_review is False

    def test_needs_review_with_value(self):
        """Test needs_review field accepts True."""
        result = GraderResult(
            grader_type="llm_judge",
            score=0.8,
            passed=True,
            needs_review=True,
        )
        assert result.needs_review is True

    def test_review_reason_field_exists(self):
        """Test that review_reason field exists with default None."""
        result = GraderResult(
            grader_type="llm_judge",
            score=0.8,
            passed=True,
        )
        assert result.review_reason is None

    def test_review_reason_with_value(self):
        """Test review_reason field accepts string."""
        result = GraderResult(
            grader_type="llm_judge",
            score=0.8,
            passed=True,
            needs_review=True,
            review_reason="High variance between judges",
        )
        assert result.review_reason == "High variance between judges"

    def test_all_new_fields_together(self):
        """Test all new fields together."""
        result = GraderResult(
            grader_type="llm_judge",
            score=0.65,
            passed=False,
            confidence=0.3,
            needs_review=True,
            review_reason="Low confidence score",
        )
        assert result.confidence == 0.3
        assert result.needs_review is True
        assert result.review_reason == "Low confidence score"


class TestCalibrationSample:
    """Test CalibrationSample model."""

    def test_basic_sample(self):
        """Test basic calibration sample creation."""
        sample = CalibrationSample(predicted=0.8, actual=True)
        assert sample.predicted == 0.8
        assert sample.actual is True
        assert sample.trial_id is None

    def test_sample_with_trial_id(self):
        """Test sample with trial_id."""
        sample = CalibrationSample(predicted=0.5, actual=False, trial_id="trial-123")
        assert sample.predicted == 0.5
        assert sample.actual is False
        assert sample.trial_id == "trial-123"

    def test_predicted_must_be_at_least_zero(self):
        """Test that predicted score below 0 is rejected."""
        with pytest.raises(Exception):  # ValidationError
            CalibrationSample(predicted=-0.1, actual=True)

    def test_predicted_must_be_at_most_one(self):
        """Test that predicted score above 1 is rejected."""
        with pytest.raises(Exception):  # ValidationError
            CalibrationSample(predicted=1.5, actual=False)

    def test_predicted_boundaries(self):
        """Test that boundary values 0 and 1 are valid."""
        sample_zero = CalibrationSample(predicted=0.0, actual=False)
        sample_one = CalibrationSample(predicted=1.0, actual=True)
        assert sample_zero.predicted == 0.0
        assert sample_one.predicted == 1.0

    def test_from_trial_factory_method(self):
        """Test from_trial() extracts score and passed from GraderResult."""
        grader_result = GraderResult(
            grader_type="llm_judge",
            score=0.75,
            passed=True,
        )
        sample = CalibrationSample.from_trial(grader_result)
        assert sample.predicted == 0.75
        assert sample.actual is True
        assert sample.trial_id is None

    def test_from_trial_with_grader_id(self):
        """Test from_trial() uses grader_id as trial_id if present."""
        grader_result = GraderResult(
            grader_type="llm_judge",
            grader_id="trial-456",
            score=0.6,
            passed=False,
        )
        sample = CalibrationSample.from_trial(grader_result)
        assert sample.predicted == 0.6
        assert sample.actual is False
        assert sample.trial_id == "trial-456"

    def test_from_trial_with_explicit_trial_id(self):
        """Test from_trial() with explicit trial_id override."""
        grader_result = GraderResult(
            grader_type="llm_judge",
            grader_id="grader-abc",
            score=0.9,
            passed=True,
        )
        sample = CalibrationSample.from_trial(grader_result, trial_id="override-xyz")
        assert sample.trial_id == "override-xyz"

    def test_extra_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(Exception):  # ValidationError
            CalibrationSample(predicted=0.5, actual=True, extra="not_allowed")


class TestCalibrationCurve:
    """Test CalibrationCurve model."""

    def test_basic_curve(self):
        """Test basic calibration curve creation."""
        from ash_hawk.types import CalibrationCurve

        samples = [
            CalibrationSample(predicted=0.8, actual=True),
            CalibrationSample(predicted=0.6, actual=False),
        ]
        curve = CalibrationCurve(samples=samples, ece=0.1, brier_score=0.15)
        assert len(curve.samples) == 2
        assert curve.ece == 0.1
        assert curve.brier_score == 0.15

    def test_compute_perfectly_calibrated(self):
        """Test compute() with perfectly calibrated samples returns ECE=0."""
        from ash_hawk.types import CalibrationCurve

        samples = [
            CalibrationSample(predicted=1.0, actual=True),
            CalibrationSample(predicted=1.0, actual=True),
            CalibrationSample(predicted=1.0, actual=True),
            CalibrationSample(predicted=0.0, actual=False),
            CalibrationSample(predicted=0.0, actual=False),
            CalibrationSample(predicted=0.0, actual=False),
        ]
        curve = CalibrationCurve.compute(samples)
        assert curve.ece == 0.0
        assert 0.0 <= curve.brier_score <= 1.0

    def test_compute_empty_samples(self):
        """Test compute() with empty samples returns ECE=0 and Brier=0."""
        from ash_hawk.types import CalibrationCurve

        curve = CalibrationCurve.compute([])
        assert curve.ece == 0.0
        assert curve.brier_score == 0.0
        assert len(curve.samples) == 0

    def test_compute_brier_score_formula(self):
        """Test Brier score calculation: mean((predicted - actual)^2)."""
        from ash_hawk.types import CalibrationCurve

        # predicted=0.8, actual=True (1.0) -> (0.8 - 1.0)^2 = 0.04
        # predicted=0.3, actual=False (0.0) -> (0.3 - 0.0)^2 = 0.09
        # mean = (0.04 + 0.09) / 2 = 0.065
        samples = [
            CalibrationSample(predicted=0.8, actual=True),
            CalibrationSample(predicted=0.3, actual=False),
        ]
        curve = CalibrationCurve.compute(samples)
        assert abs(curve.brier_score - 0.065) < 0.0001

    def test_compute_brier_score_range(self):
        """Test that Brier score is always in [0, 1] range."""
        from ash_hawk.types import CalibrationCurve

        # Worst case: all predictions wrong
        samples = [
            CalibrationSample(predicted=1.0, actual=False),
            CalibrationSample(predicted=1.0, actual=False),
        ]
        curve = CalibrationCurve.compute(samples)
        assert 0.0 <= curve.brier_score <= 1.0

    def test_extra_forbidden(self):
        """Test that extra fields are forbidden."""
        from ash_hawk.types import CalibrationCurve

        with pytest.raises(Exception):  # ValidationError
            CalibrationCurve(samples=[], ece=0.0, brier_score=0.0, extra="not_allowed")


class TestCalibrationResult:
    """Test CalibrationResult model."""

    def test_basic_result(self):
        """Test basic calibration result creation."""
        from ash_hawk.types import CalibrationCurve, CalibrationResult

        curve = CalibrationCurve(
            samples=[CalibrationSample(predicted=0.8, actual=True)],
            ece=0.1,
            brier_score=0.05,
        )
        result = CalibrationResult(
            curve=curve,
            recommended_threshold=0.7,
            grader_name="llm_judge",
        )
        assert result.curve == curve
        assert result.recommended_threshold == 0.7
        assert result.grader_name == "llm_judge"

    def test_result_with_curve_from_compute(self):
        """Test CalibrationResult with curve computed from samples."""
        from ash_hawk.types import CalibrationCurve, CalibrationResult

        samples = [
            CalibrationSample(predicted=0.9, actual=True),
            CalibrationSample(predicted=0.8, actual=True),
            CalibrationSample(predicted=0.2, actual=False),
        ]
        curve = CalibrationCurve.compute(samples)
        result = CalibrationResult(
            curve=curve,
            recommended_threshold=0.5,
            grader_name="test_grader",
        )
        assert result.curve.ece >= 0.0
        assert result.curve.brier_score >= 0.0

    def test_extra_forbidden(self):
        """Test that extra fields are forbidden."""
        from ash_hawk.types import CalibrationCurve, CalibrationResult

        curve = CalibrationCurve(samples=[], ece=0.0, brier_score=0.0)
        with pytest.raises(Exception):  # ValidationError
            CalibrationResult(
                curve=curve,
                recommended_threshold=0.5,
                grader_name="test",
                extra="not_allowed",
            )
