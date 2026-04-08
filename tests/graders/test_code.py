"""Tests for ash_hawk.graders.code module."""

import pytest

from ash_hawk.graders.code import (
    CommandResult,
    SandboxConfig,
    SandboxViolationError,
    StaticAnalysisGrader,
    StringMatchGrader,
    TestRunnerGrader,
    ToolCallGrader,
    TranscriptGrader,
    redact_secrets,
)
from ash_hawk.types import (
    EvalTranscript,
    EvalTrial,
    GraderSpec,
    TokenUsage,
)


class TestSandboxConfig:
    """Test SandboxConfig class."""

    def test_default_allowlist_includes_common_tools(self):
        config = SandboxConfig()
        assert config.is_command_allowed("pytest")
        assert config.is_command_allowed("python")
        assert config.is_command_allowed("ruff")

    def test_custom_allowlist(self):
        config = SandboxConfig(command_allowlist=["mytool", "custom*"])
        assert config.is_command_allowed("mytool")
        assert config.is_command_allowed("customrunner")
        assert not config.is_command_allowed("pytest")

    def test_path_validation_with_allowed_roots(self):
        config = SandboxConfig(allowed_roots=["/workspace"])
        assert config.validate_path_access("/workspace/file.py")
        assert config.validate_path_access("/workspace/subdir/file.py")
        assert not config.validate_path_access("/etc/passwd")

    def test_path_validation_no_roots_configured(self):
        config = SandboxConfig()
        assert not config.validate_path_access("/any/path")


class TestRedactSecrets:
    """Test secret redaction functionality."""

    def test_redact_api_key(self):
        text = "api_key = 'sk-1234567890abcdef'"
        redacted, found = redact_secrets(text)
        assert "sk-1234567890abcdef" not in redacted
        assert len(found) >= 1

    def test_redact_aws_access_key(self):
        text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        redacted, found = redact_secrets(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted

    def test_redact_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        redacted, found = redact_secrets(text)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted

    def test_no_secrets_returns_unchanged(self):
        text = "This is normal text without any secrets"
        redacted, found = redact_secrets(text)
        assert redacted == text
        assert found == []


class TestStringMatchGrader:
    """Test StringMatchGrader class."""

    @pytest.fixture
    def grader(self):
        return StringMatchGrader()

    @pytest.fixture
    def base_trial(self):
        return EvalTrial(id="t1", task_id="task1")

    @pytest.fixture
    def base_transcript(self):
        return EvalTranscript()

    @pytest.mark.asyncio
    async def test_exact_match_passes(self, grader, base_trial, base_transcript):
        base_transcript.agent_response = "Hello World"
        spec = GraderSpec(
            grader_type="string_match",
            config={"expected": "Hello World", "mode": "exact"},
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert result.passed
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_exact_match_fails(self, grader, base_trial, base_transcript):
        base_transcript.agent_response = "Hello World"
        spec = GraderSpec(
            grader_type="string_match",
            config={"expected": "Different", "mode": "exact"},
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert not result.passed
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_case_insensitive_match(self, grader, base_trial, base_transcript):
        base_transcript.agent_response = "HELLO WORLD"
        spec = GraderSpec(
            grader_type="string_match",
            config={"expected": "hello world", "mode": "exact", "case_sensitive": False},
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert result.passed

    @pytest.mark.asyncio
    async def test_regex_match(self, grader, base_trial, base_transcript):
        base_transcript.agent_response = "The answer is 42"
        spec = GraderSpec(
            grader_type="string_match",
            config={"expected": r"answer is \d+", "mode": "regex"},
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert result.passed

    @pytest.mark.asyncio
    async def test_fuzzy_match(self, grader, base_trial, base_transcript):
        base_transcript.agent_response = "Hello World!"
        spec = GraderSpec(
            grader_type="string_match",
            config={"expected": "Hello World", "mode": "fuzzy", "min_similarity": 0.9},
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert result.passed
        assert result.details["similarity"] > 0.9

    @pytest.mark.asyncio
    async def test_partial_credit_for_substring(self, grader, base_trial, base_transcript):
        base_transcript.agent_response = "The answer is Hello World today"
        spec = GraderSpec(
            grader_type="string_match",
            config={"expected": "Hello World", "mode": "exact", "partial_credit": True},
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert result.score == 0.5

    @pytest.mark.asyncio
    async def test_normalize_whitespace(self, grader, base_trial, base_transcript):
        base_transcript.agent_response = "Hello    World\n\nTest"
        spec = GraderSpec(
            grader_type="string_match",
            config={"expected": "Hello World Test", "mode": "exact", "normalize_whitespace": True},
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert result.passed


class TestTestRunnerGrader:
    """Test TestRunnerGrader class."""

    @pytest.fixture
    def grader(self):
        return TestRunnerGrader()

    @pytest.fixture
    def base_trial(self):
        return EvalTrial(id="t1", task_id="task1")

    @pytest.fixture
    def base_transcript(self):
        return EvalTranscript()

    @pytest.mark.asyncio
    async def test_requires_test_path(self, grader, base_trial, base_transcript):
        spec = GraderSpec(grader_type="test_runner", config={})
        result = await grader.grade(base_trial, base_transcript, spec)
        assert not result.passed
        assert "test_path is required" in result.error_message

    @pytest.mark.asyncio
    async def test_sandbox_violation_for_disallowed_command(
        self, grader, base_trial, base_transcript
    ):
        spec = GraderSpec(
            grader_type="test_runner",
            config={
                "test_path": "/tests",
                "command_allowlist": ["mytool"],
            },
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert not result.passed
        assert "Sandbox violation" in result.error_message


class TestStaticAnalysisGrader:
    """Test StaticAnalysisGrader class."""

    @pytest.fixture
    def grader(self):
        return StaticAnalysisGrader()

    @pytest.fixture
    def base_trial(self):
        return EvalTrial(id="t1", task_id="task1")

    @pytest.fixture
    def base_transcript(self):
        return EvalTranscript()

    @pytest.mark.asyncio
    async def test_requires_target_path(self, grader, base_trial, base_transcript):
        spec = GraderSpec(grader_type="static_analysis", config={})
        result = await grader.grade(base_trial, base_transcript, spec)
        assert not result.passed
        assert "target_path is required" in result.error_message

    @pytest.mark.asyncio
    async def test_default_tools_ruff(self, grader, base_trial, base_transcript):
        spec = GraderSpec(
            grader_type="static_analysis",
            config={"target_path": "/code"},
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert "ruff" in result.details.get("tools_run", [])


class TestToolCallGrader:
    """Test ToolCallGrader class."""

    @pytest.fixture
    def grader(self):
        return ToolCallGrader()

    @pytest.fixture
    def base_trial(self):
        return EvalTrial(id="t1", task_id="task1")

    @pytest.fixture
    def base_transcript(self):
        return EvalTranscript(
            tool_calls=[
                {"tool": "read", "input": {"path": "/file.py"}},
                {"tool": "write", "input": {"path": "/output.txt", "content": "test"}},
            ]
        )

    @pytest.mark.asyncio
    async def test_no_expected_calls_passes(self, grader, base_trial, base_transcript):
        spec = GraderSpec(grader_type="tool_call", config={})
        result = await grader.grade(base_trial, base_transcript, spec)
        assert result.passed
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_expected_tool_found(self, grader, base_trial, base_transcript):
        spec = GraderSpec(
            grader_type="tool_call",
            config={"expected_calls": [{"tool": "read"}]},
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert result.passed
        assert result.details["matched_count"] == 1

    @pytest.mark.asyncio
    async def test_expected_tool_not_found(self, grader, base_trial, base_transcript):
        spec = GraderSpec(
            grader_type="tool_call",
            config={"expected_calls": [{"tool": "execute"}]},
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert not result.passed
        assert result.details["matched_count"] == 0

    @pytest.mark.asyncio
    async def test_glob_pattern_matching(self, grader, base_trial, base_transcript):
        spec = GraderSpec(
            grader_type="tool_call",
            config={"expected_calls": [{"tool": "re*"}]},
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert result.passed

    @pytest.mark.asyncio
    async def test_input_parameter_matching(self, grader, base_trial, base_transcript):
        spec = GraderSpec(
            grader_type="tool_call",
            config={
                "expected_calls": [
                    {"tool": "read", "input": {"path": "/file.py"}, "input_match": "contains"}
                ]
            },
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert result.passed

    @pytest.mark.asyncio
    async def test_partial_credit(self, grader, base_trial, base_transcript):
        spec = GraderSpec(
            grader_type="tool_call",
            config={
                "expected_calls": [
                    {"tool": "read"},
                    {"tool": "execute"},
                ],
                "partial_credit": True,
            },
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert not result.passed
        assert result.score == 0.5

    @pytest.mark.asyncio
    async def test_min_count_constraint(self, grader, base_trial):
        transcript = EvalTranscript(
            tool_calls=[
                {"tool": "read", "input": {"path": "/a"}},
                {"tool": "read", "input": {"path": "/b"}},
            ]
        )
        spec = GraderSpec(
            grader_type="tool_call",
            config={"expected_calls": [{"tool": "read", "min_count": 2}]},
        )
        result = await grader.grade(base_trial, transcript, spec)
        assert result.passed

    @pytest.mark.asyncio
    async def test_max_count_constraint(self, grader, base_trial):
        transcript = EvalTranscript(
            tool_calls=[
                {"tool": "read", "input": {"path": "/a"}},
                {"tool": "read", "input": {"path": "/b"}},
                {"tool": "read", "input": {"path": "/c"}},
            ]
        )
        spec = GraderSpec(
            grader_type="tool_call",
            config={"expected_calls": [{"tool": "read", "max_count": 2}]},
        )
        result = await grader.grade(base_trial, transcript, spec)
        assert not result.passed


class TestTranscriptGrader:
    """Test TranscriptGrader class."""

    @pytest.fixture
    def grader(self):
        return TranscriptGrader()

    @pytest.fixture
    def base_trial(self):
        return EvalTrial(id="t1", task_id="task1")

    @pytest.fixture
    def base_transcript(self):
        return EvalTranscript(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            token_usage=TokenUsage(input=100, output=50),
            duration_seconds=5.0,
            tool_calls=[{"tool": "read"}],
        )

    @pytest.mark.asyncio
    async def test_max_turns_violation(self, grader, base_trial, base_transcript):
        spec = GraderSpec(
            grader_type="transcript",
            config={"max_turns": 0},
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert not result.passed
        assert any("Turn count" in v for v in result.details["violations"])

    @pytest.mark.asyncio
    async def test_max_tokens_violation(self, grader, base_trial, base_transcript):
        spec = GraderSpec(
            grader_type="transcript",
            config={"max_tokens": 100},
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert not result.passed

    @pytest.mark.asyncio
    async def test_max_duration_violation(self, grader, base_trial, base_transcript):
        spec = GraderSpec(
            grader_type="transcript",
            config={"max_duration_seconds": 1.0},
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert not result.passed

    @pytest.mark.asyncio
    async def test_max_tool_calls_violation(self, grader, base_trial, base_transcript):
        spec = GraderSpec(
            grader_type="transcript",
            config={"max_tool_calls": 0},
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert not result.passed

    @pytest.mark.asyncio
    async def test_no_violations_passes(self, grader, base_trial, base_transcript):
        spec = GraderSpec(
            grader_type="transcript",
            config={
                "max_turns": 10,
                "max_tokens": 1000,
                "max_duration_seconds": 60,
                "max_tool_calls": 10,
            },
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert result.passed
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_error_trace_violation(self, grader, base_trial):
        transcript = EvalTranscript(error_trace="Stack trace here")
        spec = GraderSpec(
            grader_type="transcript",
            config={"require_no_errors": True},
        )
        result = await grader.grade(base_trial, transcript, spec)
        assert not result.passed

    @pytest.mark.asyncio
    async def test_partial_credit_reduces_score(self, grader, base_trial, base_transcript):
        spec = GraderSpec(
            grader_type="transcript",
            config={"max_turns": 0, "partial_credit": True},
        )
        result = await grader.grade(base_trial, base_transcript, spec)
        assert result.score < 1.0
        assert result.score > 0.0

    @pytest.mark.asyncio
    async def test_metrics_collected(self, grader, base_trial, base_transcript):
        spec = GraderSpec(grader_type="transcript", config={})
        result = await grader.grade(base_trial, base_transcript, spec)
        assert "metrics" in result.details
        assert result.details["metrics"]["turn_count"] == 1
        assert result.details["metrics"]["message_count"] == 2


class TestCommandResult:
    """Test CommandResult dataclass."""

    def test_default_values(self):
        result = CommandResult(return_code=0, stdout="output", stderr="")
        assert result.timed_out is False
        assert result.execution_time_seconds == 0.0
        assert result.redacted_secrets == []

    def test_with_all_fields(self):
        result = CommandResult(
            return_code=1,
            stdout="",
            stderr="error",
            timed_out=True,
            execution_time_seconds=5.5,
            redacted_secrets=["secret1"],
        )
        assert result.return_code == 1
        assert result.timed_out
        assert result.execution_time_seconds == 5.5


class TestSandboxViolationError:
    """Test SandboxViolationError exception."""

    def test_is_exception(self):
        assert issubclass(SandboxViolationError, Exception)

    def test_message(self):
        error = SandboxViolationError("Command not allowed")
        assert str(error) == "Command not allowed"


class TestGraderNames:
    """Test that all graders have correct name properties."""

    def test_string_match_grader_name(self):
        assert StringMatchGrader().name == "string_match"

    def test_test_runner_grader_name(self):
        assert TestRunnerGrader().name == "test_runner"

    def test_static_analysis_grader_name(self):
        assert StaticAnalysisGrader().name == "static_analysis"

    def test_tool_call_grader_name(self):
        assert ToolCallGrader().name == "tool_call"

    def test_transcript_grader_name(self):
        assert TranscriptGrader().name == "transcript"
