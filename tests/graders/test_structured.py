"""Tests for ash_hawk.graders.structured module."""

import pytest

from ash_hawk.graders.structured import (
    FormatGrader,
    SchemaGrader,
    ToolUsageGrader,
)
from ash_hawk.types import (
    EvalTranscript,
    EvalTrial,
    GraderResult,
    GraderSpec,
)


class TestSchemaGrader:
    """Test SchemaGrader implementation."""

    def test_name(self):
        """SchemaGrader has correct name."""
        grader = SchemaGrader()
        assert grader.name == "schema"

    def test_invalid_format_raises(self):
        pass

    @pytest.mark.asyncio
    async def test_grade_with_dict_response(self):
        """SchemaGrader works with dict response."""
        grader = SchemaGrader(required_fields=["name", "age"])

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(agent_response={"name": "Alice", "age": 30})
        spec = GraderSpec(grader_type="schema")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_grade_with_json_string_response(self):
        """SchemaGrader works with JSON string response."""
        grader = SchemaGrader(required_fields=["status"])

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(agent_response='{"status": "ok"}')
        spec = GraderSpec(grader_type="schema")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_grade_missing_required_fields(self):
        """SchemaGrader fails when required fields are missing."""
        grader = SchemaGrader(required_fields=["name", "age", "email"])

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(agent_response={"name": "Alice"})
        spec = GraderSpec(grader_type="schema")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 1 / 3
        assert "Missing required fields" in result.details["errors"][0]

    @pytest.mark.asyncio
    async def test_grade_no_response(self):
        """SchemaGrader fails when no response in transcript."""
        grader = SchemaGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(agent_response=None)
        spec = GraderSpec(grader_type="schema")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        assert "No agent response" in result.details["error"]

    @pytest.mark.asyncio
    async def test_grade_invalid_json_string(self):
        """SchemaGrader fails when response is not valid JSON."""
        grader = SchemaGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(agent_response="not valid json")
        spec = GraderSpec(grader_type="schema")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_grade_with_custom_validator(self):
        """SchemaGrader supports custom validation function."""

        def validate_age(data):
            return data.get("age", 0) >= 18

        grader = SchemaGrader(custom_validator=validate_age)

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(agent_response={"name": "Bob", "age": 25})
        spec = GraderSpec(grader_type="schema")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_grade_custom_validator_fails(self):
        """SchemaGrader fails when custom validator returns False."""

        def validate_age(data):
            return data.get("age", 0) >= 18

        grader = SchemaGrader(custom_validator=validate_age)

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(agent_response={"name": "Bob", "age": 15})
        spec = GraderSpec(grader_type="schema")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert "Custom validation failed" in result.details["errors"]

    @pytest.mark.asyncio
    async def test_grade_with_config_override(self):
        """SchemaGrader reads required_fields from spec config."""
        grader = SchemaGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(agent_response={"foo": "bar"})
        spec = GraderSpec(
            grader_type="schema",
            config={"required_fields": ["foo"]},
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True


class TestFormatGrader:
    """Test FormatGrader implementation."""

    def test_name(self):
        """FormatGrader has correct name."""
        grader = FormatGrader()
        assert grader.name == "format"

    def test_invalid_format_raises(self):
        """FormatGrader raises on invalid format type."""
        with pytest.raises(ValueError, match="Invalid format"):
            FormatGrader(format_type="invalid")

    @pytest.mark.asyncio
    async def test_grade_valid_json(self):
        """FormatGrader validates JSON format."""
        grader = FormatGrader(format_type="json")

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(agent_response='{"key": "value"}')
        spec = GraderSpec(grader_type="format")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["format"] == "json"

    @pytest.mark.asyncio
    async def test_grade_invalid_json(self):
        """FormatGrader fails on invalid JSON."""
        grader = FormatGrader(format_type="json")

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(agent_response="{not valid json}")
        spec = GraderSpec(grader_type="format")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        assert "JSON parse error" in result.details["errors"][0]

    @pytest.mark.asyncio
    async def test_grade_valid_csv(self):
        """FormatGrader validates CSV format."""
        grader = FormatGrader(format_type="csv")

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(agent_response="name,age\nAlice,30\nBob,25")
        spec = GraderSpec(grader_type="format")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_grade_dict_converted_to_json(self):
        """FormatGrader converts dict response to JSON for validation."""
        grader = FormatGrader(format_type="json")

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(agent_response={"key": "value"})
        spec = GraderSpec(grader_type="format")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_grade_no_response(self):
        """FormatGrader fails when no response in transcript."""
        grader = FormatGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(agent_response=None)
        spec = GraderSpec(grader_type="format")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_grade_with_custom_validator(self):
        """FormatGrader supports custom validation."""

        def has_name_field(content):
            import json

            data = json.loads(content)
            return "name" in data

        grader = FormatGrader(format_type="json", custom_validator=has_name_field)

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(agent_response='{"name": "Alice"}')
        spec = GraderSpec(grader_type="format")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_grade_with_config_format_override(self):
        """FormatGrader reads format from spec config."""
        grader = FormatGrader(format_type="json")

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(agent_response="a,b,c")
        spec = GraderSpec(
            grader_type="format",
            config={"format": "csv"},
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.details["format"] == "csv"


class TestToolUsageGrader:
    """Test ToolUsageGrader implementation."""

    def test_name(self):
        """ToolUsageGrader has correct name."""
        grader = ToolUsageGrader()
        assert grader.name == "tool_usage"

    @pytest.mark.asyncio
    async def test_grade_no_tool_calls(self):
        """ToolUsageGrader handles empty tool calls."""
        grader = ToolUsageGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(tool_calls=[])
        spec = GraderSpec(grader_type="tool_usage")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.details["total_calls"] == 0

    @pytest.mark.asyncio
    async def test_grade_required_tools_present(self):
        """ToolUsageGrader passes when required tools are called."""
        grader = ToolUsageGrader(required_tools=["read", "write"])

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            tool_calls=[
                {"tool": "read", "input": {"path": "/tmp"}},
                {"tool": "write", "input": {"path": "/tmp/out"}},
            ]
        )
        spec = GraderSpec(grader_type="tool_usage")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert "read" in result.details["called_tools"]
        assert "write" in result.details["called_tools"]

    @pytest.mark.asyncio
    async def test_grade_required_tools_missing(self):
        """ToolUsageGrader fails when required tools are missing."""
        grader = ToolUsageGrader(required_tools=["read", "write", "delete"])

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(tool_calls=[{"tool": "read", "input": {"path": "/tmp"}}])
        spec = GraderSpec(grader_type="tool_usage")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert "Missing required tools" in result.details["errors"][0]

    @pytest.mark.asyncio
    async def test_grade_forbidden_tools_used(self):
        """ToolUsageGrader fails when forbidden tools are used."""
        grader = ToolUsageGrader(forbidden_tools=["delete"])

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(tool_calls=[{"tool": "delete", "input": {"path": "/tmp"}}])
        spec = GraderSpec(grader_type="tool_usage")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert "Used forbidden tools" in result.details["errors"][0]

    @pytest.mark.asyncio
    async def test_grade_min_calls(self):
        """ToolUsageGrader enforces minimum call count."""
        grader = ToolUsageGrader(min_calls=3)

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            tool_calls=[
                {"tool": "read", "input": {}},
                {"tool": "read", "input": {}},
            ]
        )
        spec = GraderSpec(grader_type="tool_usage")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert "Too few calls" in result.details["errors"][0]

    @pytest.mark.asyncio
    async def test_grade_max_calls(self):
        """ToolUsageGrader enforces maximum call count."""
        grader = ToolUsageGrader(max_calls=2)

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            tool_calls=[
                {"tool": "read", "input": {}},
                {"tool": "read", "input": {}},
                {"tool": "read", "input": {}},
            ]
        )
        spec = GraderSpec(grader_type="tool_usage")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert "Too many calls" in result.details["errors"][0]

    @pytest.mark.asyncio
    async def test_grade_tool_sequence(self):
        """ToolUsageGrader validates tool call sequence."""
        grader = ToolUsageGrader(tool_sequence=["read", "process", "write"])

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            tool_calls=[
                {"tool": "read", "input": {}},
                {"tool": "process", "input": {}},
                {"tool": "write", "input": {}},
            ]
        )
        spec = GraderSpec(grader_type="tool_usage")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_grade_tool_sequence_partial_match(self):
        """ToolUsageGrader finds sequence as subsequence."""
        grader = ToolUsageGrader(tool_sequence=["read", "write"])

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            tool_calls=[
                {"tool": "read", "input": {}},
                {"tool": "transform", "input": {}},
                {"tool": "write", "input": {}},
            ]
        )
        spec = GraderSpec(grader_type="tool_usage")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_grade_tool_sequence_incomplete(self):
        """ToolUsageGrader fails when sequence is incomplete."""
        grader = ToolUsageGrader(tool_sequence=["read", "process", "write"])

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            tool_calls=[
                {"tool": "read", "input": {}},
                {"tool": "process", "input": {}},
            ]
        )
        spec = GraderSpec(grader_type="tool_usage")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert "sequence" in result.details["errors"][0].lower()

    @pytest.mark.asyncio
    async def test_grade_custom_validator(self):
        """ToolUsageGrader supports custom validation."""

        def check_read_before_write(tool_calls):
            tools = [c.get("tool") for c in tool_calls]
            if "write" in tools and "read" not in tools:
                return False
            return True

        grader = ToolUsageGrader(custom_validator=check_read_before_write)

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(tool_calls=[{"tool": "write", "input": {}}])
        spec = GraderSpec(grader_type="tool_usage")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_grade_with_config_override(self):
        """ToolUsageGrader reads config from spec."""
        grader = ToolUsageGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(tool_calls=[{"tool": "read", "input": {}}])
        spec = GraderSpec(
            grader_type="tool_usage",
            config={"required_tools": ["read"]},
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_grade_unique_tools_tracking(self):
        """ToolUsageGrader tracks unique tools called."""
        grader = ToolUsageGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            tool_calls=[
                {"tool": "read", "input": {}},
                {"tool": "read", "input": {}},
                {"tool": "write", "input": {}},
            ]
        )
        spec = GraderSpec(grader_type="tool_usage")

        result = await grader.grade(trial, transcript, spec)

        assert set(result.details["unique_tools"]) == {"read", "write"}

    @pytest.mark.asyncio
    async def test_grade_tool_name_variants(self):
        """ToolUsageGrader handles different tool name keys."""
        grader = ToolUsageGrader(required_tools=["bash"])

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            tool_calls=[
                {"name": "bash", "input": {}},
                {"tool_name": "bash", "input": {}},
            ]
        )
        spec = GraderSpec(grader_type="tool_usage")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
