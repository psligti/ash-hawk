import pytest

from ash_hawk.graders.trace_assertions import TraceContentGrader
from ash_hawk.types import EvalTranscript, EvalTrial, GraderSpec


class TestTraceContentGrader:
    def test_name(self):
        assert TraceContentGrader().name == "trace_content"

    @pytest.mark.asyncio
    async def test_grade_passes_for_required_events_tools_and_markers(self):
        grader = TraceContentGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            agent_response="Loaded skill note-lark-memory and captured evidence.",
            tool_calls=[{"name": "note-lark_memory_search", "input": {"query": "prefs"}}],
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "ToolCallEvent",
                    "ts": "2026-01-01T00:00:00Z",
                    "data": {"tool": "note-lark_memory_search", "input": {"query": "prefs"}},
                },
                {
                    "schema_version": 1,
                    "event_type": "ToolResultEvent",
                    "ts": "2026-01-01T00:00:01Z",
                    "data": {"tool": "note-lark_memory_search", "result": {"items": []}},
                },
            ],
        )
        spec = GraderSpec(
            grader_type="trace_content",
            config={
                "required_event_types": ["ToolCallEvent", "ToolResultEvent"],
                "required_tool_names": ["note-lark_memory_search"],
                "required_mcp_prefixes": ["note-lark_"],
                "required_skill_markers": ["note-lark-memory"],
            },
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["violations"] == []

    @pytest.mark.asyncio
    async def test_grade_fails_when_required_conditions_missing(self):
        grader = TraceContentGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            agent_response="No skills loaded.",
            tool_calls=[{"name": "bash", "input": {"command": "ls"}}],
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "ToolCallEvent",
                    "ts": "2026-01-01T00:00:00Z",
                    "data": {"tool": "bash", "input": {"command": "ls"}},
                }
            ],
        )
        spec = GraderSpec(
            grader_type="trace_content",
            config={
                "required_event_types": ["ToolCallEvent", "ToolResultEvent"],
                "required_tool_names": ["note-lark_memory_search"],
                "required_mcp_prefixes": ["note-lark_"],
                "required_skill_markers": ["note-lark-memory"],
            },
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        violations = result.details["violations"]
        assert "missing_required_event_type:ToolResultEvent" in violations
        assert "missing_required_tool:note-lark_memory_search" in violations
        assert "missing_required_mcp_prefix:note-lark_" in violations
        assert "missing_required_skill_marker:note-lark-memory" in violations

    @pytest.mark.asyncio
    async def test_grade_respects_forbidden_patterns(self):
        grader = TraceContentGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            agent_response="Loaded disallowed skill debug-root-hack.",
            tool_calls=[{"name": "bash", "input": {"command": "pwd"}}],
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "ToolCallEvent",
                    "ts": "2026-01-01T00:00:00Z",
                    "data": {"tool": "bash", "input": {"command": "pwd"}},
                }
            ],
        )
        spec = GraderSpec(
            grader_type="trace_content",
            config={
                "forbidden_tool_names": ["bash"],
                "forbidden_skill_markers": ["debug-root-hack"],
                "forbidden_event_types": ["ToolCallEvent"],
            },
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        violations = result.details["violations"]
        assert "forbidden_tool_present:bash" in violations
        assert "forbidden_skill_marker_present:debug-root-hack" in violations
        assert "forbidden_event_type_present:ToolCallEvent" in violations
