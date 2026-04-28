from __future__ import annotations

import logging
import sys
from decimal import Decimal
from pathlib import Path

import pytest
from dawn_kestrel.provider.llm_client import LLMResponse
from dawn_kestrel.provider.provider_types import TokenUsage

from ash_hawk.thin_runtime import RuntimeGoal, create_default_harness
from ash_hawk.thin_runtime.live_eval import emit_observed_event
from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_types import (
    AuditRunResult,
    AuditToolContext,
    FailureToolContext,
    ToolExecutionPayload,
    TraceRecord,
)


def test_harness_prints_human_readable_progress_and_suppresses_console_logs(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    harness = create_default_harness(workdir=Path.cwd(), storage_root=tmp_path)
    logger = logging.getLogger("ash_hawk.thin_runtime.console_test")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    observed_events = [
        TraceRecord(
            tool=f"event_tool_{index}",
            event_type="tool_result",
            success=(index % 2 == 0),
            preview=f"preview {index}",
        )
        for index in range(7)
    ]

    def noisy_handler(call: ToolCall) -> ToolResult:
        logger.warning("suppressed logger noise")
        emit_observed_event(
            {
                "tool": "event_tool_0",
                "event_type": "tool_result",
                "success": True,
                "preview": "matched alpha.py, beta.py",
            }
        )
        emit_observed_event(
            {
                "tool": "event_tool_1",
                "event_type": "tool_result",
                "success": False,
                "preview": "permission denied while reading config",
            }
        )
        return ToolResult(
            tool_name=call.tool_name,
            success=True,
            payload=ToolExecutionPayload(
                audit_updates=AuditToolContext(
                    events=observed_events,
                    tool_usage=["load_workspace_state"],
                    run_summary={
                        "changed_file_count": "3",
                        "changed_files_preview": "alpha.py, beta.py, gamma.py",
                    },
                    diff_report={"files": 1, "diff_line_count": 4},
                    run_result=AuditRunResult(
                        run_id="eval-run-123",
                        aggregate_score=0.75,
                        aggregate_passed=True,
                    ),
                ),
                failure_updates=FailureToolContext(
                    explanations=["Workspace state loaded without validation errors"]
                ),
                message="Loaded workspace state",
            ),
        )

    harness.tools.register_handler("load_workspace_state", noisy_handler)

    try:
        result = harness.execute(
            RuntimeGoal(goal_id="goal-console", description="Show progress clearly"),
            requested_skills=["workspace-governance"],
            tool_execution_order=["load_workspace_state"],
        )
    finally:
        logger.removeHandler(handler)
        logger.propagate = True

    output = capsys.readouterr().out
    assert result.success is True
    assert "Thin Runtime Run" in output
    assert "Goal: goal-console" in output
    assert "Description: Show progress clearly" in output
    assert "Agent: coordinator" in output
    assert "Skills: workspace-governance" in output
    assert "Console: human-readable steps only. Logger output is muted for this run." in output
    assert "Decision: load_workspace_state via explicit_order." in output
    assert "Selected from explicit tool execution order override." in output
    assert "Step 1/10: load_workspace_state()" in output
    assert output.index("Decision: load_workspace_state via explicit_order.") < output.index(
        "Step 1/10: load_workspace_state()"
    )
    assert "  Observed events:" in output
    assert "    - event_tool_0 | ok | matched alpha.py, beta.py" in output
    assert "    - event_tool_1 | failed | permission denied while reading config" in output
    assert "Step 1 result: ✓ load_workspace_state()" in output
    assert "→ Loaded workspace state" in output
    assert output.index("    - event_tool_0 | ok | matched alpha.py, beta.py") < output.index(
        "Step 1 result: ✓ load_workspace_state()"
    )
    assert "Result snippet:" in output
    assert "    Summary: Loaded workspace state" in output
    assert "    Validation: run eval-run-123, score 0.75, passed" in output
    assert "    Output details:" in output
    assert "      Changed file count: 3" in output
    assert "      Changed files preview: alpha.py, beta.py, gamma.py" in output
    assert "    Diff summary:" in output
    assert "      Files: 1" in output
    assert "      Diff line count: 4" in output
    assert "    Failure signals:" in output
    assert "      - Workspace state loaded without validation errors" in output
    assert "    Tool usage: load_workspace_state" in output
    assert "      - event_tool_6 | ok | preview 6" in output
    assert "Memory consolidation: applied" in output
    assert "Run Summary" in output
    assert f"Run ID: {result.run_id}" in output
    assert "Decision trace:" in output
    assert "Recorded tool activity:" in output
    assert "load_workspace_state: ok" in output
    assert "      - event_tool_1 | failed | preview 1" in output
    assert "... +" not in output
    assert f"  Run directory: {tmp_path / 'runs' / result.run_id}" in output
    assert f"  Transcript JSON: {tmp_path / 'runs' / result.run_id / 'execution.json'}" in output
    assert f"  Run summary JSON: {tmp_path / 'runs' / result.run_id / 'summary.json'}" in output
    assert (
        f"  Session memory snapshot (global): {tmp_path / 'memory' / 'session_snapshot.json'}"
        in output
    )
    assert f"  Durable memory snapshot (global): {tmp_path / 'memory' / 'snapshot.json'}" in output
    assert "suppressed logger noise" not in output
    assert "{" not in output
    assert "}" not in output


def test_harness_suppresses_stderr_console_logs_and_restores_handler_level(
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = create_default_harness(workdir=Path.cwd())
    logger = logging.getLogger("ash_hawk.thin_runtime.console_test.stderr")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    original_level = handler.level
    logger.addHandler(handler)

    def noisy_handler(call: ToolCall) -> ToolResult:
        del call
        logger.warning("suppressed stderr noise")
        return ToolResult(tool_name="load_workspace_state", success=True)

    harness.tools.register_handler("load_workspace_state", noisy_handler)

    try:
        result = harness.execute(
            RuntimeGoal(goal_id="goal-console-stderr", description="Keep stderr clean"),
            requested_skills=["workspace-governance"],
            tool_execution_order=["load_workspace_state"],
        )
    finally:
        logger.removeHandler(handler)
        logger.propagate = True

    captured = capsys.readouterr()
    assert result.success is True
    assert handler.level == original_level
    assert "suppressed stderr noise" not in captured.err


def test_harness_suppresses_rich_style_console_handlers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = create_default_harness(workdir=Path.cwd())
    logger = logging.getLogger("ash_hawk.thin_runtime.console_test.rich")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    rich_messages: list[str] = []

    class RichHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            rich_messages.append(record.getMessage())

    handler = RichHandler()
    original_level = handler.level
    logger.addHandler(handler)

    def noisy_handler(call: ToolCall) -> ToolResult:
        del call
        logger.warning("suppressed rich noise")
        return ToolResult(tool_name="load_workspace_state", success=True)

    harness.tools.register_handler("load_workspace_state", noisy_handler)

    try:
        result = harness.execute(
            RuntimeGoal(goal_id="goal-console-rich", description="Keep rich console clean"),
            requested_skills=["workspace-governance"],
            tool_execution_order=["load_workspace_state"],
        )
    finally:
        logger.removeHandler(handler)
        logger.propagate = True

    output = capsys.readouterr().out
    assert result.success is True
    assert handler.level == original_level
    assert rich_messages == []
    assert "Goal: goal-console-rich" in output


def test_harness_prints_live_error_snippet_for_failed_tool(
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = create_default_harness(workdir=Path.cwd())

    def failing_handler(call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_name=call.tool_name,
            success=False,
            error="Tests failed with exit code 1",
            payload=ToolExecutionPayload(message="Tests failed"),
        )

    harness.tools.register_handler("load_workspace_state", failing_handler)

    result = harness.execute(
        RuntimeGoal(goal_id="goal-console-failure", description="Show failure clearly"),
        requested_skills=["workspace-governance"],
        tool_execution_order=["load_workspace_state"],
    )

    output = capsys.readouterr().out
    assert result.success is False
    assert "Step 1 result: ✗ load_workspace_state()" in output
    assert "→ Tests failed with exit code 1" in output
    assert "Result snippet:" in output
    assert "Summary: Tests failed" in output
    assert "Error: Tests failed with exit code 1" in output


def test_harness_prints_model_authored_thinking_line(capsys: pytest.CaptureFixture[str]) -> None:
    class FakeClient:
        provider_id = "test"

        def __init__(self) -> None:
            self.calls = 0

        async def complete(
            self,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]] | None = None,
            options: object = None,
        ) -> LLMResponse:
            del messages
            del tools
            del options
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    text="I will read workspace context first to ground the next action.",
                    usage=TokenUsage(input=0, output=0, reasoning=0),
                    finish_reason="tool_use",
                    cost=Decimal("0"),
                    tool_calls=[
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "load_workspace_state", "arguments": "{}"},
                            "tool": "load_workspace_state",
                            "input": {},
                        }
                    ],
                )
            return LLMResponse(
                text="done",
                usage=TokenUsage(input=0, output=0, reasoning=0),
                finish_reason="stop",
                cost=Decimal("0"),
                tool_calls=None,
            )

    harness = create_default_harness(workdir=Path.cwd())
    harness.runner.set_client_factory(lambda: FakeClient())
    result = harness.execute(
        RuntimeGoal(goal_id="goal-console-thinking", description="Show model-authored thinking"),
    )

    output = capsys.readouterr().out
    assert result.success is True
    assert "Thinking: I will read workspace context first to ground the next action." in output
    assert "Decision: load_workspace_state via dk_tool_calls." not in output


def test_harness_labels_fallback_rationale_as_decision(capsys: pytest.CaptureFixture[str]) -> None:
    class FakeClient:
        provider_id = "test"

        def __init__(self) -> None:
            self.calls = 0

        async def complete(
            self,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]] | None = None,
            options: object = None,
        ) -> LLMResponse:
            del messages
            del tools
            del options
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    text="",
                    usage=TokenUsage(input=0, output=0, reasoning=0),
                    finish_reason="tool_use",
                    cost=Decimal("0"),
                    tool_calls=[
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "load_workspace_state", "arguments": "{}"},
                            "tool": "load_workspace_state",
                            "input": {},
                        }
                    ],
                )
            return LLMResponse(
                text="done",
                usage=TokenUsage(input=0, output=0, reasoning=0),
                finish_reason="stop",
                cost=Decimal("0"),
                tool_calls=None,
            )

    harness = create_default_harness(workdir=Path.cwd())
    harness.runner.set_client_factory(lambda: FakeClient())
    result = harness.execute(
        RuntimeGoal(goal_id="goal-console-fallback", description="No model-authored rationale"),
    )

    output = capsys.readouterr().out
    assert result.success is True
    assert "Decision: load_workspace_state via dk_tool_calls." in output
    assert "Model emitted tool call(s): load_workspace_state" in output
    assert "Thinking:" not in output
