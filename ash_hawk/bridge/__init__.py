"""Thin telemetry bridge for running real dawn-kestrel agents.

This module provides a thin wrapper around dawn-kestrel's agent_v2
that captures telemetry via RuntimeHook callbacks. Ash-hawk no longer
constructs execution contexts - it runs real agents and observes.

Key components:
- TelemetrySink: Protocol for receiving telemetry events
- TranscriptData: Captured transcript from agent run
- OutcomeData: Captured outcome from agent run
- run_real_agent: Main entry point for thin bridge

Usage:
    from ash_hawk.bridge import run_real_agent, TelemetrySink

    class MySink(TelemetrySink):
        async def on_iteration(self, data: dict) -> None:
            print(f"Iteration {data['iteration']}")

    transcript, outcome = await run_real_agent(
        agent_path=Path(".dawn-kestrel/agents/bolt-merlin"),
        input="Find all TODOs",
        telemetry_sink=MySink(),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "TelemetrySink",
    "TranscriptData",
    "OutcomeData",
    "RunResult",
    "run_real_agent",
]


class TelemetrySink(Protocol):
    """Protocol for receiving telemetry events from agent runs.

    Implement this to capture real-time telemetry from agent execution.
    All methods are optional - implement only what you need.
    """

    async def on_iteration_start(self, data: dict[str, Any]) -> None:
        """Called at the start of each iteration.

        Args:
            data: Contains 'iteration', 'session_id', 'max_iterations'
        """
        ...

    async def on_iteration_end(self, data: dict[str, Any]) -> None:
        """Called at the end of each iteration.

        Args:
            data: Contains 'iteration', 'duration_ms', 'response_length'
        """
        ...

    async def on_action_decision(self, data: dict[str, Any]) -> None:
        """Called when a policy decision is made.

        Args:
            data: Contains 'decision', 'action_type', 'risk_level'
        """
        ...

    async def on_tool_result(self, data: dict[str, Any]) -> None:
        """Called after a tool execution completes.

        Args:
            data: Contains 'tool_name', 'status', 'duration_ms'
        """
        ...

    async def on_run_complete(self, data: dict[str, Any]) -> None:
        """Called when the agent run completes.

        Args:
            data: Contains 'success', 'error', 'total_iterations'
        """
        ...


@dataclass
class TranscriptData:
    """Captured transcript from agent run.

    Contains the full conversation history, tool calls, and trace events
    from a real agent execution.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    trace_events: list[dict[str, Any]] = field(default_factory=list)
    token_usage: dict[str, int] = field(
        default_factory=lambda: {
            "input": 0,
            "output": 0,
            "reasoning": 0,
            "cache_read": 0,
            "cache_write": 0,
        }
    )
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    agent_response: str = ""
    error_trace: str | None = None

    def to_eval_transcript(self) -> Any:
        """Convert to EvalTranscript for compatibility with existing graders."""
        from ash_hawk.types import EvalTranscript, TokenUsage

        return EvalTranscript(
            messages=self.messages,
            tool_calls=self.tool_calls,
            trace_events=self.trace_events,
            token_usage=TokenUsage(
                input=self.token_usage.get("input", 0),
                output=self.token_usage.get("output", 0),
                reasoning=self.token_usage.get("reasoning", 0),
                cache_read=self.token_usage.get("cache_read", 0),
                cache_write=self.token_usage.get("cache_write", 0),
            ),
            cost_usd=self.cost_usd,
            duration_seconds=self.duration_seconds,
            agent_response=self.agent_response,
            error_trace=self.error_trace,
        )


@dataclass
class OutcomeData:
    """Captured outcome from agent run.

    Indicates whether the agent succeeded or failed.
    """

    success: bool
    message: str = ""
    error: str | None = None

    def to_eval_outcome(self) -> Any:
        """Convert to EvalOutcome for compatibility."""
        from ash_hawk.types import EvalOutcome, FailureMode

        if self.success:
            return EvalOutcome.success()
        return EvalOutcome.failure(
            FailureMode.AGENT_ERROR,
            self.error or self.message,
        )


@dataclass
class RunResult:
    """Result from run_real_agent.

    Contains both the transcript and outcome from the agent run.
    """

    transcript: TranscriptData
    outcome: OutcomeData
    run_id: str = ""
    iterations: int = 0
    tools_used: list[str] = field(default_factory=list)


async def run_real_agent(
    agent_path: Path,
    input: str,
    telemetry_sink: TelemetrySink,
    fixtures: dict[str, Path] | None = None,
    overlays: dict[str, str] | None = None,
    *,
    max_iterations: int = 10,
    workdir: Path | None = None,
    run_id: str | None = None,
) -> RunResult:
    """Run a real dawn-kestrel agent_v2 with telemetry capture.

    This is the thin bridge - it does NOT construct execution contexts.
    It loads the real agent from disk and runs it with telemetry hooks.

    Args:
        agent_path: Path to .dawn-kestrel/agents/{name} directory
        input: User prompt/task for the agent
        telemetry_sink: Callback for telemetry events
        fixtures: Optional fixture paths to inject into workdir
        overlays: Optional candidate content to test before write-back
        max_iterations: Maximum iterations for the agent loop
        workdir: Working directory for agent execution (defaults to cwd)
        run_id: Optional run ID for tracking

    Returns:
        RunResult containing transcript and outcome

    Raises:
        ImportError: If dawn-kestrel is not installed
        ValueError: If agent_path does not exist or is invalid
    """
    from ash_hawk.bridge.dawn_kestrel import DawnKestrelBridge

    bridge = DawnKestrelBridge(
        agent_path=agent_path,
        telemetry_sink=telemetry_sink,
        max_iterations=max_iterations,
        workdir=workdir,
        run_id=run_id,
    )
    return await bridge.run(input, fixtures=fixtures, overlays=overlays)
