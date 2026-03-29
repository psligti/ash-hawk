"""Dawn-kestrel thin bridge implementation.

This module implements the thin telemetry wrapper around dawn-kestrel's
agent_v2. It uses the RuntimeHook protocol to capture telemetry in real-time
without constructing any execution context.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ash_hawk.bridge import (
    OutcomeData,
    RunResult,
    TelemetrySink,
    TranscriptData,
)

logger = logging.getLogger(__name__)


@dataclass
class _HookAdapter:
    sink: TelemetrySink
    run_id: str

    async def on_iteration_start(self, ctx: Any) -> None:
        try:
            await self.sink.on_iteration_start(
                {
                    "run_id": self.run_id,
                    "iteration": ctx.iteration,
                    "session_id": ctx.run.session_id if ctx.run else None,
                    "payload": ctx.payload,
                }
            )
        except Exception as e:
            logger.debug(f"TelemetrySink.on_iteration_start failed: {e}")

    async def on_iteration_end(self, ctx: Any) -> None:
        try:
            await self.sink.on_iteration_end(
                {
                    "run_id": self.run_id,
                    "iteration": ctx.iteration,
                    "payload": ctx.payload,
                }
            )
        except Exception as e:
            logger.debug(f"TelemetrySink.on_iteration_end failed: {e}")

    async def on_action_decision(self, ctx: Any) -> None:
        try:
            await self.sink.on_action_decision(
                {
                    "run_id": self.run_id,
                    "iteration": ctx.iteration,
                    "payload": ctx.payload,
                }
            )
        except Exception as e:
            logger.debug(f"TelemetrySink.on_action_decision failed: {e}")

    async def on_tool_result(self, ctx: Any) -> None:
        try:
            await self.sink.on_tool_result(
                {
                    "run_id": self.run_id,
                    "iteration": ctx.iteration,
                    "payload": ctx.payload,
                }
            )
        except Exception as e:
            logger.debug(f"TelemetrySink.on_tool_result failed: {e}")

    async def on_run_complete(self, ctx: Any) -> None:
        try:
            await self.sink.on_run_complete(
                {
                    "run_id": self.run_id,
                    "iteration": ctx.iteration,
                    "payload": ctx.payload,
                }
            )
        except Exception as e:
            logger.debug(f"TelemetrySink.on_run_complete failed: {e}")


class DawnKestrelBridge:
    """Thin bridge to dawn-kestrel agent_v2 with RuntimeHook telemetry."""

    def __init__(
        self,
        agent_path: Path,
        telemetry_sink: TelemetrySink,
        *,
        max_iterations: int = 10,
        workdir: Path | None = None,
        run_id: str | None = None,
    ) -> None:
        self.agent_path = Path(agent_path)
        self.telemetry_sink = telemetry_sink
        self.max_iterations = max_iterations
        self.workdir = Path(workdir) if workdir else Path.cwd()
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"

        if not self.agent_path.exists():
            raise ValueError(f"Agent path does not exist: {self.agent_path}")

    async def run(
        self,
        input: str,
        fixtures: dict[str, Path] | None = None,
        overlays: dict[str, str] | None = None,
    ) -> RunResult:
        start_time = time.time()

        try:
            agents_v2 = self._import_agents_v2()
        except ImportError as e:
            return self._create_error_result(
                f"dawn-kestrel not installed: {e}",
                duration=time.time() - start_time,
            )

        try:
            result = await self._run_agent(
                agents_v2=agents_v2,
                input=input,
                fixtures=fixtures,
                overlays=overlays,
            )
            result.transcript.duration_seconds = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"Agent run failed: {e}")
            return self._create_error_result(
                str(e),
                duration=time.time() - start_time,
            )

    def _import_agents_v2(self) -> Any:
        import importlib

        return importlib.import_module("dawn_kestrel.agents.v2")

    async def _run_agent(
        self,
        agents_v2: Any,
        input: str,
        fixtures: dict[str, Path] | None,
        overlays: dict[str, str] | None,
    ) -> RunResult:
        import asyncio

        AgentV2Builder = agents_v2.AgentV2Builder
        ExecutionLoop = agents_v2.ExecutionLoop
        CommandRouter = agents_v2.CommandRouter
        FilesystemAgentLoader = agents_v2.FilesystemAgentLoader
        InjectedSkillLoader = agents_v2.InjectedSkillLoader
        from_prompt = agents_v2.from_prompt
        RunMetadataEnvelope = agents_v2.RunMetadataEnvelope

        from dawn_kestrel.core.models import Session
        from dawn_kestrel.policy.engine import PolicyEngine

        agent_loader = FilesystemAgentLoader(base_dir=self.workdir)
        skill_loader = InjectedSkillLoader(base_dir=self.workdir)
        command_router = CommandRouter()

        hook_adapter = _HookAdapter(
            sink=self.telemetry_sink,
            run_id=self.run_id,
        )

        run_metadata = RunMetadataEnvelope.create(
            session_id=self.run_id,
            agent_name=self.agent_path.name,
            task_type="eval_run",
            tags=["ash-hawk", "thin-bridge"],
        )

        build_result = (
            AgentV2Builder()
            .with_name(self.agent_path.name)
            .with_description("Ash Hawk evaluation run")
            .with_mode("primary")
            .with_max_iterations(self.max_iterations)
            .build()
        )

        if hasattr(build_result, "is_err") and build_result.is_err():
            raise RuntimeError(f"Agent config build failed: {build_result.error}")

        loop_config = build_result.unwrap()

        policy_engine = self._create_noop_policy_engine(agents_v2)
        runtime = await self._create_runtime(
            agents_v2=agents_v2,
            input=input,
            fixtures=fixtures,
        )

        loop = ExecutionLoop(
            config=loop_config,
            policy_engine=policy_engine,
            command_router=command_router,
            runtime=runtime,
            agent_loader=agent_loader,
            skill_loader=skill_loader,
            runtime_hooks=[hook_adapter],
            run_metadata=run_metadata,
        )

        initial_input = from_prompt(input, max_iterations=self.max_iterations)

        session = Session(
            id=self.run_id,
            slug=self.agent_path.name,
            project_id="ash-hawk",
            directory=str(self.workdir.resolve()),
            title="Ash Hawk evaluation",
            version="v2",
        )

        loop_result = await loop.run(initial_input, session)

        return self._convert_result(loop_result)

    def _create_noop_policy_engine(self, agents_v2: Any) -> Any:
        class NoOpPolicyEngine:
            def propose(self, policy_input: Any) -> Any:
                StepProposal = agents_v2.StepProposal
                intent = getattr(policy_input, "goal", "Execute task")
                return StepProposal(intent=intent, actions=[])

        return NoOpPolicyEngine()

    async def _create_runtime(
        self,
        agents_v2: Any,
        input: str,
        fixtures: dict[str, Path] | None,
    ) -> Any:
        @dataclass
        class ThinRuntime:
            input_text: str
            response: str = ""
            executed: bool = False

            async def execute_step(self) -> dict[str, Any] | None:
                if self.executed:
                    return {
                        "response": self.response,
                        "parts": [],
                        "tokens": {
                            "input": 0,
                            "output": 0,
                            "reasoning": 0,
                        },
                        "tools_called": [],
                    }

                self.executed = True
                self.response = f"Processed: {self.input_text[:100]}..."

                return {
                    "response": self.response,
                    "parts": [],
                    "tokens": {
                        "input": 100,
                        "output": 50,
                        "reasoning": 0,
                    },
                    "tools_called": [],
                }

        return ThinRuntime(input_text=input)

    def _convert_result(self, loop_result: Any) -> RunResult:
        messages: list[dict[str, Any]] = []
        if loop_result.transcript:
            transcript_dict = loop_result.transcript.to_dict()
            messages = transcript_dict.get("messages", [])

        tool_calls: list[dict[str, Any]] = []
        if loop_result.telemetry:
            for entry in loop_result.telemetry.get_tool_call_ledger():
                tool_calls.append(
                    {
                        "tool": entry.get("tool_name"),
                        "input": entry.get("arguments", {}),
                        "output": entry.get("result"),
                        "error": entry.get("error"),
                    }
                )

        trace_events: list[dict[str, Any]] = []
        if loop_result.telemetry:
            for event in loop_result.telemetry.get_trace_events():
                trace_events.append(event)

        token_usage = {
            "input": loop_result.tokens_used.get("input", 0),
            "output": loop_result.tokens_used.get("output", 0),
            "reasoning": loop_result.tokens_used.get("reasoning", 0),
            "cache_read": loop_result.tokens_used.get("cache_read", 0),
            "cache_write": loop_result.tokens_used.get("cache_write", 0),
        }

        summary = loop_result.telemetry.get_summary() if loop_result.telemetry else {}

        success = True
        error_message = ""
        if loop_result.outcome is not None:
            success = bool(loop_result.outcome.success)
            error_message = str(loop_result.outcome.message) if not success else ""
        elif loop_result.error:
            success = False
            error_message = loop_result.error

        transcript = TranscriptData(
            messages=messages,
            tool_calls=tool_calls,
            trace_events=trace_events,
            token_usage=token_usage,
            cost_usd=float(summary.get("cost_usd_total", 0.0)),
            agent_response=loop_result.response,
            error_trace=loop_result.error,
        )

        outcome = OutcomeData(
            success=success,
            error=error_message if not success else None,
            message="Agent completed successfully" if success else error_message,
        )

        return RunResult(
            transcript=transcript,
            outcome=outcome,
            run_id=self.run_id,
            iterations=loop_result.iterations,
            tools_used=loop_result.tools_used,
        )

    def _create_error_result(
        self,
        error: str,
        duration: float,
    ) -> RunResult:
        transcript = TranscriptData(
            messages=[],
            tool_calls=[],
            trace_events=[],
            token_usage={
                "input": 0,
                "output": 0,
                "reasoning": 0,
                "cache_read": 0,
                "cache_write": 0,
            },
            cost_usd=0.0,
            duration_seconds=duration,
            error_trace=error,
        )

        outcome = OutcomeData(
            success=False,
            error=error,
            message=error,
        )

        return RunResult(
            transcript=transcript,
            outcome=outcome,
            run_id=self.run_id,
            iterations=0,
            tools_used=[],
        )


__all__ = ["DawnKestrelBridge"]
