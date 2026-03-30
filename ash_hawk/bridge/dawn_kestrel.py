"""Dawn-kestrel thin bridge implementation.

This module implements the thin telemetry wrapper around dawn-kestrel's
agent_v2. It uses the RuntimeHook protocol to capture telemetry in real-time
without constructing any execution context.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from ash_hawk.bridge import (
    OutcomeData,
    RunResult,
    TelemetrySink,
    TranscriptData,
)

logger = logging.getLogger(__name__)


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
            result = await self._run_agent(
                input=input,
                fixtures=fixtures,
                overlays=overlays,
            )
            result.transcript.duration_seconds = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"Agent run failed: {e}")
            try:
                await self.telemetry_sink.on_run_complete(
                    {
                        "run_id": self.run_id,
                        "iteration": 0,
                        "payload": {
                            "success": False,
                            "error": str(e),
                            "total_iterations": 0,
                        },
                    }
                )
            except Exception as sink_error:
                logger.debug(
                    f"TelemetrySink.on_run_complete failed during error handling: {sink_error}"
                )
            return self._create_error_result(
                str(e),
                duration=time.time() - start_time,
            )

    async def _run_agent(
        self,
        input: str,
        fixtures: dict[str, Path] | None,
        overlays: dict[str, str] | None,
    ) -> RunResult:
        from ash_hawk.agents.dawn_kestrel import DawnKestrelAgentRunner
        from ash_hawk.policy import PolicyEnforcer
        from ash_hawk.types import EvalStatus, EvalTask, ToolPermission, ToolSurfacePolicy

        provider, model = self._resolve_provider_model()
        runner = DawnKestrelAgentRunner(provider=provider, model=model)

        effective_workdir, cleanup = self._prepare_runtime_workdir(
            fixtures=fixtures, overlays=overlays
        )

        try:
            task = EvalTask(
                id=self.run_id,
                description="Ash Hawk thin bridge run",
                input={"prompt": input},
            )
            policy = ToolSurfacePolicy(
                allowed_tools=["*"],
                default_permission=ToolPermission.ALLOW,
                allowed_roots=[str(effective_workdir.resolve())],
                network_allowed=True,
                timeout_seconds=300.0,
            )
            policy_enforcer = PolicyEnforcer(policy)

            config: dict[str, Any] = {
                "workdir": str(effective_workdir.resolve()),
                "max_iterations": self.max_iterations,
                "agent_name": self.agent_path.name,
                "trial_id": self.run_id,
            }

            await self.telemetry_sink.on_iteration_start(
                {
                    "run_id": self.run_id,
                    "iteration": 1,
                    "session_id": self.run_id,
                    "payload": {
                        "input": input,
                        "workdir": str(effective_workdir.resolve()),
                        "fixtures_applied": len(fixtures or {}),
                        "overlays_applied": len(overlays or {}),
                    },
                }
            )

            transcript, outcome = await runner.run(
                task=task, policy_enforcer=policy_enforcer, config=config
            )

            for idx, tool_call in enumerate(transcript.tool_calls, start=1):
                tool_name = tool_call.get("name") or tool_call.get("tool")
                await self.telemetry_sink.on_tool_result(
                    {
                        "run_id": self.run_id,
                        "iteration": 1,
                        "payload": {
                            "index": idx,
                            "tool_name": tool_name,
                            "status": "error" if tool_call.get("error") else "ok",
                            "arguments": tool_call.get("arguments") or tool_call.get("input") or {},
                            "result": tool_call.get("output"),
                            "error": tool_call.get("error"),
                        },
                    }
                )

            await self.telemetry_sink.on_iteration_end(
                {
                    "run_id": self.run_id,
                    "iteration": 1,
                    "payload": {
                        "token_usage": transcript.token_usage.model_dump(),
                        "tool_calls": len(transcript.tool_calls),
                    },
                }
            )

            success = outcome.status == EvalStatus.COMPLETED
            error_message = outcome.error_message if not success else None

            await self.telemetry_sink.on_run_complete(
                {
                    "run_id": self.run_id,
                    "iteration": 1,
                    "payload": {
                        "success": success,
                        "error": error_message,
                        "total_iterations": 1,
                    },
                }
            )

            return RunResult(
                transcript=TranscriptData(
                    messages=transcript.messages,
                    tool_calls=transcript.tool_calls,
                    trace_events=transcript.trace_events,
                    token_usage=transcript.token_usage.model_dump(),
                    cost_usd=transcript.cost_usd,
                    duration_seconds=transcript.duration_seconds,
                    agent_response=str(transcript.agent_response or ""),
                    error_trace=transcript.error_trace,
                ),
                outcome=OutcomeData(
                    success=success,
                    message="Agent completed successfully"
                    if success
                    else (error_message or "Agent failed"),
                    error=error_message,
                ),
                run_id=self.run_id,
                iterations=1,
                tools_used=[
                    str(tool_call.get("name") or tool_call.get("tool"))
                    for tool_call in transcript.tool_calls
                    if tool_call.get("name") or tool_call.get("tool")
                ],
            )
        finally:
            cleanup()

    def _prepare_runtime_workdir(
        self,
        *,
        fixtures: dict[str, Path] | None,
        overlays: dict[str, str] | None,
    ) -> tuple[Path, Any]:
        fixture_map = fixtures or {}
        overlay_map = overlays or {}
        if not fixture_map and not overlay_map:
            return self.workdir, lambda: None

        temp_dir = tempfile.TemporaryDirectory(prefix="ash-hawk-thin-")
        temp_root = Path(temp_dir.name)

        for fixture_name, fixture_path in fixture_map.items():
            source_path = Path(fixture_path)
            if not source_path.exists():
                raise ValueError(f"Fixture path does not exist: {source_path}")
            destination = temp_root / fixture_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)

        for overlay_path, overlay_content in overlay_map.items():
            destination = temp_root / overlay_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(overlay_content, encoding="utf-8")

        return temp_root, temp_dir.cleanup

    def _resolve_provider_model(self) -> tuple[str, str]:
        import importlib

        settings_module = importlib.import_module("dawn_kestrel.core.settings")
        settings = settings_module.get_settings()
        default_account = settings.get_default_account()

        if default_account is not None:
            provider = str(default_account.provider_id.value)
        else:
            provider = str(settings.get_default_provider().value)

        default_model = str(settings.get_default_model(provider)).strip()
        if not default_model:
            raise ValueError(f"Could not resolve default model for provider '{provider}'")

        return provider, default_model

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
