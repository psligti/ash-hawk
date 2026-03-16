"""Dawn-Kestrel specific integration for post-run hooks."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ash_hawk.contracts import RunArtifact, ToolCallRecord
from ash_hawk.integration.post_run_hook import (
    DefaultPostRunReviewHook,
    HookConfig,
)

if TYPE_CHECKING:
    from ash_hawk.types import EvalTranscript


class TranscriptToArtifactConverter:
    """Converts dawn-kestrel transcripts to RunArtifacts."""

    def convert(
        self,
        transcript: EvalTranscript,
        run_id: str | None = None,
        suite_id: str | None = None,
        agent_name: str = "dawn-kestrel",
    ) -> RunArtifact:
        tool_calls = self._convert_tool_calls(transcript)

        return RunArtifact(
            run_id=run_id or f"run-{uuid4().hex[:8]}",
            suite_id=suite_id or "unknown",
            agent_name=agent_name,
            outcome="success" if not transcript.error_trace else "failure",
            tool_calls=tool_calls,
            steps=[],
            messages=transcript.messages or [],
            total_duration_ms=int(transcript.duration_seconds * 1000),
            token_usage=transcript.token_usage,
            cost_usd=transcript.cost_usd,
            error_message=transcript.error_trace,
            metadata={
                "converted_from": "dawn-kestrel-transcript",
                "agent_response": transcript.agent_response,
            },
            created_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

    def _convert_tool_calls(
        self,
        transcript: EvalTranscript,
    ) -> list[ToolCallRecord]:
        records: list[ToolCallRecord] = []
        raw_calls = transcript.tool_calls or []

        for idx, call in enumerate(raw_calls):
            if isinstance(call, dict):
                record = ToolCallRecord(
                    call_id=f"tc-{uuid4().hex[:8]}-{idx}",
                    tool_name=call.get("tool", call.get("name", "unknown")),
                    arguments=call.get("input", call.get("arguments", {})),
                    result=call.get("output", call.get("result")),
                    outcome="success" if "error" not in call else "failure",
                    error_message=call.get("error"),
                    started_at=datetime.now(UTC),
                    completed_at=datetime.now(UTC),
                )
                records.append(record)

        return records


class DawnKestrelPostRunHook(DefaultPostRunReviewHook):
    """Post-run hook specialized for dawn-kestrel agents.

    Extends DefaultPostRunReviewHook with:
    - Transcript to RunArtifact conversion
    - Dawn-kestrel specific metadata extraction
    """

    def __init__(
        self,
        config: HookConfig | None = None,
        agent_name: str = "dawn-kestrel",
    ) -> None:
        super().__init__(config)
        self._agent_name = agent_name
        self._converter = TranscriptToArtifactConverter()

    def on_transcript_complete(
        self,
        transcript: EvalTranscript,
        run_id: str | None = None,
        suite_id: str | None = None,
    ) -> None:
        artifact = self._converter.convert(
            transcript=transcript,
            run_id=run_id,
            suite_id=suite_id,
            agent_name=self._agent_name,
        )
        self.on_run_complete(artifact)

    def convert_transcript(
        self,
        transcript: EvalTranscript,
        run_id: str | None = None,
        suite_id: str | None = None,
    ) -> RunArtifact:
        return self._converter.convert(
            transcript=transcript,
            run_id=run_id,
            suite_id=suite_id,
            agent_name=self._agent_name,
        )
