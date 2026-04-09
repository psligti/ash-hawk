# type-hygiene: skip-file
from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pydantic as pd

if TYPE_CHECKING:
    from ash_hawk.types import EvalTrial

logger = logging.getLogger(__name__)

DIAGNOSIS_PROMPT = """You are analyzing a failed evaluation trial. Diagnose why it failed.

Trial ID: {trial_id}
Task ID: {task_id}
Agent Response: {agent_response}
Grader Results: {grader_results}
Error Trace: {error_trace}
Transcript Excerpt:
{transcript_excerpt}

Recent Tool Calls:
{tool_calls_excerpt}

Recent Trace Events:
{trace_excerpt}

Provide your diagnosis as JSON:
{{
    "failure_summary": "one-line summary",
    "root_cause": "detailed root cause analysis",
    "suggested_fix": "concrete fix suggestion",
    "target_files": ["file1.py", "file2.py"],
    "confidence": 0.8
}}"""


class Diagnosis(pd.BaseModel):
    """LLM-generated diagnosis of a failed evaluation trial."""

    model_config = pd.ConfigDict(extra="forbid")

    trial_id: str = pd.Field(description="ID of the failed trial")
    failure_summary: str = pd.Field(description="One-line summary of the failure")
    root_cause: str = pd.Field(description="Detailed root cause analysis")
    suggested_fix: str = pd.Field(description="Concrete fix suggestion")
    target_files: list[str] = pd.Field(
        default_factory=list, description="Files that should be modified"
    )
    confidence: float = pd.Field(
        default=0.0, ge=0.0, le=1.0, description="LLM confidence in the diagnosis"
    )


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _format_transcript_excerpt(trial: EvalTrial) -> tuple[str, str, str]:
    transcript = trial.result.transcript if trial.result else None
    if transcript is None:
        return "none", "none", "none"

    transcript_lines: list[str] = []
    for message in transcript.messages[-8:]:
        role = str(message.get("role", "unknown"))
        content = _truncate(str(message.get("content", "")), 300)
        transcript_lines.append(f"[{role}] {content}")

    tool_lines: list[str] = []
    for tool_call in transcript.tool_calls[-8:]:
        name = str(tool_call.get("name") or tool_call.get("tool") or "unknown")
        args = tool_call.get("arguments") or tool_call.get("input") or {}
        tool_lines.append(_truncate(f"{name}({json.dumps(args, default=str)})", 300))

    trace_lines: list[str] = []
    for event in transcript.trace_events[-12:]:
        event_type = str(event.get("event_type", "unknown"))
        data = event.get("data", {})
        trace_lines.append(_truncate(f"{event_type}: {json.dumps(data, default=str)}", 300))

    return (
        "\n".join(transcript_lines) if transcript_lines else "none",
        "\n".join(tool_lines) if tool_lines else "none",
        "\n".join(trace_lines) if trace_lines else "none",
    )


async def diagnose_failures(
    failures: list[EvalTrial],
    trace_dir: Path | None = None,
    agent_content: dict[str, str] | None = None,
    lesson_store: Any | None = None,
    console: Any | None = None,
) -> list[Diagnosis]:
    """Diagnose multiple failures in parallel using asyncio.gather."""
    results = await asyncio.gather(
        *[
            _diagnose_single(f, trace_dir, agent_content, lesson_store, console=console)
            for f in failures
        ],
        return_exceptions=True,
    )

    diagnoses: list[Diagnosis] = []
    for failure, result in zip(failures, results):
        if isinstance(result, Exception):
            logger.warning("Diagnosis failed for trial %s", failure.id, exc_info=result)
        elif isinstance(result, Diagnosis):
            diagnoses.append(result)

    return diagnoses


async def _diagnose_single(
    trial: EvalTrial,
    trace_dir: Path | None,
    agent_content: dict[str, str] | None = None,
    lesson_store: Any | None = None,
    console: Any | None = None,
) -> Diagnosis | None:
    transcript = trial.result.transcript if trial.result else None
    agent_response = ""
    grader_results_str = "[]"
    error_trace = ""

    if transcript is not None:
        agent_response = str(transcript.agent_response or "")[:2000]
        if trial.result and trial.result.grader_results:
            grader_results_str = json.dumps([r.model_dump() for r in trial.result.grader_results])[
                :2000
            ]
        error_trace = str(transcript.error_trace or "")[:2000]

    transcript_excerpt, tool_calls_excerpt, trace_excerpt = _format_transcript_excerpt(trial)

    prompt = DIAGNOSIS_PROMPT.format(
        trial_id=trial.id,
        task_id=trial.task_id,
        agent_response=agent_response,
        grader_results=grader_results_str,
        error_trace=error_trace,
        transcript_excerpt=transcript_excerpt,
        tool_calls_excerpt=tool_calls_excerpt,
        trace_excerpt=trace_excerpt,
    )

    agent_context = ""
    if agent_content:
        agent_files_section = "\n\n## Agent Content\n"
        for key in sorted(agent_content.keys()):
            if "AGENT.md" in key or "agent.md" in key:
                agent_files_section += f"\n### {key}\n{agent_content[key][:2000]}\n"
        skill_count = 0
        for key in sorted(agent_content.keys()):
            if "skill" in key.lower() and skill_count < 2:
                agent_files_section += f"\n### {key}\n{agent_content[key][:1000]}\n"
                skill_count += 1
        agent_context = agent_files_section

    prompt = prompt + agent_context

    # Inject past lessons to avoid repeating failed approaches
    if lesson_store is not None:
        failed_lessons = lesson_store.get_failed_attempts()
        if failed_lessons:
            lesson_section = lesson_store.format_lessons_for_prompt(failed_lessons, max_lessons=8)
            prompt += (
                "\n\n## Past Failed Attempts (DO NOT repeat these approaches)\n" + lesson_section
            )
            logger.info(
                "Injecting %d past failed lessons into diagnosis prompt for trial %s",
                len(failed_lessons),
                trial.id,
            )

    if console is not None:
        console.print(f"    [dim]Diagnosing trial {trial.id} (task={trial.task_id})...[/dim]")
    response = await _call_llm(prompt)
    if response is None:
        error_msg = error_trace or agent_response or "Unknown failure"
        if console is not None:
            console.print(
                f"    [yellow]⚠ LLM unavailable, using fallback diagnosis for trial {trial.id}[/yellow]"
            )
        return Diagnosis(
            trial_id=trial.id,
            failure_summary=error_msg[:200],
            root_cause=(
                f"LLM diagnosis unavailable. Error trace: {error_trace[:500]}"
                if error_trace
                else "LLM diagnosis unavailable"
            ),
            suggested_fix="Review transcript and error trace manually",
            target_files=[],
            confidence=0.1,
        )

    result = _parse_diagnosis_response(trial.id, response)
    if result is not None and console is not None:
        files_str = ", ".join(result.target_files[:3]) if result.target_files else "none"
        console.print(
            f"    [dim]Diagnosed trial {trial.id}: "
            f"{result.failure_summary[:60]}  "
            f"files=[{files_str}][/dim]"
        )
    elif result is None and console is not None:
        console.print(f"    [yellow]⚠ Could not parse diagnosis for trial {trial.id}[/yellow]")
    return result


async def _call_llm(prompt: str) -> str | None:
    try:
        config_module = importlib.import_module("dawn_kestrel.base.config")
        client_module = importlib.import_module("dawn_kestrel.provider.llm_client")
    except ImportError:
        logger.warning("dawn-kestrel not installed, returning stub diagnosis")
        return None

    try:
        dk_config = config_module.load_agent_config()
        provider = (
            dk_config.get("runtime.provider")
            or os.environ.get("DAWN_KESTREL_PROVIDER")
            or "anthropic"
        )
        model = (
            dk_config.get("runtime.model")
            or os.environ.get("DAWN_KESTREL_MODEL")
            or "claude-sonnet-4-20250514"
        )
        api_key = config_module.get_config_api_key(provider) or None

        LLMClient = getattr(client_module, "LLMClient")
        client = LLMClient(provider_id=provider, model=model, api_key=api_key)

        result = await client.chat_completion(
            system_prompt="You are analyzing a failed evaluation trial to diagnose the root cause.",
            user_message=prompt,
        )
        return result if isinstance(result, str) else None
    except Exception:
        logger.warning("LLM call failed", exc_info=True)
        return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first balanced-brace JSON object from text."""
    depth = 0
    start: int | None = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    parsed = json.loads(text[start : i + 1])
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    start = None
                    continue
    return None


def _parse_diagnosis_response(trial_id: str, response: str) -> Diagnosis | None:
    data = _extract_json_object(response)
    if data is None:
        logger.warning("No JSON found in diagnosis response for %s", trial_id)
        return None

    required_fields = ["failure_summary", "root_cause", "suggested_fix"]
    if not all(f in data for f in required_fields):
        logger.warning("Missing required fields in diagnosis for %s", trial_id)
        return None

    return Diagnosis(
        trial_id=trial_id,
        failure_summary=str(data["failure_summary"]),
        root_cause=str(data["root_cause"]),
        suggested_fix=str(data["suggested_fix"]),
        target_files=data.get("target_files", []),
        confidence=float(data.get("confidence", 0.0)),
    )
