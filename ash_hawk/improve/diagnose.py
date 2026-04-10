# type-hygiene: skip-file
from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pydantic as pd

from ash_hawk.improve.agentic_diagnoser import investigate_trial_with_explorer

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

Generate AT LEAST 5 distinct candidate improvement ideas for this single failure whenever the
trace contains enough signal to do so. Each idea must be meaningfully different in intervention
angle, target files, or proposed fix. Do not repeat the same idea with small wording changes.

Prefer diversity across prompting, runtime config, skills, tool behavior, verification, path
resolution, and execution flow.

Provide your diagnosis as JSON:
{{
    "ideas": [
        {{
            "failure_summary": "one-line summary",
            "root_cause": "detailed root cause analysis",
            "suggested_fix": "concrete fix suggestion",
            "target_files": ["file1.py", "file2.py"],
            "confidence": 0.8
        }}
    ]
}}

If you truly cannot produce 5 non-duplicate ideas from the evidence, return as many distinct
ideas as possible, but maximize variety."""


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
    actionable: bool = pd.Field(
        default=True,
        description="Whether this diagnosis is actionable enough to test as a hypothesis",
    )
    diagnosis_mode: Literal["llm", "fallback_llm_unavailable", "fallback_parse_failure"] = pd.Field(
        default="llm", description="How the diagnosis was produced"
    )
    degraded_reason: str | None = pd.Field(
        default=None,
        description="Why this diagnosis is degraded or non-actionable",
    )


def _fallback_diagnosis(
    trial: EvalTrial,
    *,
    failure_summary: str,
    root_cause: str,
    suggested_fix: str,
    diagnosis_mode: Literal["fallback_llm_unavailable", "fallback_parse_failure"],
    degraded_reason: str,
) -> list[Diagnosis]:
    return [
        Diagnosis(
            trial_id=trial.id,
            failure_summary=failure_summary[:200],
            root_cause=root_cause,
            suggested_fix=suggested_fix,
            target_files=[],
            confidence=0.1,
            actionable=False,
            diagnosis_mode=diagnosis_mode,
            degraded_reason=degraded_reason,
        )
    ]


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
    agent_path: Path | None = None,
    lesson_store: Any | None = None,
    console: Any | None = None,
) -> list[Diagnosis]:
    """Diagnose multiple failures in parallel using asyncio.gather."""
    results = await asyncio.gather(
        *[
            _diagnose_single(
                f,
                trace_dir,
                agent_content,
                agent_path,
                lesson_store,
                console=console,
            )
            for f in failures
        ],
        return_exceptions=True,
    )

    diagnoses: list[Diagnosis] = []
    for failure, result in zip(failures, results):
        if isinstance(result, Exception):
            logger.warning("Diagnosis failed for trial %s", failure.id, exc_info=result)
        elif isinstance(result, list):
            diagnoses.extend(result)

    return diagnoses


async def _diagnose_single(
    trial: EvalTrial,
    trace_dir: Path | None,
    agent_content: dict[str, str] | None = None,
    agent_path: Path | None = None,
    lesson_store: Any | None = None,
    console: Any | None = None,
) -> list[Diagnosis]:
    transcript = trial.result.transcript if trial.result else None
    agent_response = ""
    grader_results_str = "[]"
    error_trace = ""

    if trial.result is None:
        error_trace = f"Trial produced no result (status={trial.status})"

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

    response: str | None = None
    explorer_result = None
    if agent_path is not None:
        if console is not None:
            console.print(
                f"    [dim]Diagnosis mode:[/dim] explorer  [dim]trial={trial.id}  agent={agent_path.name}[/dim]"
            )
        explorer_raw = await investigate_trial_with_explorer(trial, agent_path, lesson_store)
        if isinstance(explorer_raw, str):
            response = explorer_raw
            explorer_result = None
        else:
            explorer_result = explorer_raw
            if explorer_result is not None:
                response = explorer_result.response
        if explorer_raw is None and console is not None:
            console.print(
                f"    [yellow]⚠ Explorer diagnosis unavailable for trial {trial.id}; falling back[/yellow]"
            )

    if response is None:
        response = await _call_llm(prompt)

    if response is None:
        error_msg = error_trace or agent_response or "Unknown failure"
        if console is not None:
            console.print(
                f"    [yellow]⚠ LLM unavailable, using fallback diagnosis for trial {trial.id}[/yellow]"
            )
        return _fallback_diagnosis(
            trial,
            failure_summary=error_msg,
            root_cause=(
                f"LLM diagnosis unavailable. Error trace: {error_trace[:500]}"
                if error_trace
                else "LLM diagnosis unavailable"
            ),
            suggested_fix="Review transcript and error trace manually",
            diagnosis_mode="fallback_llm_unavailable",
            degraded_reason="diagnosis_llm_unavailable",
        )

    results = _parse_diagnosis_response(trial.id, response)
    if results and console is not None:
        primary = results[0]
        files_str = ", ".join(primary.target_files[:3]) if primary.target_files else "none"
        console.print(f"    [dim]Diagnosis output parsed successfully for trial {trial.id}[/dim]")
        if explorer_result is not None and explorer_result.tool_calls_used is not None:
            console.print(
                "    [dim]Explorer tool budget used: "
                f"tools {explorer_result.tool_calls_used}/{explorer_result.tool_calls_max or '?'}  "
                f"reads {explorer_result.file_reads_used or 0}/{explorer_result.file_reads_max or '?'}  "
                f"search {explorer_result.search_calls_used or 0}/{explorer_result.search_calls_max or '?'}[/dim]"
            )
        console.print(
            f"    [dim]Explorer returned {len(results)} idea(s) for trial {trial.id}[/dim]"
        )
        console.print(
            f"    [dim]Primary diagnosis: "
            f"{primary.failure_summary[:60]}  "
            f"files=[{files_str}]  ideas={len(results)}[/dim]"
        )
    elif not results and console is not None:
        console.print(f"    [yellow]⚠ Could not parse diagnosis for trial {trial.id}[/yellow]")
    if not results:
        return _fallback_diagnosis(
            trial,
            failure_summary=error_trace or agent_response or "Diagnosis response was not parseable",
            root_cause="Diagnosis response was not parseable JSON",
            suggested_fix="Review the failed trial manually and retry diagnosis",
            diagnosis_mode="fallback_parse_failure",
            degraded_reason="diagnosis_parse_failure",
        )

    for result in results:
        if not result.target_files:
            result.actionable = False
            result.degraded_reason = "diagnosis_missing_target_files"

    return results


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


def _parse_diagnosis_response(trial_id: str, response: str) -> list[Diagnosis]:
    data = _extract_json_object(response)
    if data is None:
        logger.warning("No JSON found in diagnosis response for %s", trial_id)
        return []

    raw_ideas = data.get("ideas")
    if isinstance(raw_ideas, list):
        parsed_ideas = _parse_diagnosis_ideas(trial_id, raw_ideas)
        if parsed_ideas:
            return parsed_ideas

    parsed_single = _parse_diagnosis_ideas(trial_id, [data])
    if parsed_single:
        return parsed_single

    logger.warning("Missing required fields in diagnosis for %s", trial_id)
    return []


def _parse_diagnosis_ideas(trial_id: str, raw_ideas: list[Any]) -> list[Diagnosis]:
    required_fields = ["failure_summary", "root_cause", "suggested_fix"]
    diagnoses: list[Diagnosis] = []
    seen_keys: set[tuple[str, str, tuple[str, ...]]] = set()

    for raw_idea in raw_ideas:
        if not isinstance(raw_idea, dict):
            continue
        if not all(field in raw_idea for field in required_fields):
            continue

        target_files_raw = raw_idea.get("target_files", [])
        target_files = (
            [str(path) for path in target_files_raw] if isinstance(target_files_raw, list) else []
        )
        failure_summary = str(raw_idea["failure_summary"])
        root_cause = str(raw_idea["root_cause"])
        dedupe_key = (failure_summary, root_cause, tuple(target_files))
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        diagnoses.append(
            Diagnosis(
                trial_id=trial_id,
                failure_summary=failure_summary,
                root_cause=root_cause,
                suggested_fix=str(raw_idea["suggested_fix"]),
                target_files=target_files,
                confidence=float(raw_idea.get("confidence", 0.0)),
                actionable=True,
                diagnosis_mode="llm",
            )
        )

    return diagnoses
