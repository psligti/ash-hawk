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

DIAGNOSIS_MESSAGE_LIMIT = 16
DIAGNOSIS_MESSAGE_CHAR_LIMIT = 600
DIAGNOSIS_TOOL_CALL_LIMIT = 20
DIAGNOSIS_TOOL_CALL_CHAR_LIMIT = 600
DIAGNOSIS_TRACE_EVENT_LIMIT = 24
DIAGNOSIS_TRACE_EVENT_CHAR_LIMIT = 600

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

Mutable Agent Files (relative to agent root):
{agent_file_manifest}

Generate 2-4 distinct candidate improvement ideas for this single failure whenever the trace
contains enough signal to do so. Prefer fewer ideas when the evidence points to one or two likely
fixes.

Strongly prefer small, simple, localized fixes. The best idea usually changes a single file or a
tightly-coupled pair of files. Avoid sweeping refactors, broad prompt rewrites, and multi-module
changes unless the evidence clearly requires them.

Prefer executable code-path fixes first: `execute.py`, `coding_agent.py`, `prompt_builder.py`,
`tool_dispatcher.py`, or specific `tools/*` modules. Treat shared prompt or skill files as a last
resort, not a default lever.

Do not combine a local code-path fix with shared prompt/skill cleanup in the same idea unless the
evidence proves both are required together. If a local code path can plausibly explain the failure,
do not propose `prompts/*` or `skills/*` changes.

Prefer diversity only across plausible small fixes. Do not manufacture variety by suggesting large
or far-reaching changes.

Provide your diagnosis as JSON:
{{
    "ideas": [
        {{
            "failure_summary": "one-line summary",
            "root_cause": "detailed root cause analysis",
            "suggested_fix": "concrete fix suggestion",
            "target_files": ["file1.py", "file2.py"],
            "anchor_files": ["existing/file.py"],
            "confidence": 0.8
        }}
    ]
}}

Rules for file targeting:
- `target_files` must be relative to the agent root, not repo-root guesses.
- Prefer existing files from the mutable file manifest above.
- You may propose a NEW file only if it fits under the existing architecture and you also list 1-2 real existing `anchor_files` that will wire it into the system.
- Do not invent filenames that are not in the manifest unless they are clearly new files tied to those anchors.
- Shared files under `prompts/` or `skills/` should appear only when you can explain why a narrower code-path fix is insufficient.

If the evidence is narrow, return one or two focused ideas rather than broad speculative ones."""


class Diagnosis(pd.BaseModel):
    """LLM-generated diagnosis of a failed evaluation trial."""

    model_config = pd.ConfigDict(extra="forbid")

    trial_id: str = pd.Field(description="ID of the failed trial")
    cluster_id: str | None = pd.Field(default=None, description="ID of grouped failure cluster")
    family: str = pd.Field(default="unknown", description="Coarse root-cause family label")
    failure_summary: str = pd.Field(description="One-line summary of the failure")
    root_cause: str = pd.Field(description="Detailed root cause analysis")
    suggested_fix: str = pd.Field(description="Concrete fix suggestion")
    target_files: list[str] = pd.Field(
        default_factory=list, description="Files that should be modified"
    )
    anchor_files: list[str] = pd.Field(
        default_factory=list,
        description="Existing mutable files that anchor any new target file into the architecture",
    )
    confidence: float = pd.Field(
        default=0.0, ge=0.0, le=1.0, description="LLM confidence in the diagnosis"
    )
    actionable: bool = pd.Field(
        default=True,
        description="Whether this diagnosis is actionable enough to test as a hypothesis",
    )
    diagnosis_mode: Literal[
        "llm", "explorer", "fallback_llm_unavailable", "fallback_parse_failure"
    ] = pd.Field(default="llm", description="How the diagnosis was produced")
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
            family=_infer_diagnosis_family(failure_summary, root_cause, []),
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


def _format_agent_file_manifest(agent_content: dict[str, str] | None) -> str:
    if not agent_content:
        return "none"
    files = sorted(agent_content.keys())
    if len(files) > 80:
        visible = files[:80]
        visible.append(f"... (+{len(files) - 80} more)")
        files = visible
    return "\n".join(f"- {path}" for path in files)


def _infer_diagnosis_family(
    failure_summary: str,
    root_cause: str,
    target_files: list[str],
) -> str:
    haystack = " ".join([failure_summary, root_cause, *target_files]).lower()
    family_rules: list[tuple[str, tuple[str, ...]]] = [
        (
            "workspace_path_resolution",
            (
                "workspace_root",
                "wrong directory",
                "wrong working directory",
                "changed_paths: []",
                "missing_required",
                "scenario",
                "repo_diff",
                ".scenario.yaml",
            ),
        ),
        (
            "tool_loader",
            (
                "tool loader",
                "load_tools",
                "tool_names=[]",
                "no tools",
                "registered tool",
                "tools/loader",
                "empty list",
            ),
        ),
        (
            "tool_use_enforcement",
            (
                "zero tool calls",
                "no tool calls",
                "text-only",
                "writing prose",
                "must use tools",
                "tool-use",
                "tool usage",
            ),
        ),
        (
            "verification_honesty",
            (
                "falsely reported",
                "claimed",
                "reported success",
                "verification",
                "all tests pass",
                "summary",
                "completion_honesty",
            ),
        ),
        (
            "todo_management",
            ("todo", "task ids", "too_many_tasks", "task count", "todowrite", "todo_update"),
        ),
        (
            "edit_persistence",
            (
                "edit tool",
                "write tool",
                "not persisted",
                "persist",
                "old_string mismatch",
                "write was silently dropped",
                "post-edit verification",
            ),
        ),
        (
            "tool_budget_efficiency",
            (
                "tool budget",
                "excessive reads",
                "too many reads",
                "too many globs",
                "write too late",
                "over-explored",
            ),
        ),
        (
            "prompt_scope",
            (
                "system prompt",
                "coding.md",
                "skill",
                "general-coding",
                "agent prompt",
                "prompting strategy",
            ),
        ),
        (
            "integrity_validation",
            (
                "integrity validation",
                "unaffected paths",
                "pass_to_fail",
                "score_drop against unaffected",
                "regressed_unaffected_paths",
            ),
        ),
    ]
    for family, needles in family_rules:
        if any(needle in haystack for needle in needles):
            return family
    return "unknown"


def _format_transcript_excerpt(trial: EvalTrial) -> tuple[str, str, str]:
    transcript = trial.result.transcript if trial.result else None
    if transcript is None:
        return "none", "none", "none"

    transcript_lines: list[str] = []
    for message in transcript.messages[-DIAGNOSIS_MESSAGE_LIMIT:]:
        role = str(message.get("role", "unknown"))
        content = _truncate(str(message.get("content", "")), DIAGNOSIS_MESSAGE_CHAR_LIMIT)
        transcript_lines.append(f"[{role}] {content}")

    tool_lines: list[str] = []
    for tool_call in transcript.tool_calls[-DIAGNOSIS_TOOL_CALL_LIMIT:]:
        name = str(tool_call.get("name") or tool_call.get("tool") or "unknown")
        args = tool_call.get("arguments") or tool_call.get("input") or {}
        tool_lines.append(
            _truncate(
                f"{name}({json.dumps(args, default=str)})",
                DIAGNOSIS_TOOL_CALL_CHAR_LIMIT,
            )
        )

    trace_lines: list[str] = []
    for event in transcript.trace_events[-DIAGNOSIS_TRACE_EVENT_LIMIT:]:
        event_type = str(event.get("event_type", "unknown"))
        data = event.get("data", {})
        trace_lines.append(
            _truncate(
                f"{event_type}: {json.dumps(data, default=str)}",
                DIAGNOSIS_TRACE_EVENT_CHAR_LIMIT,
            )
        )

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
    personal_preferences: str = "",
    console: Any | None = None,
    audit_bundle: Any | None = None,
    audit_iteration: int | None = None,
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
                personal_preferences,
                console=console,
                audit_bundle=audit_bundle,
                audit_iteration=audit_iteration,
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
    personal_preferences: str = "",
    console: Any | None = None,
    audit_bundle: Any | None = None,
    audit_iteration: int | None = None,
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
        agent_file_manifest=_format_agent_file_manifest(agent_content),
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
    if personal_preferences:
        prompt += "\n\n" + personal_preferences

    if console is not None:
        console.print(f"    [dim]Diagnosing trial {trial.id} (task={trial.task_id})...[/dim]")

    response: str | None = None
    explorer_result = None
    if agent_path is not None:
        if console is not None:
            console.print(
                f"    [dim]Diagnosis mode:[/dim] explorer  [dim]trial={trial.id}  agent={agent_path.name}[/dim]"
            )
        explorer_raw = None
        explorer_failure_reported = False
        try:
            explorer_raw = await investigate_trial_with_explorer(
                trial,
                agent_path,
                lesson_store,
                agent_content=agent_content,
                audit_bundle=audit_bundle,
                audit_stem=(
                    f"iter-{audit_iteration:03d}/trial-{trial.id}"
                    if audit_iteration is not None
                    else trial.id
                ),
            )
        except Exception as exc:
            logger.warning("Explorer diagnosis failed for trial %s", trial.id, exc_info=exc)
            if console is not None:
                console.print(
                    f"    [yellow]⚠ Explorer diagnosis crashed for trial {trial.id}: {exc!r}; falling back[/yellow]"
                )
            explorer_failure_reported = True
        if isinstance(explorer_raw, str):
            response = explorer_raw
            explorer_result = None
        else:
            explorer_result = explorer_raw
            if explorer_result is not None:
                if explorer_result.error and console is not None:
                    console.print(
                        f"    [yellow]⚠ Explorer diagnosis failed for trial {trial.id}: {explorer_result.error}; falling back[/yellow]"
                    )
                    explorer_failure_reported = True
                response = explorer_result.response
        if explorer_raw is None and console is not None and not explorer_failure_reported:
            console.print(
                f"    [yellow]⚠ Explorer diagnosis unavailable for trial {trial.id}; falling back[/yellow]"
            )

    if not response:
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

    source_mode: Literal["llm", "explorer"] = (
        "explorer" if explorer_result is not None and not explorer_result.error else "llm"
    )
    results = _parse_diagnosis_response(trial.id, response, diagnosis_mode=source_mode)
    if audit_bundle is not None:
        audit_stem = (
            f"iter-{audit_iteration:03d}/trial-{trial.id}"
            if audit_iteration is not None
            else trial.id
        )
        audit_bundle.write_json(
            f"diagnoses/{audit_stem}/result.json",
            {
                "trial_id": trial.id,
                "source_mode": source_mode,
                "response": response,
                "parsed_diagnoses": [result.model_dump() for result in results],
                "explorer_result": None
                if explorer_result is None
                else {
                    "response": explorer_result.response,
                    "error": explorer_result.error,
                    "tool_calls_used": explorer_result.tool_calls_used,
                    "tool_calls_max": explorer_result.tool_calls_max,
                    "file_reads_used": explorer_result.file_reads_used,
                    "file_reads_max": explorer_result.file_reads_max,
                    "search_calls_used": explorer_result.search_calls_used,
                    "search_calls_max": explorer_result.search_calls_max,
                },
            },
        )
    if results and console is not None:
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
        for index, diagnosis in enumerate(results, start=1):
            files_str = ", ".join(diagnosis.target_files) if diagnosis.target_files else "none"
            console.print(
                f"    [dim]Idea {index}/{len(results)}: {diagnosis.failure_summary}  files=[{files_str}][/dim]"
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
        if agent_content is not None:
            from ash_hawk.improve.targeting import validate_diagnosis_targets

            validate_diagnosis_targets(result, set(agent_content.keys()))
        if not result.target_files:
            result.actionable = False
            result.degraded_reason = result.degraded_reason or "diagnosis_missing_target_files"

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
    candidates: list[dict[str, Any]] = []
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
                    if isinstance(parsed, dict):
                        candidates.append(parsed)
                except json.JSONDecodeError:
                    start = None
                    continue
    for candidate in candidates:
        if "ideas" in candidate:
            return candidate
    for candidate in candidates:
        if all(key in candidate for key in ("failure_summary", "root_cause", "suggested_fix")):
            return candidate
    return candidates[-1] if candidates else None


def _parse_diagnosis_response(
    trial_id: str,
    response: str,
    *,
    diagnosis_mode: Literal["llm", "explorer"] = "llm",
) -> list[Diagnosis]:
    data = _extract_json_object(response)
    if data is None:
        logger.warning("No JSON found in diagnosis response for %s", trial_id)
        return []

    raw_ideas = data.get("ideas")
    if isinstance(raw_ideas, list):
        parsed_ideas = _parse_diagnosis_ideas(trial_id, raw_ideas, diagnosis_mode=diagnosis_mode)
        if parsed_ideas:
            return parsed_ideas

    parsed_single = _parse_diagnosis_ideas(trial_id, [data], diagnosis_mode=diagnosis_mode)
    if parsed_single:
        return parsed_single

    logger.warning("Missing required fields in diagnosis for %s", trial_id)
    return []


def _parse_diagnosis_ideas(
    trial_id: str,
    raw_ideas: list[Any],
    *,
    diagnosis_mode: Literal["llm", "explorer"] = "llm",
) -> list[Diagnosis]:
    required_fields = ["failure_summary", "root_cause", "suggested_fix"]
    diagnoses: list[Diagnosis] = []
    seen_keys: set[tuple[str, tuple[str, ...], str]] = set()

    for raw_idea in raw_ideas:
        if not isinstance(raw_idea, dict):
            continue
        if not all(field in raw_idea for field in required_fields):
            continue

        target_files_raw = raw_idea.get("target_files", [])
        target_files = (
            [str(path) for path in target_files_raw] if isinstance(target_files_raw, list) else []
        )
        anchor_files_raw = raw_idea.get("anchor_files", [])
        anchor_files = (
            [str(path) for path in anchor_files_raw] if isinstance(anchor_files_raw, list) else []
        )
        if len(target_files) > 3:
            continue
        failure_summary = str(raw_idea["failure_summary"])
        root_cause = str(raw_idea["root_cause"])
        family = _infer_diagnosis_family(failure_summary, root_cause, target_files)
        dedupe_key = (family, tuple(sorted(target_files)), root_cause.strip().lower())
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        diagnoses.append(
            Diagnosis(
                trial_id=trial_id,
                family=family,
                failure_summary=failure_summary,
                root_cause=root_cause,
                suggested_fix=str(raw_idea["suggested_fix"]),
                target_files=target_files,
                anchor_files=anchor_files,
                confidence=float(raw_idea.get("confidence", 0.0)),
                actionable=True,
                diagnosis_mode=diagnosis_mode,
            )
        )

    return diagnoses
