# type-hygiene: skip-file
from __future__ import annotations

import re
from typing import Any

from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    basic_input_schema,
    context_input_schema,
    delegation_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_types import MemoryToolContext, ToolExecutionPayload


def _stringify_memory_items(scope_name: str, values: dict[str, Any]) -> list[tuple[str, bool]]:
    rendered: list[tuple[str, bool]] = []
    for key, raw_value in values.items():
        is_primary_guidance = key != "entries"
        if isinstance(raw_value, list):
            for item in raw_value:
                text = str(item).strip()
                if text:
                    rendered.append((f"{scope_name}.{key}: {text}", is_primary_guidance))
        elif isinstance(raw_value, dict):
            for nested_key, nested_value in raw_value.items():
                text = str(nested_value).strip()
                if text:
                    rendered.append(
                        (f"{scope_name}.{key}.{nested_key}: {text}", is_primary_guidance)
                    )
        else:
            text = str(raw_value).strip()
            if text:
                rendered.append((f"{scope_name}.{key}: {text}", is_primary_guidance))
    return rendered


def _query_terms(call: ToolCall) -> set[str]:
    query_text = " ".join(
        str(part)
        for part in [
            call.goal_id,
            call.agent_text or "",
            call.context.runtime.agent_text or "",
            call.context.failure.failure_family or "",
            *call.context.failure.explanations,
            *call.context.failure.concepts,
            *call.context.memory.search_results,
        ]
        if part
    )
    return {
        token
        for token in re.findall(r"[a-z0-9_]+", query_text.lower())
        if len(token) >= 3 and token not in {"the", "and", "for", "with", "that", "this"}
    }


def _rank_memory_matches(call: ToolCall) -> list[str]:
    memory_context = call.context.memory
    scope_entries = {
        "working_memory": memory_context.working_memory,
        "session_memory": memory_context.session_memory,
        "episodic_memory": memory_context.episodic_memory,
        "semantic_memory": memory_context.semantic_memory,
        "personal_memory": memory_context.personal_memory,
    }
    candidates: list[tuple[str, bool]] = []
    for scope_name, values in scope_entries.items():
        if values:
            candidates.extend(_stringify_memory_items(scope_name, values))

    if not candidates:
        return []

    terms = _query_terms(call)
    scored: list[tuple[int, int, int, str]] = []
    for index, (candidate, is_primary_guidance) in enumerate(candidates):
        normalized = candidate.lower()
        score = sum(1 for term in terms if term in normalized)
        if terms and score == 0:
            continue
        scored.append((score, 1 if is_primary_guidance else 0, -index, candidate))

    if not scored:
        prioritized = [candidate for candidate, is_primary in candidates if is_primary]
        fallback = prioritized or [candidate for candidate, _ in candidates]
        return fallback[:5]

    scored.sort(reverse=True)
    return [candidate for _, _, _, candidate in scored[:5]]


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    matches = _rank_memory_matches(call)
    payload = ToolExecutionPayload(
        memory_updates=MemoryToolContext(
            semantic_loaded=bool(call.context.memory.semantic_memory) or None,
            personal_loaded=bool(call.context.memory.personal_memory) or None,
            episodic_loaded=bool(call.context.memory.episodic_memory) or None,
            session_loaded=bool(call.context.memory.session_memory) or None,
            working_snapshot_loaded=bool(call.context.memory.working_memory) or None,
            search_results=matches,
        )
    )
    if matches:
        return True, payload, f"Surfaced {len(matches)} relevant memory entries", []
    return True, payload, "No relevant memory entries found", []


COMMAND = ToolCommand(
    name="search_knowledge",
    summary="Search knowledge.",
    when_to_use=["When this exact capability is needed"],
    when_not_to_use=["When required inputs are missing"],
    input_schema=basic_input_schema(),
    output_schema=standard_output_schema(),
    side_effects=["none"],
    risk_level="low",
    timeout_seconds=30,
    completion_criteria=[
        "Output matches declared schema",
        "Execution stays within timeout and permission bounds",
        "Errors are explicit and actionable when failure occurs",
    ],
    escalation_rules=[
        "Escalate when confidence in output validity is low",
        "Escalate when the request exceeds tool permissions",
        "Escalate when repeated retries produce the same failure",
    ],
    executor=_execute,
)


def run(call: ToolCall) -> ToolResult:
    return COMMAND.run(call)
