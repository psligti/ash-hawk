# type-hygiene: skip-file
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, cast

import pydantic as pd

from ash_hawk.types import EvalTranscript, EvalTrial

Phase1FailureBucket = Literal[
    "no_verification",
    "false_claim",
    "wrong_path",
    "no_tool_use",
    "bad_retry",
    "bad_tool_choice",
    "bad_search",
    "unknown",
]

_VERIFICATION_KEYWORDS = (
    "verify",
    "verified",
    "verification",
    "test",
    "tests",
    "pytest",
    "ruff",
    "mypy",
    "lint",
    "build",
    "validated",
)
_DELEGATION_KEYWORDS = ("delegate", "delegated", "subagent", "parallel")
_COMPLETION_KEYWORDS = (
    "done",
    "completed",
    "fixed",
    "resolved",
    "finished",
    "all set",
    "success",
)
_SEARCH_TOOLS = {"read", "grep", "glob", "ls"}
_EDIT_TOOLS = {"edit", "write", "apply_patch"}
_PATH_KEYWORDS = (
    "wrong path",
    "wrong file",
    "wrong location",
    "required path",
    "missing_required",
)
_RETRY_KEYWORDS = (
    "retry",
    "reread",
    "re-read",
    "stale context",
    "same failing action",
)


class Phase1Review(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid")

    trial_id: str = pd.Field(description="Trial identifier")
    task_id: str = pd.Field(description="Task identifier")
    scenario_path: str | None = pd.Field(default=None, description="Scenario path if available")
    task_type: str = pd.Field(description="Derived task type or scenario family")
    decision_summary: str | None = pd.Field(
        default=None,
        description="Short assistant-message summary of the run's approach",
    )
    skills_loaded: list[str] = pd.Field(default_factory=list)
    tools_used: list[str] = pd.Field(default_factory=list)
    agent_calls_used: list[str] = pd.Field(default_factory=list)
    verification_actions: list[str] = pd.Field(default_factory=list)
    final_claim: str | None = pd.Field(default=None)
    eval_score: float = pd.Field(ge=0.0, le=1.0)
    eval_pass: bool = pd.Field(description="Binary trial pass/fail")
    failure_reason: str | None = pd.Field(default=None)
    verification_before_done: bool = pd.Field(description="Whether verification preceded done")
    claim_trace_alignment: bool = pd.Field(description="Whether final claim matches trace")
    unnecessary_complexity: bool = pd.Field(description="Whether the run overused tools")
    suspicious: bool = pd.Field(description="Whether the run should be reviewed")
    failure_bucket: Phase1FailureBucket | None = pd.Field(default=None)
    reasons: list[str] = pd.Field(default_factory=list)


def _trial_scenario_path(trial: EvalTrial) -> str | None:
    snapshot = getattr(trial, "input_snapshot", None)
    if isinstance(snapshot, dict):
        snapshot_map = cast(dict[str, Any], snapshot)
        scenario_path = snapshot_map.get("scenario_path")
        if isinstance(scenario_path, str) and scenario_path:
            return scenario_path
    return None


def _scenario_family_key(path: str | None, fallback_task_id: str) -> str:
    if not path:
        return fallback_task_id
    name = path.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    stem = name.removesuffix(".scenario.yaml")
    if "__" in stem:
        return stem.split("__", maxsplit=1)[0]
    if stem.startswith("mvp_"):
        return "mvp"
    if stem.startswith("policy_"):
        return "policy"
    return stem


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _assistant_messages(transcript: EvalTranscript) -> list[str]:
    messages: list[str] = []
    for message in transcript.messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            messages.append(content.strip())
    if isinstance(transcript.agent_response, str) and transcript.agent_response.strip():
        messages.append(transcript.agent_response.strip())
    return _dedupe(messages)


def _tool_name_from_event(event: dict[str, Any]) -> str | None:
    event_type = event.get("event_type")
    data = cast(dict[str, Any], event.get("data", {}))
    if event_type == "ToolCallEvent":
        candidate = data.get("tool") or data.get("tool_name") or data.get("name")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    if event_type in {"PolicyDecisionEvent", "RejectionEvent"}:
        candidate = data.get("tool_name")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _transcript_tools(transcript: EvalTranscript) -> list[str]:
    tool_names: list[str] = []
    for event in transcript.trace_events or []:
        tool_name = _tool_name_from_event(event)
        if tool_name:
            tool_names.append(tool_name)
    for call in transcript.tool_calls or []:
        candidate = call.get("tool") or call.get("tool_name") or call.get("name")
        if isinstance(candidate, str) and candidate.strip():
            tool_names.append(candidate.strip())
    return _dedupe(tool_names)


def _verification_actions(transcript: EvalTranscript) -> list[str]:
    actions: list[str] = []
    for event in transcript.trace_events or []:
        event_map = event
        if event_map.get("event_type") != "VerificationEvent":
            continue
        data = cast(dict[str, Any], event_map.get("data", {}))
        passed = data.get("pass")
        message = data.get("message")
        if passed is True:
            if isinstance(message, str) and message.strip():
                actions.append(f"verification:{message.strip()}")
            else:
                actions.append("verification:pass")

    for call in transcript.tool_calls or []:
        call_map = call
        tool_name = call_map.get("tool") or call_map.get("tool_name") or call_map.get("name")
        command = ""
        arguments: dict[str, Any] = {}
        raw_arguments = call_map.get("arguments")
        if isinstance(raw_arguments, dict):
            arguments = cast(dict[str, Any], raw_arguments)
        else:
            raw_input = call_map.get("input")
            if isinstance(raw_input, dict):
                arguments = cast(dict[str, Any], raw_input)
            else:
                raw_input_args = call_map.get("input_args")
                if isinstance(raw_input_args, dict):
                    arguments = cast(dict[str, Any], raw_input_args)
        raw_command = arguments.get("command") or arguments.get("cmd")
        if isinstance(raw_command, str):
            command = raw_command.lower()
        if isinstance(tool_name, str) and tool_name.strip():
            tool_lower = tool_name.lower()
            if any(keyword in tool_lower for keyword in _VERIFICATION_KEYWORDS):
                actions.append(f"tool:{tool_name.strip()}")
                continue
        if command and any(keyword in command for keyword in _VERIFICATION_KEYWORDS):
            actions.append(f"command:{command[:80]}")
    return _dedupe(actions)


def _decision_summary(messages: list[str]) -> str | None:
    if not messages:
        return None
    first = messages[0]
    compact = " ".join(first.split())
    if len(compact) <= 160:
        return compact
    return compact[:159].rstrip() + "…"


def _completion_claimed(final_claim: str | None) -> bool:
    if not final_claim:
        return False
    lowered = final_claim.lower()
    return any(keyword in lowered for keyword in _COMPLETION_KEYWORDS)


def _claim_trace_alignment(
    *,
    final_claim: str | None,
    verification_actions: list[str],
    agent_calls_used: list[str],
    eval_pass: bool,
) -> tuple[bool, list[str]]:
    if not final_claim:
        return True, []

    lowered = final_claim.lower()
    reasons: list[str] = []

    if any(keyword in lowered for keyword in _DELEGATION_KEYWORDS) and not agent_calls_used:
        reasons.append("delegation_claim_without_trace")

    if any(keyword in lowered for keyword in _VERIFICATION_KEYWORDS) and not verification_actions:
        reasons.append("verification_claim_without_trace")

    if not eval_pass and _completion_claimed(final_claim):
        reasons.append("completion_claim_on_failed_trial")

    return not reasons, reasons


def _unnecessary_complexity(
    transcript: EvalTranscript, tools_used: list[str]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    total_tool_calls = len(transcript.tool_calls or [])
    search_count = sum(1 for tool in tools_used if tool in _SEARCH_TOOLS)
    edit_count = sum(1 for tool in tools_used if tool in _EDIT_TOOLS)

    if total_tool_calls >= 12 and search_count >= max(3, len(tools_used) // 2):
        reasons.append("high_search_volume")
    if total_tool_calls >= 10 and edit_count == 0:
        reasons.append("many_tools_no_edits")
    return bool(reasons), reasons


def _failure_bucket(
    *,
    eval_pass: bool,
    verified_before_done: bool,
    claim_trace_alignment: bool,
    unnecessary_complexity: bool,
    complexity_reasons: list[str],
    tools_used: list[str],
    failure_reason: str | None,
    final_claim: str | None,
) -> Phase1FailureBucket | None:
    if eval_pass and claim_trace_alignment and verified_before_done and not unnecessary_complexity:
        return None
    if not claim_trace_alignment:
        return "false_claim"
    lowered_failure = (failure_reason or "").lower()
    lowered_claim = (final_claim or "").lower()
    if not tools_used:
        return "no_tool_use"
    if any(keyword in lowered_failure or keyword in lowered_claim for keyword in _PATH_KEYWORDS):
        return "wrong_path"
    if any(keyword in lowered_failure or keyword in lowered_claim for keyword in _RETRY_KEYWORDS):
        return "bad_retry"
    if unnecessary_complexity and any(
        reason == "high_search_volume" for reason in complexity_reasons
    ):
        return "bad_search"
    if unnecessary_complexity:
        return "bad_tool_choice"
    if not verified_before_done:
        return "no_verification"
    return "unknown"


def review_trial(trial: EvalTrial) -> Phase1Review:
    transcript = trial.result.transcript if trial.result is not None else EvalTranscript()
    scenario_path = _trial_scenario_path(trial)
    task_type = _scenario_family_key(scenario_path, trial.task_id)
    messages = _assistant_messages(transcript)
    final_claim = messages[-1] if messages else None
    tools_used = _transcript_tools(transcript)
    agent_calls_used = [tool for tool in tools_used if tool == "task"]
    verification_actions = _verification_actions(transcript)
    verified_before_done = bool(verification_actions) or not final_claim
    eval_pass = bool(trial.result.aggregate_passed) if trial.result is not None else False
    eval_score = float(trial.result.aggregate_score) if trial.result is not None else 0.0
    failure_reason = None
    if trial.result is not None and trial.result.outcome.failure_mode is not None:
        failure_reason = trial.result.outcome.failure_mode.value
    claim_trace_alignment, claim_reasons = _claim_trace_alignment(
        final_claim=final_claim,
        verification_actions=verification_actions,
        agent_calls_used=agent_calls_used,
        eval_pass=eval_pass,
    )
    unnecessary_complexity, complexity_reasons = _unnecessary_complexity(transcript, tools_used)
    failure_bucket = _failure_bucket(
        eval_pass=eval_pass,
        verified_before_done=verified_before_done,
        claim_trace_alignment=claim_trace_alignment,
        unnecessary_complexity=unnecessary_complexity,
        complexity_reasons=complexity_reasons,
        tools_used=tools_used,
        failure_reason=failure_reason,
        final_claim=final_claim,
    )
    suspicious_reasons: list[str] = []
    if not eval_pass:
        suspicious_reasons.append("eval_failed")
    if not verified_before_done and not eval_pass:
        suspicious_reasons.append("missing_verification")
    suspicious_reasons.extend(claim_reasons)
    suspicious_reasons.extend(complexity_reasons)
    if eval_score >= 0.6 and not eval_pass:
        suspicious_reasons.append("high_score_failed_trial")

    return Phase1Review(
        trial_id=trial.id,
        task_id=trial.task_id,
        scenario_path=scenario_path,
        task_type=task_type,
        decision_summary=_decision_summary(messages),
        skills_loaded=[],
        tools_used=tools_used,
        agent_calls_used=agent_calls_used,
        verification_actions=verification_actions,
        final_claim=final_claim,
        eval_score=eval_score,
        eval_pass=eval_pass,
        failure_reason=failure_reason,
        verification_before_done=verified_before_done,
        claim_trace_alignment=claim_trace_alignment,
        unnecessary_complexity=unnecessary_complexity,
        suspicious=bool(suspicious_reasons),
        failure_bucket=failure_bucket,
        reasons=_dedupe(suspicious_reasons),
    )


def review_summary(summary: Any) -> list[Phase1Review]:
    trials = getattr(summary, "trials", [])
    reviews: list[Phase1Review] = []
    for trial in trials:
        if isinstance(trial, EvalTrial):
            reviews.append(review_trial(trial))
    return reviews


__all__ = ["Phase1FailureBucket", "Phase1Review", "review_summary", "review_trial"]
