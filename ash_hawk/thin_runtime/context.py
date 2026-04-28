# type-hygiene: skip-file
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ash_hawk.thin_runtime.models import (
    AgentSpec,
    ContextSnapshot,
    RuntimeGoal,
    SkillSpec,
    ToolSpec,
)

_BLOCKED_PATH_PATTERN = re.compile(r"path '([^']+)'")
_MAX_RECENT_STEPS = 3
_MAX_ACTIONABLE_FILES = 5
_MAX_REFERENCE_FILES = 4
_MAX_BLOCKED_FILES = 4


class RuntimeContextAssembler:
    def assemble(
        self,
        *,
        goal: RuntimeGoal,
        agent: AgentSpec,
        skills: list[SkillSpec],
        tools: list[ToolSpec],
        memory_snapshot: dict[str, dict[str, Any]],
        workdir: Path,
        available_skills: list[SkillSpec] | None = None,
    ) -> ContextSnapshot:
        snapshot = ContextSnapshot(
            goal={
                "goal_id": goal.goal_id,
                "description": goal.description,
                "target_score": goal.target_score,
                "max_iterations": goal.max_iterations,
            },
            runtime={
                "lead_agent": agent.name,
                "active_skills": [skill.name for skill in skills],
                "max_iterations": goal.max_iterations,
            },
            workspace={
                "workdir": str(workdir),
                "repo_root": str(workdir),
                "allowed_target_files": [],
                "changed_files": [],
            },
            evaluation={
                "baseline_summary": {},
                "targeted_validation_summary": {},
                "integrity_summary": {},
                "regressions": [],
            },
            failure={
                "failed_trials": [],
                "failure_buckets": {},
                "suspicious_reviews": [],
                "clustered_failures": [],
                "ranked_hypotheses": [],
            },
            memory={
                "working_snapshot": memory_snapshot.get("working_memory", {}),
                "session": memory_snapshot.get("session_memory", {}),
                "episodic": memory_snapshot.get("episodic_memory", {}),
                "semantic": memory_snapshot.get("semantic_memory", {}),
                "personal": memory_snapshot.get("personal_memory", {}),
            },
            tool={
                "active_tools": [tool.name for tool in tools],
                "policy_decisions": [],
                "registered_mcp_tools": [],
            },
            audit={
                "events": memory_snapshot.get("artifact_memory", {}).get("events", []),
                "artifacts": memory_snapshot.get("artifact_memory", {}).get("artifacts", []),
                "transcripts": memory_snapshot.get("artifact_memory", {}).get("transcripts", []),
            },
        )
        self.refresh(
            snapshot=snapshot,
            goal=goal,
            agent=agent,
            skills=skills,
            tools=tools,
            memory_snapshot=memory_snapshot,
            workdir=workdir,
            available_skills=available_skills,
        )
        return snapshot

    def refresh(
        self,
        *,
        snapshot: ContextSnapshot,
        goal: RuntimeGoal,
        agent: AgentSpec,
        skills: list[SkillSpec],
        tools: list[ToolSpec],
        memory_snapshot: dict[str, dict[str, Any]],
        workdir: Path,
        available_skills: list[SkillSpec] | None = None,
    ) -> None:
        del available_skills

        repl_sessions = _open_python_repl_sessions(memory_snapshot)
        _repair_evaluation_from_tool_results(snapshot)
        recent_eval_summaries = _recent_eval_summaries(snapshot.evaluation)
        top_hypothesis = _top_hypothesis(snapshot.failure)
        diagnosed_issues = _diagnosed_issues(snapshot.failure, top_hypothesis)
        file_groups = _classify_files(snapshot.workspace, snapshot.audit)
        recent_steps = _recent_steps(snapshot.audit)
        phase = _phase(snapshot, recent_eval_summaries, top_hypothesis, file_groups)
        latest_evidence = _latest_evidence(
            recent_eval_summaries=recent_eval_summaries,
            diagnosed_issues=diagnosed_issues,
            top_hypothesis=top_hypothesis,
            file_groups=file_groups,
        )
        constraints = _constraints(file_groups=file_groups, phase=phase, recent_steps=recent_steps)
        next_pressure = _next_pressure(
            phase=phase,
            recent_eval_summaries=recent_eval_summaries,
            top_hypothesis=top_hypothesis,
            file_groups=file_groups,
        )
        artifact_index = _artifact_index(
            snapshot=snapshot,
            recent_eval_summaries=recent_eval_summaries,
            repl_sessions=repl_sessions,
        )

        snapshot.goal.update(
            {
                "intent": goal.description,
                "success_focus": _success_focus(goal),
            }
        )
        snapshot.runtime.update(
            {
                "goal_intent": _goal_intent(goal, agent),
                "phase": phase,
                "recent_steps": recent_steps,
                "latest_evidence": latest_evidence,
                "constraints": constraints,
                "next_pressure": next_pressure,
                "progress_summary": _progress_summary(
                    snapshot=snapshot,
                    phase=phase,
                    recent_steps=recent_steps,
                    latest_evidence=latest_evidence,
                    file_groups=file_groups,
                ),
                "active_skill_summaries": [_skill_summary(skill) for skill in skills[:6]],
            }
        )
        snapshot.workspace.update(
            {
                "actionable_files": file_groups["actionable_files"],
                "reference_files": file_groups["reference_files"],
                "blocked_files": file_groups["blocked_files"],
                "file_summaries": _combined_file_summaries(file_groups),
                "open_python_repl_sessions": repl_sessions,
                "repo_root": str(workdir),
                "workdir": str(workdir),
            }
        )
        snapshot.evaluation.update({"recent_eval_summaries": recent_eval_summaries})
        snapshot.failure.update(
            {
                "diagnosed_issues": diagnosed_issues,
                "top_hypothesis": top_hypothesis,
            }
        )
        snapshot.tool.update(
            {
                "available_tool_summaries": [_tool_summary(tool) for tool in tools[:8]],
            }
        )
        snapshot.audit.update(
            {
                "artifact_index": artifact_index,
                "progress_artifacts": artifact_index[:6],
            }
        )


def _success_focus(goal: RuntimeGoal) -> str:
    target_fragment = (
        f"reach target score {goal.target_score:.2f}"
        if goal.target_score is not None
        else "finish the goal"
    )
    return f"{target_fragment} within {goal.max_iterations} iteration(s)."


def _goal_intent(goal: RuntimeGoal, agent: AgentSpec) -> str:
    mission = agent.mission or agent.goal or agent.description or agent.summary or ""
    mission_fragment = f" Agent mission: {mission.strip()}" if mission.strip() else ""
    target_fragment = (
        f" Target score: {goal.target_score:.2f}." if goal.target_score is not None else ""
    )
    return (
        f"Goal: {goal.description.strip()} Agent: {agent.name}."
        f" Max iterations: {goal.max_iterations}.{target_fragment}{mission_fragment}"
    ).strip()


def _recent_eval_summaries(evaluation: dict[str, Any]) -> list[str]:
    summaries: list[str] = []
    labels = [
        ("baseline_summary", "baseline"),
        ("last_eval_summary", "latest"),
        ("repeat_eval_summary", "repeat"),
        ("targeted_validation_summary", "targeted"),
        ("integrity_summary", "integrity"),
    ]
    for key, label in labels:
        raw = evaluation.get(key)
        if not isinstance(raw, dict):
            continue
        score = raw.get("score")
        status = raw.get("status")
        parts = [label]
        if isinstance(score, int | float):
            parts.append(f"score {float(score):.2f}")
        if isinstance(status, str) and status.strip():
            parts.append(status.strip())
        if len(parts) > 1:
            summaries.append(": ".join([parts[0], ", ".join(parts[1:])]))
    return summaries[:5]


def _repair_evaluation_from_tool_results(snapshot: ContextSnapshot) -> None:
    tool_results = snapshot.audit.get("tool_results", [])
    if not isinstance(tool_results, list):
        return
    field_by_tool = {
        "run_baseline_eval": "baseline_summary",
        "run_eval": "last_eval_summary",
        "run_eval_repeated": "repeat_eval_summary",
        "run_targeted_validation": "targeted_validation_summary",
        "run_integrity_validation": "integrity_summary",
    }
    for tool_result in reversed(tool_results):
        if not isinstance(tool_result, dict):
            continue
        tool_name = tool_result.get("tool")
        if not isinstance(tool_name, str):
            continue
        summary_field = field_by_tool.get(tool_name)
        if summary_field is None:
            continue
        existing = snapshot.evaluation.get(summary_field)
        if isinstance(existing, dict) and any(
            existing.get(key) not in (None, "") for key in ("score", "status", "tool")
        ):
            continue
        payload = tool_result.get("payload")
        if not isinstance(payload, dict):
            continue
        evaluation_updates = payload.get("evaluation_updates")
        if isinstance(evaluation_updates, dict):
            summary = evaluation_updates.get(summary_field)
            if isinstance(summary, dict) and any(
                summary.get(key) not in (None, "") for key in ("score", "status", "tool")
            ):
                snapshot.evaluation[summary_field] = summary
                continue
        audit_updates = payload.get("audit_updates")
        if not isinstance(audit_updates, dict):
            continue
        run_result = audit_updates.get("run_result")
        if not isinstance(run_result, dict):
            continue
        aggregate_score = run_result.get("aggregate_score")
        message = run_result.get("message")
        if not isinstance(aggregate_score, int | float) and not isinstance(message, str):
            continue
        snapshot.evaluation[summary_field] = {
            "score": float(aggregate_score) if isinstance(aggregate_score, int | float) else None,
            "status": "completed" if tool_result.get("success") else "failed",
            "tool": tool_name,
        }


def _top_hypothesis(failure: dict[str, Any]) -> str | None:
    raw_hypotheses = failure.get("ranked_hypotheses", [])
    if not isinstance(raw_hypotheses, list):
        return None
    for hypothesis in raw_hypotheses:
        if not isinstance(hypothesis, dict):
            continue
        name = str(hypothesis.get("name", "")).strip()
        if not name:
            continue
        score = hypothesis.get("score")
        if isinstance(score, int | float):
            return f"{name} ({float(score):.2f})"
        return name
    return None


def _diagnosed_issues(failure: dict[str, Any], top_hypothesis: str | None) -> list[str]:
    issues: list[str] = []
    failure_family = failure.get("failure_family")
    if isinstance(failure_family, str) and failure_family.strip():
        issues.append(f"failure family: {failure_family.strip()}")
    explanations = failure.get("explanations", [])
    if isinstance(explanations, list):
        issues.extend(str(item).strip() for item in explanations if str(item).strip())
    if top_hypothesis:
        issues.append(f"top hypothesis: {top_hypothesis}")
    deduped: list[str] = []
    for item in issues:
        if item and item not in deduped:
            deduped.append(item)
    return deduped[:6]


def _classify_files(workspace: dict[str, Any], audit: dict[str, Any]) -> dict[str, list[str]]:
    blocked_reasons = _blocked_path_reasons(audit)
    entries: list[tuple[str, str]] = []
    for key, label in (("scenario_path", "scenario"), ("agent_config", "agent config")):
        raw = workspace.get(key)
        if isinstance(raw, str) and raw.strip():
            entries.append((raw.strip(), label))
    for key, label in (
        ("mutated_files", "mutated"),
        ("changed_files", "changed"),
        ("allowed_target_files", "scoped"),
        ("scenario_required_files", "required"),
    ):
        raw = workspace.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, str) and item.strip():
                entries.append((item.strip(), label))

    actionable_files: list[str] = []
    reference_files: list[str] = []
    blocked_files: list[str] = []
    seen_paths: set[str] = set()
    for path, label in entries:
        normalized = path.strip()
        if not normalized or normalized in seen_paths:
            continue
        seen_paths.add(normalized)
        blocked_reason = blocked_reasons.get(normalized)
        if blocked_reason is None and _is_runtime_metadata_path(normalized):
            blocked_reason = "blocked: evaluation infrastructure"
        if blocked_reason is not None:
            blocked_files.append(_blocked_file_summary(normalized, blocked_reason))
            continue
        if _is_reference_file(normalized, label):
            reference_files.append(_reference_file_summary(normalized, label))
            continue
        actionable_files.append(_actionable_file_summary(normalized, label))

    return {
        "actionable_files": actionable_files[:_MAX_ACTIONABLE_FILES],
        "reference_files": reference_files[:_MAX_REFERENCE_FILES],
        "blocked_files": blocked_files[:_MAX_BLOCKED_FILES],
    }


def _combined_file_summaries(file_groups: dict[str, list[str]]) -> list[str]:
    return [
        *file_groups["actionable_files"],
        *file_groups["reference_files"],
        *file_groups["blocked_files"],
    ][:8]


def _blocked_path_reasons(audit: dict[str, Any]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for container in (
        audit.get("tool_results", []),
        audit.get("delegation_summaries", []),
        audit.get("delegations", []),
    ):
        if not isinstance(container, list):
            continue
        for item in container:
            if not isinstance(item, dict):
                continue
            for key in ("error", "summary"):
                raw = item.get(key)
                if not isinstance(raw, str) or "Access denied" not in raw:
                    continue
                match = _BLOCKED_PATH_PATTERN.search(raw)
                if match is None:
                    continue
                reasons[match.group(1)] = "blocked: policy-forbidden"
    return reasons


def _is_runtime_metadata_path(path: str) -> bool:
    parts = Path(path).parts
    return ".dawn-kestrel" in parts or ".ash-hawk" in parts


def _is_reference_file(path: str, label: str) -> bool:
    if label in {"scenario", "agent config"}:
        return True
    if label in {"changed", "scoped", "required", "mutated"} and not _is_durable_target(path):
        return True
    return False


def _is_durable_target(path: str) -> bool:
    normalized = path.strip().lower()
    filename = Path(normalized).name
    if filename == "agent.md":
        return True
    suffix = Path(normalized).suffix.lower()
    return suffix in {
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".go",
        ".rs",
        ".rb",
        ".java",
        ".kt",
        ".swift",
        ".php",
        ".c",
        ".cpp",
        ".cs",
        ".sh",
    }


def _actionable_file_summary(path: str, label: str) -> str:
    return f"{path} [{label}; {_file_kind(path)}]"


def _reference_file_summary(path: str, label: str) -> str:
    return f"{path} [reference from {label}; {_file_kind(path)}]"


def _blocked_file_summary(path: str, reason: str) -> str:
    return f"{path} [{reason}; {_file_kind(path)}]"


def _recent_steps(audit: dict[str, Any]) -> list[str]:
    tool_steps: list[str] = []
    raw_tool_results = audit.get("tool_results", [])
    if isinstance(raw_tool_results, list):
        for item in raw_tool_results[-_MAX_RECENT_STEPS:]:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool", "")).strip()
            success = bool(item.get("success"))
            payload = item.get("payload")
            detail = ""
            if isinstance(payload, dict):
                detail = str(payload.get("message", "")).strip()
            error = str(item.get("error", "")).strip()
            snippet = detail or error or ("completed" if success else "failed")
            if not tool_name:
                continue
            tool_steps.append(f"{tool_name}: {_truncate(snippet, 120)}")
    delegation_steps: list[str] = []
    raw_delegations = audit.get("delegation_summaries", [])
    if isinstance(raw_delegations, list) and raw_delegations:
        last = raw_delegations[-1]
        if isinstance(last, dict):
            agent_name = str(last.get("agent_name", "delegated agent")).strip()
            summary = str(last.get("summary", "")).strip() or str(last.get("error", "")).strip()
            if summary:
                delegation_steps.append(f"delegation {agent_name}: {_truncate(summary, 120)}")
    steps = [*tool_steps, *delegation_steps]
    deduped: list[str] = []
    for item in steps[-_MAX_RECENT_STEPS:]:
        if item not in deduped:
            deduped.append(item)
    return deduped[:_MAX_RECENT_STEPS]


def _phase(
    snapshot: ContextSnapshot,
    recent_eval_summaries: list[str],
    top_hypothesis: str | None,
    file_groups: dict[str, list[str]],
) -> str:
    if isinstance(snapshot.runtime.get("stop_reason"), str):
        return "completed"
    tool_results = snapshot.audit.get("tool_results", [])
    if not isinstance(tool_results, list) or not tool_results:
        return "bootstrap"
    if not recent_eval_summaries:
        return "bootstrap"
    repeat_summary = snapshot.evaluation.get("repeat_eval_summary", {})
    if isinstance(repeat_summary, dict) and repeat_summary.get("status"):
        return "verification"
    mutated_files = snapshot.workspace.get("mutated_files", [])
    if isinstance(mutated_files, list) and mutated_files:
        return "mutation"
    if top_hypothesis and file_groups["actionable_files"]:
        return "hypothesis"
    if snapshot.failure.get("failure_family") or snapshot.failure.get("diagnosed_issues"):
        return "diagnosis"
    return "bootstrap"


def _latest_evidence(
    *,
    recent_eval_summaries: list[str],
    diagnosed_issues: list[str],
    top_hypothesis: str | None,
    file_groups: dict[str, list[str]],
) -> list[str]:
    evidence: list[str] = []
    if recent_eval_summaries:
        evidence.append(recent_eval_summaries[0])
    if diagnosed_issues:
        evidence.append(diagnosed_issues[0])
    if top_hypothesis:
        evidence.append(f"top hypothesis: {top_hypothesis}")
    if file_groups["actionable_files"]:
        evidence.append(f"actionable target: {file_groups['actionable_files'][0]}")
    deduped: list[str] = []
    for item in evidence:
        if item not in deduped:
            deduped.append(item)
    return deduped[:4]


def _constraints(
    *,
    file_groups: dict[str, list[str]],
    phase: str,
    recent_steps: list[str],
) -> list[str]:
    constraints: list[str] = []
    if file_groups["blocked_files"]:
        constraints.append(f"Do not target blocked paths: {file_groups['blocked_files'][0]}")
    if phase in {"diagnosis", "hypothesis"} and not file_groups["actionable_files"]:
        constraints.append("Name a durable actionable file before mutating anything.")
    if recent_steps and "Access denied" in recent_steps[-1]:
        constraints.append(
            "Pivot away from policy-forbidden files and use direct code evidence instead."
        )
    return constraints[:3]


def _next_pressure(
    *,
    phase: str,
    recent_eval_summaries: list[str],
    top_hypothesis: str | None,
    file_groups: dict[str, list[str]],
) -> str:
    actionable_target = (
        file_groups["actionable_files"][0] if file_groups["actionable_files"] else None
    )
    if phase == "bootstrap":
        if not recent_eval_summaries:
            return "Establish the latest eval evidence before choosing a mutation."
        return "Move from bootstrap into diagnosis using the latest eval evidence."
    if phase == "diagnosis":
        if actionable_target:
            return f"Turn the blocker into one narrow hypothesis on {actionable_target}."
        return "Use direct reference artifacts to identify one durable target file."
    if phase == "hypothesis":
        if actionable_target and top_hypothesis:
            return f"Test {top_hypothesis} with one focused change in {actionable_target}."
        return "Choose one actionable file and one concrete hypothesis before mutating."
    if phase == "mutation":
        return "Apply one focused change, then close the loop with re-evaluation."
    if phase == "verification":
        return "Use the latest eval result to decide whether the change improved the score."
    return "Summarize the current outcome without inventing extra work."


def _artifact_index(
    *,
    snapshot: ContextSnapshot,
    recent_eval_summaries: list[str],
    repl_sessions: list[str],
) -> list[str]:
    artifacts: list[str] = []
    scenario_path = snapshot.workspace.get("scenario_path")
    if isinstance(scenario_path, str) and scenario_path.strip():
        artifacts.append(
            f"scenario_file: {scenario_path.strip()} -> use context-assembly to refresh direct context"
        )
    agent_config = snapshot.workspace.get("agent_config")
    if isinstance(agent_config, str) and agent_config.strip():
        label = (
            "blocked_agent_config_reference"
            if _is_runtime_metadata_path(agent_config)
            else "agent_config_reference"
        )
        artifacts.append(
            f"{label}: {agent_config.strip()} -> use signal-driven-workspace to pivot to actionable files"
        )
    if recent_eval_summaries:
        artifacts.append(
            "latest_eval_summary -> use improvement-loop for diagnosis and re-evaluation"
        )
    raw_tool_results = snapshot.audit.get("tool_results", [])
    if isinstance(raw_tool_results, list):
        for item in reversed(raw_tool_results):
            if not isinstance(item, dict):
                continue
            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue
            audit_updates = payload.get("audit_updates")
            if isinstance(audit_updates, dict) and isinstance(
                audit_updates.get("diff_report"), dict
            ):
                if audit_updates["diff_report"]:
                    artifacts.append(
                        "latest_diff_report -> use signal-driven-workspace to confirm the mutation"
                    )
                    break
    if snapshot.audit.get("delegation_summaries") or snapshot.audit.get("delegations"):
        artifacts.append("delegation_log -> use improvement-loop to tighten the next delegation")
    if repl_sessions:
        artifacts.append(
            "python_repl_sessions -> use direct runtime inspection when evidence is local"
        )
    raw_artifacts = snapshot.audit.get("artifacts", [])
    if isinstance(raw_artifacts, list) and raw_artifacts:
        artifacts.append(
            "execution_artifacts_available -> inspect only if the live brief is insufficient"
        )
    deduped: list[str] = []
    for item in artifacts:
        if item not in deduped:
            deduped.append(item)
    return deduped[:6]


def _progress_summary(
    *,
    snapshot: ContextSnapshot,
    phase: str,
    recent_steps: list[str],
    latest_evidence: list[str],
    file_groups: dict[str, list[str]],
) -> str:
    completed_iterations = snapshot.runtime.get("completed_iterations")
    max_iterations = snapshot.runtime.get("max_iterations")
    iteration_fragment = "iteration state pending"
    if isinstance(completed_iterations, int) and isinstance(max_iterations, int):
        iteration_fragment = f"iterations {completed_iterations}/{max_iterations}"
    parts = [f"phase {phase}", iteration_fragment]
    if recent_steps:
        parts.append(f"latest step: {recent_steps[-1]}")
    if latest_evidence:
        parts.append(f"latest evidence: {latest_evidence[0]}")
    if file_groups["actionable_files"]:
        parts.append(f"actionable: {file_groups['actionable_files'][0]}")
    elif file_groups["blocked_files"]:
        parts.append(f"blocked: {file_groups['blocked_files'][0]}")
    return "; ".join(parts)


def _open_python_repl_sessions(memory_snapshot: dict[str, dict[str, Any]]) -> list[str]:
    session_memory = memory_snapshot.get("session_memory", {})
    session_keys = ("open_python_repl_sessions", "python_repl_sessions", "repl_sessions")
    sessions: list[str] = []
    for key in session_keys:
        raw = session_memory.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, str) and item.strip():
                sessions.append(item.strip())
            elif isinstance(item, dict):
                session_id = str(item.get("session_id", "")).strip()
                command = str(item.get("command", "")).strip()
                if session_id and command:
                    sessions.append(f"{session_id}: {command}")
                elif session_id:
                    sessions.append(session_id)
    deduped: list[str] = []
    for item in sessions:
        if item not in deduped:
            deduped.append(item)
    return deduped[:5]


def _skill_summary(skill: SkillSpec) -> str:
    summary = skill.description.strip() if skill.description.strip() else skill.name
    tool_fragment = f" | tools: {', '.join(skill.tool_names[:3])}" if skill.tool_names else ""
    return f"{skill.name}: {summary}{tool_fragment}"


def _tool_summary(tool: ToolSpec) -> str:
    when = tool.when_to_use[0] if tool.when_to_use else tool.summary or tool.description
    return f"{tool.name}: {str(when).strip()}"


def _file_kind(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        return "python source"
    if suffix in {".yaml", ".yml"}:
        return "yaml config"
    if suffix == ".md":
        return "markdown instructions"
    if suffix in {".json", ".jsonl"}:
        return "json artifact"
    if suffix == ".toml":
        return "project config"
    if suffix == ".txt":
        return "text file"
    return "workspace file"


def _truncate(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3]}..."
