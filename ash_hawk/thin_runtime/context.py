# type-hygiene: skip-file
from __future__ import annotations

from pathlib import Path
from typing import Any

from ash_hawk.thin_runtime.models import (
    AgentSpec,
    ContextSnapshot,
    RuntimeGoal,
    SkillSpec,
    ToolSpec,
)


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
    ) -> ContextSnapshot:
        return ContextSnapshot(
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
