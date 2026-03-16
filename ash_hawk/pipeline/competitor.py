"""Competitor role for re-attempting failed runs with lessons applied."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ash_hawk.contracts import CuratedLesson, ReviewFinding, RunArtifact
from ash_hawk.services.comparison_service import ComparisonResult, ComparisonService
from ash_hawk.services.replay_service import ReplayConfig, ReplayService

if TYPE_CHECKING:
    from ash_hawk.agents.dawn_kestrel import DawnKestrelAgentRunner


@dataclass
class CompetitorInput:
    artifact: RunArtifact | None = None
    lessons_to_apply: list[CuratedLesson] = field(default_factory=list)
    replay_config: dict[str, Any] = field(default_factory=dict)
    use_real_replay: bool = True
    agent_runner: DawnKestrelAgentRunner | None = None


@dataclass
class CompetitorOutput:
    replay_artifact: RunArtifact | None = None
    comparison: ComparisonResult | None = None
    findings: list[ReviewFinding] = field(default_factory=list)
    improvement_achieved: bool = False
    error: str | None = None


class CompetitorRole:
    def __init__(self) -> None:
        self._comparison_service = ComparisonService()
        self._replay_service = ReplayService()

    def compete(self, input_data: CompetitorInput) -> CompetitorOutput:
        artifact = input_data.artifact
        if artifact is None:
            return CompetitorOutput(error="No artifact provided for competition")

        if artifact.is_successful():
            return CompetitorOutput(
                replay_artifact=None,
                comparison=None,
                findings=[
                    ReviewFinding(
                        finding_id=f"finding-{uuid4().hex[:8]}",
                        category="competition",
                        severity="info",
                        title="Baseline already successful",
                        description="Run was already successful, no replay needed",
                        evidence_refs=[],
                    )
                ],
                improvement_achieved=False,
            )

        lessons = input_data.lessons_to_apply
        if not lessons:
            return CompetitorOutput(
                replay_artifact=None,
                comparison=None,
                findings=[
                    ReviewFinding(
                        finding_id=f"finding-{uuid4().hex[:8]}",
                        category="competition",
                        severity="info",
                        title="No lessons available for replay",
                        description="Competitor role skipped - no lessons to apply",
                        evidence_refs=[],
                    )
                ],
                improvement_achieved=False,
            )

        if input_data.use_real_replay and input_data.agent_runner:
            replay_artifact = asyncio.run(
                self._replay_service.replay_with_lessons(
                    artifact=artifact,
                    lessons=lessons,
                    agent_runner=input_data.agent_runner,
                )
            )
        else:
            replay_artifact = self._simulate_replay_with_lessons(artifact, lessons)

            if not input_data.use_real_replay:
                replay_artifact.metadata["simulation"] = True

        comparison = self._comparison_service.compare(
            baseline=artifact,
            treatment=replay_artifact,
            lessons_applied=[lesson.lesson_id for lesson in lessons],
        )

        findings = self._generate_competition_findings(comparison, artifact, replay_artifact)

        return CompetitorOutput(
            replay_artifact=replay_artifact,
            comparison=comparison,
            findings=findings,
            improvement_achieved=comparison.metrics.score_delta > 0,
        )

    def _simulate_replay_with_lessons(
        self,
        baseline: RunArtifact,
        lessons: list[CuratedLesson],
    ) -> RunArtifact:
        simulated_tool_calls = baseline.tool_calls.copy()
        simulated_outcome = baseline.outcome
        simulated_error = baseline.error_message

        for lesson in lessons:
            payload = lesson.lesson_payload
            if lesson.lesson_type == "tool":
                tool_id = payload.get("tool_id")
                timeout = payload.get("timeout_override")
                if timeout and tool_id:
                    for tc in simulated_tool_calls:
                        if tc.tool_name == tool_id and tc.outcome == "failure":
                            if tc.error_message and "timeout" in tc.error_message.lower():
                                simulated_outcome = "success"
                                simulated_error = None
                                tc.outcome = "success"
                                tc.error_message = None
            if lesson.lesson_type == "policy":
                rule_type = payload.get("rule_type")
                if rule_type in ("engagement", "ranking"):
                    failed_calls = [tc for tc in simulated_tool_calls if tc.outcome == "failure"]
                    if failed_calls and len(failed_calls) < len(simulated_tool_calls) // 2:
                        simulated_outcome = "success"
                        simulated_error = None
        success_rate = sum(1 for tc in simulated_tool_calls if tc.outcome == "success")
        total_calls = len(simulated_tool_calls)
        if total_calls > 0 and success_rate / total_calls > 0.8:
            simulated_outcome = "success"
            simulated_error = None
        return RunArtifact(
            run_id=f"replay-{baseline.run_id}-{uuid4().hex[:8]}",
            suite_id=baseline.suite_id,
            agent_name=baseline.agent_name,
            outcome=simulated_outcome,
            tool_calls=simulated_tool_calls,
            steps=baseline.steps,
            messages=baseline.messages,
            total_duration_ms=baseline.total_duration_ms,
            token_usage=baseline.token_usage,
            cost_usd=baseline.cost_usd,
            error_message=simulated_error,
            metadata={
                **baseline.metadata,
                "replay_of": baseline.run_id,
                "lessons_applied": [lesson.lesson_id for lesson in lessons],
                "simulation": True,
            },
            created_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

    def _generate_competition_findings(
        self,
        comparison: ComparisonResult,
        baseline: RunArtifact,
        treatment: RunArtifact,
    ) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        if comparison.metrics.score_delta > 0:
            findings.append(
                ReviewFinding(
                    finding_id=f"finding-{uuid4().hex[:8]}",
                    category="competition",
                    severity="info",
                    title="Replay showed improvement",
                    description=f"Score delta: +{comparison.metrics.score_delta:.2f}",
                    evidence_refs=["comparison.metrics"],
                    recommendation="Consider approving the applied lessons",
                )
            )
        if comparison.regressions:
            for regression in comparison.regressions:
                findings.append(
                    ReviewFinding(
                        finding_id=f"finding-{uuid4().hex[:8]}",
                        category="competition",
                        severity="warning",
                        title="Regression detected in replay",
                        description=regression,
                        evidence_refs=["comparison.regressions"],
                        recommendation="Review lesson applicability before approval",
                    )
                )
        if baseline.outcome == "failure" and treatment.outcome == "success":
            findings.append(
                ReviewFinding(
                    finding_id=f"finding-{uuid4().hex[:8]}",
                    category="competition",
                    severity="info",
                    title="Failed run recovered with lessons",
                    description="Baseline failed but treatment succeeded",
                    evidence_refs=["baseline.outcome", "treatment.outcome"],
                )
            )
        return findings

    def can_replay(self, artifact: RunArtifact) -> bool:
        if artifact.is_successful():
            return False
        if not artifact.tool_calls and not artifact.steps:
            return False
        return True

    def get_replay_candidates(
        self,
        artifacts: list[RunArtifact],
        lessons: list[CuratedLesson],
    ) -> list[RunArtifact]:
        candidates = []
        for artifact in artifacts:
            if not self.can_replay(artifact):
                continue
            applicable_lessons = [
                lesson for lesson in lessons if artifact.agent_name in lesson.applies_to_agents
            ]
            if applicable_lessons:
                candidates.append(artifact)
        return candidates
