"""Research template for improve-cycle experiments.

This template provides utilities for generating research-focused
experiment configurations and lesson discovery workflows.
"""

from __future__ import annotations

from typing import Any

from ash_hawk.improve_cycle.models import (
    CuratedLesson,
    ExperimentPlan,
    ProposalType,
    RiskLevel,
    RunArtifactBundle,
)


def create_research_experiment_plan(
    lessons: list[CuratedLesson],
    run_bundle: RunArtifactBundle,
    *,
    max_iterations: int = 5,
    focus_areas: list[str] | None = None,
) -> ExperimentPlan:
    """Create a research-oriented experiment plan.

    Research experiments prioritize discovery over immediate wins,
    allowing broader exploration of lesson applicability.

    Args:
        lessons: Curated lessons to research.
        run_bundle: Current run artifact bundle.
        max_iterations: Maximum research iterations.
        focus_areas: Optional focus areas to constrain research scope.

    Returns:
        ExperimentPlan configured for research mode.
    """
    focus = focus_areas or ["behavior", "tooling", "reliability"]

    return ExperimentPlan(
        experiment_plan_id=f"research-{run_bundle.run_id}",
        lesson_ids=[lesson.lesson_id for lesson in lessons],
        mode="isolated",
        scenario_ids=run_bundle.scenario_ids,
        eval_pack_ids=[run_bundle.eval_pack_id],
        repeat_count=max(1, max_iterations // 2),
        acceptance_criteria=[
            "novel_insight_discovered",
            "evidence_quality>=0.7",
            "no_critical_regressions",
        ],
        rejection_criteria=[
            "no_new_insights",
            "evidence_inconclusive",
        ],
        rollback_criteria=[
            "research_debt_accumulated",
        ],
        max_latency_delta_pct=20.0,
        max_token_delta_pct=15.0,
        notes=f"Research experiment focusing on: {', '.join(focus)}",
    )


def generate_research_lessons(
    findings: list[dict[str, Any]],
    *,
    min_confidence: float = 0.6,
) -> list[CuratedLesson]:
    """Generate research-oriented lessons from findings.

    Research lessons prioritize exploration and hypothesis formation
    over immediate actionable changes.

    Args:
        findings: Raw findings from analysis.
        min_confidence: Minimum confidence threshold.

    Returns:
        List of research-oriented curated lessons.
    """
    lessons: list[CuratedLesson] = []
    for idx, finding in enumerate(findings):
        confidence = float(finding.get("confidence", 0.5))
        if confidence < min_confidence:
            continue

        lessons.append(
            CuratedLesson(
                lesson_id=f"research-lesson-{idx + 1}",
                proposal_id=finding.get("proposal_id", f"rp-{idx + 1}"),
                proposal_type=ProposalType.PLAYBOOK_UPDATE,
                title=f"Research: {finding.get('title', 'Unknown finding')}",
                summary=finding.get("summary", "Research-oriented finding"),
                target_surface=finding.get("surface", "research_notes"),
                approved=True,
                curation_notes="Generated for research exploration",
                confidence=confidence,
                risk_level=RiskLevel.LOW,
                lineage=[finding.get("source", "research")],
            )
        )
    return lessons
