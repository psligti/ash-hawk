"""Replay template for improve-cycle baseline comparison.

This template provides utilities for replay-based comparison
experiments and delta measurement.
"""

from __future__ import annotations

from typing import Any

from ash_hawk.improve_cycle.models import (
    CompetitorOutput,
    CuratedLesson,
    ExperimentPlan,
    MetricValue,
    ProposalType,
    RiskLevel,
    RunArtifactBundle,
)


def create_replay_experiment_plan(
    baseline_run_id: str,
    lessons: list[CuratedLesson],
    run_bundle: RunArtifactBundle,
    *,
    comparison_mode: str = "strict",
) -> ExperimentPlan:
    """Create a replay experiment plan.

    Replay experiments compare lesson-influenced runs against
    a baseline to measure measurable improvements.

    Args:
        baseline_run_id: ID of the baseline run to compare against.
        lessons: Lessons to apply during replay.
        run_bundle: Current run artifact bundle.
        comparison_mode: Comparison strictness (strict/relaxed).

    Returns:
        ExperimentPlan configured for replay comparison.
    """
    return ExperimentPlan(
        experiment_plan_id=f"replay-{run_bundle.run_id}",
        lesson_ids=[lesson.lesson_id for lesson in lessons],
        mode="ab",
        scenario_ids=run_bundle.scenario_ids,
        eval_pack_ids=[run_bundle.eval_pack_id],
        repeat_count=3,
        acceptance_criteria=[
            "score_delta>0",
            "no_regressions" if comparison_mode == "strict" else "regressions_acceptable",
        ],
        rejection_criteria=[
            "score_delta<0",
            "variance>0.03",
        ],
        rollback_criteria=[
            "latency_regression>20%",
        ],
        max_latency_delta_pct=15.0,
        max_token_delta_pct=10.0,
        notes=f"Replay comparison against baseline {baseline_run_id}",
    )


def compute_replay_deltas(
    baseline_metrics: list[MetricValue],
    replay_metrics: list[MetricValue],
) -> dict[str, Any]:
    """Compute deltas between baseline and replay metrics.

    Args:
        baseline_metrics: Metrics from the baseline run.
        replay_metrics: Metrics from the replay run.

    Returns:
        Dictionary of metric deltas with improvement flags.
    """
    baseline_by_name = {m.name: m for m in baseline_metrics}
    replay_by_name = {m.name: m for m in replay_metrics}

    deltas: dict[str, Any] = {}
    for name, replay_metric in replay_by_name.items():
        baseline_metric = baseline_by_name.get(name)
        if baseline_metric is None:
            continue

        delta = replay_metric.value - baseline_metric.value
        pct_delta = (delta / baseline_metric.value * 100) if baseline_metric.value != 0 else 0.0

        # For score, positive delta is improvement
        # For latency/token count, negative delta is improvement
        is_improvement = delta > 0 if name in {"score", "coverage"} else delta < 0

        deltas[name] = {
            "baseline": baseline_metric.value,
            "replay": replay_metric.value,
            "delta": delta,
            "pct_delta": pct_delta,
            "improved": is_improvement,
        }

    return deltas


def generate_replay_lessons(
    competitor_output: CompetitorOutput,
    *,
    improvement_threshold: float = 0.05,
) -> list[CuratedLesson]:
    """Generate lessons from replay comparison results.

    Args:
        competitor_output: Results from competitor replay.
        improvement_threshold: Minimum improvement to generate lesson.

    Returns:
        List of lessons derived from replay improvements.
    """
    lessons: list[CuratedLesson] = []

    if not competitor_output.improved:
        return lessons

    score_delta = next(
        (m.delta for m in competitor_output.metrics_after if m.name == "score"),
        0.0,
    )
    if score_delta is None or score_delta < improvement_threshold:
        return lessons

    lessons.append(
        CuratedLesson(
            lesson_id=f"replay-lesson-{competitor_output.replay_run_id}",
            proposal_id=competitor_output.baseline_run_id,
            proposal_type=ProposalType.PLAYBOOK_UPDATE,
            title=f"Replay improvement: +{score_delta:.3f} score",
            summary=competitor_output.summary,
            target_surface="replay_optimization",
            approved=True,
            curation_notes="Generated from replay comparison",
            confidence=min(0.95, 0.5 + score_delta),
            risk_level=RiskLevel.LOW,
            lineage=[competitor_output.baseline_run_id, competitor_output.replay_run_id],
        )
    )

    return lessons
