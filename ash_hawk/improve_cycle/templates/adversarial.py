"""Adversarial template for improve-cycle stress testing.

This template provides utilities for generating adversarial scenarios
and stress-test configurations to prevent overfitting and expose
hidden weaknesses in lessons.
"""

from __future__ import annotations

from typing import Any

from ash_hawk.improve_cycle.models import (
    AdversarialScenario,
    AnalystOutput,
    ExperimentPlan,
    ProposalType,
    RiskLevel,
    RunArtifactBundle,
    VerificationReport,
)


def create_adversarial_experiment_plan(
    scenarios: list[AdversarialScenario],
    run_bundle: RunArtifactBundle,
    *,
    intensity: str = "medium",
) -> ExperimentPlan:
    """Create an adversarial experiment plan.

    Adversarial experiments stress-test lessons under edge cases
    and contradictory signals.

    Args:
        scenarios: Adversarial scenarios to include.
        run_bundle: Current run artifact bundle.
        intensity: Adversarial intensity level (low/medium/high).

    Returns:
        ExperimentPlan configured for adversarial mode.
    """
    intensity_multipliers = {"low": 1, "medium": 2, "high": 3}
    repeat_multiplier = intensity_multipliers.get(intensity, 2)

    return ExperimentPlan(
        experiment_plan_id=f"adversarial-{run_bundle.run_id}",
        lesson_ids=[],  # Adversarial mode doesn't test specific lessons
        mode="adversarial",
        scenario_ids=run_bundle.scenario_ids,
        eval_pack_ids=[run_bundle.eval_pack_id],
        repeat_count=3 * repeat_multiplier,
        acceptance_criteria=[
            "robustness_under_adversarial>=0.8",
            "no_critical_breakdown",
            "graceful_degradation",
        ],
        rejection_criteria=[
            "adversarial_collapse",
            "cascading_failures",
            "unsafe_fallback",
        ],
        rollback_criteria=[
            "adversarial_damage_unrecoverable",
        ],
        max_latency_delta_pct=25.0,
        max_token_delta_pct=20.0,
        notes=f"Adversarial stress test with {len(scenarios)} scenarios at {intensity} intensity",
    )


def generate_adversarial_scenarios(
    analyst_output: AnalystOutput,
    reports: list[VerificationReport],
    *,
    max_scenarios: int = 5,
) -> list[AdversarialScenario]:
    """Generate adversarial scenarios from analysis results.

    Adversarial scenarios target identified weaknesses and stress
    the system under challenging conditions.

    Args:
        analyst_output: Analysis results with risk areas.
        reports: Verification reports showing potential weak spots.
        max_scenarios: Maximum number of scenarios to generate.

    Returns:
        List of adversarial scenarios targeting identified weaknesses.
    """
    scenarios: list[AdversarialScenario] = []

    risk_areas = analyst_output.risk_areas or ["general"]
    high_variance = any(r.variance and r.variance > 0.02 for r in reports)
    regressions = sum(r.regression_count for r in reports)

    for idx, area in enumerate(risk_areas[:max_scenarios]):
        scenario = AdversarialScenario(
            scenario_id=f"adv-scenario-{idx + 1}",
            title=f"Adversarial stress: {area}",
            target_weakness=area,
            description=(
                f"Stress test targeting {area} weakness with "
                f"{'high variance signals' if high_variance else 'normal variance'} "
                f"and {regressions} prior regressions"
            ),
            expected_failure_mode="premature_confidence" if idx % 2 == 0 else "cascade_failure",
            evaluation_hooks=[
                "robustness_check",
                "graceful_degradation",
                "evidence_consistency",
            ],
        )
        scenarios.append(scenario)

    return scenarios


def assess_adversarial_results(
    scenarios: list[AdversarialScenario],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assess adversarial experiment results.

    Args:
        scenarios: Scenarios that were tested.
        results: Results from adversarial execution.

    Returns:
        Assessment summary with robustness metrics.
    """
    if not scenarios or not results:
        return {"robustness_score": 0.0, "recommendation": "insufficient_data"}

    passed = sum(1 for r in results if r.get("passed", False))
    total = len(results)
    robustness = passed / max(1, total)

    return {
        "robustness_score": robustness,
        "scenarios_tested": len(scenarios),
        "pass_rate": robustness,
        "recommendation": (
            "promote_with_caution"
            if robustness >= 0.8
            else "hold_for_hardening"
            if robustness >= 0.6
            else "reject_requires_defense"
        ),
    }
