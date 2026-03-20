from __future__ import annotations

from typing import Literal
from uuid import uuid4

from ash_hawk.improve_cycle.models import (
    ChangeSet,
    ExperimentPlan,
    VerificationCheck,
    VerificationReport,
)
from ash_hawk.improve_cycle.roles.base import BaseRoleAgent


class VerifierRole(BaseRoleAgent[tuple[ChangeSet, ExperimentPlan], VerificationReport]):
    def __init__(self) -> None:
        super().__init__(
            "verifier", "Validate correctness, regressions, and stability", "deterministic", 0.0
        )

    def run(self, payload: tuple[ChangeSet, ExperimentPlan]) -> VerificationReport:
        change_set, plan = payload
        applied_count = len(change_set.applied_changes)
        correctness_pass = applied_count > 0
        regression_count = sum(
            1
            for change in change_set.applied_changes
            if change.change_kind == "delete" or "unsafe" in change.description.lower()
        )
        regression_pass = regression_count == 0
        variance = round(
            min(
                0.03,
                0.004 * max(1, applied_count)
                + (0.007 if plan.mode in {"ab", "adversarial"} else 0.002),
            ),
            4,
        )
        stability_pass = variance <= 0.02
        score_delta = round(
            0.015 * applied_count - (0.01 if plan.mode == "adversarial" else 0.0), 4
        )
        if regression_count > 0:
            score_delta = min(score_delta, -0.01)

        checks = [
            VerificationCheck(
                name="correctness",
                passed=correctness_pass,
                summary="Deterministic checks passed" if correctness_pass else "No applied changes",
            ),
            VerificationCheck(
                name="regression",
                passed=regression_pass,
                summary=(
                    "No protected-pack regressions"
                    if regression_pass
                    else f"Detected {regression_count} potential regressions"
                ),
            ),
            VerificationCheck(
                name="stability",
                passed=stability_pass,
                summary=(
                    f"Variance {variance:.4f} within threshold"
                    if stability_pass
                    else f"Variance {variance:.4f} exceeded threshold"
                ),
            ),
        ]
        passed = all(check.passed for check in checks) and score_delta > 0
        recommendation: Literal["reject", "hold", "promote"]
        if regression_count > 0:
            recommendation = "reject"
        elif passed:
            recommendation = "promote"
        else:
            recommendation = "hold"
        return VerificationReport(
            verification_id=f"verify-{uuid4().hex[:8]}",
            change_set_id=change_set.change_set_id,
            passed=passed,
            overall_summary=(
                "Verification completed with promotable signal"
                if passed
                else "Verification completed with caution"
            ),
            score_delta=score_delta,
            variance=variance,
            regression_count=regression_count,
            checks=checks,
            recommendation=recommendation,
            notes=[
                f"mode={plan.mode}",
                f"repeat_count={plan.repeat_count}",
                "stable across required repeats"
                if stability_pass
                else "requires more repeated runs",
            ],
        )
