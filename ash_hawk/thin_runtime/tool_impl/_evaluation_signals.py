from __future__ import annotations

from dataclasses import dataclass

from ash_hawk.thin_runtime.models import ToolCall


@dataclass(frozen=True)
class EvaluationSignals:
    baseline_score: float | None
    candidate_score: float | None
    candidate_label: str
    score_regressed: bool
    failure_family: str | None
    failure_explanations: tuple[str, ...]
    aggregate_passed: bool | None
    verification_verified: bool | None
    existing_regressions: tuple[str, ...]


def collect_evaluation_signals(call: ToolCall) -> EvaluationSignals:
    evaluation = call.context.evaluation
    failure = call.context.failure
    audit = call.context.audit

    baseline_score = _coerce_score(evaluation.baseline_summary.score)
    repeat_score = _coerce_score(evaluation.repeat_eval_summary.score)
    last_eval_score = _coerce_score(evaluation.last_eval_summary.score)
    targeted_score = _coerce_score(evaluation.targeted_validation_summary.score)
    integrity_score = _coerce_score(evaluation.integrity_summary.score)

    candidate_score = None
    candidate_label = "current evaluation"
    for label, score in (
        ("repeat evaluation", repeat_score),
        ("last evaluation", last_eval_score),
        ("targeted validation", targeted_score),
        ("integrity validation", integrity_score),
        ("baseline evaluation", baseline_score),
    ):
        if score is not None:
            candidate_score = score
            candidate_label = label
            break

    score_regressed = (
        baseline_score is not None
        and candidate_score is not None
        and candidate_label != "baseline evaluation"
        and candidate_score < baseline_score
    )

    run_result = audit.run_result
    failure_explanations = tuple(item for item in failure.explanations if item)
    existing_regressions = tuple(item for item in evaluation.regressions if item)
    failure_family = (
        failure.failure_family.strip()
        if isinstance(failure.failure_family, str) and failure.failure_family.strip()
        else None
    )

    return EvaluationSignals(
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        candidate_label=candidate_label,
        score_regressed=score_regressed,
        failure_family=failure_family,
        failure_explanations=failure_explanations,
        aggregate_passed=run_result.aggregate_passed,
        verification_verified=evaluation.verification.verified,
        existing_regressions=existing_regressions,
    )


def _coerce_score(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None
