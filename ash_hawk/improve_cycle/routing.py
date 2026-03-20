from __future__ import annotations

from ash_hawk.improve_cycle.models import FailureCategory, TriageOutput


def should_run_coach(triage: TriageOutput) -> bool:
    return triage.primary_owner in {"coach", "both"}


def should_run_architect(triage: TriageOutput) -> bool:
    return triage.primary_owner in {"architect", "both"}


def should_block_promotion(triage: TriageOutput) -> bool:
    return triage.primary_owner == "block" or triage.primary_cause.category in {
        FailureCategory.NONDETERMINISM,
        FailureCategory.ENVIRONMENTAL_FLAKE,
    }
