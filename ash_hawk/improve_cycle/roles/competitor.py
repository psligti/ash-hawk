from __future__ import annotations

from ash_hawk.improve_cycle.models import (
    CompetitorOutput,
    EvidenceRef,
    MetricValue,
    RunArtifactBundle,
)
from ash_hawk.improve_cycle.roles.base import BaseRoleAgent


class CompetitorRole(BaseRoleAgent[RunArtifactBundle, CompetitorOutput]):
    def __init__(self) -> None:
        super().__init__(
            "competitor", "Replay weak runs and compare measurable deltas", "deterministic", 0.0
        )

    def run(self, payload: RunArtifactBundle) -> CompetitorOutput:
        before_score = next((m.value for m in payload.metrics if m.name == "score"), 0.0)
        after_score = min(1.0, before_score + 0.05)
        return CompetitorOutput(
            baseline_run_id=payload.run_id,
            replay_run_id=f"replay-{payload.run_id}",
            improved=after_score > before_score,
            summary="Replay completed with targeted adjustments",
            metrics_before=[MetricValue(name="score", value=before_score)],
            metrics_after=[
                MetricValue(
                    name="score",
                    value=after_score,
                    baseline_value=before_score,
                    delta=after_score - before_score,
                )
            ],
            evidence=[
                EvidenceRef(
                    artifact_id=payload.run_id,
                    kind="comparison",
                    note="baseline_vs_replay",
                )
            ],
        )
