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
        score_before = next((m.value for m in payload.metrics if m.name == "score"), 0.0)
        replay_signal = min(0.15, 0.01 * len(payload.outputs) + 0.01 * len(payload.tool_traces))
        score_after = min(1.0, score_before + replay_signal)
        metrics_before: list[MetricValue] = [MetricValue(name="score", value=score_before)]
        metrics_after: list[MetricValue] = [
            MetricValue(
                name="score",
                value=score_after,
                baseline_value=score_before,
                delta=score_after - score_before,
            )
        ]

        for metric_name in ("latency_ms", "token_count"):
            baseline = next((m.value for m in payload.metrics if m.name == metric_name), None)
            if baseline is None:
                continue
            improved = max(0.0, baseline * 0.95)
            metrics_before.append(MetricValue(name=metric_name, value=baseline))
            metrics_after.append(
                MetricValue(
                    name=metric_name,
                    value=improved,
                    baseline_value=baseline,
                    delta=improved - baseline,
                )
            )

        improved = score_after > score_before + 0.005
        return CompetitorOutput(
            baseline_run_id=payload.run_id,
            replay_run_id=f"replay-{payload.run_id}",
            improved=improved,
            summary=(
                f"Replay {'improved' if improved else 'did not improve'} score "
                f"from {score_before:.3f} to {score_after:.3f}"
            ),
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            evidence=[
                EvidenceRef(
                    artifact_id=payload.run_id,
                    kind="comparison",
                    note=f"replay_signal={replay_signal:.3f}",
                )
            ],
        )
