"""Comparison service for before/after measurement of lesson effectiveness."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pydantic as pd

from ash_hawk.contracts import RunArtifact


class ComparisonMetrics(pd.BaseModel):
    """Metrics comparing before and after lesson application."""

    score_delta: float = pd.Field(
        default=0.0,
        description="Change in overall score (after - before)",
    )
    efficiency_delta: float | None = pd.Field(
        default=None,
        description="Change in efficiency score",
    )
    quality_delta: float | None = pd.Field(
        default=None,
        description="Change in quality score",
    )
    safety_delta: float | None = pd.Field(
        default=None,
        description="Change in safety score",
    )
    tool_calls_delta: int = pd.Field(
        default=0,
        description="Change in total tool calls",
    )
    errors_delta: int = pd.Field(
        default=0,
        description="Change in error count",
    )
    duration_delta_seconds: float = pd.Field(
        default=0.0,
        description="Change in execution duration",
    )

    model_config = pd.ConfigDict(extra="forbid")


class ComparisonResult(pd.BaseModel):
    """Full comparison result between baseline and treatment runs."""

    baseline_run_id: str = pd.Field(
        description="ID of the baseline run",
    )
    treatment_run_id: str = pd.Field(
        description="ID of the treatment run (with lessons applied)",
    )
    lessons_applied: list[str] = pd.Field(
        default_factory=list,
        description="Lesson IDs that were applied",
    )
    metrics: ComparisonMetrics = pd.Field(
        description="Comparison metrics",
    )
    improvements: list[str] = pd.Field(
        default_factory=list,
        description="Areas that improved",
    )
    regressions: list[str] = pd.Field(
        default_factory=list,
        description="Areas that regressed",
    )
    statistical_significance: float | None = pd.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="P-value for the comparison (if computed)",
    )
    created_at: datetime = pd.Field(
        default_factory=lambda: datetime.now(),
        description="When the comparison was computed",
    )

    model_config = pd.ConfigDict(extra="forbid")

    @pd.computed_field(return_type=bool)
    def is_improvement(self) -> bool:
        return self.metrics.score_delta > 0

    @pd.computed_field(return_type=bool)
    def has_regression(self) -> bool:
        return len(self.regressions) > 0


@dataclass
class ArtifactStats:
    score: float = 0.0
    efficiency_score: float | None = None
    quality_score: float | None = None
    safety_score: float | None = None
    total_tool_calls: int = 0
    error_count: int = 0
    duration_seconds: float = 0.0


class ComparisonService:
    """Computes before/after comparisons for lesson effectiveness."""

    def extract_stats(self, artifact: RunArtifact) -> ArtifactStats:
        total_tool_calls = len(artifact.tool_calls)

        error_count = sum(1 for tc in artifact.tool_calls if tc.outcome == "failure")

        duration_seconds = 0.0
        if artifact.total_duration_ms:
            duration_seconds = artifact.total_duration_ms / 1000.0
        elif artifact.created_at and artifact.completed_at:
            duration_seconds = (artifact.completed_at - artifact.created_at).total_seconds()

        score = artifact.get_tool_success_rate()
        quality_score = 1.0 if artifact.is_successful() else 0.0

        metadata = artifact.metadata
        efficiency_score = metadata.get("efficiency_score")
        safety_score = metadata.get("safety_score")

        return ArtifactStats(
            score=score,
            efficiency_score=efficiency_score,
            quality_score=quality_score,
            safety_score=safety_score,
            total_tool_calls=total_tool_calls,
            error_count=error_count,
            duration_seconds=duration_seconds,
        )

    def compare(
        self,
        baseline: RunArtifact,
        treatment: RunArtifact,
        lessons_applied: list[str] | None = None,
    ) -> ComparisonResult:
        baseline_stats = self.extract_stats(baseline)
        treatment_stats = self.extract_stats(treatment)

        metrics = ComparisonMetrics(
            score_delta=treatment_stats.score - baseline_stats.score,
            efficiency_delta=self._safe_delta(
                treatment_stats.efficiency_score,
                baseline_stats.efficiency_score,
            ),
            quality_delta=self._safe_delta(
                treatment_stats.quality_score,
                baseline_stats.quality_score,
            ),
            safety_delta=self._safe_delta(
                treatment_stats.safety_score,
                baseline_stats.safety_score,
            ),
            tool_calls_delta=treatment_stats.total_tool_calls - baseline_stats.total_tool_calls,
            errors_delta=treatment_stats.error_count - baseline_stats.error_count,
            duration_delta_seconds=treatment_stats.duration_seconds
            - baseline_stats.duration_seconds,
        )

        improvements: list[str] = []
        regressions: list[str] = []

        if metrics.score_delta > 0:
            improvements.append(f"Overall score improved by {metrics.score_delta:.2f}")
        elif metrics.score_delta < 0:
            regressions.append(f"Overall score decreased by {abs(metrics.score_delta):.2f}")

        if metrics.efficiency_delta is not None and metrics.efficiency_delta > 0:
            improvements.append(f"Efficiency improved by {metrics.efficiency_delta:.2f}")
        elif metrics.efficiency_delta is not None and metrics.efficiency_delta < 0:
            regressions.append(f"Efficiency decreased by {abs(metrics.efficiency_delta):.2f}")

        if metrics.tool_calls_delta < 0:
            improvements.append(f"Reduced tool calls by {abs(metrics.tool_calls_delta)}")
        elif metrics.tool_calls_delta > 0:
            regressions.append(f"Increased tool calls by {metrics.tool_calls_delta}")

        if metrics.errors_delta < 0:
            improvements.append(f"Reduced errors by {abs(metrics.errors_delta)}")
        elif metrics.errors_delta > 0:
            regressions.append(f"Increased errors by {metrics.errors_delta}")

        if metrics.duration_delta_seconds < 0:
            improvements.append(f"Faster execution by {abs(metrics.duration_delta_seconds):.1f}s")
        elif metrics.duration_delta_seconds > 0:
            regressions.append(f"Slower execution by {metrics.duration_delta_seconds:.1f}s")

        return ComparisonResult(
            baseline_run_id=baseline.run_id,
            treatment_run_id=treatment.run_id,
            lessons_applied=lessons_applied or [],
            metrics=metrics,
            improvements=improvements,
            regressions=regressions,
        )

    def _safe_delta(
        self,
        after: float | None,
        before: float | None,
    ) -> float | None:
        if after is None or before is None:
            return None
        return after - before

    def compare_multiple(
        self,
        baselines: list[RunArtifact],
        treatments: list[RunArtifact],
        lessons_applied: list[str] | None = None,
    ) -> dict[str, Any]:
        if not baselines or not treatments:
            return {"error": "Empty baseline or treatment list"}

        all_results: list[ComparisonResult] = []
        for baseline in baselines:
            for treatment in treatments:
                result = self.compare(baseline, treatment, lessons_applied)
                all_results.append(result)

        avg_score_delta = sum(r.metrics.score_delta for r in all_results) / len(all_results)
        avg_efficiency_delta = self._average_deltas(
            [r.metrics.efficiency_delta for r in all_results]
        )
        avg_tool_calls_delta = sum(r.metrics.tool_calls_delta for r in all_results) / len(
            all_results
        )
        avg_errors_delta = sum(r.metrics.errors_delta for r in all_results) / len(all_results)

        improvements_count = sum(1 for r in all_results if r.metrics.score_delta > 0)
        regressions_count = sum(1 for r in all_results if len(r.regressions) > 0)

        return {
            "total_comparisons": len(all_results),
            "average_score_delta": avg_score_delta,
            "average_efficiency_delta": avg_efficiency_delta,
            "average_tool_calls_delta": avg_tool_calls_delta,
            "average_errors_delta": avg_errors_delta,
            "improvements_count": improvements_count,
            "regressions_count": regressions_count,
            "improvement_rate": improvements_count / len(all_results) if all_results else 0,
            "detailed_results": [r.model_dump() for r in all_results],
        }

    def _average_deltas(self, deltas: list[float | None]) -> float | None:
        valid = [d for d in deltas if d is not None]
        if not valid:
            return None
        return sum(valid) / len(valid)
