# type-hygiene: skip-file  # dynamic trace data payloads are intentionally Any
from __future__ import annotations

from math import sqrt
from typing import Any

import pydantic as pd


class DivergencePoint(pd.BaseModel):
    step_index: int = pd.Field(description="Aligned step index for divergence.")
    dimension: str = pd.Field(description="Emotion dimension for divergence.")
    score_a: float = pd.Field(description="Aligned score for run A.")
    score_b: float = pd.Field(description="Aligned score for run B.")
    delta: float = pd.Field(description="Difference between run A and run B.")
    run_a_outcome: str = pd.Field(description="Outcome label for run A.")
    run_b_outcome: str = pd.Field(description="Outcome label for run B.")
    label: str = pd.Field(description="Narrative label for divergence.")

    model_config = pd.ConfigDict(extra="forbid")


class CrossRunComparison(pd.BaseModel):
    run_ids: tuple[str, str] = pd.Field(description="Run IDs being compared.")
    divergences: list[DivergencePoint] = pd.Field(description="Detected divergences.")
    correlation: float = pd.Field(description="Overall Pearson correlation.")
    summary: str = pd.Field(description="Narrative summary of comparison.")

    model_config = pd.ConfigDict(extra="forbid")


class CrossRunComparator:
    def compare(
        self,
        run_a_scores: list[dict[str, Any]],
        run_b_scores: list[dict[str, Any]],
        run_a_outcome: str,
        run_b_outcome: str,
        dimensions: list[str],
        threshold: float = 0.4,
    ) -> CrossRunComparison:
        run_a_id = self._infer_run_id(run_a_scores, "run_a")
        run_b_id = self._infer_run_id(run_b_scores, "run_b")
        run_ids = (run_a_id, run_b_id)

        if not run_a_scores or not run_b_scores:
            summary = self._generate_summary([], 0.0, (run_a_outcome, run_b_outcome))
            return CrossRunComparison(
                run_ids=run_ids,
                divergences=[],
                correlation=0.0,
                summary=summary,
            )

        max_len = max(len(run_a_scores), len(run_b_scores))
        percent_grid = self._build_percent_grid(max_len)
        divergences: list[DivergencePoint] = []

        aligned_pairs: list[tuple[float, float]] = []
        for dimension in dimensions:
            series_a = self._build_series(run_a_scores, dimension)
            series_b = self._build_series(run_b_scores, dimension)
            if not series_a or not series_b:
                continue

            aligned_a = [self._interpolate(series_a, percent) for percent in percent_grid]
            aligned_b = [self._interpolate(series_b, percent) for percent in percent_grid]
            aligned_pairs.extend(zip(aligned_a, aligned_b, strict=False))

            for idx, (score_a, score_b) in enumerate(zip(aligned_a, aligned_b, strict=False)):
                delta = score_a - score_b
                if abs(delta) < threshold:
                    continue
                label = self._label_divergence(dimension, delta)
                divergences.append(
                    DivergencePoint(
                        step_index=int(round(percent_grid[idx])),
                        dimension=dimension,
                        score_a=score_a,
                        score_b=score_b,
                        delta=delta,
                        run_a_outcome=run_a_outcome,
                        run_b_outcome=run_b_outcome,
                        label=label,
                    )
                )

        correlation = self._compute_correlation(aligned_pairs)
        summary = self._generate_summary(divergences, correlation, (run_a_outcome, run_b_outcome))

        return CrossRunComparison(
            run_ids=run_ids,
            divergences=divergences,
            correlation=correlation,
            summary=summary,
        )

    def _compute_correlation(self, pairs: list[tuple[float, float]]) -> float:
        if len(pairs) < 2:
            return 0.0
        scores_a = [pair[0] for pair in pairs]
        scores_b = [pair[1] for pair in pairs]
        mean_a = sum(scores_a) / len(scores_a)
        mean_b = sum(scores_b) / len(scores_b)

        numerator = sum((a - mean_a) * (b - mean_b) for a, b in pairs)
        denom_a = sum((a - mean_a) ** 2 for a in scores_a)
        denom_b = sum((b - mean_b) ** 2 for b in scores_b)
        if denom_a == 0 or denom_b == 0:
            return 0.0
        return numerator / sqrt(denom_a * denom_b)

    def _generate_summary(
        self,
        divergences: list[DivergencePoint],
        correlation: float,
        outcomes: tuple[str, str],
    ) -> str:
        run_a_outcome, run_b_outcome = outcomes
        if not divergences:
            return (
                f"Runs track closely (correlation {correlation:.2f}). Outcomes: "
                f"{run_a_outcome} vs {run_b_outcome}."
            )

        strongest = max(divergences, key=lambda item: abs(item.delta))
        direction = "higher" if strongest.delta > 0 else "lower"
        return (
            f"Detected {len(divergences)} divergences (correlation {correlation:.2f}). "
            f"Largest gap: run A {direction} on {strongest.dimension} at "
            f"{strongest.step_index}% completion. Outcomes: {run_a_outcome} vs {run_b_outcome}."
        )

    @staticmethod
    def _build_percent_grid(length: int) -> list[float]:
        if length <= 1:
            return [0.0]
        step = 100.0 / (length - 1)
        return [step * idx for idx in range(length)]

    @staticmethod
    def _infer_run_id(run_scores: list[dict[str, Any]], fallback: str) -> str:
        if run_scores:
            run_id = run_scores[0].get("run_id")
            if isinstance(run_id, str) and run_id:
                return run_id
        return fallback

    @staticmethod
    def _build_series(step_scores: list[dict[str, Any]], dimension: str) -> list[float]:
        series: list[float] = []
        for step in step_scores:
            score = CrossRunComparator._extract_score(step, dimension)
            if score is not None:
                series.append(score)
        return series

    @staticmethod
    def _extract_score(step: dict[str, Any], dimension: str) -> float | None:
        if isinstance(step.get("scores"), dict):
            raw = step["scores"].get(dimension)
        else:
            raw = step.get(dimension)
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _interpolate(series: list[float], percent: float) -> float:
        if len(series) == 1:
            return series[0]
        position = (percent / 100.0) * (len(series) - 1)
        lower = int(position)
        upper = min(lower + 1, len(series) - 1)
        if lower == upper:
            return series[lower]
        weight = position - lower
        return (1 - weight) * series[lower] + weight * series[upper]

    @staticmethod
    def _label_divergence(dimension: str, delta: float) -> str:
        if delta > 0:
            return f"run A higher on {dimension}"
        if delta < 0:
            return f"run A lower on {dimension}"
        return f"run A matches run B on {dimension}"
