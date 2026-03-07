"""Calibration runner for analyzing eval reliability."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from ash_hawk.calibration.brier import compute_brier_score
from ash_hawk.calibration.ece import compute_ece_with_bins
from ash_hawk.calibration.types import (
    CalibrationBin,
    CalibrationConfig,
    CalibrationResult,
    GroundTruth,
)


class CalibrationRunner:
    """Runner for computing calibration metrics on eval results."""

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self._config = config or CalibrationConfig()
        self._results: list[
            tuple[str, float, bool, bool]
        ] = []  # task_id, pred_score, pred_passed, actual_passed

    def add_result(
        self,
        task_id: str,
        predicted_score: float,
        predicted_passed: bool,
        actual_passed: bool,
    ) -> None:
        """Add a single eval result for calibration analysis."""
        self._results.append((task_id, predicted_score, predicted_passed, actual_passed))

    def add_results_from_ground_truth(
        self,
        eval_results: dict[str, tuple[float, bool]],
        ground_truths: Sequence[GroundTruth],
    ) -> None:
        """Add results by matching eval outputs to ground truth labels.

        Args:
            eval_results: Dict mapping task_id to (score, passed)
            ground_truths: Sequence of GroundTruth labels
        """
        gt_map = {gt.task_id: gt for gt in ground_truths}
        for task_id, (score, pred_passed) in eval_results.items():
            if task_id in gt_map:
                gt = gt_map[task_id]
                self.add_result(task_id, score, pred_passed, gt.expected_passed)

    def compute(self) -> CalibrationResult:
        """Compute calibration metrics from accumulated results."""
        if len(self._results) < self._config.min_samples:
            raise ValueError(
                f"Need at least {self._config.min_samples} samples, got {len(self._results)}"
            )

        predicted_scores = [r[1] for r in self._results]
        actual_outcomes = [r[3] for r in self._results]

        ece, bin_details = compute_ece_with_bins(
            predicted_scores, actual_outcomes, self._config.n_bins
        )
        brier = compute_brier_score(predicted_scores, actual_outcomes)

        # Build per-bin accuracy list
        per_bin: list[CalibrationBin] = []
        bin_width = 1.0 / self._config.n_bins
        for i, (pred_mean, actual_acc, count) in enumerate(bin_details):
            per_bin.append(
                CalibrationBin(
                    predicted_mean=pred_mean,
                    actual_accuracy=actual_acc,
                    count=count,
                    predicted_range=(i * bin_width, (i + 1) * bin_width),
                )
            )

        # Find disagreement tasks
        disagreement = self._find_disagreement_tasks()

        return CalibrationResult(
            ece=ece,
            brier_score=brier,
            num_samples=len(self._results),
            pass_rate=sum(1 for r in self._results if r[3]) / len(self._results),
            mean_predicted=sum(predicted_scores) / len(predicted_scores),
            mean_actual=sum(actual_outcomes) / len(actual_outcomes),
            per_bin_accuracy=per_bin,
            disagreement_tasks=disagreement,
            generated_at=datetime.now(UTC).isoformat(),
        )

    def _find_disagreement_tasks(self) -> list[str]:
        """Find tasks where prediction differs significantly from actual."""
        disagreement = []
        for task_id, pred_score, pred_passed, actual_passed in self._results:
            # Disagreement: predicted passed but score below threshold
            # or predicted failed but score above threshold
            if pred_passed != actual_passed:
                disagreement.append(task_id)
            elif (
                abs(pred_score - (1.0 if actual_passed else 0.0))
                > self._config.disagreement_threshold
            ):
                disagreement.append(task_id)
        return disagreement

    def load_ground_truth(self, path: Path) -> list[GroundTruth]:
        """Load ground truth labels from JSON file.

        Expected format:
        {
            "labels": [
                {"task_id": "task-001", "expected_passed": true, "notes": "..."},
                ...
            ]
        }
        """
        data = json.loads(path.read_text())
        labels = []
        for item in data.get("labels", []):
            labels.append(
                GroundTruth(
                    task_id=item["task_id"],
                    expected_passed=item["expected_passed"],
                    expected_score=item.get("expected_score"),
                    notes=item.get("notes"),
                )
            )
        return labels

    def clear(self) -> None:
        """Clear accumulated results."""
        self._results.clear()


def run_calibration(
    eval_results: dict[str, tuple[float, bool]],
    ground_truths: Sequence[GroundTruth],
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    """Convenience function to run calibration analysis.

    Args:
        eval_results: Dict mapping task_id to (score, passed)
        ground_truths: Sequence of GroundTruth labels
        config: Optional calibration configuration

    Returns:
        CalibrationResult with ECE, Brier score, and bin details
    """
    runner = CalibrationRunner(config)
    runner.add_results_from_ground_truth(eval_results, ground_truths)
    return runner.compute()


__all__ = ["CalibrationRunner", "run_calibration"]
