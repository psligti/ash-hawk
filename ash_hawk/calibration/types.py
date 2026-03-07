"""Calibration types for eval score reliability."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class GroundTruth:
    """Ground truth label for a single eval task."""

    task_id: str
    expected_passed: bool
    expected_score: float | None = None
    notes: str | None = None


@dataclass(frozen=True)
class CalibrationBin:
    """A single bin in the calibration reliability diagram."""

    predicted_mean: float
    actual_accuracy: float
    count: int
    predicted_range: tuple[float, float]


@dataclass
class CalibrationResult:
    """Result of calibration analysis."""

    ece: float
    brier_score: float
    num_samples: int
    pass_rate: float
    mean_predicted: float
    mean_actual: float
    per_bin_accuracy: list[CalibrationBin]
    disagreement_tasks: list[str]
    generated_at: str

    @property
    def is_well_calibrated(self) -> bool:
        """Return True if ECE < 0.1 (commonly accepted threshold)."""
        return self.ece < 0.1

    def to_dict(self) -> dict[str, Any]:
        return {
            "ece": self.ece,
            "brier_score": self.brier_score,
            "num_samples": self.num_samples,
            "pass_rate": self.pass_rate,
            "mean_predicted": self.mean_predicted,
            "mean_actual": self.mean_actual,
            "per_bin_accuracy": [
                {
                    "predicted_mean": b.predicted_mean,
                    "actual_accuracy": b.actual_accuracy,
                    "count": b.count,
                    "predicted_range": b.predicted_range,
                }
                for b in self.per_bin_accuracy
            ],
            "disagreement_tasks": self.disagreement_tasks,
            "generated_at": self.generated_at,
            "is_well_calibrated": self.is_well_calibrated,
        }


@dataclass
class CalibrationConfig:
    """Configuration for calibration runner."""

    n_bins: int = 10
    disagreement_threshold: float = 0.3
    pass_threshold: float = 0.7
    min_samples: int = 10


__all__ = [
    "GroundTruth",
    "CalibrationBin",
    "CalibrationResult",
    "CalibrationConfig",
]
