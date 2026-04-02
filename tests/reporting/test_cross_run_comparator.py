# type-hygiene: skip-file  # test file — mock/factory types are intentionally loose

import pydantic as pd
import pytest

from ash_hawk.reporting.cross_run_comparator import (
    CrossRunComparator,
    CrossRunComparison,
    DivergencePoint,
)


class TestDivergencePoint:
    def test_model_creation(self) -> None:
        point = DivergencePoint(
            step_index=50,
            dimension="joy",
            score_a=0.7,
            score_b=0.2,
            delta=0.5,
            run_a_outcome="passed",
            run_b_outcome="failed",
            label="run A higher on joy",
        )

        assert point.step_index == 50
        assert point.dimension == "joy"

    def test_extra_forbid(self) -> None:
        with pytest.raises(pd.ValidationError):
            DivergencePoint.model_validate(
                {
                    "step_index": 50,
                    "dimension": "joy",
                    "score_a": 0.7,
                    "score_b": 0.2,
                    "delta": 0.5,
                    "run_a_outcome": "passed",
                    "run_b_outcome": "failed",
                    "label": "run A higher on joy",
                    "extra_field": "nope",
                }
            )


class TestCrossRunComparison:
    def test_model_creation(self) -> None:
        comparison = CrossRunComparison(
            run_ids=("run-a", "run-b"),
            divergences=[],
            correlation=1.0,
            summary="Runs track closely (correlation 1.00). Outcomes: ok vs ok.",
        )

        assert comparison.run_ids == ("run-a", "run-b")
        assert comparison.correlation == 1.0

    def test_extra_forbid(self) -> None:
        with pytest.raises(pd.ValidationError):
            CrossRunComparison.model_validate(
                {
                    "run_ids": ("run-a", "run-b"),
                    "divergences": [],
                    "correlation": 1.0,
                    "summary": "Runs track closely.",
                    "extra_field": "nope",
                }
            )


class TestCrossRunComparator:
    def test_compare_with_empty_run_a(self) -> None:
        comparator = CrossRunComparator()
        comparison = comparator.compare(
            [],
            [{"scores": {"joy": 0.5}}],
            "passed",
            "failed",
            ["joy"],
        )

        assert comparison.divergences == []
        assert comparison.correlation == 0.0

    def test_compare_with_empty_run_b(self) -> None:
        comparator = CrossRunComparator()
        comparison = comparator.compare(
            [{"scores": {"joy": 0.5}}],
            [],
            "passed",
            "failed",
            ["joy"],
        )

        assert comparison.divergences == []
        assert comparison.correlation == 0.0

    def test_compare_identical_runs(self) -> None:
        comparator = CrossRunComparator()
        run_scores = [
            {"scores": {"joy": 0.2}},
            {"scores": {"joy": 0.8}},
        ]
        comparison = comparator.compare(
            run_scores,
            run_scores,
            "passed",
            "passed",
            ["joy"],
        )

        assert comparison.divergences == []
        assert comparison.correlation == 1.0

    def test_compare_divergent_runs(self) -> None:
        comparator = CrossRunComparator()
        run_a = [
            {"scores": {"joy": 0.9}},
            {"scores": {"joy": 0.9}},
        ]
        run_b = [
            {"scores": {"joy": 0.0}},
            {"scores": {"joy": 0.0}},
        ]
        comparison = comparator.compare(
            run_a,
            run_b,
            "passed",
            "failed",
            ["joy"],
        )

        assert comparison.divergences
        assert all(abs(point.delta) >= 0.4 for point in comparison.divergences)

    def test_compare_with_single_step_each(self) -> None:
        comparator = CrossRunComparator()
        run_a = [{"scores": {"joy": 0.5}}]
        run_b = [{"scores": {"joy": 0.5}}]
        comparison = comparator.compare(
            run_a,
            run_b,
            "passed",
            "passed",
            ["joy"],
        )

        assert comparison.divergences == []
        assert comparison.correlation == 0.0

    def test_compute_correlation_identical_series(self) -> None:
        comparator = CrossRunComparator()
        compute = getattr(comparator, "_compute_correlation")
        pairs = [(0.1, 0.1), (0.9, 0.9)]

        assert compute(pairs) == 1.0

    def test_compute_correlation_with_few_pairs(self) -> None:
        comparator = CrossRunComparator()
        compute = getattr(comparator, "_compute_correlation")

        assert compute([(0.1, 0.1)]) == 0.0

    def test_compute_correlation_zero_variance(self) -> None:
        comparator = CrossRunComparator()
        compute = getattr(comparator, "_compute_correlation")
        pairs = [(1.0, 2.0), (1.0, 3.0)]

        assert compute(pairs) == 0.0

    def test_interpolate_linear(self) -> None:
        comparator = CrossRunComparator()
        interpolate = getattr(comparator, "_interpolate")

        assert interpolate([0.0, 10.0], 50.0) == 5.0

    def test_build_percent_grid_spacing(self) -> None:
        comparator = CrossRunComparator()
        grid = getattr(comparator, "_build_percent_grid")

        assert grid(3) == [0.0, 50.0, 100.0]
        assert grid(1) == [0.0]

    def test_label_divergence(self) -> None:
        comparator = CrossRunComparator()
        labeler = getattr(comparator, "_label_divergence")

        assert labeler("joy", 0.2) == "run A higher on joy"
        assert labeler("joy", -0.2) == "run A lower on joy"
        assert labeler("joy", 0.0) == "run A matches run B on joy"

    def test_infer_run_id(self) -> None:
        comparator = CrossRunComparator()
        infer = getattr(comparator, "_infer_run_id")

        assert infer([{"run_id": "run-123"}], "fallback") == "run-123"
        assert infer([], "fallback") == "fallback"
