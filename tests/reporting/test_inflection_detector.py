# type-hygiene: skip-file  # test file — mock/factory types are intentionally loose

import pydantic as pd
import pytest

from ash_hawk.reporting.inflection_detector import InflectionDetector, InflectionPoint


class TestInflectionPoint:
    def test_model_creation(self) -> None:
        point = InflectionPoint(
            step_index=1,
            dimension="joy",
            direction="positive_shift",
            magnitude=0.5,
            score_before=0.1,
            score_after=0.6,
            event_type="verification",
            event_summary="verified",
            label="joy increased from 0.10 to 0.60",
        )

        assert point.step_index == 1
        assert point.dimension == "joy"

    def test_extra_forbid(self) -> None:
        with pytest.raises(pd.ValidationError):
            InflectionPoint.model_validate(
                {
                    "step_index": 1,
                    "dimension": "joy",
                    "direction": "positive_shift",
                    "magnitude": 0.5,
                    "score_before": 0.1,
                    "score_after": 0.6,
                    "event_type": "verification",
                    "event_summary": "verified",
                    "label": "joy increased",
                    "extra_field": "nope",
                }
            )


class TestInflectionDetector:
    def test_detect_with_empty_list(self) -> None:
        detector = InflectionDetector()
        assert detector.detect([], []) == []

    def test_detect_with_single_step(self) -> None:
        detector = InflectionDetector()
        steps = [{"step_index": 0, "scores": {"joy": 0.2}}]
        assert detector.detect(steps, []) == []

    def test_detect_with_all_failed_data_quality(self) -> None:
        detector = InflectionDetector()
        steps = [
            {"step_index": 0, "scores": {"joy": 0.2}},
            {"step_index": 1, "scores": {"joy": 0.8}},
        ]
        assert detector.detect(steps, [], data_quality="all_failed") == []

    def test_detect_with_empty_data_quality(self) -> None:
        detector = InflectionDetector()
        steps = [
            {"step_index": 0, "scores": {"joy": 0.2}},
            {"step_index": 1, "scores": {"joy": 0.8}},
        ]
        assert detector.detect(steps, [], data_quality="empty") == []

    def test_detect_positive_shift(self) -> None:
        detector = InflectionDetector()
        steps = [
            {"step_index": 0, "scores": {"joy": 0.0}},
            {"step_index": 1, "scores": {"joy": 0.6}},
        ]
        events = [{"step_index": 1, "event_type": "verification success"}]
        points = detector.detect(steps, events)

        assert len(points) == 1
        assert points[0].direction == "positive_shift"

    def test_detect_negative_shift(self) -> None:
        detector = InflectionDetector()
        steps = [
            {"step_index": 0, "scores": {"joy": 0.8}},
            {"step_index": 1, "scores": {"joy": 0.1}},
        ]
        points = detector.detect(steps, [])

        assert len(points) == 1
        assert points[0].direction == "negative_shift"

    def test_detect_direction_reversal(self) -> None:
        detector = InflectionDetector()
        steps = [
            {"step_index": 0, "scores": {"joy": 0.0}},
            {"step_index": 1, "scores": {"joy": 0.6}},
            {"step_index": 2, "scores": {"joy": -0.3}},
        ]
        points = detector.detect(steps, [])

        assert any(point.direction == "reversal" for point in points)

    def test_detect_below_threshold(self) -> None:
        detector = InflectionDetector()
        steps = [
            {"step_index": 0, "scores": {"joy": 0.2}},
            {"step_index": 1, "scores": {"joy": 0.3}},
        ]
        points = detector.detect(steps, [])

        assert points == []

    def test_detect_handles_multiple_dimensions(self) -> None:
        detector = InflectionDetector()
        steps = [
            {"step_index": 0, "scores": {"joy": 0.0, "sadness": 0.5}},
            {"step_index": 1, "scores": {"joy": 0.7, "sadness": 0.0}},
        ]
        points = detector.detect(steps, [])

        dimensions = {point.dimension for point in points}
        assert dimensions == {"joy", "sadness"}

    def test_label_inflection_human_readable(self) -> None:
        detector = InflectionDetector()
        labeler = getattr(detector, "_label_inflection")
        label = labeler(
            "joy",
            0.1,
            0.6,
            "positive_shift",
            {"event_type": "verification success"},
        )
        generic_label = labeler(
            "joy",
            0.6,
            0.1,
            "negative_shift",
            {},
        )

        assert label == "verification boosted joy"
        assert "joy dropped" in generic_label

    def test_determine_direction_reversal(self) -> None:
        detector = InflectionDetector()
        determiner = getattr(detector, "_determine_direction")

        assert determiner(-0.5, 0.4) == "reversal"
        assert determiner(0.4, None) == "positive_shift"
        assert determiner(-0.4, None) == "negative_shift"

    def test_clamp_magnitude(self) -> None:
        detector = InflectionDetector()
        clamp = getattr(detector, "_clamp_magnitude")

        assert clamp(3.0) == 2.0
        assert clamp(-1.0) == 0.0
        assert clamp(1.5) == 1.5
