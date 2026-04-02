# type-hygiene: skip-file  # dynamic trace data payloads are intentionally Any
from __future__ import annotations

from typing import Any, Literal

import pydantic as pd


class InflectionPoint(pd.BaseModel):
    step_index: int = pd.Field(description="Step index where the inflection was detected.")
    dimension: str = pd.Field(description="Emotion dimension for the inflection.")
    direction: Literal["positive_shift", "negative_shift", "reversal"] = pd.Field(
        description="Direction of the inflection shift.",
    )
    magnitude: float = pd.Field(description="Magnitude of the score change (clamped).")
    score_before: float = pd.Field(description="Score before the inflection.")
    score_after: float = pd.Field(description="Score after the inflection.")
    event_type: str = pd.Field(description="Associated event type.")
    event_summary: str = pd.Field(description="Associated event summary.")
    label: str = pd.Field(description="Human-readable label for the inflection.")

    model_config = pd.ConfigDict(extra="forbid")


class InflectionDetector:
    def __init__(self, threshold: float = 0.3) -> None:
        self._threshold = threshold

    def detect(
        self,
        step_scores: list[dict[str, Any]],
        events: list[dict[str, Any]],
        *,
        data_quality: str = "all_scored",
    ) -> list[InflectionPoint]:
        if data_quality in {"all_failed", "empty"}:
            return []
        if len(step_scores) < 2:
            return []

        event_by_step = self._index_events(events)
        dimensions = self._extract_dimensions(step_scores)
        inflections: list[InflectionPoint] = []

        for dimension in dimensions:
            prev_delta: float | None = None
            prev_score: float | None = None

            for idx in range(1, len(step_scores)):
                score_before = self._extract_score(step_scores[idx - 1], dimension)
                score_after = self._extract_score(step_scores[idx], dimension)
                if score_before is None or score_after is None:
                    prev_delta = None
                    prev_score = score_after
                    continue

                delta = score_after - score_before
                if abs(delta) < self._threshold:
                    prev_delta = delta
                    prev_score = score_after
                    continue

                direction = self._determine_direction(delta, prev_delta)
                magnitude = self._clamp_magnitude(abs(delta))

                event = event_by_step.get(idx, {})
                event_type = self._event_type(event)
                event_summary = self._event_summary(event)
                label = self._label_inflection(
                    dimension,
                    score_before,
                    score_after,
                    direction,
                    event,
                )

                inflections.append(
                    InflectionPoint(
                        step_index=idx,
                        dimension=dimension,
                        direction=direction,
                        magnitude=magnitude,
                        score_before=score_before,
                        score_after=score_after,
                        event_type=event_type,
                        event_summary=event_summary,
                        label=label,
                    )
                )

                prev_delta = delta
                prev_score = score_after

            if prev_score is None:
                continue

        return inflections

    def _label_inflection(
        self,
        dimension: str,
        score_before: float,
        score_after: float,
        direction: Literal["positive_shift", "negative_shift", "reversal"],
        event: dict[str, Any],
    ) -> str:
        event_type = self._event_type(event)
        normalized = event_type.lower()

        if "tool_failure" in normalized or "tool failure" in normalized:
            return f"tool failure triggered {dimension}"
        if "verification" in normalized and "success" in normalized:
            return f"verification boosted {dimension}"
        if "retry" in normalized:
            return f"retry loop increased {dimension}"

        if direction == "reversal":
            return (
                f"{dimension} reversed after shifting from {score_before:.2f} to {score_after:.2f}"
            )
        if direction == "positive_shift":
            return f"{dimension} increased from {score_before:.2f} to {score_after:.2f}"
        return f"{dimension} dropped from {score_before:.2f} to {score_after:.2f}"

    @staticmethod
    def _determine_direction(
        delta: float,
        prev_delta: float | None,
    ) -> Literal["positive_shift", "negative_shift", "reversal"]:
        if prev_delta is not None and delta * prev_delta < 0:
            return "reversal"
        if delta >= 0:
            return "positive_shift"
        return "negative_shift"

    @staticmethod
    def _clamp_magnitude(value: float) -> float:
        return max(0.0, min(2.0, value))

    @staticmethod
    def _extract_dimensions(step_scores: list[dict[str, Any]]) -> list[str]:
        if not step_scores:
            return []
        first = step_scores[0]
        if isinstance(first.get("scores"), dict):
            return list(first["scores"].keys())
        return [key for key in first.keys() if key not in {"step_index", "events", "confidence"}]

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
    def _index_events(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        event_by_step: dict[int, dict[str, Any]] = {}
        for event in events:
            step_index = event.get("step_index")
            if isinstance(step_index, int):
                event_by_step[step_index] = event
        return event_by_step

    @staticmethod
    def _event_type(event: dict[str, Any]) -> str:
        value = event.get("event_type") or event.get("type") or "unknown"
        return str(value)

    @staticmethod
    def _event_summary(event: dict[str, Any]) -> str:
        value = event.get("event_summary") or event.get("summary") or ""
        return str(value)
