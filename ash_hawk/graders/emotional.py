# type-hygiene: skip-file  # dynamic trace data payloads are intentionally Any
from __future__ import annotations

import asyncio
import importlib
from collections.abc import Awaitable, Sequence
from typing import Any, Protocol

from ash_hawk.graders.base import Grader
from ash_hawk.graders.emotion_config import EmotionGraderConfig
from ash_hawk.types import EvalTranscript, EvalTrial, GraderResult, GraderSpec


class _StepEmotionScore(Protocol):
    step_index: int
    event_type: str
    scores: dict[str, float]
    confidence: dict[str, float]
    reasoning: str

    def model_dump(self) -> dict[str, Any]: ...


class _EmotionScorer(Protocol):
    async def score_step(
        self, step: dict[str, Any], context: list[dict[str, Any]], llm_client: Any
    ) -> _StepEmotionScore: ...


class EmotionalGrader(Grader):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            self._config = EmotionGraderConfig()
        else:
            self._config = EmotionGraderConfig(**config)
        scorer_module = importlib.import_module("ash_hawk.graders.emotion_scorer")
        scorer_class = getattr(scorer_module, "EmotionScorer")
        self._scorer: _EmotionScorer = scorer_class(self._config)
        self._client = None
        self._resolved_provider: str | None = None
        self._resolved_model: str | None = None

    @property
    def name(self) -> str:
        return "emotional"

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        details: dict[str, Any]
        trace_events = transcript.trace_events
        if not trace_events:
            details = {
                "step_emotions": [],
                "emotion_trajectory": {},
                "inflection_points": [],
                "dimension_summaries": {},
                "data_quality": "empty",
            }
            return GraderResult(
                grader_type=self.name,
                score=0.5,
                passed=True,
                details=details,
            )

        try:
            llm_client = self._get_llm_client()
        except ImportError as exc:
            details = {
                "step_emotions": [],
                "emotion_trajectory": {},
                "inflection_points": [],
                "dimension_summaries": {},
                "data_quality": "all_failed",
            }
            return GraderResult(
                grader_type=self.name,
                score=0.5,
                passed=True,
                details=details,
                error_message=str(exc),
            )

        step_inputs: list[dict[str, Any]] = [dict(item) for item in trace_events]
        if not step_inputs:
            details = {
                "step_emotions": [],
                "emotion_trajectory": {},
                "inflection_points": [],
                "dimension_summaries": {},
                "data_quality": "empty",
            }
            return GraderResult(
                grader_type=self.name,
                score=0.5,
                passed=True,
                details=details,
            )

        tasks: list[Awaitable[_StepEmotionScore]] = []
        for idx, step in enumerate(step_inputs):
            step_payload = dict(step)
            step_payload["_index"] = idx
            context = step_inputs[:idx]
            tasks.append(self._scorer.score_step(step_payload, context, llm_client))

        step_scores = list(await self._gather_scores(tasks))

        failure_count = 0
        for score in step_scores:
            if self._is_failure_score(score):
                failure_count += 1

        if failure_count == len(step_scores):
            data_quality = "all_failed"
        elif failure_count > 0:
            data_quality = "partial_failure"
        else:
            data_quality = "all_scored"

        emotion_trajectory = self._build_trajectory(step_scores)
        dimension_summaries = self._build_dimension_summaries(step_scores)
        details = {
            "step_emotions": [score.model_dump() for score in step_scores],
            "emotion_trajectory": emotion_trajectory,
            "inflection_points": [],
            "dimension_summaries": dimension_summaries,
            "data_quality": data_quality,
        }

        aggregate_score = self._compute_aggregate_score(step_scores)
        return GraderResult(
            grader_type=self.name,
            score=aggregate_score,
            passed=True,
            details=details,
        )

    async def _gather_scores(
        self, tasks: Sequence[Awaitable[_StepEmotionScore]]
    ) -> list[_StepEmotionScore]:
        return [score for score in await asyncio.gather(*tasks)]

    def _build_trajectory(self, step_scores: list[_StepEmotionScore]) -> dict[str, list[float]]:
        trajectory: dict[str, list[float]] = {name: [] for name in self._dimension_names()}
        for score in step_scores:
            for name, value in score.scores.items():
                trajectory.setdefault(name, []).append(value)
        return trajectory

    def _build_dimension_summaries(
        self, step_scores: list[_StepEmotionScore]
    ) -> dict[str, dict[str, float]]:
        summaries: dict[str, dict[str, float]] = {}
        for name in self._dimension_names():
            values = [score.scores.get(name, 0.0) for score in step_scores]
            confidences = [score.confidence.get(name, 0.0) for score in step_scores]
            if not values:
                summaries[name] = {
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "confidence_mean": 0.0,
                }
                continue
            summaries[name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "confidence_mean": sum(confidences) / len(confidences),
            }
        return summaries

    def _dimension_names(self) -> list[str]:
        return [dimension.name for dimension in self._config.dimensions]

    def _compute_aggregate_score(self, step_scores: list[_StepEmotionScore]) -> float:
        values = [value for score in step_scores for value in score.scores.values()]
        if not values:
            return 0.5
        mean_value = sum(values) / len(values)
        normalized = (mean_value + 1.0) / 2.0
        return max(0.0, min(1.0, normalized))

    def _is_failure_score(self, score: _StepEmotionScore) -> bool:
        if score.reasoning:
            return False
        if any(value != 0.0 for value in score.scores.values()):
            return False
        if any(value != 0.0 for value in score.confidence.values()):
            return False
        return True

    def _get_llm_client(self) -> Any:
        if self._client is None:
            try:
                settings_module = importlib.import_module("dawn_kestrel.base.config")
                client_module = importlib.import_module("dawn_kestrel.provider.llm_client")
            except ImportError as exc:
                raise ImportError("dawn-kestrel is required for emotional grading") from exc

            dk_config = settings_module.load_agent_config()
            model_config = self._config.model_config_ref
            provider = model_config.provider or dk_config.get("runtime.provider") or "anthropic"
            model = (
                model_config.model or dk_config.get("runtime.model") or "claude-sonnet-4-20250514"
            )
            api_key = settings_module.get_config_api_key(provider) or None
            self._resolved_provider = provider
            self._resolved_model = model
            llm_client = getattr(client_module, "LLMClient")
            self._client = llm_client(provider_id=provider, model=model, api_key=api_key)
        return self._client


__all__ = ["EmotionalGrader"]
