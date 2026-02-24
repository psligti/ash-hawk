"""Composite grader for running multiple graders.

This module provides a CompositeGrader that combines multiple graders
with different scoring modes: weighted, all-or-nothing, and threshold.
"""

from __future__ import annotations

import asyncio
import logging
from enum import StrEnum
from typing import TYPE_CHECKING

from ash_hawk.graders.base import Grader
from ash_hawk.types import EvalTranscript, EvalTrial, GraderResult, GraderSpec

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CompositeMode(StrEnum):
    """Modes for combining multiple grader results."""

    WEIGHTED = "weighted"
    ALL_OR_NOTHING = "all_or_nothing"
    THRESHOLD = "threshold"


class CompositeGrader(Grader):
    """A grader that runs multiple graders and combines their results.

    Supports three modes:
    - weighted: Combine scores using weights (sum of weighted scores / sum of weights)
    - all_or_nothing: All graders must pass for overall pass
    - threshold: Combined score must meet or exceed a threshold

    Graders are run in parallel where safe (when mode allows independent execution).
    """

    def __init__(
        self,
        graders: list[Grader],
        mode: CompositeMode = CompositeMode.WEIGHTED,
        threshold: float = 0.7,
        weights: list[float] | None = None,
        run_parallel: bool = True,
    ):
        """Initialize the composite grader.

        Args:
            graders: List of grader instances to run.
            mode: How to combine results (weighted, all_or_nothing, threshold).
            threshold: Minimum score for threshold mode (default 0.7).
            weights: Optional weights for each grader (default: all 1.0).
            run_parallel: Whether to run graders in parallel (default: True).

        Raises:
            ValueError: If graders list is empty or weights don't match graders count.
        """
        if not graders:
            raise ValueError("CompositeGrader requires at least one grader")

        self._graders = graders
        self._mode = CompositeMode(mode) if isinstance(mode, str) else mode
        self._threshold = threshold
        self._run_parallel = run_parallel

        if weights is not None:
            if len(weights) != len(graders):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of graders ({len(graders)})"
                )
            self._weights = weights
        else:
            self._weights = [1.0] * len(graders)

    @property
    def name(self) -> str:
        """Return the grader name."""
        return "composite"

    @property
    def mode(self) -> CompositeMode:
        """Return the composite mode."""
        return self._mode

    @property
    def threshold(self) -> float:
        """Return the threshold for threshold mode."""
        return self._threshold

    @property
    def graders(self) -> list[Grader]:
        """Return the list of graders."""
        return self._graders

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        """Grade by running all graders and combining their results.

        Args:
            trial: The trial being evaluated.
            transcript: The execution transcript.
            spec: The grader specification (config may override mode/threshold).

        Returns:
            A GraderResult with combined score and pass/fail status.
        """
        config = spec.config
        mode = CompositeMode(config.get("mode", self._mode))
        threshold = config.get("threshold", self._threshold)
        run_parallel = config.get("run_parallel", self._run_parallel)

        # Get weights from config or use instance weights
        weights = config.get("weights", self._weights)
        if isinstance(weights, list) and len(weights) != len(self._graders):
            logger.warning("Config weights count mismatch, using instance weights")
            weights = self._weights

        # Run all graders
        if run_parallel:
            results = await self._run_graders_parallel(trial, transcript, spec)
        else:
            results = await self._run_graders_sequential(trial, transcript, spec)

        # Combine results based on mode
        score, passed = self._combine_results(results, mode, threshold, weights)

        return GraderResult(
            grader_type=self.name,
            score=score,
            passed=passed,
            details={
                "mode": str(mode),
                "threshold": threshold if mode == CompositeMode.THRESHOLD else None,
                "grader_results": [
                    {
                        "grader_type": r.grader_type,
                        "score": r.score,
                        "passed": r.passed,
                    }
                    for r in results
                ],
                "weights": weights if mode == CompositeMode.WEIGHTED else None,
            },
        )

    async def _run_graders_parallel(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> list[GraderResult]:
        """Run all graders in parallel using asyncio.gather.

        Args:
            trial: The trial being evaluated.
            transcript: The execution transcript.
            spec: The grader specification.

        Returns:
            List of GraderResults from all graders.
        """
        tasks = [grader.grade(trial, transcript, spec) for grader in self._graders]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Grader %s failed with error: %s",
                    self._graders[i].name,
                    result,
                )
                processed_results.append(
                    GraderResult(
                        grader_type=self._graders[i].name,
                        score=0.0,
                        passed=False,
                        error_message=str(result),
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    async def _run_graders_sequential(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> list[GraderResult]:
        """Run all graders sequentially.

        Args:
            trial: The trial being evaluated.
            transcript: The execution transcript.
            spec: The grader specification.

        Returns:
            List of GraderResults from all graders.
        """
        results = []
        for grader in self._graders:
            try:
                result = await grader.grade(trial, transcript, spec)
                results.append(result)
            except Exception as e:
                logger.error(
                    "Grader %s failed with error: %s",
                    grader.name,
                    e,
                )
                results.append(
                    GraderResult(
                        grader_type=grader.name,
                        score=0.0,
                        passed=False,
                        error_message=str(e),
                    )
                )
        return results

    def _combine_results(
        self,
        results: list[GraderResult],
        mode: CompositeMode,
        threshold: float,
        weights: list[float],
    ) -> tuple[float, bool]:
        """Combine multiple grader results based on the specified mode.

        Args:
            results: List of grader results to combine.
            mode: How to combine the results.
            threshold: Threshold for threshold mode.
            weights: Weights for weighted mode.

        Returns:
            Tuple of (combined_score, passed).
        """
        if mode == CompositeMode.WEIGHTED:
            return self._combine_weighted(results, weights)
        elif mode == CompositeMode.ALL_OR_NOTHING:
            return self._combine_all_or_nothing(results)
        elif mode == CompositeMode.THRESHOLD:
            return self._combine_threshold(results, threshold, weights)
        else:
            # Default to weighted
            return self._combine_weighted(results, weights)

    def _combine_weighted(
        self,
        results: list[GraderResult],
        weights: list[float],
    ) -> tuple[float, bool]:
        """Combine results using weighted average.

        Args:
            results: List of grader results.
            weights: Weights for each result.

        Returns:
            Tuple of (weighted_score, passed) where passed is True if
            weighted_score >= 0.5.
        """
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0, False

        weighted_sum = sum(r.score * w for r, w in zip(results, weights, strict=True))
        score = weighted_sum / total_weight
        passed = score >= 0.5

        return score, passed

    def _combine_all_or_nothing(
        self,
        results: list[GraderResult],
    ) -> tuple[float, bool]:
        """Combine results requiring all graders to pass.

        Args:
            results: List of grader results.

        Returns:
            Tuple of (average_score, passed) where passed is True only
            if all graders passed.
        """
        all_passed = all(r.passed for r in results)
        avg_score = sum(r.score for r in results) / len(results) if results else 0.0
        return avg_score, all_passed

    def _combine_threshold(
        self,
        results: list[GraderResult],
        threshold: float,
        weights: list[float],
    ) -> tuple[float, bool]:
        """Combine results checking if weighted score meets threshold.

        Args:
            results: List of grader results.
            threshold: Minimum required score.
            weights: Weights for each result.

        Returns:
            Tuple of (weighted_score, passed) where passed is True if
            weighted_score >= threshold.
        """
        score, _ = self._combine_weighted(results, weights)
        passed = score >= threshold
        return score, passed


__all__ = ["CompositeGrader", "CompositeMode"]
