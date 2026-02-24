"""Base class for all graders.

This module defines the abstract base class that all graders must implement.
Graders evaluate trial results and return structured GraderResult objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ash_hawk.types import EvalTranscript, EvalTrial, GraderResult, GraderSpec


class Grader(ABC):
    """Abstract base class for all graders.

    Graders evaluate the output of a trial against specified criteria
    and return a GraderResult with a score and pass/fail status.

    Graders are stateless - they should not maintain state between calls
    and should not be coupled to specific storage backends.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name/identifier for this grader.

        This name is used to register and look up graders in the registry.

        Returns:
            The grader's unique name (e.g., 'string_match', 'test_runner').
        """
        ...

    @abstractmethod
    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        """Grade a trial based on its transcript and grader specification.

        Args:
            trial: The trial being evaluated, containing task info and results.
            transcript: The complete execution transcript with messages, tool calls, etc.
            spec: The grader specification with configuration and parameters.

        Returns:
            A GraderResult with score (0.0-1.0), pass/fail status, and optional details.
        """
        ...

    def __repr__(self) -> str:
        """Return string representation of the grader."""
        return f"{self.__class__.__name__}(name={self.name!r})"


class PassThroughGrader(Grader):
    """A simple grader that always passes with score 1.0.

    Useful for testing and as a placeholder grader.
    """

    @property
    def name(self) -> str:
        """Return the grader name."""
        return "pass_through"

    async def grade(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
        spec: GraderSpec,
    ) -> GraderResult:
        """Always return a passing result with score 1.0.

        Args:
            trial: The trial being evaluated (unused).
            transcript: The execution transcript (unused).
            spec: The grader specification (unused).

        Returns:
            A GraderResult with score=1.0 and passed=True.
        """
        return GraderResult(
            grader_type=self.name,
            score=1.0,
            passed=True,
            details={"message": "Pass-through grader always passes"},
        )


__all__ = ["Grader", "PassThroughGrader"]
