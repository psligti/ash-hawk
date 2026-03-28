"""Guardrail system for self-improvement loops.

This module provides safety mechanisms to prevent runaway improvement cycles
and detect when progress has plateaued.

Guardrails include:
    - Consecutive held-out score drops (overfitting detection)
    - Maximum reverts (prevents infinite retry loops)
    - Plateau detection (stops when improvement stalls)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class GuardrailConfig:
    """Configuration for improvement loop guardrails.

    Attributes:
        max_consecutive_holdout_drops: Stop if held-out score drops this many
            times in a row (indicates overfitting to training data).
        max_reverts: Maximum number of skill reverts allowed before stopping
            (prevents infinite retry loops).
        plateau_window: Number of consecutive iterations to check for plateau.
        plateau_threshold: Minimum score improvement required to not be
            considered a plateau (e.g., 0.02 = 2% improvement required).
    """

    max_consecutive_holdout_drops: int = 3
    max_reverts: int = 5
    plateau_window: int = 5
    plateau_threshold: float = 0.02


@dataclass
class GuardrailState:
    """Mutable state for guardrail tracking.

    Attributes:
        consecutive_holdout_drops: Current count of consecutive score drops.
        total_reverts: Total reverts so far.
        recent_scores: List of recent held-out scores for plateau detection.
        stop_reason: Human-readable reason for why should_stop() returned True.
    """

    consecutive_holdout_drops: int = 0
    total_reverts: int = 0
    recent_scores: list[float] = field(default_factory=list[float])
    stop_reason: str | None = None


class GuardrailChecker:
    """Checks guardrail conditions for improvement loops.

    Tracks held-out score changes, reverts, and plateau detection to
    determine when an improvement loop should stop.

    Example:
        >>> config = GuardrailConfig(max_consecutive_holdout_drops=3)
        >>> checker = GuardrailChecker(config)
        >>> checker.record_iteration(0.75, applied=True)
        >>> checker.should_stop()
        False
    """

    def __init__(self, config: GuardrailConfig | None = None) -> None:
        """Initialize the guardrail checker.

        Args:
            config: Guardrail configuration. Uses defaults if not provided.
        """
        self._config = config or GuardrailConfig()
        self._state = GuardrailState()

    @property
    def config(self) -> GuardrailConfig:
        """Current guardrail configuration."""
        return self._config

    @property
    def state(self) -> GuardrailState:
        return GuardrailState(
            consecutive_holdout_drops=self._state.consecutive_holdout_drops,
            total_reverts=self._state.total_reverts,
            recent_scores=list(self._state.recent_scores),
            stop_reason=self._state.stop_reason,
        )

    def record_iteration(
        self,
        score: float,
        applied: bool,
        reverted: bool = False,
    ) -> None:
        """Record an iteration result for guardrail tracking.

        Args:
            score: Held-out score for this iteration (0.0 to 1.0).
            applied: Whether the skill change was applied (vs discarded).
            reverted: Whether this iteration was a revert of a previous change.
        """
        if reverted:
            self._state.total_reverts += 1

        if applied:
            self._update_score_tracking(score)

    def _update_score_tracking(self, score: float) -> None:
        """Update score tracking for drop and plateau detection.

        Args:
            score: Current held-out score.
        """
        if self._state.recent_scores:
            previous_score = self._state.recent_scores[-1]
            if score < previous_score:
                self._state.consecutive_holdout_drops += 1
            else:
                self._state.consecutive_holdout_drops = 0

        self._state.recent_scores.append(score)
        if len(self._state.recent_scores) > self._config.plateau_window:
            self._state.recent_scores = self._state.recent_scores[-self._config.plateau_window :]

    def should_stop(self) -> bool:
        """Check if any guardrail condition has been triggered.

        Returns:
            True if the improvement loop should stop, False otherwise.
        """
        if self._state.consecutive_holdout_drops >= self._config.max_consecutive_holdout_drops:
            self._state.stop_reason = (
                f"Consecutive holdout drops ({self._state.consecutive_holdout_drops}) "
                f"exceeded maximum ({self._config.max_consecutive_holdout_drops})"
            )
            return True

        if self._state.total_reverts >= self._config.max_reverts:
            self._state.stop_reason = (
                f"Total reverts ({self._state.total_reverts}) "
                f"exceeded maximum ({self._config.max_reverts})"
            )
            return True

        if self._is_plateaued():
            self._state.stop_reason = (
                f"Score plateaued over last {self._config.plateau_window} iterations "
                f"(improvement < {self._config.plateau_threshold:.2%})"
            )
            return True

        return False

    def _is_plateaued(self) -> bool:
        """Check if scores have plateaued.

        Returns:
            True if the score has not improved significantly over the
            plateau window, False otherwise.
        """
        if len(self._state.recent_scores) < self._config.plateau_window:
            return False

        window = self._state.recent_scores[-self._config.plateau_window :]
        improvement = max(window) - min(window)

        return improvement < self._config.plateau_threshold

    def reset(self) -> None:
        """Reset all guardrail state.

        Useful for starting a new improvement session with the same config.
        """
        self._state = GuardrailState()

    @property
    def stop_reason(self) -> str | None:
        """Human-readable reason for why the loop should stop.

        Returns None if should_stop() has not returned True.
        """
        return self._state.stop_reason


__all__ = [
    "GuardrailConfig",
    "GuardrailChecker",
    "GuardrailState",
]
