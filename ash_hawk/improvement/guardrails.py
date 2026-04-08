from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GuardrailConfig:
    """Configuration for improvement loop safety guardrails."""

    max_consecutive_holdout_drops: int = 3  # Stop if holdout score drops N times in a row
    max_reverts: int = 5  # Stop if N total reverts in a cycle
    plateau_window: int = 5  # Window size for plateau detection
    plateau_threshold: float = 0.02  # Score change below this = plateau


@dataclass
class IterationRecord:
    """Record of a single improvement iteration for guardrail tracking."""

    iteration: int
    score: float
    applied: bool  # True = kept, False = reverted
    holdout_score: float | None = None  # Score on holdout set (if available)


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    should_stop: bool
    reason: str | None = None
    consecutive_drops: int = 0
    total_reverts: int = 0
    plateau_detected: bool = False


class GuardrailChecker:
    """Track iteration history and determine if the improvement loop should stop.

    Checks:
    1. Max consecutive holdout score drops (overfitting protection)
    2. Max total reverts (spinning wheels protection)
    3. Plateau detection (no meaningful improvement for N iterations)
    """

    def __init__(self, config: GuardrailConfig | None = None) -> None:
        """Initialize with config (uses defaults if None)."""
        self._config = config or GuardrailConfig()
        self._history: list[IterationRecord] = []
        self._last_reason: str | None = None

    def record_iteration(self, record: IterationRecord) -> GuardrailResult:
        """Record an iteration and check all guardrails.

        Logs:
        - logger.info("Guardrail check: iter=%d score=%.4f applied=%s holdout=%.4f", ...)
        - logger.warning("Guardrail triggered: %s", reason) if should_stop

        Returns GuardrailResult with current state.
        """
        self._history.append(record)

        holdout_str = f"{record.holdout_score:.4f}" if record.holdout_score is not None else "N/A"
        logger.info(
            "Guardrail check: iter=%d score=%.4f applied=%s holdout=%s",
            record.iteration,
            record.score,
            record.applied,
            holdout_str,
        )

        return self._evaluate()

    def check(self) -> GuardrailResult:
        """Check all guardrails without recording a new iteration.

        Useful for pre-flight checks.
        """
        return self._evaluate()

    def _evaluate(self) -> GuardrailResult:
        """Run all guardrail checks and aggregate results."""
        total_reverts = sum(1 for r in self._history if not r.applied)
        holdout_stop, consecutive_drops = self._check_holdout_drops()
        revert_stop, _ = self._check_max_reverts()
        plateau = self._check_plateau()

        reasons: list[str] = []
        if holdout_stop:
            reasons.append(
                f"Holdout score dropped {consecutive_drops} consecutive times "
                f"(max {self._config.max_consecutive_holdout_drops})"
            )
        if revert_stop:
            reasons.append(
                f"Total reverts ({total_reverts}) exceeded max ({self._config.max_reverts})"
            )
        if plateau:
            reasons.append(
                f"Score plateaued (change < {self._config.plateau_threshold} over "
                f"{self._config.plateau_window} iterations)"
            )

        should_stop = holdout_stop or revert_stop or plateau
        reason = "; ".join(reasons) if reasons else None

        if should_stop and reason is not None:
            logger.warning("Guardrail triggered: %s", reason)
            self._last_reason = reason

        return GuardrailResult(
            should_stop=should_stop,
            reason=reason,
            consecutive_drops=consecutive_drops,
            total_reverts=total_reverts,
            plateau_detected=plateau,
        )

    def _check_holdout_drops(self) -> tuple[bool, int]:
        """Check for consecutive holdout score drops.

        Returns (should_stop, consecutive_drop_count).
        """
        consecutive = 0
        for i in range(len(self._history) - 1, 0, -1):
            curr = self._history[i]
            prev = self._history[i - 1]
            if curr.holdout_score is None or prev.holdout_score is None:
                break
            if curr.holdout_score < prev.holdout_score:
                consecutive += 1
            else:
                break

        return (consecutive >= self._config.max_consecutive_holdout_drops, consecutive)

    def _check_max_reverts(self) -> tuple[bool, int]:
        """Check if total reverts exceed limit.

        Returns (should_stop, total_revert_count).
        """
        total = sum(1 for r in self._history if not r.applied)
        return (total >= self._config.max_reverts, total)

    def _check_plateau(self) -> bool:
        """Check if scores have plateaued.

        A plateau is when the absolute score change over the last plateau_window
        iterations is less than plateau_threshold.
        """
        window = self._config.plateau_window
        if len(self._history) < window:
            return False

        recent = self._history[-window:]
        oldest = recent[0].score
        newest = recent[-1].score
        return abs(newest - oldest) < self._config.plateau_threshold

    @property
    def stop_reason(self) -> str | None:
        """Human-readable reason for the last stop trigger."""
        return self._last_reason

    @property
    def history(self) -> list[IterationRecord]:
        """Read-only access to iteration history."""
        return list(self._history)

    def reset(self) -> None:
        """Clear all history."""
        self._history.clear()
        self._last_reason = None
