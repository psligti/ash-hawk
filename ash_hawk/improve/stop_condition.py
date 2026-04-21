from __future__ import annotations

import logging

import pydantic as pd

logger = logging.getLogger(__name__)

TESTED_MUTATION_OUTCOMES = {
    "applied",
    "applied_with_tolerated_regressions",
    "targeted_regression",
    "intolerable_regression",
    "net_negative",
    "below_keep_threshold",
}


class ScoreRecord(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid")

    iteration: int = pd.Field(description="Iteration number")
    score: float = pd.Field(ge=0.0, le=1.0, description="Pass rate score")
    applied: bool = pd.Field(description="Whether the change was kept")
    delta: float = pd.Field(default=0.0, description="Score change from previous")
    mutation_outcome: str = pd.Field(default="applied", description="Mutation execution outcome")


class StopResult(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid")

    should_stop: bool = pd.Field(description="Whether the loop should stop")
    reasons: list[str] = pd.Field(default_factory=list, description="Why it should stop")
    total_reverts: int = pd.Field(default=0, description="Total reverted hypotheses")
    iterations_since_best: int = pd.Field(default=0, description="Iterations since best score")
    plateau_detected: bool = pd.Field(default=False, description="Whether plateau was detected")


class StopConditionConfig(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid")

    max_reverts: int = pd.Field(default=5, description="Max total reverts before stopping")
    convergence_window: int = pd.Field(default=5, description="Window size for plateau detection")
    variance_threshold: float = pd.Field(
        default=0.001, description="Score variance below this = plateau"
    )
    max_iterations_without_improvement: int = pd.Field(
        default=10, description="Stop if no improvement for N iterations"
    )
    max_consecutive_regressions: int = pd.Field(
        default=3, description="Stop if N consecutive negative deltas"
    )
    max_consecutive_noop_mutations: int = pd.Field(
        default=4,
        description="Stop if N consecutive mutations produce no useful candidate",
    )
    min_mutation_yield: float = pd.Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Stop when kept/tested yield drops below this threshold",
    )
    min_yield_sample_size: int = pd.Field(
        default=5,
        ge=1,
        description="Minimum tested mutations before yield threshold is enforced",
    )


class StopCondition:
    """Single source of truth for when the improvement loop should stop.

    Replaces the former GuardrailChecker + ConvergenceDetector with one class.
    """

    def __init__(self, config: StopConditionConfig | None = None) -> None:
        self._config = config or StopConditionConfig()
        self._history: list[ScoreRecord] = []
        self._best_score: float = 0.0
        self._iterations_since_best: int = 0

    def record(self, record: ScoreRecord) -> StopResult:
        if not self._history or record.score > self._best_score:
            self._best_score = record.score
            self._iterations_since_best = 0
        else:
            self._iterations_since_best += 1

        self._history.append(record)

        logger.info(
            "Stop check: iter=%d score=%.4f delta=%.4f applied=%s best=%.4f stale=%d",
            record.iteration,
            record.score,
            record.delta,
            record.applied,
            self._best_score,
            self._iterations_since_best,
        )

        result = self.check()

        if result.should_stop:
            logger.warning("Stop condition triggered: %s", "; ".join(result.reasons))

        return result

    def check(self) -> StopResult:
        if len(self._history) < 1:
            return StopResult(should_stop=False)

        total_reverts = sum(1 for r in self._history if not r.applied)
        tested_reverts = sum(
            1
            for r in self._history
            if not r.applied and r.mutation_outcome in TESTED_MUTATION_OUTCOMES
        )
        reasons: list[str] = []
        plateau = False

        max_reverts_hit = self._check_max_reverts(tested_reverts)
        regression_hit = self._check_regression()
        plateau = self._check_plateau()
        no_improvement_hit = self._check_no_improvement()
        noop_stall_hit = self._check_noop_stall()
        yield_hit = self._check_yield_threshold()

        if max_reverts_hit:
            reasons.append(
                f"Tested reverts ({tested_reverts}) exceeded max ({self._config.max_reverts})"
            )
        if regression_hit:
            reasons.append(
                f"Score regressed for {self._config.max_consecutive_regressions} "
                f"consecutive iterations"
            )
        if plateau:
            reasons.append(
                f"Score plateaued (variance < {self._config.variance_threshold} over "
                f"{self._config.convergence_window} iterations)"
            )
        if no_improvement_hit:
            reasons.append(
                f"No improvement for {self._iterations_since_best} iterations "
                f"(max {self._config.max_iterations_without_improvement})"
            )
        if noop_stall_hit:
            reasons.append(
                f"Mutation generation stalled for {self._config.max_consecutive_noop_mutations} consecutive attempts"
            )
        if yield_hit:
            reasons.append(
                f"Mutation yield dropped to {self._yield_ratio():.0%} "
                f"(min {self._config.min_mutation_yield:.0%} after {self._config.min_yield_sample_size} samples)"
            )

        return StopResult(
            should_stop=len(reasons) > 0,
            reasons=reasons,
            total_reverts=total_reverts,
            iterations_since_best=self._iterations_since_best,
            plateau_detected=plateau,
        )

    def _check_max_reverts(self, tested_reverts: int) -> bool:
        return tested_reverts >= self._config.max_reverts

    def _check_regression(self) -> bool:
        window = self._config.max_consecutive_regressions
        if len(self._history) < window:
            return False
        recent = self._history[-window:]
        return all(r.delta < 0 for r in recent)

    def _check_plateau(self) -> bool:
        window = self._config.convergence_window
        if len(self._history) < window:
            return False

        recent = self._history[-window:]
        scores = [r.score for r in recent]
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)

        return variance < self._config.variance_threshold

    def _check_no_improvement(self) -> bool:
        return self._iterations_since_best >= self._config.max_iterations_without_improvement

    def _check_noop_stall(self) -> bool:
        window = self._config.max_consecutive_noop_mutations
        if len(self._history) < window:
            return False
        stalled_outcomes = {
            "no_file_changes",
            "mutation_cli_timeout",
            "mutation_cli_error",
            "mutation_zero_tools",
            "mutation_parse_failed",
            "post_mutation_eval_failed",
        }
        recent = self._history[-window:]
        return all(record.mutation_outcome in stalled_outcomes for record in recent)

    def _yield_ratio(self) -> float:
        tested = len(self._history)
        if tested == 0:
            return 0.0
        kept = sum(1 for record in self._history if record.applied)
        return kept / tested

    def _check_yield_threshold(self) -> bool:
        if self._config.min_mutation_yield <= 0.0:
            return False
        if len(self._history) < self._config.min_yield_sample_size:
            return False
        return self._yield_ratio() < self._config.min_mutation_yield

    @property
    def history(self) -> list[ScoreRecord]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()
        self._best_score = 0.0
        self._iterations_since_best = 0
