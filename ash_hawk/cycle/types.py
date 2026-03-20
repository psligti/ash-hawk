"""Types for improvement cycle orchestration."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

import pydantic as pd


class ConvergenceStatus(StrEnum):
    """Status of convergence checking."""

    CONVERGED = "converged"  # Score plateaued
    IMPROVING = "improving"  # Still making progress
    STAGNANT = "stagnant"  # No improvement, not converged
    REGRESSING = "regressing"  # Score going down


class CycleStatus(StrEnum):
    """Status of a cycle run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CONVERGED = "converged"
    FAILED = "failed"
    PAUSED = "paused"


class CycleConfig(pd.BaseModel):
    """Configuration for an improvement cycle."""

    cycle_id: str = pd.Field(description="Unique cycle identifier")
    experiment_id: str = pd.Field(description="Experiment ID for lesson isolation")
    target_agent: str = pd.Field(description="Agent to improve (e.g., bolt-merlin)")
    max_iterations: int = pd.Field(default=100, ge=1, le=1000, description="Maximum iterations")
    convergence_threshold: float = pd.Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Variance threshold for convergence detection",
    )
    convergence_window: int = pd.Field(
        default=5,
        ge=3,
        le=20,
        description="Number of iterations to check for convergence",
    )
    stop_on_convergence: bool = pd.Field(
        default=False,
        description="Stop early when convergence is reached instead of running all iterations",
    )
    promotion_success_threshold: int = pd.Field(
        default=3,
        ge=1,
        le=10,
        description="Consecutive improvements needed to promote lessons",
    )
    min_score_improvement: float = pd.Field(
        default=0.02,
        ge=0.0,
        le=0.5,
        description="Minimum score delta to count as improvement",
    )
    eval_pack: str | None = pd.Field(
        default=None,
        description="Specific eval pack to use (default: auto-detect by agent)",
    )
    baseline_run_id: str | None = pd.Field(
        default=None,
        description="Optional baseline run for comparison",
    )
    checkpoint_interval: int = pd.Field(
        default=10,
        ge=1,
        le=50,
        description="Save checkpoint every N iterations",
    )
    max_lessons_per_iteration: int = pd.Field(
        default=4,
        ge=1,
        le=50,
        description="Maximum approved lessons applied in each iteration",
    )
    scenario_paths: list[str] = pd.Field(
        default_factory=list,
        description="Scenario paths used to produce real evaluation scores per iteration",
    )
    scenario_parallelism: int | None = pd.Field(
        default=None,
        ge=1,
        le=256,
        description="Optional parallelism override for per-iteration scenario execution",
    )
    fail_on_scenario_error: bool = pd.Field(
        default=True,
        description="Fail iteration when scenario execution errors instead of silent fallback",
    )
    metadata: dict[str, Any] = pd.Field(default_factory=dict)

    model_config = pd.ConfigDict(extra="forbid")


class IterationResult(pd.BaseModel):
    """Result of a single iteration."""

    iteration_num: int = pd.Field(description="Iteration number (1-indexed)")
    run_artifact_id: str = pd.Field(description="ID of the run artifact")
    score: float = pd.Field(ge=0.0, le=1.0, description="Evaluation score")
    score_delta: float | None = pd.Field(
        default=None,
        description="Change from previous iteration",
    )
    lessons_generated: int = pd.Field(default=0, ge=0, description="Lessons created this iteration")
    lessons_applied: int = pd.Field(default=0, ge=0, description="Lessons applied this iteration")
    status: CycleStatus = pd.Field(description="Iteration status")
    error_message: str | None = pd.Field(default=None)
    started_at: datetime
    completed_at: datetime | None = pd.Field(default=None)
    metadata: dict[str, Any] = pd.Field(default_factory=dict)

    model_config = pd.ConfigDict(extra="forbid")

    def duration_seconds(self) -> float | None:
        """Calculate iteration duration."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()


class CycleResult(pd.BaseModel):
    """Result of a complete cycle run."""

    cycle_id: str = pd.Field(description="Cycle identifier")
    config: CycleConfig = pd.Field(description="Original configuration")
    total_iterations: int = pd.Field(ge=0, description="Iterations completed")
    iterations: list[Any] = pd.Field(
        default_factory=list,
        description="All iteration results",
    )
    best_score: float = pd.Field(default=0.0, ge=0.0, le=1.0, description="Best score achieved")
    best_iteration: int | None = pd.Field(default=None, description="Iteration with best score")
    final_score: float = pd.Field(default=0.0, ge=0.0, le=1.0, description="Final iteration score")
    initial_score: float = pd.Field(
        default=0.0, ge=0.0, le=1.0, description="First iteration score"
    )
    total_lessons_generated: int = pd.Field(default=0, ge=0)
    lessons_promoted: list[str] = pd.Field(
        default_factory=list,
        description="Lesson IDs promoted to global store",
    )
    convergence_status: ConvergenceStatus = pd.Field(
        default=ConvergenceStatus.IMPROVING,
        description="Final convergence status",
    )
    status: CycleStatus = pd.Field(default=CycleStatus.PENDING)
    started_at: datetime | None = pd.Field(default=None)
    completed_at: datetime | None = pd.Field(default=None)
    error_message: str | None = pd.Field(default=None)
    metadata: dict[str, Any] = pd.Field(default_factory=dict)

    model_config = pd.ConfigDict(extra="forbid")

    @pd.computed_field(return_type=float)
    def improvement_delta(self) -> float:
        """Total improvement from start to end."""
        return self.final_score - self.initial_score

    @pd.computed_field(return_type=float)
    def avg_iteration_time(self) -> float:
        if not self.iterations:
            return 0.0
        times: list[float] = []
        for iteration_obj in self.iterations:
            if not isinstance(iteration_obj, IterationResult):
                continue
            iteration = iteration_obj
            duration = iteration.duration_seconds()
            if duration is not None:
                times.append(duration)
        return sum(times) / len(times) if times else 0.0


class CycleCheckpoint(pd.BaseModel):
    """Checkpoint for resumable cycle execution."""

    cycle_id: str
    config: CycleConfig
    iterations: list[IterationResult]
    current_iteration: int
    status: CycleStatus
    saved_at: datetime
    pending_lessons: list[str] = pd.Field(
        default_factory=list,
        description="Lessons pending promotion check",
    )
    consecutive_improvements: int = pd.Field(
        default=0,
        ge=0,
        description="Count of consecutive score improvements",
    )

    model_config = pd.ConfigDict(extra="forbid")
