"""Minimal types for auto-research improvement cycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class CycleStatus(StrEnum):
    """Status of an auto-research cycle."""

    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class TargetType(StrEnum):
    """Type of improvement target."""

    AGENT = "agent"
    SKILL = "skill"
    POLICY = "policy"
    TOOL = "tool"


@dataclass
class IterationResult:
    """Result of a single improvement iteration."""

    iteration_num: int
    score_before: float
    score_after: float
    improvement_text: str = ""
    applied: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def delta(self) -> float:
        """Score change from this iteration."""
        return self.score_after - self.score_before


@dataclass
class CycleResult:
    """Result of a complete auto-research cycle."""

    agent_name: str
    target_path: str
    scenario_paths: list[str]
    target_type: TargetType | None = None
    status: CycleStatus = CycleStatus.RUNNING
    iterations: list[IterationResult] = field(default_factory=list)
    initial_score: float = 0.0
    final_score: float = 0.0
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    error_message: str | None = None

    @property
    def improvement_delta(self) -> float:
        """Total improvement from initial to final."""
        return self.final_score - self.initial_score

    @property
    def total_iterations(self) -> int:
        """Number of iterations completed."""
        return len(self.iterations)


__all__ = ["CycleResult", "CycleStatus", "IterationResult", "TargetType"]
