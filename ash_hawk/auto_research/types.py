"""Minimal types for auto-research improvement cycle."""

# type-hygiene: skip-file  # pre-existing Any — lever values and result payloads are heterogeneous

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class CycleStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    CONVERGED = "converged"
    ERROR = "error"


class TargetType(StrEnum):
    """Type of improvement target."""

    AGENT = "agent"
    SKILL = "skill"
    POLICY = "policy"
    TOOL = "tool"


class ConvergenceReason(StrEnum):
    """Reason for convergence detection."""

    PLATEAU = "plateau"
    NO_IMPROVEMENT = "no_improvement"
    REGRESSION = "regression"


class PromotionStatus(StrEnum):
    PENDING = "pending"
    PROMOTED = "promoted"
    FAILED = "failed"


@dataclass
class ConvergenceResult:
    """Result of convergence detection check."""

    converged: bool
    reason: ConvergenceReason | None = None
    score_variance: float = 0.0
    confidence: float = 0.0
    recent_scores: list[float] = field(default_factory=list)
    iterations_since_improvement: int = 0


@dataclass
class IterationResult:
    """Result of a single improvement iteration."""

    iteration_num: int
    score_before: float
    score_after: float
    improvement_text: str = ""
    applied: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    category_scores: dict[str, float] | None = None

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
        return len(self.iterations)

    @property
    def applied_iterations(self) -> list[IterationResult]:
        return [i for i in self.iterations if i.applied]


@dataclass
class ToolUsagePattern:
    """Pattern of tool usage across transcripts."""

    tool_name: str
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_duration_seconds: float = 0.0
    common_sequences: list[tuple[str, ...]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


@dataclass
class DecisionPattern:
    """Pattern of decision-making across transcripts."""

    pattern_type: str
    frequency: int
    success_rate: float = 0.0
    example_sequences: list[list[str]] = field(default_factory=list)
    description: str = ""


@dataclass
class FailurePattern:
    """Pattern of failures across transcripts."""

    failure_type: str
    frequency: int
    affected_tools: list[str] = field(default_factory=list)
    recovery_attempts: int = 0
    recovery_success_rate: float = 0.0
    example_contexts: list[str] = field(default_factory=list)


@dataclass
class IntentPatterns:
    """Aggregated intent patterns from transcript analysis."""

    transcript_count: int
    dominant_tools: list[str] = field(default_factory=list)
    tool_usage_patterns: list[ToolUsagePattern] = field(default_factory=list)
    decision_patterns: list[DecisionPattern] = field(default_factory=list)
    failure_patterns: list[FailurePattern] = field(default_factory=list)
    inferred_intent: str | None = None
    confidence: float = 0.0


@dataclass
class PromotedLesson:
    lesson_id: str
    improvement_text: str
    score_delta: float
    target_type: TargetType
    target_name: str
    source_experiment: str
    promotion_status: PromotionStatus = PromotionStatus.PENDING
    note_id: str | None = None
    error_message: str | None = None
    promoted_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class LeverDimension:
    name: str
    values: list[Any]
    weight: float = 1.0
    mutation_rate: float = 0.2
    mutation_strategy: str = "random"


@dataclass
class LeverConfiguration:
    agent: str
    skills: tuple[str, ...]
    tools: tuple[str, ...]
    context_strategy: str
    prompt_preset: str
    timeout_multiplier: float = 1.0
    model_routing: str = ""

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "skills": list(self.skills),
            "tools": list(self.tools),
            "context_strategy": self.context_strategy,
            "prompt_preset": self.prompt_preset,
            "timeout_multiplier": self.timeout_multiplier,
            "model_routing": self.model_routing,
        }


DEFAULT_LEVER_SPACE: dict[str, LeverDimension] = {
    "agent": LeverDimension(
        name="agent",
        values=["orchestrator", "master_orchestrator", "explore", "consult"],
        weight=0.3,
        mutation_rate=0.1,
    ),
    "skills": LeverDimension(
        name="skills",
        values=[[], ["council"], ["frontend-ui-ux"], ["playwright"], ["git-master"]],
        weight=0.2,
        mutation_rate=0.3,
    ),
    "tools": LeverDimension(
        name="tools",
        values=[
            ["read", "edit", "write"],
            ["read", "edit", "write", "bash"],
            ["read", "edit", "write", "bash", "grep", "glob"],
        ],
        weight=0.2,
        mutation_rate=0.2,
    ),
    "context_strategy": LeverDimension(
        name="context_strategy",
        values=["file-based", "dynamic", "composite"],
        weight=0.15,
        mutation_rate=0.15,
    ),
    "prompt_preset": LeverDimension(
        name="prompt_preset",
        values=["balanced", "delegation_heavy", "precision", "throughput"],
        weight=0.15,
        mutation_rate=0.25,
    ),
    "timeout_multiplier": LeverDimension(
        name="timeout_multiplier",
        values=[0.75, 1.0, 1.25, 1.5, 2.0],
        weight=0.0,
        mutation_rate=0.1,
    ),
    "model_routing": LeverDimension(
        name="model_routing",
        values=["default", "fast", "reasoning", "creative"],
        weight=0.1,
        mutation_rate=0.2,
        mutation_strategy="categorical",
    ),
}


@dataclass
class EnhancedCycleConfig:
    enable_multi_target: bool = True
    max_parallel_targets: int = 4
    enable_lever_search: bool = False
    lever_space: dict[str, LeverDimension] | None = None
    enable_intent_analysis: bool = True
    enable_knowledge_promotion: bool = True
    enable_skill_cleanup: bool = True
    note_lark_enabled: bool = True
    iterations_per_target: int = 50
    improvement_threshold: float = 0.02
    min_improvement_for_promotion: float = 0.05
    min_consecutive_successes: int = 3
    convergence_window: int = 5
    convergence_variance_threshold: float = 0.001
    project_name: str = "ash-hawk"


@dataclass
class EvolvableConfig:
    """Configuration for evolvable block-coordinate optimization phase."""

    enabled: bool = False
    max_experiments: int = 100
    dimensions: list[str] = field(default_factory=list)
    experiment_log_path: str = ".ash-hawk/evolvable-experiments.jsonl"
    improvement_threshold: float = 0.02
    safety_threshold: float = -0.05
    model_routing_enabled: bool = True


@dataclass
class EvolvableCycleResult:
    """Result of an evolvable optimization phase."""

    total_experiments: int = 0
    best_score: float = 0.0
    baseline_score: float = 0.0
    improvement: float = 0.0
    dimensions_explored: list[str] = field(default_factory=list)
    reverted_experiments: int = 0
    best_configuration: dict[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class EnhancedCycleResult:
    agent_name: str
    config: EnhancedCycleConfig
    status: CycleStatus = CycleStatus.RUNNING
    target_results: dict[str, CycleResult] = field(default_factory=dict)
    intent_patterns: IntentPatterns | None = None
    promoted_lessons: list[PromotedLesson] = field(default_factory=list)
    lever_result: Any = None
    cleanup_result: Any = None
    overall_improvement: float = 0.0
    converged: bool = False
    convergence_reason: str | None = None
    total_iterations: int = 0
    total_duration_seconds: float = 0.0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None

    @property
    def total_promoted(self) -> int:
        return len(self.promoted_lessons)


@dataclass
class MultiTargetResult:
    agent_name: str
    target_results: dict[str, CycleResult] = field(default_factory=dict)
    overall_improvement: float = 0.0
    best_target: str = ""
    converged: bool = False
    convergence_reason: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class CleanupResult:
    cleaned_skills: list[str] = field(default_factory=list)
    kept_skills: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    @property
    def total_processed(self) -> int:
        return len(self.cleaned_skills) + len(self.kept_skills)

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0


__all__ = [
    "CleanupResult",
    "ConvergenceReason",
    "ConvergenceResult",
    "CycleResult",
    "CycleStatus",
    "DEFAULT_LEVER_SPACE",
    "DecisionPattern",
    "EnhancedCycleConfig",
    "EnhancedCycleResult",
    "EvolvableConfig",
    "EvolvableCycleResult",
    "FailurePattern",
    "IntentPatterns",
    "IterationResult",
    "LeverConfiguration",
    "LeverDimension",
    "MultiTargetResult",
    "PromotedLesson",
    "PromotionStatus",
    "TargetType",
    "ToolUsagePattern",
]
