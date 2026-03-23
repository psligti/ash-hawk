"""Data types for auto-research improvement cycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal


class CycleStatus(StrEnum):
    """Status of an auto-research cycle."""

    RUNNING = "running"
    COMPLETED = "completed"
    CONVERGED = "converged"
    STOPPED = "stopped"
    ERROR = "error"


class ImprovementType(StrEnum):
    """Type of improvement target."""

    SKILL = "skill"
    POLICY = "policy"
    TOOL = "tool"
    AGENT = "agent"


@dataclass(frozen=True)
class RepoConfig:
    """Configuration discovered from repository.

    Attributes:
        agent_name: Name of the agent being improved.
        agent_runner: Runner type (e.g., dawn_kestrel).
        scenarios: List of discovered scenario paths.
        improvement_targets: List of target files to improve.
        pyproject_path: Path to pyproject.toml if found.
    """

    agent_name: str | None = None
    agent_runner: str = "dawn_kestrel"
    scenarios: list[Path] = field(default_factory=list)
    improvement_targets: list[Path] = field(default_factory=list)
    pyproject_path: Path | None = None


@dataclass
class AnalysisResult:
    """Result of agentic transcript analysis.

    Attributes:
        scenario_name: Name of the analyzed scenario.
        root_cause: Identified root cause of failure.
        missing_guidance: What was absent from skill/policy.
        proposed_fix: Concrete fix suggestion.
        confidence: Confidence level (0.0-1.0).
        transcript_excerpt: Relevant excerpt from transcript.
    """

    scenario_name: str
    root_cause: str
    missing_guidance: str
    proposed_fix: str
    confidence: float = 0.5
    transcript_excerpt: str = ""


@dataclass
class ImprovementResult:
    """Result of agentic improvement generation.

    Attributes:
        target_type: Type of improvement (skill, policy, tool).
        target_path: Path to the target file.
        original_content: Content before improvement.
        updated_content: Content after improvement (markdown with frontmatter).
        change_name: Name of the change (from frontmatter 'name' field).
        rationale: Why this change was made.
        analysis_ref: Reference to the analysis that prompted this.
    """

    target_type: ImprovementType
    target_path: Path
    original_content: str
    updated_content: str
    change_name: str
    rationale: str
    analysis_ref: AnalysisResult | None = None


@dataclass
class ToolCopyLifecycle:
    """Tracks lifecycle of a tool working copy in .dawn-kestrel.

    Attributes:
        original_path: Path to the original tool file.
        copy_path: Path to the .dawn-kestrel working copy.
        target_type: Type of tool (skill, policy, tool, agent).
        existed_before: Whether the copy already existed before this cycle.
        created_this_cycle: Whether this copy was created during this cycle.
    """

    original_path: Path
    copy_path: Path
    target_type: ImprovementType
    existed_before: bool = False
    created_this_cycle: bool = False


@dataclass
class IterationResult:
    """Result of a single improvement iteration.

    Attributes:
        iteration_num: The iteration number (0-indexed).
        score_before: Score before applying improvements.
        score_after: Score after applying improvements.
        score_delta: Change in score.
        analyses: Analysis results from this iteration.
        improvements: Improvements generated and applied.
        applied: Whether improvements were kept (vs reverted).
        timestamp: When this iteration completed.
    """

    iteration_num: int
    score_before: float = 0.0
    score_after: float = 0.0
    score_delta: float = 0.0
    analyses: list[AnalysisResult] = field(default_factory=list)
    improvements: list[ImprovementResult] = field(default_factory=list)
    applied: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class CycleResult:
    """Result of a complete auto-research cycle.

    Attributes:
        experiment_id: Unique identifier for this experiment.
        agent_name: Name of the agent being improved.
        status: Final status of the cycle.
        iterations: List of all iteration results.
        initial_score: Score at the start.
        final_score: Score at the end.
        best_score: Best score achieved.
        improvement_delta: Total improvement (final - initial).
        total_lessons_created: Number of lessons created.
        converged: Whether the cycle converged.
        started_at: When the cycle started.
        completed_at: When the cycle completed.
        error_message: Error message if status is ERROR.
    """

    experiment_id: str
    agent_name: str
    status: CycleStatus = CycleStatus.RUNNING
    iterations: list[IterationResult] = field(default_factory=list)
    initial_score: float = 0.0
    final_score: float = 0.0
    best_score: float = 0.0
    improvement_delta: float = 0.0
    total_lessons_created: int = 0
    converged: bool = False
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    error_message: str | None = None

    @property
    def total_iterations(self) -> int:
        """Total number of iterations completed."""
        return len(self.iterations)


__all__ = [
    "AnalysisResult",
    "CycleResult",
    "CycleStatus",
    "ImprovementResult",
    "ImprovementType",
    "IterationResult",
    "RepoConfig",
    "ToolCopyLifecycle",
]
