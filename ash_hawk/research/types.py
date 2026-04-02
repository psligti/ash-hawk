"""Foundation types for the Research Supervisor."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal


class ResearchAction(StrEnum):
    FIX = "fix"
    OBSERVE = "observe"
    EXPERIMENT = "experiment"
    EVALUATE = "evaluate"
    RESTRUCTURE = "restructure"
    PROMOTE = "promote"


class CauseCategory(StrEnum):
    PROMPT_QUALITY = "prompt_quality"
    TOOL_MISUSE = "tool_misuse"
    CONTEXT_OVERFLOW = "context_overflow"
    DELEGATION_FAILURE = "delegation_failure"
    ORCHESTRATION_BRANCH = "orchestration_branch"
    TIMEOUT_MISALLOCATION = "timeout_misallocation"
    UNKNOWN = "unknown"


class TargetSurface(StrEnum):
    PROMPT = "prompt"
    POLICY = "policy"
    TOOL = "tool"
    DELEGATION = "delegation"
    ORCHESTRATION = "orchestration"
    EVAL_QUESTION = "eval_question"


class HypothesisStatus(StrEnum):
    ACTIVE = "active"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    UNRESOLVABLE = "unresolvable"


@dataclass
class ResearchLoopConfig:
    iterations: int = 10
    uncertainty_threshold: float = 0.6
    d_step_interval: int = 5
    prune_interval: int = 3
    max_diagnoses_per_run: int = 50
    safety_threshold: float = -0.05
    human_approval_required: bool = True
    storage_path: Path = field(default_factory=lambda: Path(".ash-hawk/research"))
    improvement_threshold: float = 0.02
    min_active_targets: int = 3
    max_hypotheses: int = 20
    max_evidence_per_hypothesis: int = 50


@dataclass
class ResearchDecision:
    action: ResearchAction
    rationale: str
    target: str | None
    expected_info_gain: float
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ResearchLoopResult:
    decisions: list[ResearchDecision] = field(default_factory=list)
    diagnoses_count: int = 0
    strategies_promoted: list[str] = field(default_factory=list)
    uncertainty_before: float = 0.0
    uncertainty_after: float = 0.0
    improvement_delta: float = 0.0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None

    @property
    def total_decisions(self) -> int:
        return len(self.decisions)

    @property
    def observe_vs_fix_ratio(self) -> float:
        observe = sum(1 for d in self.decisions if d.action == ResearchAction.OBSERVE)
        fix = sum(1 for d in self.decisions if d.action == ResearchAction.FIX)
        return observe / fix if fix > 0 else 0.0


__all__ = [
    "CauseCategory",
    "HypothesisStatus",
    "ResearchAction",
    "ResearchDecision",
    "ResearchLoopConfig",
    "ResearchLoopResult",
    "TargetSurface",
]
