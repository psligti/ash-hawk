from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pydantic as pd

from ash_hawk.contracts import CuratedLesson, ImprovementProposal

if TYPE_CHECKING:
    pass


class QualityGateResult(pd.BaseModel):
    passed: bool = pd.Field(description="Whether the proposal passed the quality gate")
    failures: list[str] = pd.Field(
        default_factory=list,
        description="List of reasons why the proposal failed",
    )
    warnings: list[str] = pd.Field(
        default_factory=list,
        description="List of warnings (non-blocking issues)",
    )

    model_config = pd.ConfigDict(extra="forbid")


@dataclass
class QualityGate:
    min_confidence: float = 0.7
    require_evidence: bool = True
    require_rationale: bool = True
    allowed_risk_levels: list[str] = field(default_factory=lambda: ["low", "medium"])
    min_evidence_count: int = 1
    max_risk_for_auto_approve: str = "medium"

    def check(self, proposal: ImprovementProposal) -> QualityGateResult:
        failures: list[str] = []
        warnings: list[str] = []

        if proposal.confidence is not None and proposal.confidence < self.min_confidence:
            failures.append(
                f"Confidence {proposal.confidence:.2f} below minimum {self.min_confidence}"
            )

        if self.require_rationale and not proposal.rationale.strip():
            failures.append("Missing rationale")

        if self.require_evidence and len(proposal.evidence_refs) < self.min_evidence_count:
            if self.min_evidence_count > 0:
                failures.append(
                    f"Insufficient evidence: {len(proposal.evidence_refs)} < {self.min_evidence_count}"
                )

        if proposal.risk_level not in self.allowed_risk_levels:
            failures.append(
                f"Risk level '{proposal.risk_level}' not in allowed: {self.allowed_risk_levels}"
            )

        if proposal.confidence is not None and proposal.confidence < 0.5:
            warnings.append("Very low confidence score")

        if proposal.risk_level == "high" and proposal.confidence is not None:
            if proposal.confidence < 0.9:
                warnings.append("High risk proposal with low confidence")

        return QualityGateResult(
            passed=len(failures) == 0,
            failures=failures,
            warnings=warnings,
        )


@dataclass
class CurationConfig:
    auto_approve: bool = True
    min_confidence: float = 0.7
    require_evidence: bool = True
    require_rationale: bool = True
    allowed_risk_levels: list[str] = field(default_factory=lambda: ["low", "medium"])
    min_evidence_count: int = 1
    require_human_approval_for_high_risk: bool = True
    max_lessons_per_run: int = 10

    def to_quality_gate(self) -> QualityGate:
        return QualityGate(
            min_confidence=self.min_confidence,
            require_evidence=self.require_evidence,
            require_rationale=self.require_rationale,
            allowed_risk_levels=self.allowed_risk_levels,
            min_evidence_count=self.min_evidence_count,
        )


class CuratorRole:
    def __init__(self, config: CurationConfig | None = None) -> None:
        self._config = config or CurationConfig()
        self._quality_gate = self._config.to_quality_gate()

    def curate(
        self,
        proposals: list[ImprovementProposal],
        auto_appro: bool | None = None,
    ) -> list[CuratedLesson]:
        lessons: list[CuratedLesson] = []

        use_auto_approve = auto_appro if auto_appro is not None else self._config.auto_approve

        for proposal in proposals:
            if proposal.status != "pending":
                continue

            gate_result = self._quality_gate.check(proposal)

            if not gate_result.passed:
                proposal.status = "rejected"
                proposal.rejection_reason = "; ".join(gate_result.failures)
                continue

            if not use_auto_approve:
                continue

            if self._config.require_human_approval_for_high_risk and proposal.risk_level == "high":
                continue

            lesson = self._create_lesson(proposal)
            lessons.append(lesson)

            if len(lessons) >= self._config.max_lessons_per_run:
                break

        return lessons

    def _create_lesson(self, proposal: ImprovementProposal) -> CuratedLesson:
        return CuratedLesson(
            lesson_id=f"lesson-{uuid4().hex[:8]}",
            source_proposal_id=proposal.proposal_id,
            applies_to_agents=[proposal.target_agent],
            lesson_type=proposal.proposal_type,
            title=proposal.title,
            description=proposal.rationale,
            lesson_payload=proposal.diff_payload or {},
            validation_status="approved",
            version=1,
            created_at=datetime.now(UTC),
            experiment_id=proposal.experiment_id,
            strategy=proposal.strategy,
            sub_strategies=proposal.sub_strategies,
        )

    def reject(self, proposal: ImprovementProposal, reason: str) -> ImprovementProposal:
        proposal.status = "rejected"
        proposal.rejection_reason = reason
        proposal.reviewed_at = datetime.now(UTC)
        return proposal

    def defer(self, proposal: ImprovementProposal) -> ImprovementProposal:
        proposal.status = "pending"
        proposal.reviewed_at = datetime.now(UTC)
        return proposal

    def evaluate_proposal(self, proposal: ImprovementProposal) -> QualityGateResult:
        return self._quality_gate.check(proposal)


__all__ = [
    "CurationConfig",
    "CuratorRole",
    "QualityGate",
    "QualityGateResult",
]
