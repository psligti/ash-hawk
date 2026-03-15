"""Ash Hawk contracts for cross-agent improvement engine.

These contracts define the interface for evaluation, improvement proposals,
and curated lessons across all Dawn Kestrel-based agents.

Key contracts:
- ReviewRequest: Request to evaluate a completed run
- ReviewResult: Result of the evaluation with findings
- ImprovementProposal: Structured proposal for behavioral change
- CuratedLesson: Approved and versioned lesson for persistence
"""

from __future__ import annotations

from ash_hawk.contracts.curated_lesson import CuratedLesson
from ash_hawk.contracts.improvement_proposal import ImprovementProposal
from ash_hawk.contracts.review_request import ReviewRequest
from ash_hawk.contracts.review_result import ReviewFinding, ReviewMetrics, ReviewResult

__all__ = [
    "ReviewRequest",
    "ReviewResult",
    "ReviewFinding",
    "ReviewMetrics",
    "ImprovementProposal",
    "CuratedLesson",
]
