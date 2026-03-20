from __future__ import annotations

from ash_hawk.pipeline.coach import CoachRole
from ash_hawk.pipeline.types import PipelineContext, PipelineRole


def _context() -> PipelineContext:
    return PipelineContext(
        run_artifact_id="run-coach-001",
        review_request_id="review-coach-001",
        role=PipelineRole.COACH,
        target_agent="test-agent",
        experiment_id="exp-coach-001",
    )


class TestCoachRole:
    def test_generate_proposals_uses_broad_signals(self) -> None:
        coach = CoachRole()
        proposals = coach.generate_proposals(
            _context(),
            ["Tool call timed out", "Permission denied", "Policy rule mismatch detected"],
            {
                "root_causes": [
                    "Timeout issues detected (2 occurrences) - consider adjusting tool timeouts",
                    "Permission issues with tools: bash",
                ],
                "risk_areas": ["command_execution"],
                "findings": [
                    {
                        "category": "efficiency",
                        "description": "Tool read called 4 times - consider batching",
                    },
                    {
                        "category": "root_cause",
                        "description": "Tool bash failed 2 time(s)",
                    },
                ],
            },
        )

        assert len(proposals) >= 2
        assert any(p.title.startswith("Coach contrarian probe") for p in proposals)
        assert all(p.evidence_refs for p in proposals)
        assert all(p.diff_payload.get("experiment_steps") for p in proposals)

    def test_generate_proposals_returns_empty_without_signals(self) -> None:
        coach = CoachRole()
        proposals = coach.generate_proposals(_context(), [], {})
        assert proposals == []
