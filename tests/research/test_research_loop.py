from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ash_hawk.research.diagnosis import DiagnosisReport
from ash_hawk.research.research_loop import ResearchLoop
from ash_hawk.research.types import ResearchAction, ResearchDecision, ResearchLoopConfig
from ash_hawk.research.uncertainty import HypothesisState


def _diagnosis(recommended_action: str = "fix", uncertainty_level: float = 0.2) -> DiagnosisReport:
    return DiagnosisReport(
        diagnosis_id="diag-1",
        run_id="run-1",
        timestamp=datetime.now(UTC).isoformat(),
        uncertainty_level=uncertainty_level,
        recommended_action=recommended_action,
    )


class TestResearchLoop:
    @pytest.mark.asyncio
    async def test_creation_with_defaults(self, tmp_path: Path) -> None:
        loop = ResearchLoop(storage_path=tmp_path)

        result = await loop.run([])

        assert result.diagnoses_count == ResearchLoopConfig().iterations

    @pytest.mark.asyncio
    async def test_run_no_llm_client_completes(self, tmp_path: Path) -> None:
        loop = ResearchLoop(storage_path=tmp_path)

        result = await loop.run([])

        assert result.completed_at is not None
        assert result.diagnoses_count == ResearchLoopConfig().iterations

    @pytest.mark.asyncio
    async def test_run_returns_result(self, tmp_path: Path) -> None:
        loop = ResearchLoop(storage_path=tmp_path)

        result = await loop.run([])

        assert result.decisions is not None

    @pytest.mark.asyncio
    async def test_run_respects_iterations(self, tmp_path: Path) -> None:
        config = ResearchLoopConfig(iterations=2, max_diagnoses_per_run=10)
        loop = ResearchLoop(config=config, storage_path=tmp_path)

        result = await loop.run([])

        assert result.diagnoses_count == 2
        assert len(result.decisions) == 2

    @pytest.mark.asyncio
    async def test_run_respects_diagnosis_budget(self, tmp_path: Path) -> None:
        config = ResearchLoopConfig(iterations=5, max_diagnoses_per_run=2)
        loop = ResearchLoop(config=config, storage_path=tmp_path)

        result = await loop.run([])

        assert result.diagnoses_count == 2
        assert len(result.decisions) == 2

    @pytest.mark.asyncio
    async def test_run_saves_state(self, tmp_path: Path) -> None:
        loop = ResearchLoop(storage_path=tmp_path)

        await loop.run([])

        assert (tmp_path / "uncertainty" / "uncertainty.json").exists()
        assert (tmp_path / "targets" / "targets.json").exists()
        assert (tmp_path / "strategies" / "strategies.json").exists()

    def test_decide_returns_observe_when_uncertainty_high(self) -> None:
        loop = ResearchLoopHarness()
        loop.set_hypotheses({})

        decision = loop.decide(_diagnosis(), 0)

        assert decision.action is ResearchAction.OBSERVE

    def test_decide_returns_fix_when_uncertainty_low(self) -> None:
        loop = ResearchLoopHarness()
        loop.set_hypotheses(
            {
                "hyp-1": HypothesisState(
                    hypothesis_id="hyp-1",
                    description="High",
                    confidence=0.9,
                )
            }
        )

        decision = loop.decide(_diagnosis(), 0)

        assert decision.action is ResearchAction.FIX

    def test_decide_returns_promote_when_recommended(self) -> None:
        loop = ResearchLoopHarness()
        loop.set_hypotheses(
            {
                "hyp-1": HypothesisState(
                    hypothesis_id="hyp-1",
                    description="High",
                    confidence=0.9,
                )
            }
        )

        decision = loop.decide(_diagnosis(recommended_action="promote"), 0)

        assert decision.action is ResearchAction.PROMOTE

    def test_has_competing_hypotheses(self) -> None:
        loop = ResearchLoopHarness()
        loop.set_hypotheses(
            {
                "hyp-1": HypothesisState(
                    hypothesis_id="hyp-1",
                    description="low",
                    confidence=0.2,
                ),
                "hyp-2": HypothesisState(
                    hypothesis_id="hyp-2",
                    description="low",
                    confidence=0.3,
                ),
            }
        )

        assert loop.has_competing_hypotheses() is True


class ResearchLoopHarness(ResearchLoop):
    def set_hypotheses(self, hypotheses: dict[str, HypothesisState]) -> None:
        self._uncertainty.hypotheses = hypotheses

    def decide(self, diagnosis: DiagnosisReport, iteration: int) -> ResearchDecision:
        return self._decide(diagnosis, iteration)

    def has_competing_hypotheses(self) -> bool:
        return self._has_competing_hypotheses()
