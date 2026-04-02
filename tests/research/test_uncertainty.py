from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ash_hawk.research.uncertainty import HypothesisState, InformationGap, UncertaintyModel


class MockDiagnosisReport:
    def __init__(
        self,
        hypotheses: list[object] | None = None,
        missing_signals: list[object] | None = None,
        uncertainty_level: float = 0.5,
        cause_categories: list[object] | None = None,
    ) -> None:
        self.hypotheses = hypotheses or []
        self.missing_signals = missing_signals or []
        self.uncertainty_level = uncertainty_level
        self.cause_categories = cause_categories or []


class MockCompetingHypothesis:
    def __init__(
        self,
        hypothesis_id: str,
        cause_category: str,
        description: str,
        confidence: float,
        supporting_evidence: list[str] | None = None,
        missing_evidence: list[str] | None = None,
    ) -> None:
        self.hypothesis_id = hypothesis_id
        self.cause_category = cause_category
        self.description = description
        self.confidence = confidence
        self.supporting_evidence = supporting_evidence or []
        self.missing_evidence = missing_evidence or []


class TestUncertaintyModel:
    def test_hypothesis_state_is_resolved(self) -> None:
        hypothesis = HypothesisState(
            hypothesis_id="hyp-1",
            description="Resolved",
            confidence=0.9,
            status="confirmed",
        )

        assert hypothesis.is_resolved() is True

    def test_information_gap_creation(self) -> None:
        gap = InformationGap(
            gap_id="gap-1",
            description="Need data",
            proposed_measurement="metric",
            expected_impact=0.5,
        )

        assert gap.gap_id == "gap-1"
        assert gap.proposed_by == "diagnosis"
        assert gap.status == "identified"

    def test_uncertainty_model_defaults(self) -> None:
        model = UncertaintyModel()

        assert model.hypotheses == {}
        assert model.unresolved_questions == []
        assert model.information_gaps == []

    def test_load_from_missing_path_returns_fresh_model(self, tmp_path: Path) -> None:
        model = UncertaintyModel.load(tmp_path / "missing")

        assert model.hypotheses == {}
        assert model.unresolved_questions == []

    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        model = UncertaintyModel(storage_path=tmp_path)
        model.hypotheses["hyp-1"] = HypothesisState(
            hypothesis_id="hyp-1",
            description="Hyp",
            confidence=0.7,
            evidence_for=["e1"],
            missing_measurements=["m1"],
            sample_count=2,
            last_updated=datetime.now(UTC).isoformat(),
        )
        model.unresolved_questions = ["missing-a"]
        model.information_gaps.append(
            InformationGap(
                gap_id="gap-001",
                description="Missing measurement",
                proposed_measurement="missing-a",
                expected_impact=0.4,
            )
        )

        await model.save()
        loaded = UncertaintyModel.load(tmp_path)

        assert "hyp-1" in loaded.hypotheses
        assert loaded.unresolved_questions == ["missing-a"]
        assert loaded.information_gaps[0].gap_id == "gap-001"

    def test_load_with_corrupted_json_falls_back(self, tmp_path: Path) -> None:
        path = tmp_path / "uncertainty.json"
        path.write_text("{not-json}")

        model = UncertaintyModel.load(tmp_path)

        assert model.hypotheses == {}

    def test_update_from_diagnosis_adds_hypotheses(self) -> None:
        model = UncertaintyModel()
        report = MockDiagnosisReport(
            hypotheses=[
                MockCompetingHypothesis(
                    hypothesis_id="hyp-1",
                    cause_category="tool_misuse",
                    description="Tool issue",
                    confidence=0.4,
                )
            ],
            missing_signals=["signal-a"],
        )

        model.update_from_diagnosis(report)

        assert "hyp-1" in model.hypotheses
        assert model.unresolved_questions == ["signal-a"]
        assert model.information_gaps[0].proposed_measurement == "signal-a"

    def test_update_from_diagnosis_merges_existing(self) -> None:
        model = UncertaintyModel()
        model.hypotheses["hyp-1"] = HypothesisState(
            hypothesis_id="hyp-1",
            description="Same",
            confidence=0.2,
            sample_count=1,
        )
        report = MockDiagnosisReport(
            hypotheses=[
                MockCompetingHypothesis(
                    hypothesis_id="hyp-2",
                    cause_category="tool_misuse",
                    description="Same",
                    confidence=0.8,
                )
            ]
        )

        model.update_from_diagnosis(report)

        hypothesis = model.hypotheses["hyp-1"]
        assert hypothesis.sample_count == 2
        assert abs(hypothesis.confidence - 0.5) < 1e-6

    def test_uncertainty_level_property(self) -> None:
        model = UncertaintyModel()
        model.hypotheses["hyp-1"] = HypothesisState(
            hypothesis_id="hyp-1",
            description="High",
            confidence=0.8,
        )

        assert abs(model.uncertainty_level - 0.2) < 1e-6

    def test_should_observe_before_fixing_threshold(self) -> None:
        model = UncertaintyModel()
        model.hypotheses["hyp-1"] = HypothesisState(
            hypothesis_id="hyp-1",
            description="Low",
            confidence=0.1,
        )

        assert model.should_observe_before_fixing(threshold=0.6) is True

    def test_rank_missing_measurements_sorts_by_impact(self) -> None:
        model = UncertaintyModel()
        model.information_gaps = [
            InformationGap(
                gap_id="gap-1",
                description="Low",
                proposed_measurement="m1",
                expected_impact=0.1,
            ),
            InformationGap(
                gap_id="gap-2",
                description="High",
                proposed_measurement="m2",
                expected_impact=0.9,
            ),
        ]

        ranked = model.rank_missing_measurements()

        assert ranked[0][0] == "gap-2"

    def test_get_competing_hypotheses_filters_by_confidence(self) -> None:
        model = UncertaintyModel()
        model.hypotheses = {
            "hyp-1": HypothesisState(
                hypothesis_id="hyp-1",
                description="low",
                confidence=0.2,
            ),
            "hyp-2": HypothesisState(
                hypothesis_id="hyp-2",
                description="high",
                confidence=0.8,
            ),
        }

        competing = model.get_competing_hypotheses()

        assert [hyp.hypothesis_id for hyp in competing] == ["hyp-1"]

    def test_hypothesis_capping_at_max_hypotheses(self) -> None:
        model = UncertaintyModel(max_hypotheses=1)
        report = MockDiagnosisReport(
            hypotheses=[
                MockCompetingHypothesis(
                    hypothesis_id="hyp-1",
                    cause_category="tool",
                    description="Low",
                    confidence=0.1,
                ),
                MockCompetingHypothesis(
                    hypothesis_id="hyp-2",
                    cause_category="tool",
                    description="High",
                    confidence=0.9,
                ),
            ]
        )

        model.update_from_diagnosis(report)

        assert len(model.hypotheses) == 1
        assert next(iter(model.hypotheses.values())).description == "High"

    @pytest.mark.asyncio
    async def test_evidence_capping_at_max_evidence(self, tmp_path: Path) -> None:
        model = UncertaintyModel(storage_path=tmp_path, max_evidence=2)
        model.hypotheses["hyp-1"] = HypothesisState(
            hypothesis_id="hyp-1",
            description="Evidence",
            confidence=0.5,
            evidence_for=["a", "b", "c"],
            evidence_against=["d", "e", "f"],
            missing_measurements=["g", "h", "i"],
        )

        await model.save()

        hypothesis = model.hypotheses["hyp-1"]
        assert hypothesis.evidence_for == ["b", "c"]
        assert hypothesis.evidence_against == ["e", "f"]
        assert hypothesis.missing_measurements == ["h", "i"]
