from __future__ import annotations

import json

import pytest

from ash_hawk.research.diagnosis import DiagnosisEngine
from ash_hawk.research.types import CauseCategory


class MockLLMClient:
    def __init__(self, response: str | None = None) -> None:
        self._response = response

    async def complete(
        self, messages: list[dict[str, str]], options: dict[str, float] | None = None
    ):
        return type("Response", (), {"text": self._response})()


def _response_payload(
    hypotheses: list[dict[str, object]], missing_signals: list[str] | None = None
) -> str:
    payload = {
        "hypotheses": hypotheses,
        "missing_signals": missing_signals or [],
    }
    return json.dumps(payload)


class TestDiagnosisEngine:
    @pytest.mark.asyncio
    async def test_competing_hypothesis_creation(self) -> None:
        response = _response_payload(
            [
                {
                    "cause_category": "tool_misuse",
                    "description": "Tool misused",
                    "confidence": 0.7,
                    "supporting_evidence": ["trace"],
                    "missing_evidence": ["logs"],
                }
            ]
        )
        engine = DiagnosisEngine(llm_client=MockLLMClient(response))

        report = await engine.diagnose({"a": 0.1}, [], {})

        assert len(report.hypotheses) == 1
        hypothesis = report.hypotheses[0]
        assert hypothesis.description == "Tool misused"
        assert hypothesis.cause_category == CauseCategory.TOOL_MISUSE
        assert hypothesis.supporting_evidence == ["trace"]
        assert hypothesis.missing_evidence == ["logs"]

    def test_diagnosis_report_defaults(self) -> None:
        from ash_hawk.research.diagnosis import DiagnosisReport

        report = DiagnosisReport(diagnosis_id="d1", run_id="r1", timestamp="now")

        assert report.cause_categories == []
        assert report.hypotheses == []
        assert report.primary_hypothesis is None
        assert report.uncertainty_level == 0.0
        assert report.missing_signals == []
        assert report.recommended_action == "observe"

    @pytest.mark.asyncio
    async def test_diagnose_no_llm_client_fallback(self) -> None:
        engine = DiagnosisEngine()

        report = await engine.diagnose({"score": 0.2}, [], {})

        assert report.cause_categories == [CauseCategory.UNKNOWN]
        assert report.uncertainty_level == 0.55
        assert report.recommended_action == "experiment"

    @pytest.mark.asyncio
    async def test_diagnose_valid_json(self) -> None:
        response = _response_payload(
            [
                {
                    "cause_category": "prompt_quality",
                    "description": "Prompt issue",
                    "confidence": 0.6,
                }
            ],
            ["signal-a"],
        )
        engine = DiagnosisEngine(llm_client=MockLLMClient(response))

        report = await engine.diagnose({"score": 0.2}, [], {})

        assert report.primary_hypothesis is not None
        assert report.missing_signals == ["signal-a"]

    @pytest.mark.asyncio
    async def test_diagnose_invalid_json_fallback(self) -> None:
        engine = DiagnosisEngine(llm_client=MockLLMClient("not-json"))

        report = await engine.diagnose({"score": 0.2}, [], {})

        assert report.cause_categories == [CauseCategory.UNKNOWN]
        assert report.hypotheses == []

    @pytest.mark.asyncio
    async def test_diagnose_malformed_hypothesis_fallback(self) -> None:
        response = _response_payload([{"cause_category": "tool_misuse"}])
        engine = DiagnosisEngine(llm_client=MockLLMClient(response))

        report = await engine.diagnose({"score": 0.2}, [], {})

        assert report.hypotheses == []
        assert report.recommended_action == "experiment"

    @pytest.mark.asyncio
    async def test_confidence_is_clamped(self) -> None:
        response = _response_payload(
            [
                {
                    "cause_category": "tool_misuse",
                    "description": "Overconfident",
                    "confidence": 1.5,
                }
            ]
        )
        engine = DiagnosisEngine(llm_client=MockLLMClient(response))

        report = await engine.diagnose({"score": 0.2}, [], {})

        assert report.hypotheses[0].confidence == 1.0

    @pytest.mark.asyncio
    async def test_primary_hypothesis_selection(self) -> None:
        response = _response_payload(
            [
                {
                    "cause_category": "tool_misuse",
                    "description": "Low confidence",
                    "confidence": 0.2,
                },
                {
                    "cause_category": "prompt_quality",
                    "description": "High confidence",
                    "confidence": 0.8,
                },
            ]
        )
        engine = DiagnosisEngine(llm_client=MockLLMClient(response))

        report = await engine.diagnose({"score": 0.2}, [], {})

        assert report.primary_hypothesis is not None
        assert report.primary_hypothesis == report.hypotheses[1].hypothesis_id

    @pytest.mark.asyncio
    async def test_uncertainty_level_calculation(self) -> None:
        response = _response_payload(
            [
                {
                    "cause_category": "tool_misuse",
                    "description": "Moderate",
                    "confidence": 0.7,
                }
            ]
        )
        engine = DiagnosisEngine(llm_client=MockLLMClient(response))

        report = await engine.diagnose({"score": 0.2}, [], {})

        assert abs(report.uncertainty_level - 0.3) < 1e-6

    @pytest.mark.asyncio
    async def test_recommended_action_for_uncertainty(self) -> None:
        response = _response_payload(
            [
                {
                    "cause_category": "prompt_quality",
                    "description": "Mixed",
                    "confidence": 0.5,
                }
            ]
        )
        engine = DiagnosisEngine(llm_client=MockLLMClient(response))

        report = await engine.diagnose({"score": 0.2}, [], {})

        assert report.recommended_action == "experiment"

    @pytest.mark.asyncio
    async def test_diagnose_with_empty_eval_results(self) -> None:
        engine = DiagnosisEngine(llm_client=MockLLMClient("{}"))

        report = await engine.diagnose({}, [], {})

        assert report.cause_categories == [CauseCategory.UNKNOWN]
        assert report.uncertainty_level == 0.55

    @pytest.mark.asyncio
    async def test_cause_categories_populated(self) -> None:
        response = _response_payload(
            [
                {
                    "cause_category": "tool_misuse",
                    "description": "Tool",
                    "confidence": 0.4,
                },
                {
                    "cause_category": "prompt_quality",
                    "description": "Prompt",
                    "confidence": 0.3,
                },
            ]
        )
        engine = DiagnosisEngine(llm_client=MockLLMClient(response))

        report = await engine.diagnose({"score": 0.2}, [], {})

        assert report.cause_categories == [CauseCategory.TOOL_MISUSE, CauseCategory.PROMPT_QUALITY]

    @pytest.mark.asyncio
    async def test_missing_signals_captured(self) -> None:
        response = _response_payload(
            [
                {
                    "cause_category": "prompt_quality",
                    "description": "Missing",
                    "confidence": 0.4,
                }
            ],
            ["signal-1", "signal-2"],
        )
        engine = DiagnosisEngine(llm_client=MockLLMClient(response))

        report = await engine.diagnose({"score": 0.2}, [], {})

        assert report.missing_signals == ["signal-1", "signal-2"]
