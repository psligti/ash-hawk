# type-hygiene: skip-file
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ash_hawk.improve.diagnose import Diagnosis, _parse_diagnosis_response, diagnose_failures
from ash_hawk.types import EvalOutcome, EvalStatus, EvalTrial, TrialResult


def _make_trial(trial_id: str = "trial-1") -> EvalTrial:
    return EvalTrial(
        id=trial_id,
        task_id="task-1",
        status=EvalStatus.ERROR,
        attempt_number=1,
        input_snapshot="test",
        result=TrialResult(
            trial_id=trial_id,
            outcome=EvalOutcome.failure("agent_error", "err"),
        ),
    )


class TestParseDiagnosis:
    def test_valid_json_response(self):
        response = (
            '{"failure_summary": "bad", "root_cause": "bug", '
            '"suggested_fix": "fix it", "target_files": ["a.py"], "confidence": 0.9}'
        )
        result = _parse_diagnosis_response("trial-1", response)
        assert result is not None
        assert result.trial_id == "trial-1"
        assert result.failure_summary == "bad"
        assert result.root_cause == "bug"
        assert result.confidence == 0.9

    def test_missing_required_field(self):
        response = '{"failure_summary": "bad"}'
        result = _parse_diagnosis_response("trial-1", response)
        assert result is None

    def test_invalid_json(self):
        result = _parse_diagnosis_response("trial-1", "not json at all")
        assert result is None

    def test_default_confidence(self):
        response = '{"failure_summary": "bad", "root_cause": "bug", "suggested_fix": "fix"}'
        result = _parse_diagnosis_response("trial-1", response)
        assert result is not None
        assert result.confidence == 0.0


class TestDiagnoseFailures:
    @pytest.mark.asyncio
    async def test_llm_returns_none_skips(self):
        trial = _make_trial()
        with patch(
            "ash_hawk.improve.diagnose._call_llm", new_callable=AsyncMock, return_value=None
        ):
            results = await diagnose_failures([trial])
        assert results == []
