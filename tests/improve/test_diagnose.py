# type-hygiene: skip-file
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ash_hawk.improve.diagnose import (
    _parse_diagnosis_response,
    diagnose_failures,
)
from ash_hawk.types import EvalOutcome, EvalStatus, EvalTranscript, EvalTrial, TrialResult


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
            '{"ideas": ['
            '{"failure_summary": "bad", "root_cause": "bug", '
            '"suggested_fix": "fix it", "target_files": ["a.py"], "confidence": 0.9}'
            "]}"
        )
        result = _parse_diagnosis_response("trial-1", response)
        assert len(result) == 1
        assert result[0].trial_id == "trial-1"
        assert result[0].failure_summary == "bad"
        assert result[0].root_cause == "bug"
        assert result[0].confidence == 0.9

    def test_missing_required_field(self):
        response = '{"failure_summary": "bad"}'
        result = _parse_diagnosis_response("trial-1", response)
        assert result == []

    def test_invalid_json(self):
        result = _parse_diagnosis_response("trial-1", "not json at all")
        assert result == []

    def test_default_confidence(self):
        response = '{"failure_summary": "bad", "root_cause": "bug", "suggested_fix": "fix"}'
        result = _parse_diagnosis_response("trial-1", response)
        assert len(result) == 1
        assert result[0].confidence == 0.0

    def test_nested_json_target_files(self):
        response = (
            "Here is my diagnosis:\n"
            '{"ideas": ['
            '{"failure_summary": "bad", "root_cause": "nested issue", '
            '"suggested_fix": "fix it", "target_files": ["a.py", "b.py"], '
            '"confidence": 0.7, "extra": {"nested": true}}'
            "]}\n"
            "End of analysis."
        )
        result = _parse_diagnosis_response("trial-1", response)
        assert len(result) == 1
        assert result[0].target_files == ["a.py", "b.py"]
        assert result[0].confidence == 0.7

    def test_parses_multiple_unique_ideas(self):
        response = (
            '{"ideas": ['
            '{"failure_summary": "bad1", "root_cause": "bug1", '
            '"suggested_fix": "fix1", "target_files": ["a.py"], "confidence": 0.9},'
            '{"failure_summary": "bad2", "root_cause": "bug2", '
            '"suggested_fix": "fix2", "target_files": ["b.py"], "confidence": 0.8}'
            "]}"
        )
        result = _parse_diagnosis_response("trial-1", response)
        assert len(result) == 2
        assert [d.target_files for d in result] == [["a.py"], ["b.py"]]

    def test_dedupes_duplicate_ideas(self):
        response = (
            '{"ideas": ['
            '{"failure_summary": "bad", "root_cause": "bug", '
            '"suggested_fix": "fix1", "target_files": ["a.py"], "confidence": 0.9},'
            '{"failure_summary": "bad", "root_cause": "bug", '
            '"suggested_fix": "fix2", "target_files": ["a.py"], "confidence": 0.8}'
            "]}"
        )
        result = _parse_diagnosis_response("trial-1", response)
        assert len(result) == 1


class TestDiagnoseFailures:
    @pytest.mark.asyncio
    async def test_llm_returns_none_returns_fallback(self):
        trial = _make_trial()
        with patch(
            "ash_hawk.improve.diagnose._call_llm", new_callable=AsyncMock, return_value=None
        ):
            results = await diagnose_failures([trial])
        assert len(results) == 1
        d = results[0]
        assert d.trial_id == "trial-1"
        assert d.confidence == 0.1
        assert d.target_files == []
        assert "unavailable" in d.root_cause.lower()
        assert d.actionable is False
        assert d.diagnosis_mode == "fallback_llm_unavailable"
        assert d.degraded_reason == "diagnosis_llm_unavailable"

    @pytest.mark.asyncio
    async def test_unparseable_response_returns_non_actionable_fallback(self):
        trial = _make_trial()
        with patch(
            "ash_hawk.improve.diagnose._call_llm",
            new_callable=AsyncMock,
            return_value="not valid json",
        ):
            results = await diagnose_failures([trial])

        assert len(results) == 1
        d = results[0]
        assert d.actionable is False
        assert d.diagnosis_mode == "fallback_parse_failure"
        assert d.degraded_reason == "diagnosis_parse_failure"

    @pytest.mark.asyncio
    async def test_diagnosis_prompt_includes_transcript_context(self):
        captured_prompt: dict[str, str] = {}
        trial = _make_trial()
        assert trial.result is not None
        trial.result.transcript = EvalTranscript(
            messages=[{"role": "assistant", "content": "I should inspect the repo."}],
            tool_calls=[{"name": "grep", "arguments": {"pattern": "bug"}}],
            trace_events=[{"event_type": "ToolCallEvent", "data": {"name": "grep"}}],
            agent_response="Need more context",
            error_trace="Traceback line",
        )

        async def fake_call_llm(prompt: str) -> str:
            captured_prompt["value"] = prompt
            return (
                '{"ideas": ['
                '{"failure_summary": "bad", "root_cause": "bug", '
                '"suggested_fix": "fix it", "target_files": ["a.py"], "confidence": 0.8}'
                "]}"
            )

        with patch("ash_hawk.improve.diagnose._call_llm", side_effect=fake_call_llm):
            results = await diagnose_failures([trial])

        assert len(results) == 1
        prompt = captured_prompt["value"]
        assert "Transcript Excerpt" in prompt
        assert "I should inspect the repo." in prompt
        assert 'grep({"pattern": "bug"})' in prompt
        assert "ToolCallEvent" in prompt

    @pytest.mark.asyncio
    async def test_flattens_multiple_ideas_from_single_failure(self):
        trial = _make_trial()

        with patch(
            "ash_hawk.improve.diagnose._call_llm",
            new_callable=AsyncMock,
            return_value=(
                '{"ideas": ['
                '{"failure_summary": "bad1", "root_cause": "bug1", '
                '"suggested_fix": "fix1", "target_files": ["a.py"], "confidence": 0.9},'
                '{"failure_summary": "bad2", "root_cause": "bug2", '
                '"suggested_fix": "fix2", "target_files": ["b.py"], "confidence": 0.8}'
                "]}"
            ),
        ):
            results = await diagnose_failures([trial])

        assert len(results) == 2
        assert [d.target_files for d in results] == [["a.py"], ["b.py"]]

    @pytest.mark.asyncio
    async def test_prefers_agentic_diagnosis_when_agent_path_available(self, tmp_path):
        trial = _make_trial()
        agent_dir = tmp_path / "bolt_merlin" / "agent"
        agent_dir.mkdir(parents=True)

        with (
            patch(
                "ash_hawk.improve.diagnose.investigate_trial_with_explorer",
                new_callable=AsyncMock,
                return_value=(
                    '{"ideas": ['
                    '{"failure_summary": "agentic", "root_cause": "repo search", '
                    '"suggested_fix": "change runtime", "target_files": ["runtime/x.py"], '
                    '"confidence": 0.9}'
                    "]}"
                ),
            ),
            patch(
                "ash_hawk.improve.diagnose._call_llm",
                new_callable=AsyncMock,
                return_value='{"ideas": []}',
            ) as mock_llm,
        ):
            results = await diagnose_failures([trial], agent_path=agent_dir)

        assert len(results) == 1
        assert results[0].target_files == ["runtime/x.py"]
        assert mock_llm.await_count == 0

    @pytest.mark.asyncio
    async def test_explorer_exception_falls_back_to_llm(self, tmp_path):
        trial = _make_trial()
        agent_dir = tmp_path / "bolt_merlin" / "agent"
        agent_dir.mkdir(parents=True)

        with (
            patch(
                "ash_hawk.improve.diagnose.investigate_trial_with_explorer",
                new_callable=AsyncMock,
                side_effect=RuntimeError("explorer blew up"),
            ),
            patch(
                "ash_hawk.improve.diagnose._call_llm",
                new_callable=AsyncMock,
                return_value=(
                    '{"ideas": ['
                    '{"failure_summary": "fallback", "root_cause": "llm path", '
                    '"suggested_fix": "fix it", "target_files": ["a.py"], "confidence": 0.8}'
                    "]}"
                ),
            ) as mock_llm,
        ):
            results = await diagnose_failures([trial], agent_path=agent_dir)

        assert len(results) == 1
        assert results[0].target_files == ["a.py"]
        assert mock_llm.await_count == 1
