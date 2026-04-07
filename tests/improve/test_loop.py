# type-hygiene: skip-file
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ash_hawk.improve.loop import ImprovementResult, improve


def _make_mock_summary(pass_rate: float, n_trials: int = 2):
    from ash_hawk.types import (
        EvalOutcome,
        EvalRunSummary,
        EvalStatus,
        EvalTrial,
        RunEnvelope,
        SuiteMetrics,
        TrialResult,
    )

    trials = []
    for i in range(n_trials):
        passed = i < int(pass_rate * n_trials)
        trials.append(
            EvalTrial(
                id=f"trial-{i}",
                task_id=f"task-{i}",
                status=EvalStatus.COMPLETED,
                attempt_number=1,
                input_snapshot=f"input-{i}",
                result=TrialResult(
                    trial_id=f"trial-{i}",
                    outcome=(
                        EvalOutcome.success()
                        if passed
                        else EvalOutcome.failure("agent_error", "failed")
                    ),
                    aggregate_passed=passed,
                    aggregate_score=1.0 if passed else 0.0,
                ),
            )
        )

    envelope = RunEnvelope.model_construct(
        run_id="run-1",
        suite_id="test-suite",
        suite_hash="abc123",
        harness_version="0.0.0",
        agent_name="test-agent",
        provider="test",
        model="test-model",
    )
    return EvalRunSummary(
        envelope=envelope,
        metrics=SuiteMetrics(
            suite_id="test-suite",
            run_id="run-1",
            total_tasks=n_trials,
            completed_tasks=n_trials,
            passed_tasks=int(pass_rate * n_trials),
            failed_tasks=n_trials - int(pass_rate * n_trials),
            pass_rate=pass_rate,
            created_at="2026-01-01T00:00:00",
        ),
        trials=trials,
    )


class TestImprove:
    @pytest.mark.asyncio
    async def test_target_reached_exits_early(self, tmp_path):
        with (
            patch("ash_hawk.improve.loop._run_eval", new_callable=AsyncMock) as mock_run,
            patch("ash_hawk.improve.loop.diagnose_failures", new_callable=AsyncMock),
        ):
            mock_run.return_value = _make_mock_summary(1.0)
            result = await improve("suite.yaml", max_iterations=5, output_dir=tmp_path)

        assert result.initial_pass_rate == 1.0
        assert result.final_pass_rate == 1.0
        assert result.patches_proposed == []
        mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_max_iterations_respected(self, tmp_path):
        with (
            patch("ash_hawk.improve.loop._run_eval", new_callable=AsyncMock) as mock_run,
            patch("ash_hawk.improve.loop.diagnose_failures", new_callable=AsyncMock) as mock_diag,
            patch("ash_hawk.improve.loop.propose_patch", new_callable=AsyncMock),
        ):
            mock_run.return_value = _make_mock_summary(0.0)
            mock_diag.return_value = []
            result = await improve("suite.yaml", max_iterations=3, output_dir=tmp_path)

        assert mock_run.call_count == 3

    @pytest.mark.asyncio
    async def test_run_eval_failure_continues(self, tmp_path):
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("eval failed")
            return _make_mock_summary(1.0)

        with patch("ash_hawk.improve.loop._run_eval", side_effect=side_effect):
            result = await improve("suite.yaml", max_iterations=3, output_dir=tmp_path)

        assert result.final_pass_rate == 1.0
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_improvement_result_fields(self, tmp_path):
        with (
            patch("ash_hawk.improve.loop._run_eval", new_callable=AsyncMock) as mock_run,
            patch("ash_hawk.improve.loop.diagnose_failures", new_callable=AsyncMock),
        ):
            mock_run.return_value = _make_mock_summary(1.0)
            result = await improve(
                "suite.yaml", max_iterations=5, trace_dir=Path("/tmp/traces"), output_dir=tmp_path
            )

        assert isinstance(result, ImprovementResult)
        assert result.initial_pass_rate == 1.0
        assert result.patches_applied == []
        assert result.trace_path == Path("/tmp/traces")
