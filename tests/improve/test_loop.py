from __future__ import annotations

import os
import subprocess
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from rich.console import Console

from ash_hawk.improve.diagnose import Diagnosis
from ash_hawk.improve.loop import ImprovementResult, improve
from ash_hawk.improve.patch import ProposedPatch


def _git_test_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
        env.pop(key, None)
    return env


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


def _init_git_agent_repo(tmp_path: Path) -> Path:
    git_env = _git_test_env()
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env=git_env,
    )
    agent_dir = tmp_path / "bolt_merlin" / "agent"
    agent_dir.mkdir(parents=True)
    (tmp_path / "bolt_merlin" / "__init__.py").write_text("", encoding="utf-8")
    (agent_dir / "__init__.py").write_text("", encoding="utf-8")
    (agent_dir / "prompt.md").write_text("bad", encoding="utf-8")
    subprocess.run(
        ["git", "add", "."],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env=git_env,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "initial",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env=git_env,
    )
    return agent_dir


class TestImprove:
    @pytest.mark.asyncio
    async def test_target_reached_exits_early(self, tmp_path):
        with (
            patch("ash_hawk.improve.loop._run_eval", new_callable=AsyncMock) as mock_run,
            patch("ash_hawk.improve.loop.diagnose_failures", new_callable=AsyncMock),
        ):
            mock_run.return_value = _make_mock_summary(1.0)
            result = await improve(
                "suite.yaml", max_iterations=5, output_dir=tmp_path, eval_repeats=1
            )

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
            result = await improve(
                "suite.yaml", max_iterations=3, output_dir=tmp_path, eval_repeats=1
            )

        assert mock_run.call_count == 1
        assert result.stop_reasons == ["no_actionable_diagnoses"]

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
            result = await improve(
                "suite.yaml", max_iterations=3, output_dir=tmp_path, eval_repeats=1
            )

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
                "suite.yaml",
                max_iterations=5,
                trace_dir=Path("/tmp/traces"),
                output_dir=tmp_path,
                eval_repeats=1,
            )

        assert isinstance(result, ImprovementResult)
        assert result.iterations == 1
        assert result.initial_pass_rate == 1.0
        assert result.patches_applied == []
        assert result.trace_path == Path("/tmp/traces")

    @pytest.mark.asyncio
    async def test_kept_mutation_is_evaluated_in_workspace_and_synced_back(self, tmp_path):
        agent_dir = _init_git_agent_repo(tmp_path)
        target_file = agent_dir / "prompt.md"

        diagnosis = Diagnosis(
            trial_id="trial-0",
            failure_summary="bad prompt",
            root_cause="prompt needs update",
            suggested_fix="update prompt",
            target_files=["prompt.md"],
            confidence=0.9,
        )

        async def run_eval_side_effect(_suite_path, _agent_name, _timeout, agent_path):
            prompt_file = Path(agent_path) / "prompt.md"
            pass_rate = 1.0 if prompt_file.read_text(encoding="utf-8") == "better" else 0.0
            return _make_mock_summary(pass_rate, n_trials=1)

        async def patch_side_effect(*args, **kwargs):
            agent_source_path = args[1]
            (Path(agent_source_path) / "prompt.md").write_text("better", encoding="utf-8")
            return ProposedPatch(
                diagnosis=diagnosis,
                file_path="prompt.md",
                description="update prompt",
                rationale="prompt needs update",
                agent_relative_path="(agent-edited)",
                content="updated in workspace",
            )

        with (
            patch("ash_hawk.improve.loop._run_eval", new_callable=AsyncMock) as mock_run,
            patch("ash_hawk.improve.loop.diagnose_failures", new_callable=AsyncMock) as mock_diag,
            patch(
                "ash_hawk.improve.loop.propose_patch_via_agent",
                new_callable=AsyncMock,
                side_effect=patch_side_effect,
            ),
        ):
            mock_run.side_effect = run_eval_side_effect
            mock_diag.return_value = [diagnosis]

            result = await improve(
                "suite.yaml",
                agent_name="bolt_merlin",
                agent_path=agent_dir,
                max_iterations=2,
                eval_repeats=1,
                integrity_repeats=2,
                score_threshold=0.1,
                lessons_dir=tmp_path / "lessons",
                output_dir=tmp_path,
            )

        assert result.final_pass_rate == 1.0
        assert result.iterations == 2
        assert result.patches_applied == ["prompt.md"]
        assert result.mutation_history[0]["kept"] is True
        assert result.mutation_history[0]["applied_files"] == ["prompt.md"]
        assert target_file.read_text(encoding="utf-8") == "better"
        assert mock_run.await_count == 5

    @pytest.mark.asyncio
    async def test_no_op_agent_edit_is_rejected_without_re_evaluation(self, tmp_path):
        agent_dir = _init_git_agent_repo(tmp_path)
        target_file = agent_dir / "prompt.md"

        diagnosis = Diagnosis(
            trial_id="trial-0",
            failure_summary="bad prompt",
            root_cause="prompt needs update",
            suggested_fix="update prompt",
            target_files=["prompt.md"],
            confidence=0.9,
        )

        with (
            patch("ash_hawk.improve.loop._run_eval", new_callable=AsyncMock) as mock_run,
            patch("ash_hawk.improve.loop.diagnose_failures", new_callable=AsyncMock) as mock_diag,
            patch(
                "ash_hawk.improve.loop.propose_patch_via_agent",
                new_callable=AsyncMock,
                return_value=ProposedPatch(
                    diagnosis=diagnosis,
                    file_path="prompt.md",
                    description="update prompt",
                    rationale="prompt needs update",
                    agent_relative_path="(agent-edited)",
                    content="claimed change",
                ),
            ),
        ):
            mock_run.return_value = _make_mock_summary(0.0, n_trials=1)
            mock_diag.return_value = [diagnosis]

            result = await improve(
                "suite.yaml",
                agent_name="bolt_merlin",
                agent_path=agent_dir,
                max_iterations=1,
                eval_repeats=1,
                integrity_repeats=2,
                score_threshold=0.1,
                lessons_dir=tmp_path / "lessons",
                output_dir=tmp_path,
            )

        assert result.final_pass_rate == 0.0
        assert result.patches_applied == []
        assert result.mutation_history == []
        assert target_file.read_text(encoding="utf-8") == "bad"
        assert mock_run.await_count == 1

    @pytest.mark.asyncio
    async def test_console_clarifies_outer_pass_and_single_hypothesis(self, tmp_path):
        diagnosis = Diagnosis(
            trial_id="trial-0",
            failure_summary="bad prompt",
            root_cause="prompt needs update",
            suggested_fix="update prompt",
            target_files=["prompt.md"],
            confidence=0.9,
        )
        output = StringIO()
        console = Console(file=output, force_terminal=False)

        with (
            patch("ash_hawk.improve.loop._run_eval", new_callable=AsyncMock) as mock_run,
            patch("ash_hawk.improve.loop.diagnose_failures", new_callable=AsyncMock) as mock_diag,
            patch("ash_hawk.improve.loop.propose_patch", new_callable=AsyncMock),
        ):
            mock_run.return_value = _make_mock_summary(0.0, n_trials=1)
            mock_diag.return_value = [diagnosis]

            await improve(
                "suite.yaml",
                max_iterations=1,
                lessons_dir=tmp_path / "lessons",
                output_dir=tmp_path,
                eval_repeats=1,
                console=console,
            )

        rendered = output.getvalue()
        assert "Outer pass 1/1" in rendered
        assert "runs the full suite" in rendered
        assert "One failing trial produced one diagnosis" in rendered

    @pytest.mark.asyncio
    async def test_non_actionable_diagnosis_stops_without_testing_hypothesis(self, tmp_path):
        diagnosis = Diagnosis(
            trial_id="trial-0",
            failure_summary="trial timed out",
            root_cause="LLM diagnosis unavailable",
            suggested_fix="review manually",
            target_files=[],
            confidence=0.1,
            actionable=False,
            diagnosis_mode="fallback_llm_unavailable",
            degraded_reason="diagnosis_llm_unavailable",
        )
        output = StringIO()
        console = Console(file=output, force_terminal=False)

        with (
            patch("ash_hawk.improve.loop._run_eval", new_callable=AsyncMock) as mock_run,
            patch("ash_hawk.improve.loop.diagnose_failures", new_callable=AsyncMock) as mock_diag,
            patch("ash_hawk.improve.loop.propose_patch", new_callable=AsyncMock) as mock_patch,
        ):
            mock_run.return_value = _make_mock_summary(0.0, n_trials=1)
            mock_diag.return_value = [diagnosis]

            result = await improve(
                "suite.yaml",
                max_iterations=2,
                lessons_dir=tmp_path / "lessons",
                output_dir=tmp_path,
                eval_repeats=1,
                console=console,
            )

        rendered = output.getvalue()
        assert "Non-actionable diagnoses: 1" in rendered
        assert "No actionable diagnoses this pass" in rendered
        assert result.stop_reasons == ["no_actionable_diagnoses"]
        assert result.mutation_history == []
        assert mock_patch.await_count == 0

        iter_log = (tmp_path / "iter-000.json").read_text(encoding="utf-8")
        assert '"hypothesis_ranked": 0' in iter_log
        assert '"hypothesis_outcome": "no_actionable_diagnoses"' in iter_log
