from __future__ import annotations

import os
import subprocess
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from rich.console import Console

from ash_hawk.improve.diagnose import Diagnosis
from ash_hawk.improve.loop import (
    ImprovementResult,
    _resolve_adapter_override,
    _run_eval_n_times,
    improve,
)
from ash_hawk.improve.patch import ProposedPatch


def _git_test_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
        env.pop(key, None)
    return env


def _make_mock_summary(
    pass_rate: float,
    n_trials: int = 2,
    *,
    mean_score: float | None = None,
):
    from ash_hawk.types import (
        EvalOutcome,
        EvalRunSummary,
        EvalStatus,
        EvalTrial,
        RunEnvelope,
        SuiteMetrics,
        TrialResult,
    )

    aggregate_score = pass_rate if mean_score is None else mean_score
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
                    aggregate_score=1.0 if passed else aggregate_score,
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
            mean_score=aggregate_score,
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
    def test_resolve_adapter_override_normalizes_hyphenated_names(self) -> None:
        class StubRegistry:
            def get(self, name: str) -> object | None:
                if name == "bolt_merlin":
                    return object()
                return None

        with patch(
            "ash_hawk.scenario.registry.get_default_adapter_registry", return_value=StubRegistry()
        ):
            assert _resolve_adapter_override("bolt-merlin") == "bolt_merlin"

    def test_resolve_adapter_override_returns_none_when_not_registered(self) -> None:
        class StubRegistry:
            def get(self, name: str) -> object | None:
                return None

        with patch(
            "ash_hawk.scenario.registry.get_default_adapter_registry", return_value=StubRegistry()
        ):
            assert _resolve_adapter_override("custom-agent") is None

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
    async def test_partial_scores_drive_improve_metric_even_when_pass_rate_is_zero(self, tmp_path):
        with (
            patch("ash_hawk.improve.loop._run_eval", new_callable=AsyncMock) as mock_run,
            patch("ash_hawk.improve.loop.diagnose_failures", new_callable=AsyncMock),
        ):
            mock_run.return_value = _make_mock_summary(0.0, n_trials=1, mean_score=0.6944)
            result = await improve(
                "suite.yaml",
                max_iterations=5,
                output_dir=tmp_path,
                eval_repeats=1,
                target=0.6,
            )

        assert result.initial_pass_rate == pytest.approx(0.6944)
        assert result.final_pass_rate == pytest.approx(0.6944)
        assert result.convergence_achieved is True
        mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_eval_n_times_uses_median_for_multi_run_selection(self):
        summaries = [
            _make_mock_summary(0.0, n_trials=1, mean_score=0.8222),
            _make_mock_summary(0.0, n_trials=1, mean_score=0.7778),
            _make_mock_summary(0.0, n_trials=1, mean_score=0.7667),
        ]

        with patch("ash_hawk.improve.loop._run_eval", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = summaries
            stats, last_summary, errors = await _run_eval_n_times(
                "suite.yaml",
                "agent",
                300.0,
                None,
                3,
            )

        assert stats is not None
        assert stats.selected_score == pytest.approx(0.7778)
        assert stats.mean_score == pytest.approx((0.8222 + 0.7778 + 0.7667) / 3)
        assert stats.min_score == pytest.approx(0.7667)
        assert stats.max_score == pytest.approx(0.8222)
        assert last_summary is summaries[-1]
        assert errors == []

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
    async def test_kept_hypotheses_continue_from_updated_baseline(self, tmp_path):
        agent_dir = _init_git_agent_repo(tmp_path)
        second_file = agent_dir / "config.md"
        second_file.write_text("bad", encoding="utf-8")

        first = Diagnosis(
            trial_id="trial-0",
            failure_summary="bad prompt",
            root_cause="prompt needs update",
            suggested_fix="update prompt",
            target_files=["prompt.md"],
            confidence=0.9,
        )
        second = Diagnosis(
            trial_id="trial-1",
            failure_summary="bad config",
            root_cause="config needs update",
            suggested_fix="update config",
            target_files=["config.md"],
            confidence=0.8,
        )

        async def run_eval_side_effect(_suite_path, _agent_name, _timeout, agent_path):
            prompt_value = (Path(agent_path) / "prompt.md").read_text(encoding="utf-8")
            config_value = (Path(agent_path) / "config.md").read_text(encoding="utf-8")
            score = 0.0
            if prompt_value == "better":
                score += 0.5
            if config_value == "better":
                score += 0.5
            return _make_mock_summary(0.0, n_trials=1, mean_score=score)

        async def patch_side_effect(*args, **kwargs):
            agent_source_path = Path(args[1])
            diagnosis = args[0]
            target = diagnosis.target_files[0]
            (agent_source_path / target).write_text("better", encoding="utf-8")
            return ProposedPatch(
                diagnosis=diagnosis,
                file_path=target,
                description=f"update {target}",
                rationale=diagnosis.root_cause,
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
            mock_diag.return_value = [first, second]

            result = await improve(
                "suite.yaml",
                agent_name="bolt_merlin",
                agent_path=agent_dir,
                max_iterations=1,
                eval_repeats=1,
                integrity_repeats=1,
                score_threshold=0.1,
                lessons_dir=tmp_path / "lessons",
                output_dir=tmp_path,
            )

        assert result.final_pass_rate == pytest.approx(1.0)
        assert sorted(result.patches_applied) == ["config.md", "prompt.md"]
        assert len(result.mutation_history) == 2
        assert result.mutation_history[0]["baseline_score_before"] == pytest.approx(0.0)
        assert result.mutation_history[0]["selected_score_after"] == pytest.approx(0.5)
        assert result.mutation_history[1]["baseline_score_before"] == pytest.approx(0.5)
        assert result.mutation_history[1]["selected_score_after"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_integrity_validation_reverts_mean_flattering_mutation(self, tmp_path):
        agent_dir = _init_git_agent_repo(tmp_path)

        diagnosis = Diagnosis(
            trial_id="trial-0",
            failure_summary="bad prompt",
            root_cause="prompt needs update",
            suggested_fix="update prompt",
            target_files=["prompt.md"],
            confidence=0.9,
        )

        eval_summaries = [
            _make_mock_summary(0.0, n_trials=1, mean_score=0.7798),
            _make_mock_summary(0.0, n_trials=1, mean_score=0.9000),
            _make_mock_summary(0.0, n_trials=1, mean_score=0.8222),
            _make_mock_summary(0.0, n_trials=1, mean_score=0.7778),
            _make_mock_summary(0.0, n_trials=1, mean_score=0.7667),
        ]

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
            mock_run.side_effect = eval_summaries
            mock_diag.return_value = [diagnosis]

            result = await improve(
                "suite.yaml",
                agent_name="bolt_merlin",
                agent_path=agent_dir,
                max_iterations=1,
                eval_repeats=1,
                integrity_repeats=3,
                score_threshold=0.005,
                lessons_dir=tmp_path / "lessons",
                output_dir=tmp_path,
            )

        assert result.final_pass_rate == pytest.approx(0.7798)
        assert len(result.mutation_history) == 1
        assert result.mutation_history[0]["kept"] is False
        assert result.mutation_history[0]["selected_score_after"] == pytest.approx(0.7778)
        assert result.mutation_history[0]["mean_score_after"] == pytest.approx(0.7889)

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
