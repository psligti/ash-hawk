from __future__ import annotations

import json
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
    _acceptance_decision,
    _build_phase2_gate_report,
    _compute_phase2_metrics,
    _filter_actionable_diagnoses_to_agent_source,
    _is_retry_eligible,
    _mutation_timeout_seconds,
    _resolve_adapter_override,
    _retry_timeout_seconds,
    _run_eval,
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


def _make_scenario_summary(items: list[tuple[str, float, bool]]):
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
    total_score = 0.0
    passed_tasks = 0
    for idx, (scenario_path, score, passed) in enumerate(items):
        total_score += score
        passed_tasks += 1 if passed else 0
        trials.append(
            EvalTrial(
                id=f"trial-{idx}",
                task_id=f"task-{idx}",
                status=EvalStatus.COMPLETED,
                attempt_number=1,
                input_snapshot={"scenario_path": scenario_path},
                result=TrialResult(
                    trial_id=f"trial-{idx}",
                    outcome=EvalOutcome.success()
                    if passed
                    else EvalOutcome.failure("agent_error", "failed"),
                    aggregate_passed=passed,
                    aggregate_score=score,
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
            total_tasks=len(items),
            completed_tasks=len(items),
            passed_tasks=passed_tasks,
            failed_tasks=len(items) - passed_tasks,
            pass_rate=passed_tasks / len(items),
            mean_score=total_score / len(items),
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


def _git_stdout(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=_git_test_env(),
    )
    return result.stdout.strip()


class TestImprove:
    def test_acceptance_decision_keeps_positive_delta_with_tolerable_regression(self) -> None:
        decision = _acceptance_decision(
            delta=0.0944,
            regressions=[
                {
                    "scenario_path": "/tmp/other.scenario.yaml",
                    "reason": "score_drop",
                    "score_drop": 0.03,
                    "before_passed": True,
                    "after_passed": True,
                    "before_score": 1.0,
                    "after_score": 0.97,
                }
            ],
            score_threshold=0.02,
        )

        assert decision.kept is True
        assert decision.net_benefit == pytest.approx(0.0794)
        assert len(decision.tolerable_regressions) == 1
        assert decision.intolerable_regressions == []

    def test_acceptance_decision_rejects_intolerable_pass_to_fail(self) -> None:
        decision = _acceptance_decision(
            delta=0.0944,
            regressions=[
                {
                    "scenario_path": "/tmp/other.scenario.yaml",
                    "reason": "pass_to_fail",
                    "score_drop": 0.1667,
                    "before_passed": True,
                    "after_passed": False,
                    "before_score": 1.0,
                    "after_score": 0.8333,
                }
            ],
            score_threshold=0.02,
        )

        assert decision.kept is False
        assert decision.rejection_reason == "intolerable_regression"
        assert len(decision.intolerable_regressions) == 1

    def test_acceptance_decision_rejects_large_score_drop(self) -> None:
        decision = _acceptance_decision(
            delta=0.03,
            regressions=[
                {
                    "scenario_path": "/tmp/other.scenario.yaml",
                    "reason": "score_drop",
                    "score_drop": 0.08,
                    "before_passed": True,
                    "after_passed": True,
                    "before_score": 1.0,
                    "after_score": 0.92,
                }
            ],
            score_threshold=0.02,
        )

        assert decision.kept is False
        assert decision.rejection_reason == "intolerable_regression"

    def test_acceptance_decision_rejects_net_negative_from_many_small_regressions(self) -> None:
        decision = _acceptance_decision(
            delta=0.03,
            regressions=[
                {
                    "scenario_path": "/tmp/a.scenario.yaml",
                    "reason": "score_drop",
                    "score_drop": 0.04,
                    "before_passed": True,
                    "after_passed": True,
                    "before_score": 1.0,
                    "after_score": 0.96,
                },
                {
                    "scenario_path": "/tmp/b.scenario.yaml",
                    "reason": "score_drop",
                    "score_drop": 0.04,
                    "before_passed": True,
                    "after_passed": True,
                    "before_score": 1.0,
                    "after_score": 0.96,
                },
            ],
            score_threshold=0.02,
        )

        assert decision.kept is False
        assert decision.rejection_reason == "net_negative"

    def test_filter_actionable_diagnoses_to_agent_source(self) -> None:
        agent_diag = Diagnosis(
            trial_id="trial-agent",
            failure_summary="agent bug",
            root_cause="agent prompt issue",
            suggested_fix="fix prompt",
            target_files=["prompts/coding.md"],
            confidence=0.8,
        )
        external_diag = Diagnosis(
            trial_id="trial-external",
            failure_summary="grader bug",
            root_cause="repo diff mismatch",
            suggested_fix="fix grader",
            target_files=["graders/repo_diff.py"],
            confidence=0.9,
        )

        kept, filtered = _filter_actionable_diagnoses_to_agent_source(
            [agent_diag, external_diag],
            {"prompts/coding.md": "...", "tools/edit.py": "..."},
        )

        assert [d.trial_id for d in kept] == ["trial-agent"]
        assert [d.trial_id for d in filtered] == ["trial-external"]

    def test_mutation_timeout_seconds_is_scaled_and_clamped(self) -> None:
        assert _mutation_timeout_seconds(300.0) == pytest.approx(360.0)
        assert _mutation_timeout_seconds(900.0) == pytest.approx(360.0)
        assert _mutation_timeout_seconds(60.0) == pytest.approx(120.0)

    def test_retry_timeout_seconds_uses_relaxed_cap(self) -> None:
        assert _retry_timeout_seconds(300.0) == pytest.approx(450.0)
        assert _retry_timeout_seconds(60.0) == pytest.approx(120.0)

    def test_retry_eligibility_requires_transient_and_activity(self) -> None:
        assert (
            _is_retry_eligible(
                "mutation_cli_timeout",
                changed_paths=["prompt.md"],
                mutation_llm_calls=0,
                retry_count=0,
            )
            is True
        )
        assert (
            _is_retry_eligible(
                "post_mutation_eval_failed",
                changed_paths=[],
                mutation_llm_calls=2,
                retry_count=0,
            )
            is True
        )
        assert (
            _is_retry_eligible(
                "post_mutation_eval_failed",
                changed_paths=[],
                mutation_llm_calls=0,
                retry_count=0,
            )
            is False
        )
        assert (
            _is_retry_eligible(
                "mutation_cli_timeout",
                changed_paths=["prompt.md"],
                mutation_llm_calls=0,
                retry_count=1,
            )
            is False
        )

    def test_compute_phase2_metrics_aggregates_expected_fields(self) -> None:
        metrics = _compute_phase2_metrics(
            initial_score=0.4,
            final_score=0.55,
            initial_pass_rate=0.6,
            final_pass_rate=0.5,
            mutation_history=[
                {
                    "kept": True,
                    "rejection_reason": None,
                    "net_benefit": 0.15,
                    "mutation_wall_seconds": 12.0,
                    "mutation_llm_calls": 3,
                },
                {
                    "kept": False,
                    "rejection_reason": "targeted_regression",
                    "net_benefit": -0.02,
                    "mutation_wall_seconds": 8.0,
                    "mutation_llm_calls": 2,
                },
                {
                    "kept": False,
                    "rejection_reason": "mutation_cli_timeout",
                    "net_benefit": -0.01,
                    "mutation_wall_seconds": 5.0,
                    "mutation_llm_calls": 1,
                },
            ],
            iteration_logs=[
                {"diagnoses": [{"trial": "a"}, {"trial": "b"}], "hypothesis_ranked": 2},
                {"diagnoses": [{"trial": "c"}], "hypothesis_ranked": 1},
            ],
            phase1_reviews=[
                {"suspicious": True, "failure_bucket": "bad_retry"},
                {"suspicious": False, "failure_bucket": "wrong_path"},
            ],
            stop_reasons=["plateau"],
        )

        assert metrics["tested_count"] == 3
        assert metrics["kept_count"] == 1
        assert metrics["keep_rate"] == pytest.approx(1 / 3)
        assert metrics["timeout_count"] == 1
        assert metrics["delta_score"] == pytest.approx(0.15)
        assert metrics["delta_pass_rate"] == pytest.approx(-0.1)
        assert metrics["phase1_suspicious_rate"] == pytest.approx(0.5)
        assert metrics["funnel"]["generated"] == 3
        assert metrics["funnel"]["ranked"] == 3
        assert metrics["funnel"]["attempted"] == 3
        assert metrics["funnel"]["kept"] == 1
        assert metrics["score_pass_rate_divergence"] is True

    def test_phase2_gate_report_fails_when_window_underperforms(self, tmp_path: Path) -> None:
        improve_runs = tmp_path / "improve-runs"
        improve_runs.mkdir()
        config = {
            "suite_paths": ["/tmp/a.yaml"],
            "agent_name": "build",
            "score_threshold": 0.02,
            "eval_repeats": 1,
            "integrity_repeats": 3,
        }
        for idx, delta in enumerate([0.0, 0.005]):
            run_dir = improve_runs / f"improve-old{idx}"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "initial_score": 0.5,
                        "final_score": 0.5 + delta,
                        "phase2_metrics": {
                            "delta_score": delta,
                            "timeout_rate": 0.5,
                            "keep_rate": 0.0,
                            "net_benefit_total": -0.1,
                            "kept_count": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )

        current_dir = improve_runs / "improve-current"
        current_dir.mkdir()
        (current_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

        gate = _build_phase2_gate_report(
            improve_runs_root=improve_runs,
            current_config=config,
            current_run_id="improve-current",
            current_initial_score=0.5,
            current_final_score=0.505,
            current_phase2_metrics={
                "delta_score": 0.005,
                "timeout_rate": 0.5,
                "keep_rate": 0.0,
                "net_benefit_total": -0.05,
                "kept_count": 0,
            },
        )

        assert gate["status"] == "fail"
        assert gate["window_size"] == 3
        assert gate["cohort_comparability"]["cohort_size"] == 3
        assert gate["cohort_comparability"]["total_runs"] == 3

    @pytest.mark.asyncio
    async def test_run_eval_passes_progress_callback_when_console_present(self) -> None:
        captured: dict[str, object] = {}
        summary = _make_mock_summary(1.0, n_trials=1, mean_score=1.0)

        async def fake_run_scenarios_async(paths, **kwargs):
            captured["paths"] = paths
            captured["on_trial_progress"] = kwargs.get("on_trial_progress")
            callback = kwargs.get("on_trial_progress")
            if callback is not None:
                await callback(1, 0, 1, "task-0")
            return summary

        with patch(
            "ash_hawk.scenario.runner.run_scenarios_async", side_effect=fake_run_scenarios_async
        ):
            result = await _run_eval(
                ["/tmp/a.scenario.yaml"],
                "bolt_merlin",
                30.0,
                console=Console(file=StringIO(), force_terminal=False),
                phase_label="baseline evaluation",
            )

        assert result == summary
        assert captured["paths"] == [str(Path("/tmp/a.scenario.yaml").resolve())]
        assert captured["on_trial_progress"] is not None

    @pytest.mark.asyncio
    async def test_improve_resolves_relative_suite_paths_before_eval(self, tmp_path: Path) -> None:
        scenario_path = tmp_path / "demo.scenario.yaml"
        scenario_path.write_text(
            "id: demo\ndescription: demo\nsut:\n  adapter: bolt_merlin\ninputs: {}\ngraders: []\nbudgets:\n  max_time_seconds: 60\n",
            encoding="utf-8",
        )
        relative_path = os.path.relpath(scenario_path, Path.cwd())

        captured_paths: list[list[str]] = []

        async def side_effect(paths, *args, **kwargs):
            captured_paths.append(list(paths))
            return _make_mock_summary(1.0, n_trials=1)

        with (
            patch("ash_hawk.improve.loop._run_eval", new_callable=AsyncMock) as mock_run,
            patch("ash_hawk.improve.loop.diagnose_failures", new_callable=AsyncMock),
        ):
            mock_run.side_effect = side_effect
            await improve(relative_path, max_iterations=1, output_dir=tmp_path, eval_repeats=1)

        assert captured_paths == [[str(scenario_path.resolve())]]

    @pytest.mark.asyncio
    async def test_improve_artifacts_survive_cwd_shift_during_eval(self, tmp_path: Path) -> None:
        original_cwd = Path.cwd()
        os.chdir(tmp_path)
        try:
            agent_dir = _init_git_agent_repo(tmp_path)
            shifted_dir = tmp_path / "shifted-cwd"
            shifted_dir.mkdir()

            async def side_effect(*args, **kwargs):
                os.chdir(shifted_dir)
                return _make_mock_summary(1.0, n_trials=1)

            with patch("ash_hawk.improve.loop._run_eval", new_callable=AsyncMock) as mock_run:
                mock_run.side_effect = side_effect
                result = await improve(
                    "suite.yaml",
                    agent_name="bolt_merlin",
                    agent_path=agent_dir,
                    max_iterations=1,
                    eval_repeats=1,
                    integrity_repeats=2,
                )

            assert result.final_score == 1.0
            assert result.final_pass_rate == 1.0
            improve_runs = list((tmp_path / ".ash-hawk" / "improve-runs").glob("improve-*"))
            assert len(improve_runs) == 1
            run_dir = improve_runs[0]
            assert (run_dir / "config.json").exists()
            assert (run_dir / "run.json").exists()
            assert (run_dir / "validations" / "iter-000" / "baseline" / "repeat-1.json").exists()
            assert (tmp_path / ".ash-hawk" / "improve" / "iter-000.json").exists()
        finally:
            os.chdir(original_cwd)

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

        assert result.initial_score == 1.0
        assert result.final_score == 1.0
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

        assert result.final_score == 1.0
        assert result.final_pass_rate == 1.0
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_all_eval_failures_record_memory_episode(self, tmp_path):
        async def always_fail(*args, **kwargs):
            raise RuntimeError("eval failed")

        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch("ash_hawk.improve.loop._run_eval", side_effect=always_fail):
                result = await improve(
                    "suite.yaml", max_iterations=1, output_dir=tmp_path, eval_repeats=1
                )

            assert result.final_score == 0.0
            memory_dir = tmp_path / ".ash-hawk" / "memory" / "episodic"
            episode_files = list(memory_dir.glob("*.jsonl"))
            assert len(episode_files) == 1
            content = episode_files[0].read_text(encoding="utf-8")
            assert '"outcome": "baseline_eval_failed"' in content
        finally:
            os.chdir(original_cwd)

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
        assert result.initial_score == 1.0
        assert result.initial_pass_rate == 1.0
        assert result.patches_applied == []
        assert result.trace_path == Path("/tmp/traces")

        run_dirs = list((tmp_path / "improve-runs").glob("*"))
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]
        assert (run_dir / "config.json").exists()
        assert (run_dir / "run.json").exists()
        assert (run_dir / "summary.md").exists()
        assert (run_dir / "events.jsonl").exists()

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

        assert result.initial_score == pytest.approx(0.6944)
        assert result.final_score == pytest.approx(0.6944)
        assert result.initial_pass_rate == pytest.approx(0.0)
        assert result.final_pass_rate == pytest.approx(0.0)
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
        assert stats.score_stddev == pytest.approx(0.0239787406)
        assert stats.pass_rate_stddev == pytest.approx(0.0)
        assert last_summary is summaries[-1]
        assert errors == []

    @pytest.mark.asyncio
    async def test_kept_mutation_is_evaluated_in_workspace_and_synced_back(self, tmp_path):
        agent_dir = _init_git_agent_repo(tmp_path)
        repo_root = tmp_path
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
                execution_metrics={"llm_completion_count": 3},
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

        assert result.final_score == 1.0
        assert result.final_pass_rate == 1.0
        assert result.iterations == 2
        assert result.patches_applied == ["prompt.md"]
        assert result.mutation_history[0]["kept"] is True
        assert result.mutation_history[0]["applied_files"] == ["prompt.md"]
        assert result.mutation_history[0]["mutation_llm_calls"] == 3
        assert result.mutation_history[0]["mutation_wall_seconds"] >= 0.0
        current_branch = _git_stdout(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
        assert result.mutation_history[0]["git_branch"] == current_branch
        assert result.mutation_history[0]["git_commit"]
        assert result.mutation_history[0]["git_commit_paths"] == ["bolt_merlin/agent/prompt.md"]
        assert result.mutation_history[0]["git_commit_title"].startswith(
            "fix: improve trial-0 via lesson "
        )
        assert target_file.read_text(encoding="utf-8") == "better"
        assert mock_run.await_count == 3

        assert _git_stdout(repo_root, "rev-list", "--count", "HEAD") == "2"
        head_subject = _git_stdout(repo_root, "log", "-1", "--pretty=%s")
        head_body = _git_stdout(repo_root, "log", "-1", "--pretty=%b")
        assert head_subject == result.mutation_history[0]["git_commit_title"]
        assert "Trial: trial-0" in head_body
        assert "Lesson: " in head_body
        assert "Applied files: prompt.md" in head_body
        assert result.mutation_history[0]["causality_kind"] == "direct"
        assert result.mutation_history[0]["diagnosis_family"] == "unknown"

        run_dir = next((tmp_path / "improve-runs").glob("*"))
        assert (run_dir / "commits" / "iter-000" / "rank-1.json").exists()
        assert (run_dir / "hypotheses" / "iter-000" / "rank-1.json").exists()
        assert (run_dir / "iteration_logs" / "iter-000.json").exists()
        summary_text = (run_dir / "summary.md").read_text(encoding="utf-8")
        assert "Mutation wall-clock (tested):" in summary_text
        assert "Mutation LLM calls (tested):" in summary_text
        assert "Phase timing (all iterations):" in summary_text
        assert "mutation_generation=" in summary_text
        assert "fast_validation=" in summary_text
        assert "Keep rate:" in summary_text
        assert "Timeout rate:" in summary_text
        assert "Net benefit:" in summary_text
        assert "Mutation funnel:" in summary_text
        assert "Progress ledger:" in summary_text
        assert "⚠ Divergence:" in summary_text
        assert "Phase 2 cohort comparability:" in summary_text
        assert "Phase 2 gate (latest comparable window):" in summary_text
        assert "Memory episodes (this run):" in summary_text
        assert "Memory semantic rules:" in summary_text
        assert "Personal memory preferences:" in summary_text
        assert "Memory conversion skips:" in summary_text
        assert "Memory backfilled episodes:" in summary_text
        assert "Memory observability:" in summary_text
        iter_log = json.loads((run_dir / "iteration_logs" / "iter-000.json").read_text())
        assert iter_log["hypothesis_outcome"] == "kept"
        assert iter_log["kept"] is True
        assert iter_log["mutation_llm_calls"] == 3
        assert iter_log["mutation_wall_seconds"] >= 0.0
        assert iter_log["phase_durations"]["baseline_eval"] >= 0.0
        assert iter_log["phase_durations"]["diagnosis"] >= 0.0
        assert iter_log["phase_durations"]["mutation_generation"] >= 0.0
        assert iter_log["phase_durations"]["fast_validation"] >= 0.0
        run_json = json.loads((run_dir / "run.json").read_text())
        assert run_json["mutation_history"][0]["git_commit"]
        assert run_json["mutation_history"][0]["causality_kind"] == "direct"
        assert run_json["memory_summary"]["episodes_this_run"] >= 1
        assert "observability" in run_json["memory_summary"]
        assert (run_dir / "memory_summary.json").exists()

    @pytest.mark.asyncio
    async def test_kept_mutation_commit_fails_with_preexisting_staged_changes(self, tmp_path):
        agent_dir = _init_git_agent_repo(tmp_path)
        unrelated_file = tmp_path / "README.md"
        unrelated_file.write_text("draft\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "README.md"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
            env=_git_test_env(),
        )

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

            with pytest.raises(ValueError, match="pre-existing staged changes"):
                await improve(
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

        assert result.final_score == pytest.approx(1.0)
        assert result.final_pass_rate == pytest.approx(0.0)
        assert sorted(result.patches_applied) == ["config.md", "prompt.md"]
        assert len(result.mutation_history) == 2
        assert result.mutation_history[0]["baseline_score_before"] == pytest.approx(0.0)
        assert result.mutation_history[0]["selected_score_after"] == pytest.approx(0.5)
        assert result.mutation_history[1]["baseline_score_before"] == pytest.approx(0.5)
        assert result.mutation_history[1]["selected_score_after"] == pytest.approx(1.0)
        assert result.mutation_history[1]["hypothesis_rank"] == 1

    @pytest.mark.asyncio
    async def test_fast_validation_targets_only_impacted_evals(self, tmp_path):
        agent_dir = _init_git_agent_repo(tmp_path)
        scenario_a = str(tmp_path / "a.scenario.yaml")
        scenario_b = str(tmp_path / "b.scenario.yaml")
        diagnosis = Diagnosis(
            trial_id="trial-0",
            failure_summary="bad prompt",
            root_cause="prompt needs update",
            suggested_fix="update prompt",
            target_files=["prompt.md"],
            confidence=0.9,
        )

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

        baseline = _make_scenario_summary([(scenario_a, 0.0, False), (scenario_b, 1.0, True)])
        targeted = _make_scenario_summary([(scenario_a, 1.0, True)])
        integrity = _make_scenario_summary([(scenario_a, 1.0, True), (scenario_b, 1.0, True)])

        with (
            patch("ash_hawk.improve.loop._run_eval", new_callable=AsyncMock) as mock_run,
            patch("ash_hawk.improve.loop.diagnose_failures", new_callable=AsyncMock) as mock_diag,
            patch(
                "ash_hawk.improve.loop.propose_patch_via_agent",
                new_callable=AsyncMock,
                side_effect=patch_side_effect,
            ),
        ):
            mock_run.side_effect = [baseline, targeted, integrity, integrity]
            mock_diag.return_value = [diagnosis]

            result = await improve(
                [scenario_a, scenario_b],
                agent_name="bolt_merlin",
                agent_path=agent_dir,
                max_iterations=1,
                eval_repeats=1,
                integrity_repeats=2,
                score_threshold=0.05,
                lessons_dir=tmp_path / "lessons",
                output_dir=tmp_path,
            )

        called_paths = [call.args[0] for call in mock_run.await_args_list]
        assert called_paths[0] == [scenario_a, scenario_b]
        assert called_paths[1] == [scenario_a]
        assert called_paths[2] == [scenario_a, scenario_b]
        assert len(called_paths) == 3
        assert result.final_score == pytest.approx(1.0)
        assert result.final_pass_rate == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_regression_gate_reverts_unrelated_eval_regression(self, tmp_path):
        agent_dir = _init_git_agent_repo(tmp_path)
        scenario_a = str(tmp_path / "a.scenario.yaml")
        scenario_b = str(tmp_path / "b.scenario.yaml")
        diagnosis = Diagnosis(
            trial_id="trial-0",
            failure_summary="bad prompt",
            root_cause="prompt needs update",
            suggested_fix="update prompt",
            target_files=["prompt.md"],
            confidence=0.9,
        )

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

        baseline = _make_scenario_summary([(scenario_a, 0.0, False), (scenario_b, 1.0, True)])
        targeted = _make_scenario_summary([(scenario_a, 1.0, True)])
        integrity = _make_scenario_summary([(scenario_a, 1.0, True), (scenario_b, 0.0, False)])

        with (
            patch("ash_hawk.improve.loop._run_eval", new_callable=AsyncMock) as mock_run,
            patch("ash_hawk.improve.loop.diagnose_failures", new_callable=AsyncMock) as mock_diag,
            patch(
                "ash_hawk.improve.loop.propose_patch_via_agent",
                new_callable=AsyncMock,
                side_effect=patch_side_effect,
            ),
        ):
            mock_run.side_effect = [baseline, targeted, integrity]
            mock_diag.return_value = [diagnosis]

            result = await improve(
                [scenario_a, scenario_b],
                agent_name="bolt_merlin",
                agent_path=agent_dir,
                max_iterations=1,
                eval_repeats=1,
                integrity_repeats=1,
                score_threshold=0.05,
                lessons_dir=tmp_path / "lessons",
                output_dir=tmp_path,
            )

        assert len(result.mutation_history) == 1
        assert result.mutation_history[0]["kept"] is False
        assert result.mutation_history[0]["regressed_paths"]
        assert result.mutation_history[0]["regressed_paths"][0]["scenario_path"] == scenario_b

    @pytest.mark.asyncio
    async def test_regression_gate_keeps_tolerable_unrelated_score_drop(self, tmp_path):
        agent_dir = _init_git_agent_repo(tmp_path)
        scenario_a = str(tmp_path / "a.scenario.yaml")
        scenario_b = str(tmp_path / "b.scenario.yaml")
        diagnosis = Diagnosis(
            trial_id="trial-0",
            failure_summary="bad prompt",
            root_cause="prompt needs update",
            suggested_fix="update prompt",
            target_files=["prompt.md"],
            confidence=0.9,
        )

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

        baseline = _make_scenario_summary([(scenario_a, 0.0, False), (scenario_b, 1.00, True)])
        targeted = _make_scenario_summary([(scenario_a, 1.0, True)])
        integrity = _make_scenario_summary([(scenario_a, 1.0, True), (scenario_b, 0.91, True)])

        with (
            patch("ash_hawk.improve.loop._run_eval", new_callable=AsyncMock) as mock_run,
            patch("ash_hawk.improve.loop.diagnose_failures", new_callable=AsyncMock) as mock_diag,
            patch(
                "ash_hawk.improve.loop.propose_patch_via_agent",
                new_callable=AsyncMock,
                side_effect=patch_side_effect,
            ),
        ):
            mock_run.side_effect = [baseline, targeted, integrity]
            mock_diag.return_value = [diagnosis]

            result = await improve(
                [scenario_a, scenario_b],
                agent_name="bolt_merlin",
                agent_path=agent_dir,
                max_iterations=1,
                eval_repeats=1,
                integrity_repeats=1,
                score_threshold=0.05,
                lessons_dir=tmp_path / "lessons",
                output_dir=tmp_path,
            )

        assert len(result.mutation_history) == 1
        assert result.mutation_history[0]["kept"] is True
        assert result.mutation_history[0]["rejection_reason"] is None
        assert result.mutation_history[0]["tolerated_regressions"]
        assert result.mutation_history[0]["tolerated_regressions"][0]["scenario_path"] == scenario_b
        assert result.mutation_history[0]["intolerable_regressions"] == []
        assert result.mutation_history[0]["net_benefit"] == pytest.approx(0.41)

    @pytest.mark.asyncio
    async def test_clustered_failures_only_test_one_representative(self, tmp_path):
        agent_dir = _init_git_agent_repo(tmp_path)
        scenario_a = str(tmp_path / "a.scenario.yaml")
        scenario_b = str(tmp_path / "b.scenario.yaml")
        first = Diagnosis(
            trial_id="trial-0",
            failure_summary="bad auth prompt",
            root_cause="auth needs update",
            suggested_fix="update prompt",
            target_files=["prompt.md"],
            confidence=0.6,
        )
        second = Diagnosis(
            trial_id="trial-1",
            failure_summary="same auth prompt",
            root_cause="auth needs update",
            suggested_fix="update prompt again",
            target_files=["prompt.md"],
            confidence=0.9,
        )

        async def patch_side_effect(*args, **kwargs):
            agent_source_path = args[1]
            (Path(agent_source_path) / "prompt.md").write_text("better", encoding="utf-8")
            return ProposedPatch(
                diagnosis=args[0],
                file_path="prompt.md",
                description="update prompt",
                rationale="prompt needs update",
                agent_relative_path="(agent-edited)",
                content="updated in workspace",
            )

        baseline = _make_scenario_summary([(scenario_a, 0.0, False), (scenario_b, 0.0, False)])
        targeted = _make_scenario_summary([(scenario_a, 1.0, True), (scenario_b, 1.0, True)])
        integrity = _make_scenario_summary([(scenario_a, 1.0, True), (scenario_b, 1.0, True)])

        with (
            patch("ash_hawk.improve.loop._run_eval", new_callable=AsyncMock) as mock_run,
            patch("ash_hawk.improve.loop.diagnose_failures", new_callable=AsyncMock) as mock_diag,
            patch(
                "ash_hawk.improve.loop.propose_patch_via_agent",
                new_callable=AsyncMock,
                side_effect=patch_side_effect,
            ) as mock_patch,
        ):
            mock_run.side_effect = [baseline, targeted, integrity]
            mock_diag.return_value = [first, second]

            result = await improve(
                [scenario_a, scenario_b],
                agent_name="bolt_merlin",
                agent_path=agent_dir,
                max_iterations=1,
                eval_repeats=1,
                integrity_repeats=1,
                score_threshold=0.05,
                lessons_dir=tmp_path / "lessons",
                output_dir=tmp_path,
            )

        assert mock_patch.await_count == 1
        assert len(result.mutation_history) == 1

    @pytest.mark.asyncio
    async def test_integrity_validation_reverts_mean_flattering_mutation(self, tmp_path):
        agent_dir = _init_git_agent_repo(tmp_path)
        scenario_a = str(tmp_path / "a.scenario.yaml")
        scenario_b = str(tmp_path / "b.scenario.yaml")

        diagnosis = Diagnosis(
            trial_id="trial-0",
            failure_summary="bad prompt",
            root_cause="prompt needs update",
            suggested_fix="update prompt",
            target_files=["prompt.md"],
            confidence=0.9,
        )

        eval_summaries = [
            _make_scenario_summary([(scenario_a, 0.60, False), (scenario_b, 0.80, True)]),
            _make_scenario_summary([(scenario_a, 0.66, True)]),
            _make_scenario_summary([(scenario_a, 0.66, True), (scenario_b, 0.60, False)]),
            _make_scenario_summary([(scenario_a, 0.62, True), (scenario_b, 0.60, False)]),
            _make_scenario_summary([(scenario_a, 0.60, True), (scenario_b, 0.60, False)]),
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
                [scenario_a, scenario_b],
                agent_name="bolt_merlin",
                agent_path=agent_dir,
                max_iterations=1,
                eval_repeats=1,
                integrity_repeats=3,
                score_threshold=0.05,
                lessons_dir=tmp_path / "lessons",
                output_dir=tmp_path,
            )

        assert result.final_score == pytest.approx(0.70)
        assert len(result.mutation_history) == 1
        assert result.mutation_history[0]["kept"] is False
        assert result.mutation_history[0]["selected_score_after"] == pytest.approx(0.61)
        assert result.mutation_history[0]["mean_score_after"] == pytest.approx(
            (0.63 + 0.61 + 0.60) / 3
        )

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

        assert result.final_score == 0.0
        assert result.final_pass_rate == 0.0
        assert result.patches_applied == []
        assert result.mutation_history == []
        assert target_file.read_text(encoding="utf-8") == "bad"
        assert mock_run.await_count == 1

    @pytest.mark.asyncio
    async def test_changed_files_recovered_even_when_patch_output_is_unstructured(self, tmp_path):
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
            agent_source_path = Path(args[1])
            (agent_source_path / "prompt.md").write_text("better", encoding="utf-8")
            return ProposedPatch(
                diagnosis=diagnosis,
                file_path="prompt.md",
                description="recovered from changed files",
                rationale="prompt needs update",
                agent_relative_path=None,
                content=None,
                failure_reason="mutation_parse_failed",
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
                max_iterations=1,
                eval_repeats=1,
                integrity_repeats=1,
                score_threshold=0.1,
                lessons_dir=tmp_path / "lessons",
                output_dir=tmp_path,
            )

        assert result.final_score == 1.0
        assert result.final_pass_rate == 1.0
        assert result.patches_applied == ["prompt.md"]
        assert result.mutation_history[0]["kept"] is True
        assert result.mutation_history[0]["applied_files"] == ["prompt.md"]
        assert target_file.read_text(encoding="utf-8") == "better"
        assert mock_run.await_count == 2

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

    @pytest.mark.asyncio
    async def test_agent_scoped_filter_reports_no_agent_scoped_diagnoses(self, tmp_path):
        agent_dir = _init_git_agent_repo(tmp_path)
        diagnosis = Diagnosis(
            trial_id="trial-0",
            failure_summary="external docs issue",
            root_cause="needs docs outside agent source",
            suggested_fix="edit docs elsewhere",
            target_files=["docs/README.md"],
            confidence=0.8,
            actionable=True,
            diagnosis_mode="explorer",
        )
        output = StringIO()
        console = Console(file=output, force_terminal=False)

        with (
            patch("ash_hawk.improve.loop._run_eval", new_callable=AsyncMock) as mock_run,
            patch("ash_hawk.improve.loop.diagnose_failures", new_callable=AsyncMock) as mock_diag,
            patch(
                "ash_hawk.improve.loop.propose_patch_via_agent", new_callable=AsyncMock
            ) as mock_patch,
        ):
            mock_run.return_value = _make_mock_summary(0.0, n_trials=1)
            mock_diag.return_value = [diagnosis]

            result = await improve(
                "suite.yaml",
                agent_name="bolt_merlin",
                agent_path=agent_dir,
                max_iterations=1,
                lessons_dir=tmp_path / "lessons",
                output_dir=tmp_path,
                eval_repeats=1,
                console=console,
            )

        rendered = output.getvalue()
        assert "Filtered non-agent diagnoses: 1" in rendered
        assert "none targeted mutable agent files" in rendered
        assert result.stop_reasons == ["no_agent_scoped_diagnoses"]
        assert mock_patch.await_count == 0

        iter_log = (tmp_path / "iter-000.json").read_text(encoding="utf-8")
        assert '"hypothesis_outcome": "no_agent_scoped_diagnoses"' in iter_log

    def test_agent_scoped_filter_keeps_anchored_new_file(self) -> None:
        diagnosis = Diagnosis(
            trial_id="trial-0",
            failure_summary="missing helper module",
            root_cause="needs helper file anchored to dispatcher",
            suggested_fix="add helper module and wire dispatcher",
            target_files=["bolt_merlin/agent/tools/verification_retry.py"],
            anchor_files=["bolt_merlin/agent/tool_dispatcher.py"],
            confidence=0.8,
            actionable=True,
        )

        kept, filtered = _filter_actionable_diagnoses_to_agent_source(
            [diagnosis],
            {"tool_dispatcher.py": "x", "tools/edit.py": "y"},
        )

        assert len(kept) == 1
        assert filtered == []
        assert kept[0].target_files == ["tools/verification_retry.py"]
        assert kept[0].anchor_files == ["tool_dispatcher.py"]
