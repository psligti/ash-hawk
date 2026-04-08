from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ash_hawk.auto_research.cycle_runner import (
    CycleConfig,
    CycleResult,
    CycleStatus,
    run_cycle,
)
from ash_hawk.improve.loop import ImprovementResult
from ash_hawk.improvement.guardrails import GuardrailConfig


def _make_improvement_result(
    initial: float = 0.5,
    final: float = 0.8,
    iterations: int = 3,
    convergence_achieved: bool = False,
    mutations: list[dict] | None = None,
) -> ImprovementResult:
    if mutations is None:
        mutations = [
            {
                "iteration": 0,
                "mean_pass_rate_before": 0.5,
                "mean_pass_rate_after": 0.6,
                "improvement": 0.1,
                "kept": True,
                "lesson_id": "les-001",
            },
            {
                "iteration": 1,
                "mean_pass_rate_before": 0.6,
                "mean_pass_rate_after": 0.55,
                "improvement": -0.05,
                "kept": False,
                "lesson_id": "les-002",
            },
            {
                "iteration": 2,
                "mean_pass_rate_before": 0.6,
                "mean_pass_rate_after": 0.8,
                "improvement": 0.2,
                "kept": True,
                "lesson_id": "les-003",
            },
        ]
    return ImprovementResult(
        iterations=iterations,
        initial_pass_rate=initial,
        final_pass_rate=final,
        patches_proposed=[],
        patches_applied=[],
        trace_path=None,
        mutation_history=mutations,
        convergence_achieved=convergence_achieved,
    )


_IMPROVE_PATH = "ash_hawk.improve.loop.improve"


class TestRunCycleBasic:
    @pytest.mark.asyncio
    async def test_basic_cycle_completes(self, tmp_path: Path) -> None:
        mock_result = _make_improvement_result(iterations=2)

        with patch(_IMPROVE_PATH, new_callable=AsyncMock) as mock_improve:
            mock_improve.return_value = mock_result

            cfg = CycleConfig(
                max_iterations=2,
                eval_repeats=1,
                lessons_dir=tmp_path / "lessons",
                output_dir=tmp_path / "output",
            )
            result = await run_cycle(
                suite_path="suite.yaml",
                agent_name="test-agent",
                config=cfg,
            )

        assert isinstance(result, CycleResult)
        assert result.cycle_id.startswith("cycle-")
        assert result.agent_name == "test-agent"
        assert result.total_iterations == 2
        assert result.applied_count == 2
        assert result.reverted_count == 1

    @pytest.mark.asyncio
    async def test_single_scenario_file(self, tmp_path: Path) -> None:
        suite_file = tmp_path / "test.yaml"
        suite_file.write_text("test: data")

        mock_result = _make_improvement_result(
            initial=1.0, final=1.0, convergence_achieved=True, mutations=[]
        )

        with patch(_IMPROVE_PATH, new_callable=AsyncMock) as mock_improve:
            mock_improve.return_value = mock_result

            cfg = CycleConfig(
                max_iterations=1,
                eval_repeats=1,
                lessons_dir=tmp_path / "lessons",
                output_dir=tmp_path / "output",
            )
            result = await run_cycle(
                suite_path=str(suite_file),
                config=cfg,
            )

        assert result.train_scenarios == [str(suite_file)]
        assert result.holdout_scenarios == []

    @pytest.mark.asyncio
    async def test_convergence_target_reached(self, tmp_path: Path) -> None:
        mock_result = _make_improvement_result(
            initial=0.5,
            final=1.0,
            convergence_achieved=True,
            mutations=[
                {
                    "iteration": 0,
                    "mean_pass_rate_before": 0.5,
                    "mean_pass_rate_after": 1.0,
                    "improvement": 0.5,
                    "kept": True,
                    "lesson_id": "les-100",
                },
            ],
        )

        with patch(_IMPROVE_PATH, new_callable=AsyncMock) as mock_improve:
            mock_improve.return_value = mock_result

            cfg = CycleConfig(
                max_iterations=5,
                eval_repeats=1,
                target_pass_rate=1.0,
                lessons_dir=tmp_path / "lessons",
            )
            result = await run_cycle(
                suite_path="suite.yaml",
                config=cfg,
            )

        assert result.status == CycleStatus.COMPLETED
        assert result.final_score == 1.0


class TestGuardrailStop:
    @pytest.mark.asyncio
    async def test_guardrail_stops_on_max_reverts(self, tmp_path: Path) -> None:
        mutations = [
            {
                "iteration": i,
                "mean_pass_rate_before": 0.5,
                "mean_pass_rate_after": 0.45,
                "improvement": -0.05,
                "kept": False,
                "lesson_id": f"les-{i:03d}",
            }
            for i in range(3)
        ]
        mock_result = _make_improvement_result(
            initial=0.5, final=0.45, iterations=3, mutations=mutations
        )

        with patch(_IMPROVE_PATH, new_callable=AsyncMock) as mock_improve:
            mock_improve.return_value = mock_result

            guardrail_cfg = GuardrailConfig(max_reverts=2)
            cfg = CycleConfig(
                max_iterations=5,
                eval_repeats=1,
                guardrail_config=guardrail_cfg,
                lessons_dir=tmp_path / "lessons",
            )
            result = await run_cycle(
                suite_path="suite.yaml",
                config=cfg,
            )

        assert result.status == CycleStatus.GUARDRAIL_STOPPED
        assert result.guardrail_reason is not None
        assert result.reverted_count >= 2


class TestConvergenceStop:
    @pytest.mark.asyncio
    async def test_convergence_no_improvement(self, tmp_path: Path) -> None:
        flat_mutations = [
            {
                "iteration": i,
                "mean_pass_rate_before": 0.5,
                "mean_pass_rate_after": 0.5,
                "improvement": 0.001,
                "kept": True,
                "lesson_id": f"les-{i:03d}",
            }
            for i in range(12)
        ]
        mock_result = _make_improvement_result(
            initial=0.5, final=0.5, iterations=1, mutations=flat_mutations
        )

        with patch(_IMPROVE_PATH, new_callable=AsyncMock) as mock_improve:
            mock_improve.return_value = mock_result

            cfg = CycleConfig(
                max_iterations=1,
                eval_repeats=1,
                max_iterations_without_improvement=2,
                convergence_variance_threshold=0.01,
                lessons_dir=tmp_path / "lessons",
            )
            result = await run_cycle(
                suite_path="suite.yaml",
                config=cfg,
            )

        assert result.convergence_result is not None
        assert result.convergence_result.converged is True


class TestCycleResultProperties:
    def test_improvement_delta(self) -> None:
        result = CycleResult(
            cycle_id="test",
            agent_name="build",
            status=CycleStatus.COMPLETED,
            initial_score=0.4,
            final_score=0.8,
        )
        assert result.improvement_delta == pytest.approx(0.4)

    def test_improvement_delta_negative(self) -> None:
        result = CycleResult(
            cycle_id="test",
            agent_name="build",
            status=CycleStatus.GUARDRAIL_STOPPED,
            initial_score=0.7,
            final_score=0.3,
        )
        assert result.improvement_delta == pytest.approx(-0.4)

    @pytest.mark.parametrize(
        "status,expected",
        [
            (CycleStatus.COMPLETED, True),
            (CycleStatus.CONVERGED, True),
            (CycleStatus.GUARDRAIL_STOPPED, False),
            (CycleStatus.ERROR, False),
            (CycleStatus.RUNNING, False),
            (CycleStatus.PENDING, False),
        ],
    )
    def test_success_property(self, status: CycleStatus, expected: bool) -> None:
        result = CycleResult(
            cycle_id="test",
            agent_name="build",
            status=status,
        )
        assert result.success is expected


class TestCycleResultFields:
    @pytest.mark.asyncio
    async def test_result_fields_populated(self, tmp_path: Path) -> None:
        mock_result = _make_improvement_result(
            initial=1.0, final=1.0, convergence_achieved=True, mutations=[]
        )

        with patch(_IMPROVE_PATH, new_callable=AsyncMock) as mock_improve:
            mock_improve.return_value = mock_result

            cfg = CycleConfig(
                max_iterations=1,
                eval_repeats=1,
                lessons_dir=tmp_path / "lessons",
                output_dir=tmp_path / "output",
            )
            result = await run_cycle(
                suite_path="suite.yaml",
                agent_name="build",
                config=cfg,
            )

        assert result.cycle_id.startswith("cycle-")
        assert result.agent_name == "build"
        assert result.duration_seconds >= 0.0
        assert result.status in (
            CycleStatus.COMPLETED,
            CycleStatus.CONVERGED,
            CycleStatus.GUARDRAIL_STOPPED,
        )
        assert isinstance(result.train_scenarios, list)
        assert isinstance(result.holdout_scenarios, list)
        assert isinstance(result.applied_count, int)
        assert isinstance(result.reverted_count, int)
        assert isinstance(result.promoted_lessons, int)
