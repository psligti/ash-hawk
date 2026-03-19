from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ash_hawk.cycle import (
    ConvergenceChecker,
    ConvergenceStatus,
    CycleConfig,
    CycleResult,
    CycleStatus,
    IterationResult,
    create_cycle_id,
)


class TestConvergenceChecker:
    def test_improving_trend(self) -> None:
        config = CycleConfig(
            cycle_id="test",
            experiment_id="exp-1",
            target_agent="bolt-merlin",
            min_score_improvement=0.05,
        )
        checker = ConvergenceChecker(config)

        for score in [0.3, 0.4, 0.5, 0.6]:
            checker.add_score(score)

        assert checker.check_convergence() == ConvergenceStatus.IMPROVING
        assert checker.get_best_score() == 0.6
        assert checker.get_latest_score() == 0.6

    def test_converged_plateau(self) -> None:
        config = CycleConfig(
            cycle_id="test",
            experiment_id="exp-1",
            target_agent="bolt-merlin",
            convergence_threshold=0.01,
        )
        checker = ConvergenceChecker(config)

        for _ in range(5):
            checker.add_score(0.5)

        assert checker.check_convergence() == ConvergenceStatus.CONVERGED

    def test_regressing(self) -> None:
        config = CycleConfig(
            cycle_id="test",
            experiment_id="exp-1",
            target_agent="bolt-merlin",
            min_score_improvement=0.01,
            convergence_window=3,
        )
        checker = ConvergenceChecker(config)

        for score in [0.7, 0.7, 0.7, 0.5, 0.4, 0.3]:
            checker.add_score(score)

        assert checker.check_convergence() == ConvergenceStatus.REGRESSING

    def test_should_promote_lessons(self) -> None:
        config = CycleConfig(
            cycle_id="test",
            experiment_id="exp-1",
            target_agent="bolt-merlin",
            promotion_success_threshold=2,
            min_score_improvement=0.05,
        )
        checker = ConvergenceChecker(config)

        for score in [0.3, 0.4, 0.5]:
            checker.add_score(score)

        assert checker.should_promote_lessons() is True
        assert checker.get_consecutive_improvements() == 2

    def test_reset_improvement_counter(self) -> None:
        config = CycleConfig(
            cycle_id="test",
            experiment_id="exp-1",
            target_agent="bolt-merlin",
            promotion_success_threshold=3,
            min_score_improvement=0.05,
        )
        checker = ConvergenceChecker(config)

        for score in [0.3, 0.4]:
            checker.add_score(score)

        assert checker.get_consecutive_improvements() == 1

        checker.add_score(0.44)

        assert checker.get_consecutive_improvements() == 0
        checker.reset_improvement_counter()
        assert checker.get_consecutive_improvements() == 0


class TestCycleConfig:
    def test_default_values(self) -> None:
        config = CycleConfig(
            cycle_id="test",
            experiment_id="exp-1",
            target_agent="bolt-merlin",
        )

        assert config.max_iterations == 100
        assert config.convergence_threshold == 0.01
        assert config.convergence_window == 5
        assert config.stop_on_convergence is False
        assert config.promotion_success_threshold == 3
        assert config.min_score_improvement == 0.02

    def test_custom_values(self) -> None:
        config = CycleConfig(
            cycle_id="test",
            experiment_id="exp-1",
            target_agent="bolt-merlin",
            max_iterations=50,
            convergence_threshold=0.05,
            stop_on_convergence=True,
            promotion_success_threshold=5,
        )

        assert config.max_iterations == 50
        assert config.convergence_threshold == 0.05
        assert config.stop_on_convergence is True
        assert config.promotion_success_threshold == 5


class TestIterationResult:
    def test_duration_calculation(self) -> None:
        started = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        completed = started + timedelta(seconds=30)

        result = IterationResult(
            iteration_num=1,
            run_artifact_id="run-1",
            score=0.5,
            status=CycleStatus.COMPLETED,
            started_at=started,
            completed_at=completed,
        )

        assert result.duration_seconds() == 30.0

    def test_duration_none_if_not_completed(self) -> None:
        result = IterationResult(
            iteration_num=1,
            run_artifact_id="run-1",
            score=0.5,
            status=CycleStatus.RUNNING,
            started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        )

        assert result.duration_seconds() is None


class TestCycleResult:
    def test_improvement_delta(self) -> None:
        config = CycleConfig(
            cycle_id="test",
            experiment_id="exp-1",
            target_agent="bolt-merlin",
        )
        result = CycleResult(
            cycle_id="test",
            config=config,
            total_iterations=5,
            initial_score=0.3,
            final_score=0.7,
        )

        improvement = result.model_dump()["improvement_delta"]
        assert isinstance(improvement, float)
        assert abs(improvement - 0.4) < 1e-9

    def test_negative_improvement(self) -> None:
        config = CycleConfig(
            cycle_id="test",
            experiment_id="exp-1",
            target_agent="bolt-merlin",
        )
        result = CycleResult(
            cycle_id="test",
            config=config,
            total_iterations=5,
            initial_score=0.7,
            final_score=0.3,
        )

        improvement = result.model_dump()["improvement_delta"]
        assert isinstance(improvement, float)
        assert abs(improvement + 0.4) < 1e-9


class TestCreateCycleId:
    def test_generates_unique_ids(self) -> None:
        ids = [create_cycle_id() for _ in range(10)]
        assert len(ids) == len(set(ids))
        assert all(id.startswith("cycle-") for id in ids)
