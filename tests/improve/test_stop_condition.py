from __future__ import annotations

import pytest

from ash_hawk.improve.stop_condition import (
    ScoreRecord,
    StopCondition,
    StopConditionConfig,
    StopResult,
)


def _record(
    iteration: int = 0,
    score: float = 0.5,
    applied: bool = True,
    delta: float = 0.1,
    mutation_outcome: str = "applied",
) -> ScoreRecord:
    return ScoreRecord(
        iteration=iteration,
        score=score,
        applied=applied,
        delta=delta,
        mutation_outcome=mutation_outcome,
    )


class TestScoreRecord:
    def test_valid_record(self):
        r = _record()
        assert r.iteration == 0
        assert r.score == 0.5
        assert r.applied is True
        assert r.delta == 0.1

    def test_extra_fields_rejected(self):
        with pytest.raises(Exception):
            ScoreRecord(iteration=0, score=0.5, applied=True, delta=0.0, extra="bad")


class TestStopResult:
    def test_no_stop(self):
        r = StopResult(should_stop=False)
        assert not r.should_stop
        assert r.reasons == []

    def test_with_reasons(self):
        r = StopResult(should_stop=True, reasons=["plateau"])
        assert r.should_stop
        assert "plateau" in r.reasons


class TestStopConditionMaxReverts:
    def test_no_stop_under_limit(self):
        sc = StopCondition(StopConditionConfig(max_reverts=3))
        for i in range(2):
            result = sc.record(_record(iteration=i, applied=False, delta=-0.1))
        assert not result.should_stop

    def test_stops_at_limit(self):
        sc = StopCondition(StopConditionConfig(max_reverts=3))
        for i in range(3):
            result = sc.record(_record(iteration=i, applied=False, delta=-0.1))
        assert result.should_stop
        assert any("reverts" in r for r in result.reasons)

    def test_noop_mutations_do_not_count_toward_revert_limit(self):
        sc = StopCondition(StopConditionConfig(max_reverts=2, max_consecutive_noop_mutations=99))
        sc.record(
            _record(iteration=0, applied=False, delta=0.0, mutation_outcome="mutation_cli_timeout")
        )
        result = sc.record(
            _record(iteration=1, applied=False, delta=0.0, mutation_outcome="no_file_changes")
        )
        assert not result.should_stop

        result = sc.record(
            _record(iteration=2, applied=False, delta=0.0, mutation_outcome="targeted_regression")
        )
        assert not result.should_stop

        result = sc.record(
            _record(iteration=3, applied=False, delta=0.0, mutation_outcome="targeted_regression")
        )
        assert result.should_stop
        assert any("Tested reverts" in r for r in result.reasons)


class TestStopConditionRegression:
    def test_no_stop_few_regressions(self):
        sc = StopCondition(StopConditionConfig(max_consecutive_regressions=3))
        for i in range(2):
            result = sc.record(_record(iteration=i, applied=False, delta=-0.1))
        assert not result.should_stop

    def test_stops_on_consecutive_regressions(self):
        sc = StopCondition(StopConditionConfig(max_consecutive_regressions=3))
        sc.record(_record(iteration=0, applied=True, delta=0.1))
        for i in range(1, 4):
            result = sc.record(_record(iteration=i, applied=False, delta=-0.1))
        assert result.should_stop
        assert any("regressed" in r for r in result.reasons)

    def test_mixed_deltas_no_stop(self):
        sc = StopCondition(StopConditionConfig(max_consecutive_regressions=3))
        sc.record(_record(iteration=0, delta=-0.1))
        sc.record(_record(iteration=1, delta=0.05))
        result = sc.record(_record(iteration=2, delta=-0.1))
        assert not result.should_stop


class TestStopConditionPlateau:
    def test_no_stop_insufficient_history(self):
        sc = StopCondition(StopConditionConfig(convergence_window=5, variance_threshold=0.001))
        for i in range(4):
            result = sc.record(_record(iteration=i, score=0.5))
        assert not result.should_stop

    def test_stops_on_plateau(self):
        sc = StopCondition(StopConditionConfig(convergence_window=5, variance_threshold=0.001))
        for i in range(5):
            result = sc.record(_record(iteration=i, score=0.5001))
        assert result.should_stop
        assert any("plateau" in r.lower() for r in result.reasons)

    def test_no_stop_with_variance(self):
        sc = StopCondition(StopConditionConfig(convergence_window=5, variance_threshold=0.001))
        for i in range(5):
            result = sc.record(_record(iteration=i, score=0.3 + i * 0.1))
        assert not result.should_stop


class TestStopConditionNoImprovement:
    def test_stops_when_stale(self):
        sc = StopCondition(StopConditionConfig(max_iterations_without_improvement=3))
        sc.record(_record(iteration=0, score=0.8, applied=True, delta=0.1))
        for i in range(1, 4):
            result = sc.record(_record(iteration=i, score=0.7, applied=False, delta=-0.1))
        assert result.should_stop
        assert any("improvement" in r for r in result.reasons)

    def test_resets_on_new_best(self):
        sc = StopCondition(StopConditionConfig(max_iterations_without_improvement=3))
        sc.record(_record(iteration=0, score=0.5, applied=True, delta=0.1))
        sc.record(_record(iteration=1, score=0.4, applied=False, delta=-0.1))
        sc.record(_record(iteration=2, score=0.4, applied=False, delta=0.0))
        result = sc.record(_record(iteration=3, score=0.9, applied=True, delta=0.5))
        assert not result.should_stop


class TestStopConditionReset:
    def test_reset_clears_history(self):
        sc = StopCondition(StopConditionConfig(max_reverts=2))
        sc.record(_record(applied=False))
        sc.reset()
        result = sc.record(_record(applied=False))
        assert not result.should_stop


class TestStopConditionMutationStall:
    def test_stops_on_consecutive_noop_mutations(self):
        sc = StopCondition(StopConditionConfig(max_consecutive_noop_mutations=3, max_reverts=99))
        sc.record(
            _record(iteration=0, applied=False, delta=0.0, mutation_outcome="no_file_changes")
        )
        sc.record(
            _record(iteration=1, applied=False, delta=0.0, mutation_outcome="mutation_cli_timeout")
        )
        result = sc.record(
            _record(
                iteration=2, applied=False, delta=0.0, mutation_outcome="post_mutation_eval_failed"
            )
        )

        assert result.should_stop
        assert any("stalled" in reason for reason in result.reasons)

    def test_targeted_regression_does_not_count_as_stall(self):
        sc = StopCondition(StopConditionConfig(max_consecutive_noop_mutations=3, max_reverts=99))
        sc.record(
            _record(iteration=0, applied=False, delta=0.0, mutation_outcome="mutation_cli_timeout")
        )
        sc.record(
            _record(
                iteration=1,
                applied=False,
                delta=0.0,
                mutation_outcome="targeted_regression",
            )
        )
        result = sc.record(
            _record(iteration=2, applied=False, delta=0.0, mutation_outcome="mutation_cli_timeout")
        )

        assert not result.should_stop


class TestStopConditionMutationYield:
    def test_does_not_check_yield_until_sample_size_reached(self):
        sc = StopCondition(
            StopConditionConfig(min_mutation_yield=0.5, min_yield_sample_size=4, max_reverts=99)
        )
        sc.record(
            _record(iteration=0, applied=False, delta=0.0, mutation_outcome="no_file_changes")
        )
        sc.record(
            _record(iteration=1, applied=False, delta=0.0, mutation_outcome="no_file_changes")
        )
        result = sc.record(
            _record(iteration=2, applied=False, delta=0.0, mutation_outcome="no_file_changes")
        )

        assert not result.should_stop

    def test_stops_when_yield_below_threshold(self):
        sc = StopCondition(
            StopConditionConfig(min_mutation_yield=0.5, min_yield_sample_size=4, max_reverts=99)
        )
        sc.record(_record(iteration=0, applied=True, delta=0.1, mutation_outcome="applied"))
        sc.record(
            _record(iteration=1, applied=False, delta=0.0, mutation_outcome="targeted_regression")
        )
        sc.record(
            _record(iteration=2, applied=False, delta=0.0, mutation_outcome="targeted_regression")
        )
        result = sc.record(
            _record(iteration=3, applied=False, delta=0.0, mutation_outcome="targeted_regression")
        )

        assert result.should_stop
        assert any("yield" in reason.lower() for reason in result.reasons)

    def test_yield_disabled_by_default(self):
        sc = StopCondition(StopConditionConfig(max_reverts=99))
        for i in range(8):
            result = sc.record(
                _record(
                    iteration=i, applied=False, delta=0.0, mutation_outcome="targeted_regression"
                )
            )

        assert not any("yield" in reason.lower() for reason in result.reasons)

    def test_applied_mutation_resets_noop_stall_window(self):
        sc = StopCondition(StopConditionConfig(max_consecutive_noop_mutations=3, max_reverts=99))
        sc.record(
            _record(iteration=0, applied=False, delta=0.0, mutation_outcome="no_file_changes")
        )
        sc.record(_record(iteration=1, applied=True, delta=0.2, mutation_outcome="applied"))
        result = sc.record(
            _record(iteration=2, applied=False, delta=0.0, mutation_outcome="no_file_changes")
        )

        assert not result.should_stop
