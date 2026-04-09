from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ash_hawk.improve.diagnose import Diagnosis
from ash_hawk.improve.hypothesis_ranker import (
    HypothesisRanker,
    HypothesisRanking,
    RankedHypothesis,
)
from ash_hawk.improve.lesson_store import Lesson


def _diagnosis(
    trial_id: str = "trial-1",
    confidence: float = 0.5,
    target_files: list[str] | None = None,
    root_cause: str = "bug",
    failure_summary: str = "failed",
) -> Diagnosis:
    return Diagnosis(
        trial_id=trial_id,
        failure_summary=failure_summary,
        root_cause=root_cause,
        suggested_fix="fix it",
        target_files=target_files or ["main.py"],
        confidence=confidence,
    )


def _make_lesson(
    target_files: list[str] | None = None,
    outcome: str = "kept",
    score_delta: float = 0.3,
) -> Lesson:
    return Lesson(
        lesson_id="l1",
        trial_id="trial-1",
        hypothesis_summary="fix",
        root_cause="bug",
        target_files=target_files or ["main.py"],
        outcome=outcome,
        score_before=0.3,
        score_after=0.6,
        score_delta=score_delta,
        iteration=1,
    )


class TestRank:
    def test_sorts_by_impact_desc(self) -> None:
        high = _diagnosis(trial_id="high", confidence=0.9, target_files=["a.py", "b.py"])
        low = _diagnosis(trial_id="low", confidence=0.3, target_files=["c.py"])

        ranker = HypothesisRanker()
        result = ranker.rank([low, high])

        assert result.hypotheses[0].diagnosis.trial_id == "high"
        assert result.hypotheses[1].diagnosis.trial_id == "low"

    def test_empty_diagnoses(self) -> None:
        ranker = HypothesisRanker()
        result = ranker.rank([])

        assert result.hypotheses == []
        assert result.total_candidates == 0
        assert result.filtered_as_tried == 0
        assert result.ranked_count == 0

    def test_assigns_ranks(self) -> None:
        d1 = _diagnosis(trial_id="a", confidence=0.8)
        d2 = _diagnosis(trial_id="b", confidence=0.5)

        ranker = HypothesisRanker()
        result = ranker.rank([d2, d1])

        assert result.hypotheses[0].rank == 1
        assert result.hypotheses[1].rank == 2

    def test_filters_tried_hypotheses(self) -> None:
        d1 = _diagnosis(trial_id="tried", confidence=0.9)
        d2 = _diagnosis(trial_id="novel", confidence=0.5)

        store = MagicMock()
        store.has_been_tried.side_effect = lambda s, r: s == "tried_summary"
        d1.failure_summary = "tried_summary"
        store.find_similar.return_value = []

        ranker = HypothesisRanker(lesson_store=store)
        result = ranker.rank([d1, d2], filter_tried=True)

        assert result.filtered_as_tried == 1
        assert result.ranked_count == 1
        assert result.hypotheses[0].diagnosis.trial_id == "novel"

    def test_no_filter_when_disabled(self) -> None:
        d1 = _diagnosis(trial_id="tried", confidence=0.9)
        d1.failure_summary = "tried_summary"

        store = MagicMock()
        store.has_been_tried.return_value = True
        store.find_similar.return_value = []

        ranker = HypothesisRanker(lesson_store=store)
        result = ranker.rank([d1], filter_tried=False)

        assert result.filtered_as_tried == 0
        assert result.ranked_count == 1


class TestComputeNovelty:
    def test_no_store_returns_max_novelty(self) -> None:
        ranker = HypothesisRanker(lesson_store=None)
        d = _diagnosis()
        score, tried, ids = ranker._compute_novelty(d)

        assert score == 1.0
        assert tried is False
        assert ids == []

    def test_already_tried_returns_zero(self) -> None:
        store = MagicMock()
        store.has_been_tried.return_value = True

        ranker = HypothesisRanker(lesson_store=store)
        d = _diagnosis()
        score, tried, ids = ranker._compute_novelty(d)

        assert score == 0.0
        assert tried is True

    def test_similar_lessons_reduces_novelty(self) -> None:
        store = MagicMock()
        store.has_been_tried.return_value = False
        lesson_mock = MagicMock()
        lesson_mock.lesson_id = "lesson-1"
        store.find_similar.return_value = [lesson_mock]

        ranker = HypothesisRanker(lesson_store=store)
        d = _diagnosis()
        score, tried, ids = ranker._compute_novelty(d)

        assert score == 0.8
        assert tried is False
        assert ids == ["lesson-1"]

    def test_store_exception_defaults_safe(self) -> None:
        store = MagicMock()
        store.has_been_tried.side_effect = RuntimeError("boom")
        store.find_similar.side_effect = RuntimeError("boom")

        ranker = HypothesisRanker(lesson_store=store)
        d = _diagnosis()
        score, tried, ids = ranker._compute_novelty(d)

        assert score == 1.0
        assert tried is False
        assert ids == []


class TestComputeImpact:
    def test_no_store_uses_confidence(self) -> None:
        ranker = HypothesisRanker()
        d = _diagnosis(confidence=0.5)
        impact = ranker._compute_impact(d)
        assert impact == pytest.approx(0.5, abs=0.01)

    def test_measured_delta_from_kept_lessons(self) -> None:
        store = MagicMock()
        store.load_for_target.return_value = [
            _make_lesson(outcome="kept", score_delta=0.4),
            _make_lesson(outcome="kept", score_delta=0.6),
        ]

        ranker = HypothesisRanker(lesson_store=store)
        d = _diagnosis(confidence=0.5)
        impact = ranker._compute_impact(d)

        assert impact == pytest.approx(0.5, abs=0.01)

    def test_all_reverted_penalizes(self) -> None:
        store = MagicMock()
        store.load_for_target.return_value = [
            _make_lesson(outcome="reverted", score_delta=-0.1),
        ]

        ranker = HypothesisRanker(lesson_store=store)
        d = _diagnosis(confidence=0.5)
        impact = ranker._compute_impact(d)

        assert impact == pytest.approx(0.3, abs=0.01)

    def test_no_past_data_uses_confidence(self) -> None:
        store = MagicMock()
        store.load_for_target.return_value = []

        ranker = HypothesisRanker(lesson_store=store)
        d = _diagnosis(confidence=0.7)
        impact = ranker._compute_impact(d)

        assert impact == pytest.approx(0.7, abs=0.01)

    def test_store_exception_uses_confidence(self) -> None:
        store = MagicMock()
        store.load_for_target.side_effect = RuntimeError("boom")

        ranker = HypothesisRanker(lesson_store=store)
        d = _diagnosis(confidence=0.8)
        impact = ranker._compute_impact(d)

        assert impact == pytest.approx(0.8, abs=0.01)


class TestEnsureDiversity:
    def test_demotes_duplicate_file(self) -> None:
        ranker = HypothesisRanker()
        d1 = _diagnosis(trial_id="a", confidence=0.9, target_files=["same.py"])
        d2 = _diagnosis(trial_id="b", confidence=0.8, target_files=["same.py"])
        d3 = _diagnosis(trial_id="c", confidence=0.7, target_files=["other.py"])

        h1 = RankedHypothesis(
            diagnosis=d1,
            rank=1,
            novelty_score=1.0,
            confidence=0.9,
            estimated_impact=0.9,
            already_tried=False,
        )
        h2 = RankedHypothesis(
            diagnosis=d2,
            rank=2,
            novelty_score=1.0,
            confidence=0.8,
            estimated_impact=0.8,
            already_tried=False,
        )
        h3 = RankedHypothesis(
            diagnosis=d3,
            rank=3,
            novelty_score=1.0,
            confidence=0.7,
            estimated_impact=0.7,
            already_tried=False,
        )

        result = ranker._ensure_diversity([h1, h2, h3])

        assert result[0].diagnosis.trial_id == "a"
        assert result[1].diagnosis.trial_id == "c"
        assert result[2].diagnosis.trial_id == "b"

    def test_no_duplicates_keeps_all(self) -> None:
        ranker = HypothesisRanker()
        d1 = _diagnosis(trial_id="a", target_files=["a.py"])
        d2 = _diagnosis(trial_id="b", target_files=["b.py"])

        h1 = RankedHypothesis(
            diagnosis=d1,
            rank=1,
            novelty_score=1.0,
            confidence=0.9,
            estimated_impact=0.9,
            already_tried=False,
        )
        h2 = RankedHypothesis(
            diagnosis=d2,
            rank=2,
            novelty_score=1.0,
            confidence=0.8,
            estimated_impact=0.8,
            already_tried=False,
        )

        result = ranker._ensure_diversity([h1, h2])
        assert len(result) == 2

    def test_empty_target_files_not_grouped(self) -> None:
        ranker = HypothesisRanker()
        d1 = _diagnosis(trial_id="a", target_files=[])
        d2 = _diagnosis(trial_id="b", target_files=[])

        h1 = RankedHypothesis(
            diagnosis=d1,
            rank=1,
            novelty_score=1.0,
            confidence=0.9,
            estimated_impact=0.9,
            already_tried=False,
        )
        h2 = RankedHypothesis(
            diagnosis=d2,
            rank=2,
            novelty_score=1.0,
            confidence=0.8,
            estimated_impact=0.8,
            already_tried=False,
        )

        result = ranker._ensure_diversity([h1, h2])
        assert len(result) == 2


class TestHypothesisRankingModel:
    def test_fields(self) -> None:
        ranking = HypothesisRanking(
            hypotheses=[],
            total_candidates=5,
            filtered_as_tried=2,
            ranked_count=3,
        )
        assert ranking.total_candidates == 5
        assert ranking.filtered_as_tried == 2
        assert ranking.ranked_count == 3
