from __future__ import annotations

import json
from pathlib import Path

import pytest

from ash_hawk.auto_research.types import IterationResult
from ash_hawk.research.strategy_promoter import (
    PromotedStrategy,
    StrategyPattern,
    StrategyPromoter,
)


class TestStrategyPromoter:
    def test_strategy_pattern_success_rate(self) -> None:
        pattern = StrategyPattern(
            pattern_id="pattern-1",
            name="pattern",
            description="desc",
            trigger_condition="trigger",
            action="action",
            success_count=2,
            total_applications=4,
        )

        assert pattern.success_rate == 0.5

    def test_promoted_strategy_creation(self) -> None:
        strategy = PromotedStrategy(
            strategy_id="strategy-1",
            source_pattern_id="pattern-1",
            name="name",
            description="desc",
            trigger_condition="trigger",
            action_template="action",
            success_rate=0.8,
        )

        assert strategy.strategy_id == "strategy-1"
        assert strategy.success_rate == 0.8

    def test_detect_patterns_empty_iterations(self) -> None:
        promoter = StrategyPromoter()

        patterns = promoter.detect_patterns([])

        assert patterns == []

    def test_detect_patterns_groups_by_improvement_text(self) -> None:
        promoter = StrategyPromoter()
        iterations = [
            IterationResult(
                iteration_num=1,
                score_before=0.5,
                score_after=0.6,
                improvement_text="Improve prompt for accuracy",
                applied=True,
                category_scores={"prompt": 0.1, "tool": 0.2},
            ),
            IterationResult(
                iteration_num=2,
                score_before=0.6,
                score_after=0.7,
                improvement_text="Improve prompt for accuracy",
                applied=True,
                category_scores={"prompt": 0.2},
            ),
        ]

        patterns = promoter.detect_patterns(iterations)

        assert len(patterns) == 1
        pattern = patterns[0]
        assert pattern.total_applications == 2
        assert pattern.success_count == 2
        assert set(pattern.affected_surfaces) == {"prompt", "tool"}

    def test_should_promote_insufficient_success_count(self) -> None:
        promoter = StrategyPromoter()
        pattern = StrategyPattern(
            pattern_id="pattern-1",
            name="pattern",
            description="desc",
            trigger_condition="trigger",
            action="action",
            success_count=2,
            total_applications=2,
        )

        assert promoter.should_promote(pattern) is False

    def test_should_promote_low_success_rate(self) -> None:
        promoter = StrategyPromoter()
        pattern = StrategyPattern(
            pattern_id="pattern-1",
            name="pattern",
            description="desc",
            trigger_condition="trigger",
            action="action",
            success_count=3,
            total_applications=5,
        )

        assert promoter.should_promote(pattern) is False

    def test_should_promote_meets_criteria(self) -> None:
        promoter = StrategyPromoter()
        pattern = StrategyPattern(
            pattern_id="pattern-1",
            name="pattern",
            description="desc",
            trigger_condition="trigger",
            action="action",
            success_count=3,
            total_applications=4,
        )

        assert promoter.should_promote(pattern) is True

    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        promoter = StrategyPromoter(storage_path=tmp_path)
        iterations = [
            IterationResult(
                iteration_num=1,
                score_before=0.5,
                score_after=0.6,
                improvement_text="Improve prompt for accuracy",
                applied=True,
                category_scores={"prompt": 0.1},
            ),
            IterationResult(
                iteration_num=2,
                score_before=0.6,
                score_after=0.7,
                improvement_text="Improve prompt for accuracy",
                applied=True,
                category_scores={"prompt": 0.2},
            ),
        ]
        promoter.detect_patterns(iterations)

        await promoter.save()
        loaded = StrategyPromoter.load(tmp_path)

        assert len(loaded.get_candidate_patterns()) == 1
        assert loaded.get_promoted_strategies() == []

    def test_get_candidate_patterns(self) -> None:
        promoter = StrategyPromoter()
        iterations = [
            IterationResult(
                iteration_num=1,
                score_before=0.5,
                score_after=0.6,
                improvement_text="Use structured prompts",
                applied=True,
            ),
            IterationResult(
                iteration_num=2,
                score_before=0.6,
                score_after=0.7,
                improvement_text="Use structured prompts",
                applied=True,
            ),
        ]
        promoter.detect_patterns(iterations)

        candidates = promoter.get_candidate_patterns()

        assert len(candidates) == 1

    def test_get_promoted_strategies(self, tmp_path: Path) -> None:
        payload: dict[str, object] = {
            "patterns": {},
            "promoted": {
                "strategy-1": {
                    "strategy_id": "strategy-1",
                    "source_pattern_id": "pattern-1",
                    "name": "name",
                    "description": "desc",
                    "trigger_condition": "trigger",
                    "action_template": "action",
                    "success_rate": 0.8,
                    "affected_surfaces": ["prompt"],
                    "promoted_at": "",
                    "artifact_path": None,
                }
            },
        }
        file_path = tmp_path / "strategies.json"
        file_path.write_text(json.dumps(payload))

        promoter = StrategyPromoter.load(tmp_path)
        promoted = promoter.get_promoted_strategies()

        assert [strategy.strategy_id for strategy in promoted] == ["strategy-1"]
