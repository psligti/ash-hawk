from __future__ import annotations

from datetime import UTC, datetime

from ash_hawk.research.types import (
    CauseCategory,
    ResearchAction,
    ResearchDecision,
    ResearchLoopConfig,
    ResearchLoopResult,
    TargetSurface,
)


class TestResearchTypes:
    def test_research_action_values(self) -> None:
        values = {action.value for action in ResearchAction}

        assert values == {
            "fix",
            "observe",
            "experiment",
            "evaluate",
            "restructure",
            "promote",
        }

    def test_cause_category_values(self) -> None:
        values = {category.value for category in CauseCategory}

        assert values == {
            "prompt_quality",
            "tool_misuse",
            "context_overflow",
            "delegation_failure",
            "orchestration_branch",
            "timeout_misallocation",
            "unknown",
        }

    def test_target_surface_values(self) -> None:
        values = {surface.value for surface in TargetSurface}

        assert values == {
            "prompt",
            "policy",
            "tool",
            "delegation",
            "orchestration",
            "eval_question",
        }

    def test_research_loop_config_defaults(self) -> None:
        config = ResearchLoopConfig()

        assert config.iterations == 10
        assert config.uncertainty_threshold == 0.6
        assert config.d_step_interval == 5
        assert config.prune_interval == 3
        assert config.max_diagnoses_per_run == 50
        assert config.human_approval_required is True
        assert str(config.storage_path) == ".ash-hawk/research"
        assert config.min_active_targets == 3

    def test_research_loop_result_properties(self) -> None:
        decisions = [
            ResearchDecision(
                action=ResearchAction.OBSERVE,
                rationale="observe",
                target=None,
                expected_info_gain=0.1,
                confidence=0.5,
            ),
            ResearchDecision(
                action=ResearchAction.FIX,
                rationale="fix",
                target=None,
                expected_info_gain=0.2,
                confidence=0.8,
            ),
        ]
        result = ResearchLoopResult(decisions=decisions)

        assert result.total_decisions == 2
        assert result.observe_vs_fix_ratio == 1.0

    def test_research_loop_result_ratio_handles_zero_fixes(self) -> None:
        decisions = [
            ResearchDecision(
                action=ResearchAction.OBSERVE,
                rationale="observe",
                target=None,
                expected_info_gain=0.1,
                confidence=0.5,
            )
        ]
        result = ResearchLoopResult(decisions=decisions)

        assert result.observe_vs_fix_ratio == 0.0

    def test_research_decision_dataclass(self) -> None:
        decision = ResearchDecision(
            action=ResearchAction.FIX,
            rationale="tune prompt",
            target="prompt",
            expected_info_gain=0.4,
            confidence=0.9,
        )

        assert decision.action is ResearchAction.FIX
        assert decision.rationale == "tune prompt"
        assert decision.target == "prompt"
        assert decision.expected_info_gain == 0.4
        assert decision.confidence == 0.9
        assert isinstance(decision.timestamp, datetime)
        assert decision.timestamp.tzinfo == UTC
