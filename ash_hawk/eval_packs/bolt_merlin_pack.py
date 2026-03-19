"""Bolt Merlin agent-specific evaluator pack for execution quality."""

from __future__ import annotations

from ash_hawk.eval_packs.base import EvalPack, EvalPackConfig
from ash_hawk.strategies import Strategy, SubStrategy

BoltMerlinEvalPack = EvalPack(
    pack_id="bolt-merlin-eval",
    name="Bolt Merlin Execution Evaluation Pack",
    description="Evaluates execution quality for bolt-merlin agent focusing on tool efficiency, error recovery, and task completion",
    target_agents=["bolt-merlin"],
    grader_configs={
        "tool-efficiency": EvalPackConfig(
            grader_name="deterministic",
            weight=0.20,
            pass_threshold=0.75,
            params={"check_type": "tool_efficiency"},
        ),
        "error-recovery": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.20,
            pass_threshold=0.70,
            params={
                "rubric": "Evaluate how well the agent recovered from tool errors and failures",
            },
        ),
        "task-completion": EvalPackConfig(
            grader_name="deterministic",
            weight=0.25,
            pass_threshold=0.85,
            params={"check_type": "task_completion"},
        ),
        "change-precision": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.20,
            pass_threshold=0.75,
            params={
                "rubric": "Evaluate whether code changes are precise, minimal, and correct",
            },
        ),
        "retry-behavior": EvalPackConfig(
            grader_name="deterministic",
            weight=0.15,
            pass_threshold=0.70,
            params={"check_type": "retry_analysis"},
        ),
    },
    strategy_focus=[
        Strategy.TOOL_QUALITY,
        Strategy.SKILL_QUALITY,
    ],
    global_pass_threshold=0.75,
    metadata={
        "category": "execution",
        "agent_type": "execution-agent",
        "sub_strategies": [
            SubStrategy.TOOL_EFFICIENCY,
            SubStrategy.ERROR_RECOVERY,
            SubStrategy.RETRY_BEHAVIOR,
            SubStrategy.CHANGE_PRECISION,
        ],
    },
)
