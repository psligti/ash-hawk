"""Iron Rook agent-specific evaluator pack for PR review quality."""

from __future__ import annotations

from ash_hawk.eval_packs.base import EvalPack, EvalPackConfig
from ash_hawk.strategies import Strategy, SubStrategy

IronRookEvalPack = EvalPack(
    pack_id="iron-rook-eval",
    name="Iron Rook PR Review Evaluation Pack",
    description="Evaluates PR review quality for iron-rook agent focusing on policy compliance, change precision, and evidence quality",
    target_agents=["iron-rook"],
    grader_configs={
        "policy-compliance": EvalPackConfig(
            grader_name="deterministic",
            weight=0.25,
            pass_threshold=0.85,
            params={"check_type": "policy_compliance"},
        ),
        "change-precision": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.25,
            pass_threshold=0.75,
            params={
                "rubric": "Evaluate whether feedback is precise, actionable, and correctly identifies issues",
            },
        ),
        "evidence-quality": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.20,
            pass_threshold=0.70,
            params={
                "rubric": "Evaluate quality of code citations, references, and supporting evidence for feedback",
            },
        ),
        "review-completeness": EvalPackConfig(
            grader_name="deterministic",
            weight=0.15,
            pass_threshold=0.80,
            params={"check_type": "review_coverage"},
        ),
        "task-completion": EvalPackConfig(
            grader_name="deterministic",
            weight=0.15,
            pass_threshold=0.85,
            params={"check_type": "task_completion"},
        ),
    },
    strategy_focus=[
        Strategy.POLICY_QUALITY,
        Strategy.AGENT_BEHAVIOR,
    ],
    global_pass_threshold=0.75,
    metadata={
        "category": "pr-review",
        "agent_type": "policy-agent",
        "sub_strategies": [
            SubStrategy.EVIDENCE_QUALITY,
            SubStrategy.CHANGE_PRECISION,
            SubStrategy.TASK_COMPLETION,
        ],
    },
)
