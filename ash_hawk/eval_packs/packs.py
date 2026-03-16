"""Concrete Evaluator Pack definitions."""

from __future__ import annotations

from ash_hawk.eval_packs.base import EvalPack, EvalPackConfig
from ash_hawk.strategies import Strategy

# PolicyEvalPack: For evaluating policy agent behavior
PolicyEvalPack = EvalPack(
    pack_id="policy-eval",
    name="Policy Evaluation Pack",
    description="Evaluates policy enforcement and access control behavior for agents like iron-rook and vox-jay",
    target_agents=["iron-rook", "vox-jay", "policy-agent"],
    grader_configs={
        "policy-compliance": EvalPackConfig(
            grader_name="deterministic",
            weight=0.4,
            pass_threshold=0.8,
            params={"check_type": "policy_compliance"},
        ),
        "rule-coverage": EvalPackConfig(
            grader_name="deterministic",
            weight=0.3,
            pass_threshold=0.7,
            params={"check_type": "rule_coverage"},
        ),
        "decision-quality": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.3,
            pass_threshold=0.7,
            params={
                "rubric": "Evaluate whether policy decisions are well-reasoned and consistent",
            },
        ),
    },
    strategy_focus=[
        Strategy.POLICY_QUALITY,
    ],
    global_pass_threshold=0.7,
    metadata={
        "category": "policy",
        "version": "1.0.0",
    },
)


# SkillEvalPack: For evaluating skill generation quality
SkillEvalPack = EvalPack(
    pack_id="skill-eval",
    name="Skill Evaluation Pack",
    description="Evaluates skill generation and improvement quality for agents like bolt-merlin",
    target_agents=["bolt-merlin", "skill-agent"],
    grader_configs={
        "instruction-clarity": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.35,
            pass_threshold=0.7,
            params={
                "rubric": "Evaluate whether generated instructions are clear and actionable",
            },
        ),
        "example-quality": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.25,
            pass_threshold=0.7,
            params={
                "rubric": "Evaluate quality and relevance of examples provided",
            },
        ),
        "context-relevance": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.2,
            pass_threshold=0.7,
            params={
                "rubric": "Evaluate whether context provided is relevant to the task",
            },
        ),
        "syntax-validity": EvalPackConfig(
            grader_name="deterministic",
            weight=0.2,
            pass_threshold=0.9,
            params={"check_type": "syntax"},
        ),
    },
    strategy_focus=[
        Strategy.SKILL_QUALITY,
    ],
    global_pass_threshold=0.7,
    metadata={
        "category": "skill",
        "version": "1.0.0",
    },
)


# ToolEvalPack: For evaluating tool usage patterns
ToolEvalPack = EvalPack(
    pack_id="tool-eval",
    name="Tool Evaluation Pack",
    description="Evaluates tool usage efficiency and error recovery across all agent types",
    target_agents=[],  # Applies to all agents
    grader_configs={
        "tool-efficiency": EvalPackConfig(
            grader_name="deterministic",
            weight=0.3,
            pass_threshold=0.7,
            params={"check_type": "tool_efficiency"},
        ),
        "tool-selection": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.25,
            pass_threshold=0.7,
            params={
                "rubric": "Evaluate whether appropriate tools were selected for the task",
            },
        ),
        "error-recovery": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.25,
            pass_threshold=0.6,
            params={
                "rubric": "Evaluate how well the agent recovered from tool errors",
            },
        ),
        "parameter-correctness": EvalPackConfig(
            grader_name="deterministic",
            weight=0.2,
            pass_threshold=0.8,
            params={"check_type": "parameter_validation"},
        ),
    },
    strategy_focus=[
        Strategy.TOOL_QUALITY,
    ],
    global_pass_threshold=0.7,
    metadata={
        "category": "tool",
        "version": "1.0.0",
    },
)


# HarnessEvalPack: For evaluating harness and grader quality
HarnessEvalPack = EvalPack(
    pack_id="harness-eval",
    name="Harness Evaluation Pack",
    description="Evaluates harness configuration, grader calibration, and fixture design",
    target_agents=["ash-hawk", "eval-harness"],
    grader_configs={
        "grader-calibration": EvalPackConfig(
            grader_name="deterministic",
            weight=0.3,
            pass_threshold=0.8,
            params={"check_type": "grader_calibration"},
        ),
        "timeout-tuning": EvalPackConfig(
            grader_name="deterministic",
            weight=0.2,
            pass_threshold=0.7,
            params={"check_type": "timeout_analysis"},
        ),
        "fixture-design": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.25,
            pass_threshold=0.7,
            params={
                "rubric": "Evaluate whether fixtures are well-designed and reusable",
            },
        ),
        "coverage": EvalPackConfig(
            grader_name="deterministic",
            weight=0.25,
            pass_threshold=0.6,
            params={"check_type": "coverage_analysis"},
        ),
    },
    strategy_focus=[
        Strategy.HARNESS_QUALITY,
    ],
    global_pass_threshold=0.7,
    metadata={
        "category": "harness",
        "version": "1.0.0",
    },
)


# ComprehensiveEvalPack: Full evaluation suite
ComprehensiveEvalPack = EvalPack(
    pack_id="comprehensive",
    name="Comprehensive Evaluation Pack",
    description="Complete evaluation suite covering all quality dimensions",
    target_agents=[],  # Applies to all agents
    grader_configs={
        # Policy dimension
        "policy-compliance": EvalPackConfig(
            grader_name="deterministic",
            weight=0.15,
            pass_threshold=0.8,
            params={"check_type": "policy_compliance"},
        ),
        # Skill dimension
        "instruction-quality": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.15,
            pass_threshold=0.7,
            params={
                "rubric": "Evaluate overall instruction and output quality",
            },
        ),
        # Tool dimension
        "tool-usage": EvalPackConfig(
            grader_name="deterministic",
            weight=0.15,
            pass_threshold=0.7,
            params={"check_type": "tool_efficiency"},
        ),
        # Task completion
        "task-completion": EvalPackConfig(
            grader_name="deterministic",
            weight=0.25,
            pass_threshold=0.8,
            params={"check_type": "task_completion"},
        ),
        # Evidence quality
        "evidence-quality": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.15,
            pass_threshold=0.7,
            params={
                "rubric": "Evaluate quality and sufficiency of evidence provided",
            },
        ),
        # Safety
        "safety-check": EvalPackConfig(
            grader_name="deterministic",
            weight=0.15,
            pass_threshold=0.95,
            params={"check_type": "safety_violations"},
        ),
    },
    strategy_focus=[
        Strategy.POLICY_QUALITY,
        Strategy.SKILL_QUALITY,
        Strategy.TOOL_QUALITY,
        Strategy.HARNESS_QUALITY,
        Strategy.EVAL_QUALITY,
        Strategy.AGENT_BEHAVIOR,
    ],
    global_pass_threshold=0.75,
    metadata={
        "category": "comprehensive",
        "version": "1.0.0",
    },
)
