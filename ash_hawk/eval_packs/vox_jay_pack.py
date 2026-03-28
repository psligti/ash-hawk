"""Vox Jay agent-specific evaluator pack for content quality."""

from __future__ import annotations

from ash_hawk.eval_packs.base import EvalPack, EvalPackConfig
from ash_hawk.strategies import Strategy, SubStrategy


PromoterScoringEvalPack = EvalPack(
    pack_id="promoter-scoring-eval",
    name="Promoter Scoring Evaluation Pack",
    description="Evaluates promoter scorer's ability to differentiate 'engagement-worthy' accounts from 'nice-thought' accounts",
    target_agents=["vox-jay"],
    grader_configs={
        "engagement-worthy-accuracy": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.35,
            pass_threshold=0.80,
            params={
                "rubric": "Score 1.0 if HIGH_VALUE accounts are correctly identified (>=70%), 0.5 if partially correct (50-70%), 0.0 if poor (<50%)",
            },
        ),
        "nice-thought-accuracy": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.35,
            pass_threshold=0.70,
            params={
                "rubric": "Score 1.0 if LOW/MODERATE accounts are correctly identified, 0.5 if partially correct, 0.0 if poor",
            },
        ),
        "false-positive-rate": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.20,
            pass_threshold=0.90,
            params={
                "rubric": "Score 1.0 if no nice-thought accounts are marked as HIGH_VALUE, 0.5 if 1-2 false positives, 0.0 if 3+ false positives",
            },
        ),
        "signal-differentiation": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.10,
            pass_threshold=0.75,
            params={
                "rubric": "Evaluate whether the scorer correctly identifies key differentiating signals: amplification, accessibility, engagement, helpfulness, niche focus",
            },
        ),
    },
    strategy_focus=[
        Strategy.SKILL_QUALITY,
        Strategy.AGENT_BEHAVIOR,
    ],
    global_pass_threshold=0.75,
    metadata={
        "category": "evaluation",
        "agent_type": "scoring-agent",
        "sub_strategies": [
            SubStrategy.CLASSIFICATION_ACCURACY,
            SubStrategy.FALSE_POSITIVE_RATE,
        ],
    },
)


VoxJayEvalPack = EvalPack(
    pack_id="vox-jay-eval",
    name="Vox Jay Content Evaluation Pack",
    description="Evaluates content generation quality for vox-jay agent focusing on voice/tone, playbook adherence, and instruction clarity",
    target_agents=["vox-jay"],
    grader_configs={
        "voice-tone": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.25,
            pass_threshold=0.75,
            params={
                "rubric": "Evaluate whether content matches the expected voice, tone, and style guidelines",
            },
        ),
        "playbook-adherence": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.25,
            pass_threshold=0.80,
            params={
                "rubric": "Evaluate how well the content follows established playbooks and templates",
            },
        ),
        "instruction-clarity": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.20,
            pass_threshold=0.75,
            params={
                "rubric": "Evaluate whether instructions in the content are clear, actionable, and unambiguous",
            },
        ),
        "content-relevance": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.15,
            pass_threshold=0.70,
            params={
                "rubric": "Evaluate whether the content is relevant to the task and context",
            },
        ),
        "engagement-quality": EvalPackConfig(
            grader_name="llm_judge",
            weight=0.15,
            pass_threshold=0.70,
            params={
                "rubric": "Evaluate whether the content is engaging and appropriate for the audience",
            },
        ),
    },
    strategy_focus=[
        Strategy.SKILL_QUALITY,
        Strategy.AGENT_BEHAVIOR,
    ],
    global_pass_threshold=0.75,
    metadata={
        "category": "content",
        "agent_type": "content-agent",
        "sub_strategies": [
            SubStrategy.VOICE_TONE,
            SubStrategy.PLAYBOOK_ADHERENCE,
            SubStrategy.INSTRUCTION_CLARITY,
            SubStrategy.ENGAGEMENT_PROXY,
        ],
    },
)
