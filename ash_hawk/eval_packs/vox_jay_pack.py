"""Vox Jay agent-specific evaluator pack for content quality."""

from __future__ import annotations

from ash_hawk.eval_packs.base import EvalPack, EvalPackConfig
from ash_hawk.strategies import Strategy, SubStrategy

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
