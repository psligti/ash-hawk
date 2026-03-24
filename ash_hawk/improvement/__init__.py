"""Improvement module for agent behavior optimization.

This module provides tools for capturing prompts and data structures
for the improvement agent, based on the control level hierarchy:

    Agent (highest) > Skill (medium) > Tool (lowest)

Key Components:
    - PromptCapture: Save agent, skill, and tool prompts to .dawn-kestrel/
    - ControlLevel: Enum for improvement target hierarchy
    - Finding: Dataclass for review findings
    - ReviewMetrics: Dataclass for aggregate metrics

The improvement agent (ash_hawk.agents.ImprovementAgentRunner) uses
the prompt at ash_hawk/prompts/improvement/target_selection.md to
make agentic decisions about where improvements should happen.

Storage Structure:
    .dawn-kestrel/
    ├── agent.md              # Agent system prompt
    ├── skills/
    │   └── {skill_name}.md   # Skill definitions
    └── tools/
        └── {tool_name}.md    # Tool descriptions
"""

from __future__ import annotations

from ash_hawk.improvement.decision_engine import (
    CONTROL_LEVEL_TO_LESSON_TYPE,
    CONTROL_LEVEL_TO_STRATEGY,
    ControlLevel,
    Finding,
    ReviewMetrics,
)
from ash_hawk.improvement.prompt_capture import (
    DEFAULT_DAWN_KESTREL_DIR,
    CapturedPrompt,
    PromptCapture,
)

__all__ = [
    "ControlLevel",
    "Finding",
    "ReviewMetrics",
    "CapturedPrompt",
    "PromptCapture",
    "DEFAULT_DAWN_KESTREL_DIR",
    "CONTROL_LEVEL_TO_LESSON_TYPE",
    "CONTROL_LEVEL_TO_STRATEGY",
]
