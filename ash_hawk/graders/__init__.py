"""Grader infrastructure for ash-hawk evaluation harness.

This module provides the base grader abstract class and a registry
for managing and discovering graders.
"""

from ash_hawk.graders.base import Grader, PassThroughGrader
from ash_hawk.graders.llm_boolean import LLMBooleanJudgeGrader
from ash_hawk.graders.llm_boolean_specialized import BooleanJudgeGrader, create_boolean_graders
from ash_hawk.graders.prompt_stack_optimizer import (
    PromptStackOptimizerConfig,
    PromptStackOptimizerGrader,
)
from ash_hawk.graders.registry import (
    ENTRY_POINT_GROUP,
    GraderRegistry,
    get_default_registry,
)

__all__ = [
    "Grader",
    "PassThroughGrader",
    "GraderRegistry",
    "get_default_registry",
    "ENTRY_POINT_GROUP",
    "PromptStackOptimizerGrader",
    "PromptStackOptimizerConfig",
    "LLMBooleanJudgeGrader",
    "BooleanJudgeGrader",
    "create_boolean_graders",
]
