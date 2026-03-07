"""Preset types for multi-grader composition factories."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

PresetName = Literal[
    "security_review",
    "code_quality",
    "general",
    "minimal",
]


class GraderPresetConfig(BaseModel):
    """Configuration for expanding a grader preset into grader specs."""

    model_config = ConfigDict(extra="forbid")

    preset: PresetName
    llm_judge: bool = True
    llm_weight: float = 0.4
    pass_threshold: float = 0.7
    rubric: str | None = None
    string_match: bool = True
    expected_keywords: list[str] | None = None
    tool_call: bool = True
    expected_tools: list[str] | None = None
    schema_validation: bool = False
    schema_type: str | None = None
    transcript: bool = True
    max_turns: int = 50


__all__ = ["PresetName", "GraderPresetConfig"]
