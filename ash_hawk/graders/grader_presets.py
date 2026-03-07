"""Grader preset factories for common multi-grader compositions."""

from __future__ import annotations

from typing import Any

from ash_hawk.graders.preset_types import GraderPresetConfig, PresetName

PRESET_DEFAULTS: dict[PresetName, dict[str, Any]] = {
    "security_review": {
        "llm_judge": True,
        "llm_weight": 0.4,
        "rubric": "security_review",
        "string_match": True,
        "expected_keywords": ["severity", "vulnerability", "risk"],
        "tool_call": True,
        "transcript": True,
        "max_turns": 50,
    },
    "code_quality": {
        "llm_judge": True,
        "llm_weight": 0.5,
        "rubric": "code_quality",
        "string_match": False,
        "tool_call": False,
        "schema": True,
        "transcript": True,
        "max_turns": 30,
    },
    "general": {
        "llm_judge": True,
        "llm_weight": 0.4,
        "rubric": "general",
        "string_match": True,
        "tool_call": False,
        "transcript": True,
        "max_turns": 50,
    },
    "minimal": {
        "llm_judge": True,
        "llm_weight": 1.0,
        "rubric": "general",
        "string_match": False,
        "tool_call": False,
        "transcript": False,
    },
}


def get_default_rubric(preset: PresetName) -> str:
    """Get the default rubric name for a preset.

    Args:
        preset: Preset name

    Returns:
        Default rubric name
    """
    defaults = PRESET_DEFAULTS.get(preset, {})
    rubric = defaults.get("rubric", "general")
    return str(rubric) if rubric else "general"


def expand_preset(config: GraderPresetConfig) -> list[dict[str, Any]]:
    """Expand a preset configuration into a list of grader specs.

    This function converts a high-level preset configuration into
    the detailed grader specs needed for EvalTask.

    Args:
        config: Preset configuration

    Returns:
        List of grader spec dictionaries ready for use in EvalTask

    Example:
        >>> config = GraderPresetConfig(preset="security_review")
        >>> specs = expand_preset(config)
        >>> len(specs)  # Returns 4 graders
        4
    """
    specs: list[dict[str, Any]] = []

    # Merge preset defaults with config overrides
    defaults = PRESET_DEFAULTS.get(config.preset, {})
    effective = {**defaults, **config.model_dump(exclude_unset=True)}

    # LLM Judge grader
    if effective.get("llm_judge", True):
        llm_spec = {
            "grader_type": "llm_judge",
            "weight": effective.get("llm_weight", 0.4),
            "required": True,
            "config": {
                "rubric": effective.get("rubric", get_default_rubric(config.preset)),
                "pass_threshold": effective.get("pass_threshold", 0.7),
            },
        }
        specs.append(llm_spec)

    # String match grader
    if effective.get("string_match", False):
        keywords = effective.get("expected_keywords")
        string_spec = {
            "grader_type": "string_match",
            "weight": 0.1,
            "required": False,
            "config": {
                "mode": "contains",
                "contains": keywords or [],
            },
        }
        specs.append(string_spec)

    # Tool call grader
    if effective.get("tool_call", False):
        tools = effective.get("expected_tools")
        tool_spec = {
            "grader_type": "tool_call",
            "weight": 0.0,
            "required": False,
            "config": {
                "expected_calls": [{"tool": t} for t in (tools or [])],
                "partial_credit": True,
            },
        }
        specs.append(tool_spec)

    # Schema grader
    if effective.get("schema", False):
        schema_type = effective.get("schema_type")
        schema_spec = {
            "grader_type": "schema",
            "weight": 0.15,
            "required": True,
            "config": {
                "schema_type": schema_type or "default",
            },
        }
        specs.append(schema_spec)

    # Transcript grader
    if effective.get("transcript", True):
        transcript_spec = {
            "grader_type": "transcript",
            "weight": 0.1,
            "required": False,
            "config": {
                "max_turns": effective.get("max_turns", 50),
            },
        }
        specs.append(transcript_spec)

    return specs


def get_preset_names() -> list[PresetName]:
    """Get list of available preset names."""
    return list(PRESET_DEFAULTS.keys())


__all__ = [
    "PRESET_DEFAULTS",
    "expand_preset",
    "get_default_rubric",
    "get_preset_names",
]
