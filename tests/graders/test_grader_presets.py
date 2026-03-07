"""Tests for grader presets."""

from __future__ import annotations

import pytest

from ash_hawk.graders.grader_presets import (
    PRESET_DEFAULTS,
    expand_preset,
    get_default_rubric,
    get_preset_names,
)
from ash_hawk.graders.preset_types import GraderPresetConfig


class TestExpandPreset:
    """Tests for expand_preset function."""

    def test_security_review_preset(self) -> None:
        config = GraderPresetConfig(preset="security_review")
        specs = expand_preset(config)

        assert len(specs) >= 3
        grader_types = {s["grader_type"] for s in specs}
        assert "llm_judge" in grader_types
        assert "string_match" in grader_types
        assert "tool_call" in grader_types
        assert "transcript" in grader_types

    def test_minimal_preset(self) -> None:
        config = GraderPresetConfig(preset="minimal")
        specs = expand_preset(config)

        assert len(specs) == 1
        assert specs[0]["grader_type"] == "llm_judge"
        assert specs[0]["weight"] == 1.0

    def test_override_pass_threshold(self) -> None:
        config = GraderPresetConfig(preset="security_review", pass_threshold=0.65)
        specs = expand_preset(config)

        llm_spec = next(s for s in specs if s["grader_type"] == "llm_judge")
        assert llm_spec["config"]["pass_threshold"] == 0.65

    def test_override_expected_tools(self) -> None:
        config = GraderPresetConfig(
            preset="security_review",
            expected_tools=["bash", "grep"],
        )
        specs = expand_preset(config)

        tool_spec = next(s for s in specs if s["grader_type"] == "tool_call")
        expected_tools = {c["tool"] for c in tool_spec["config"]["expected_calls"]}
        assert "bash" in expected_tools
        assert "grep" in expected_tools

    def test_disable_string_match(self) -> None:
        config = GraderPresetConfig(preset="general", string_match=False)
        specs = expand_preset(config)

        grader_types = {s["grader_type"] for s in specs}
        assert "string_match" not in grader_types

    def test_general_preset(self) -> None:
        config = GraderPresetConfig(preset="general")
        specs = expand_preset(config)

        grader_types = {s["grader_type"] for s in specs}
        assert "llm_judge" in grader_types
        assert "string_match" in grader_types
        assert "transcript" in grader_types
        assert "tool_call" not in grader_types

    def test_code_quality_preset(self) -> None:
        config = GraderPresetConfig(preset="code_quality")
        specs = expand_preset(config)

        grader_types = {s["grader_type"] for s in specs}
        assert "llm_judge" in grader_types
        assert "schema" in grader_types
        assert "transcript" in grader_types


class TestGetDefaultRubric:
    """Tests for get_default_rubric function."""

    def test_security_review_rubric(self) -> None:
        assert get_default_rubric("security_review") == "security_review"

    def test_general_rubric(self) -> None:
        assert get_default_rubric("general") == "general"

    def test_minimal_rubric(self) -> None:
        assert get_default_rubric("minimal") == "general"


class TestGetPresetNames:
    """Tests for get_preset_names function."""

    def test_returns_all_presets(self) -> None:
        names = get_preset_names()
        assert "security_review" in names
        assert "code_quality" in names
        assert "general" in names
        assert "minimal" in names

    def test_matches_defaults_dict(self) -> None:
        names = get_preset_names()
        assert set(names) == set(PRESET_DEFAULTS.keys())


class TestGraderPresetConfig:
    """Tests for GraderPresetConfig validation."""

    def test_valid_preset_name(self) -> None:
        config = GraderPresetConfig(preset="security_review")
        assert config.preset == "security_review"

    def test_invalid_preset_name_raises(self) -> None:
        with pytest.raises(Exception):  # Pydantic validation error
            GraderPresetConfig(preset="invalid_preset")  # type: ignore[arg-type]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(Exception):  # Pydantic validation error
            GraderPresetConfig(preset="general", unknown_field=True)  # type: ignore[call-arg]

    def test_default_values(self) -> None:
        config = GraderPresetConfig(preset="general")
        assert config.llm_judge is True
        assert config.llm_weight == 0.4
        assert config.pass_threshold == 0.7
        assert config.max_turns == 50
