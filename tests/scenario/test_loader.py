from __future__ import annotations

from pathlib import Path
from typing import Any

import pydantic as pd
import pytest
import yaml

from ash_hawk.scenario.loader import discover_scenarios, load_scenario


def _scenario_data() -> dict[str, Any]:
    return {
        "schema_version": "v1",
        "id": "scenario-1",
        "description": "Test scenario",
        "sut": {
            "type": "coding_agent",
            "adapter": "local",
            "config": {"foo": "bar"},
        },
        "inputs": {"prompt": "Do the thing"},
        "tools": {
            "allowed_tools": ["read"],
            "mocks": {},
            "fault_injection": {},
        },
        "budgets": {
            "max_steps": 5,
            "max_tool_calls": 10,
            "max_tokens": 1000,
            "max_time_seconds": 30.0,
        },
        "expectations": {
            "must_events": ["event_a"],
            "must_not_events": [],
            "ordering_rules": [],
            "diff_assertions": [],
            "output_assertions": [],
        },
        "graders": [
            {
                "grader_type": "string_match",
                "config": {"expected": "ok"},
            }
        ],
    }


def test_load_scenario_valid(tmp_path: Path) -> None:
    scenario_path = tmp_path / "demo.scenario.yaml"
    scenario_path.write_text(yaml.safe_dump(_scenario_data()), encoding="utf-8")

    scenario = load_scenario(scenario_path)

    assert scenario.id == "scenario-1"
    assert scenario.sut.type == "coding_agent"


def test_load_scenario_unknown_field_raises(tmp_path: Path) -> None:
    scenario_path = tmp_path / "invalid.scenario.yml"
    data = _scenario_data()
    data["unknown_field"] = "nope"
    scenario_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(pd.ValidationError):
        load_scenario(scenario_path)


def test_discover_scenarios(tmp_path: Path) -> None:
    first = tmp_path / "alpha.scenario.yaml"
    second = tmp_path / "beta.scenario.yml"
    ignored = tmp_path / "gamma.yaml"

    first.write_text("schema_version: v1", encoding="utf-8")
    second.write_text("schema_version: v1", encoding="utf-8")
    ignored.write_text("schema_version: v1", encoding="utf-8")

    discovered = discover_scenarios(tmp_path)

    assert set(discovered) == {first.resolve(), second.resolve()}
