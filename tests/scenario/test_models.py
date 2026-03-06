from __future__ import annotations

from typing import Any

import pydantic as pd
import pytest

from ash_hawk.scenario.models import ScenarioV1


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
            "allowed_tools": ["read", "write"],
            "mocks": {"mock_tool": {"ok": True}},
            "fault_injection": {"delay_ms": 5},
        },
        "budgets": {
            "max_steps": 5,
            "max_tool_calls": 10,
            "max_tokens": 1000,
            "max_time_seconds": 30.0,
        },
        "expectations": {
            "must_events": ["event_a"],
            "must_not_events": ["event_b"],
            "ordering_rules": [{"before": "event_a", "after": "event_c"}],
            "diff_assertions": [{"path": "output.txt"}],
            "output_assertions": [{"contains": "done"}],
        },
        "graders": [
            {
                "grader_type": "string_match",
                "config": {"expected": "ok"},
                "weight": 1.0,
                "required": True,
                "timeout_seconds": 5.0,
            }
        ],
    }


def test_scenario_v1_valid() -> None:
    scenario = ScenarioV1.model_validate(_scenario_data())

    assert scenario.schema_version == "v1"
    assert scenario.id == "scenario-1"
    assert scenario.sut.adapter == "local"
    assert scenario.tools.allowed_tools == ["read", "write"]
    assert scenario.budgets.max_steps == 5
    assert scenario.expectations.must_events == ["event_a"]
    assert scenario.graders[0].grader_type == "string_match"


def test_scenario_v1_extra_field_forbidden() -> None:
    data = _scenario_data()
    data["extra_field"] = "not_allowed"

    with pytest.raises(pd.ValidationError):
        ScenarioV1.model_validate(data)
