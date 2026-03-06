from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import ash_hawk.scenario.registry as registry_module
from ash_hawk.scenario.adapters import ScenarioAdapter
from ash_hawk.scenario.registry import ScenarioAdapterRegistry
from ash_hawk.scenario.runner import run_scenarios


class MockScenarioAdapter:
    @property
    def name(self) -> str:
        return "mock_adapter"

    def run_scenario(
        self,
        scenario: dict[str, Any],
        workdir: Path,
        tooling_harness: dict[str, Any],
        budgets: dict[str, Any],
    ) -> tuple[Any, list[Any], dict[str, Any]]:
        del workdir, budgets
        call = tooling_harness["call"]
        call("read", {"path": "input.txt"})
        return "ok", [], {"note": "artifact"}


def test_scenario_runner_smoke(tmp_path, monkeypatch) -> None:
    registry = ScenarioAdapterRegistry()
    adapter = MockScenarioAdapter()
    assert isinstance(adapter, ScenarioAdapter)
    registry.register(adapter)

    monkeypatch.setattr(registry_module, "_default_registry", registry)
    monkeypatch.chdir(tmp_path)

    scenario_path = tmp_path / "demo.scenario.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "v1",
                "id": "demo",
                "description": "Smoke scenario",
                "sut": {"type": "coding_agent", "adapter": "mock_adapter", "config": {}},
                "inputs": {"prompt": "run"},
                "tools": {
                    "allowed_tools": ["read"],
                    "mocks": {
                        "read": {
                            "input": {"path": "input.txt"},
                            "result": {"status": "ok"},
                        }
                    },
                    "fault_injection": {},
                },
                "budgets": {
                    "max_steps": 3,
                    "max_tool_calls": 5,
                    "max_tokens": 100,
                    "max_time_seconds": 10.0,
                },
                "expectations": {
                    "must_events": [],
                    "must_not_events": [],
                    "ordering_rules": [],
                    "diff_assertions": [],
                    "output_assertions": [],
                },
                "graders": [],
            }
        ),
        encoding="utf-8",
    )

    summary = run_scenarios([str(scenario_path)])
    assert summary.trials
    trace_events = summary.trials[0].result.transcript.trace_events
    assert trace_events
    assert any(event.get("event_type") == "ToolCallEvent" for event in trace_events)
    assert any(event.get("event_type") == "ArtifactEvent" for event in trace_events)
