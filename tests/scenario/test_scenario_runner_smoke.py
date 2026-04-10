from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import ash_hawk.scenario.registry as registry_module
from ash_hawk.scenario.adapters import ScenarioAdapter
from ash_hawk.scenario.models import ScenarioAdapterResult, ScenarioTraceEvent, ScenarioV1
from ash_hawk.scenario.registry import ScenarioAdapterRegistry
from ash_hawk.scenario.runner import ScenarioRunner, run_scenarios


class MockScenarioAdapter:
    @property
    def name(self) -> str:
        return "mock_adapter"

    def run_scenario(
        self,
        scenario: dict[str, object],
        workdir: Path,
        tooling_harness: dict[str, object],
        budgets: dict[str, object],
    ) -> ScenarioAdapterResult:
        del workdir, budgets
        call = tooling_harness["call"]
        call("read", {"path": "input.txt"})
        trace_events = [
            ScenarioTraceEvent(
                event_type="ModelMessageEvent",
                data={"role": "assistant", "content": "ok"},
            )
        ]
        return ScenarioAdapterResult(
            final_output="ok",
            trace_events=trace_events,
            artifacts={"note": "artifact"},
        )


class WorkspaceInspectingAdapter:
    @property
    def name(self) -> str:
        return "workspace_adapter"

    def run_scenario(
        self,
        scenario: dict[str, object],
        workdir: Path,
        tooling_harness: dict[str, object],
        budgets: dict[str, object],
    ) -> ScenarioAdapterResult:
        del scenario, tooling_harness, budgets
        auth_content = (workdir / "auth.py").read_text(encoding="utf-8")
        readme_content = (workdir / "README.md").read_text(encoding="utf-8")
        return ScenarioAdapterResult(
            final_output="workspace-ok",
            trace_events=[
                ScenarioTraceEvent(
                    event_type="ModelMessageEvent",
                    data={
                        "role": "assistant",
                        "content": f"{auth_content}\n---\n{readme_content}",
                    },
                )
            ],
            artifacts={},
        )


def test_scenario_runner_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    trial = summary.trials[0]
    assert trial.result is not None
    assert trial.result.transcript.tool_calls


def test_scenario_runner_overrides_task_timeout_from_constructor(tmp_path: Path) -> None:
    runner = ScenarioRunner(storage_path=tmp_path / "storage", scenario_timeout_seconds=420.0)
    scenario = ScenarioV1.model_validate(
        {
            "schema_version": "v1",
            "id": "timeout-override-demo",
            "description": "Timeout override scenario",
            "sut": {"type": "coding_agent", "adapter": "mock_adapter", "config": {}},
            "inputs": {"prompt": "run"},
            "tools": {
                "allowed_tools": ["read"],
                "mocks": {},
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
    )

    task = runner._scenario_to_task(scenario, tmp_path / "demo.scenario.yaml")
    assert task.timeout_seconds == 420.0


def test_scenario_runner_seeds_workspace_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ScenarioAdapterRegistry()
    adapter = WorkspaceInspectingAdapter()
    assert isinstance(adapter, ScenarioAdapter)
    registry.register(adapter)

    monkeypatch.setattr(registry_module, "_default_registry", registry)
    monkeypatch.chdir(tmp_path)

    scenario_path = tmp_path / "workspace-demo.scenario.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "v1",
                "id": "workspace-demo",
                "description": "Workspace seeded scenario",
                "sut": {"type": "coding_agent", "adapter": "workspace_adapter", "config": {}},
                "inputs": {"prompt": "run"},
                "tools": {
                    "allowed_tools": [],
                    "mocks": {},
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
                "workspace": {
                    "auth.py": "return False\n",
                    "README.md": "# Seeded README\n",
                },
            }
        ),
        encoding="utf-8",
    )

    summary = run_scenarios([str(scenario_path)])
    assert summary.trials
    trial = summary.trials[0]
    assert trial.result is not None
    assert (
        trial.result.transcript.messages[-1]["content"] == "return False\n\n---\n# Seeded README\n"
    )
