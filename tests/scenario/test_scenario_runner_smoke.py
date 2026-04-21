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


class WorkspaceEditingAdapter:
    @property
    def name(self) -> str:
        return "workspace_edit_adapter"

    def run_scenario(
        self,
        scenario: dict[str, object],
        workdir: Path,
        tooling_harness: dict[str, object],
        budgets: dict[str, object],
    ) -> ScenarioAdapterResult:
        del scenario, budgets
        target = workdir / "src" / "api.py"
        target.write_text("def handler():\n    return 'ok'\n", encoding="utf-8")
        call = tooling_harness["call"]
        call(
            "edit",
            {
                "filePath": str(target),
                "path": "src/api.py",
            },
        )
        return ScenarioAdapterResult(
            final_output="edited",
            trace_events=[
                ScenarioTraceEvent(
                    event_type="ModelMessageEvent",
                    data={"role": "assistant", "content": "edited"},
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


def test_scenario_runner_expands_pack_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ScenarioAdapterRegistry()
    adapter = MockScenarioAdapter()
    assert isinstance(adapter, ScenarioAdapter)
    registry.register(adapter)

    monkeypatch.setattr(registry_module, "_default_registry", registry)
    monkeypatch.chdir(tmp_path)

    scenarios_dir = tmp_path / "evals" / "scenarios"
    nested_dir = scenarios_dir / "curriculum" / "foundation"
    nested_dir.mkdir(parents=True)

    for path, scenario_id in (
        (nested_dir / "first.scenario.yaml", "first"),
        (nested_dir / "second.scenario.yaml", "second"),
    ):
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "v1",
                    "id": scenario_id,
                    "description": f"{scenario_id} scenario",
                    "sut": {"type": "coding_agent", "adapter": "mock_adapter", "config": {}},
                    "inputs": {"prompt": "run"},
                    "tools": {
                        "allowed_tools": ["read"],
                        "mocks": {
                            "read": {"input": {"path": "input.txt"}, "result": {"status": "ok"}}
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

    pack = scenarios_dir / "phase1_evidence_pack.yaml"
    pack.write_text(
        yaml.safe_dump(
            {
                "schema_version": "v1",
                "id": "phase1",
                "scenarios": [
                    {"scenario": "./curriculum/foundation/first.scenario.yaml"},
                    {"scenario": "./curriculum/foundation/second.scenario.yaml"},
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = run_scenarios([str(pack)])
    assert len(summary.trials) == 2
    assert {trial.task_id for trial in summary.trials} == {"first", "second"}


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


def test_scenario_runner_propagates_sandbox_workdir_to_repo_diff_grader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ScenarioAdapterRegistry()
    adapter = WorkspaceEditingAdapter()
    assert isinstance(adapter, ScenarioAdapter)
    registry.register(adapter)

    monkeypatch.setattr(registry_module, "_default_registry", registry)
    monkeypatch.chdir(tmp_path)

    scenario_path = tmp_path / "workspace-repo-diff.scenario.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "v1",
                "id": "workspace-repo-diff",
                "description": "Workspace repo diff scenario",
                "sut": {"type": "coding_agent", "adapter": "workspace_edit_adapter", "config": {}},
                "inputs": {"prompt": "run"},
                "workspace": {"src/api.py": "def handler():\n    return 'old'\n"},
                "tools": {
                    "allowed_tools": ["edit"],
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
                "graders": [
                    {
                        "grader_type": "repo_diff",
                        "config": {
                            "required_file_changes": [{"path": "src/api.py"}],
                            "semantic_assertions": [
                                {"path": "src/api.py", "must_contain_after": ["return 'ok'"]}
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = run_scenarios([str(scenario_path)])
    assert summary.trials
    trial = summary.trials[0]

    assert trial.input_snapshot["workdir"] != str(tmp_path)
    assert trial.result is not None
    assert trial.result.aggregate_passed is True
    grader_result = trial.result.grader_results[0]
    assert grader_result.passed is True
