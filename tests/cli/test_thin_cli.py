from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from ash_hawk.cli.main import cli
from ash_hawk.improvement.improver_agent import DiffProposal


def test_thin_improve_reports_updated_targets_and_score_delta(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("id: test\n", encoding="utf-8")

    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "outcome": {"success": False, "error": "failed"},
                "grader_results": [],
                "aggregate_score": 0.25,
            }
        ),
        encoding="utf-8",
    )

    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    target_file = agent_dir / "AGENT.md"
    target_file.write_text("old\n", encoding="utf-8")

    class _FakeImprover:
        async def analyze_failures(self, context: Any) -> list[DiffProposal]:
            _ = context
            return [
                DiffProposal(
                    file_path=target_file,
                    diff="--- a\n+++ b\n@@\n-old\n+new\n",
                    description="Update agent prompt",
                )
            ]

    class _FakeApplyResult:
        success = True
        file_path = target_file
        backup_path = None
        error = None

    class _FakeApplier:
        async def apply(
            self, file_path: Path, diff: str, dry_run: bool, backup: bool
        ) -> _FakeApplyResult:
            _ = file_path, diff, dry_run, backup
            target_file.write_text("new\n", encoding="utf-8")
            return _FakeApplyResult()

    class _FakeScenario:
        graders = [object()]

    class _FakeGradedResult:
        aggregate_score = 0.6

    class _FakeThinRunner:
        def __init__(self, workdir: Path, max_iterations: int) -> None:
            _ = workdir, max_iterations

        async def run_with_grading(self, scenario: Any, scenario_file: Path) -> _FakeGradedResult:
            _ = scenario, scenario_file
            return _FakeGradedResult()

    monkeypatch.setattr("ash_hawk.improvement.improver_agent.ImproverAgent", _FakeImprover)
    monkeypatch.setattr("ash_hawk.improvement.applier.DiffApplier", _FakeApplier)
    monkeypatch.setattr("ash_hawk.scenario.loader.load_scenario", lambda _: _FakeScenario())
    monkeypatch.setattr("ash_hawk.scenario.thin_runner.ThinScenarioRunner", _FakeThinRunner)
    monkeypatch.setattr("ash_hawk.cli.thin.ThinScenarioRunner", _FakeThinRunner)

    result = CliRunner().invoke(
        cli,
        [
            "thin",
            "improve",
            str(scenario_path),
            str(transcript_path),
            "--agent",
            str(agent_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Updated targets:" in result.output
    assert target_file.name in result.output
    assert "Score Delta" in result.output
    assert "Before: 0.250" in result.output
    assert "After:  0.600" in result.output
    assert "Delta:  +0.350" in result.output


def test_thin_improve_reports_reverted_target_on_apply_failure(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("id: test\n", encoding="utf-8")

    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        json.dumps({"run_id": "run-2", "outcome": {"success": False, "error": "failed"}}),
        encoding="utf-8",
    )

    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    target_file = agent_dir / "AGENT.md"
    target_file.write_text("old\n", encoding="utf-8")
    backup_file = target_file.with_suffix(".bak")
    backup_file.write_text("old\n", encoding="utf-8")

    class _FakeImprover:
        async def analyze_failures(self, context: Any) -> list[DiffProposal]:
            _ = context
            return [
                DiffProposal(
                    file_path=target_file,
                    diff="bad diff",
                    description="Broken patch",
                )
            ]

    class _FakeApplyResult:
        success = False
        file_path = target_file
        backup_path = backup_file
        error = "patch failed"

    class _FakeApplier:
        async def apply(
            self, file_path: Path, diff: str, dry_run: bool, backup: bool
        ) -> _FakeApplyResult:
            _ = file_path, diff, dry_run, backup
            target_file.write_text("broken\n", encoding="utf-8")
            return _FakeApplyResult()

    monkeypatch.setattr("ash_hawk.improvement.improver_agent.ImproverAgent", _FakeImprover)
    monkeypatch.setattr("ash_hawk.improvement.applier.DiffApplier", _FakeApplier)
    monkeypatch.setattr(
        "ash_hawk.scenario.loader.load_scenario", lambda _: type("S", (), {"graders": []})()
    )

    class _FakeThinRunner:
        def __init__(self, workdir: Path, max_iterations: int) -> None:
            _ = workdir, max_iterations

        async def run_with_grading(self, scenario: Any, scenario_file: Path) -> Any:
            _ = scenario, scenario_file
            raise RuntimeError("should not be called")

    monkeypatch.setattr("ash_hawk.scenario.thin_runner.ThinScenarioRunner", _FakeThinRunner)
    monkeypatch.setattr("ash_hawk.cli.thin.ThinScenarioRunner", _FakeThinRunner)

    result = CliRunner().invoke(
        cli,
        [
            "thin",
            "improve",
            str(scenario_path),
            str(transcript_path),
            "--agent",
            str(agent_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Reverted targets:" in result.output
    assert target_file.name in result.output
    assert target_file.read_text(encoding="utf-8") == "old\n"
