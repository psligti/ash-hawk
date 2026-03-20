from __future__ import annotations

from pathlib import Path
from typing import Any

from click.testing import CliRunner
from pytest import MonkeyPatch

from ash_hawk.cli.main import cli
from ash_hawk.cycle.types import ConvergenceStatus, CycleConfig, CycleStatus, IterationResult


class _DummyResult:
    def __init__(self) -> None:
        self.status = CycleStatus.CONVERGED
        self.total_iterations = 1
        self.best_score = 0.6
        self.final_score = 0.6
        self.improvement_delta = 0.1
        self.total_lessons_generated = 0
        self.lessons_promoted: list[str] = []
        self.convergence_status = ConvergenceStatus.CONVERGED
        self.iterations: list[IterationResult] = []


def test_improve_cycle_reads_yaml_config(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "improve_cycle.yaml"
    cfg_path.write_text(
        """
improve_cycle:
  competitor:
    enabled: false
  triage:
    enabled: false
  verifier:
    enabled: true
    min_repeats: 6
    max_variance: 0.01
    max_latency_delta_pct: 25
    max_token_delta_pct: 14
    require_regression_pass: true
  adversary:
    enabled: false
  promotion:
    default_scope: global
    low_risk_success_threshold: 3
    medium_risk_success_threshold: 5
""".strip(),
        encoding="utf-8",
    )
    scenario_path = tmp_path / "s.scenario.yaml"
    scenario_path.write_text(
        "version: v1\nid: s\ndescription: s\nsut:\n  adapter: mock\n  config: {}\ninputs: {}\nexpectations: []\ngraders: []\nbudgets:\n  max_time_seconds: 10\n",
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    class FakeRunner:
        def __init__(self, config: CycleConfig):
            captured["config"] = config

        async def run_cycle(self):
            return _DummyResult()

    monkeypatch.setattr("ash_hawk.cycle.create_cycle_id", lambda: "cid")
    monkeypatch.setattr("ash_hawk.cycle.CycleRunner", FakeRunner)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "improve",
            "cycle",
            "--agent",
            "bolt-merlin",
            "--config",
            str(cfg_path),
            "--iterations",
            "1",
            "--scenario-path",
            str(scenario_path),
        ],
    )
    assert result.exit_code == 0
    config = captured["config"]
    assert isinstance(config, CycleConfig)
    metadata = config.metadata
    assert metadata["enable_competitor"] is False
    assert metadata["enable_triage"] is False
    assert metadata["enable_adversary"] is False
    assert metadata["min_verification_runs"] == 6
    assert metadata["max_latency_delta_pct"] == 25
    assert metadata["max_token_delta_pct"] == 14
    assert metadata["promotion_scope"] == "global"
