from __future__ import annotations

from pathlib import Path

import click
import pytest

from ash_hawk.cli.improve import resolve_cycle_scenario_paths


class TestResolveCycleScenarioPaths:
    def test_resolves_directory_to_all_scenarios(self, tmp_path: Path) -> None:
        root = tmp_path / "scenarios"
        root.mkdir(parents=True)
        (root / "a.scenario.yaml").write_text(
            "version: v1\nid: a\ndescription: a\nsut:\n  adapter: mock\n  config: {}\ninputs: {}\nexpectations: []\ngraders: []\nbudgets:\n  max_time_seconds: 10\n",
            encoding="utf-8",
        )
        nested = root / "nested"
        nested.mkdir()
        (nested / "b.scenario.yml").write_text(
            "version: v1\nid: b\ndescription: b\nsut:\n  adapter: mock\n  config: {}\ninputs: {}\nexpectations: []\ngraders: []\nbudgets:\n  max_time_seconds: 10\n",
            encoding="utf-8",
        )

        paths = resolve_cycle_scenario_paths([str(root)])
        assert len(paths) == 2
        assert all(path.endswith((".scenario.yaml", ".scenario.yml")) for path in paths)

    def test_raises_for_missing_path(self) -> None:
        with pytest.raises(click.ClickException, match="Scenario path not found"):
            resolve_cycle_scenario_paths(["/tmp/does-not-exist-ash-hawk"])
