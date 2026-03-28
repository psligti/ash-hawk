from __future__ import annotations

from pathlib import Path

from ash_hawk.scenario.adapters.sdk_dawn_kestrel import SdkDawnKestrelAdapter
from ash_hawk.scenario.models import JSONValue, ScenarioAdapterResult


class BoltMerlinScenarioAdapter(SdkDawnKestrelAdapter):
    name: str = "bolt_merlin"

    def run_scenario(
        self,
        scenario: dict[str, JSONValue],
        workdir: Path,
        tooling_harness: dict[str, object],
        budgets: dict[str, JSONValue],
    ) -> ScenarioAdapterResult:
        scenario_copy = dict(scenario)
        sut_raw = scenario_copy.get("sut", {})
        sut = dict(sut_raw) if isinstance(sut_raw, dict) else {}
        config_raw = sut.get("config", {})
        config = dict(config_raw) if isinstance(config_raw, dict) else {}
        run_config_raw = config.get("run_config", {})
        run_config = dict(run_config_raw) if isinstance(run_config_raw, dict) else {}
        run_config["agent_name"] = run_config.get("agent_name", "bolt-merlin")
        config["run_config"] = run_config
        sut["config"] = config
        scenario_copy["sut"] = sut
        return super().run_scenario(scenario_copy, workdir, tooling_harness, budgets)


__all__ = ["BoltMerlinScenarioAdapter"]
