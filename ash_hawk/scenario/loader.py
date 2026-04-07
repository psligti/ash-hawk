# type-hygiene: skip-file
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ash_hawk.scenario.models import ScenarioV1


def load_scenario(path: str | Path) -> ScenarioV1:
    scenario_path = Path(path)
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

    content = scenario_path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    if not isinstance(data, dict):
        raise ValueError("Scenario YAML must be a mapping")

    return ScenarioV1.model_validate(data)


def discover_scenarios(search_root: str | Path) -> list[Path]:
    root = Path(search_root).resolve()
    patterns = ("*.scenario.yaml", "*.scenario.yml")

    if root.is_file():
        return [root] if any(root.match(pattern) for pattern in patterns) else []

    if not root.exists():
        raise FileNotFoundError(f"Scenario search root not found: {root}")

    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root.rglob(pattern))

    return sorted({path.resolve() for path in matches})


def load_scenarios(search_root: str | Path) -> list[ScenarioV1]:
    return [load_scenario(path) for path in discover_scenarios(search_root)]


__all__ = ["discover_scenarios", "load_scenario", "load_scenarios"]
