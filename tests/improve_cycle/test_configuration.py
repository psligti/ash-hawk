from __future__ import annotations

from pathlib import Path

import pytest

from ash_hawk.improve_cycle.configuration import ImproveCycleConfig, load_improve_cycle_config


def test_load_improve_cycle_config_from_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "improve_cycle.yaml"
    cfg_path.write_text(
        """
improve_cycle:
  competitor:
    enabled: false
  verifier:
    enabled: true
    min_repeats: 7
    max_variance: 0.02
    max_latency_delta_pct: 20
    max_token_delta_pct: 12
    require_regression_pass: true
  promotion:
    default_scope: global
    low_risk_success_threshold: 2
    medium_risk_success_threshold: 4
""".strip(),
        encoding="utf-8",
    )

    config = ImproveCycleConfig.from_yaml(cfg_path)
    assert config.competitor.enabled is False
    assert config.verifier.min_repeats == 7
    assert config.promotion.default_scope == "global"


def test_load_improve_cycle_config_uses_packaged_defaults() -> None:
    config = load_improve_cycle_config(None)
    assert config.triage.enabled is True
    assert config.verifier.min_repeats >= 1


def test_load_improve_cycle_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ImproveCycleConfig.from_yaml(tmp_path / "missing.yaml")
