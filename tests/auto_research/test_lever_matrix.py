from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ash_hawk.auto_research.lever_matrix import LeverMatrixSearch, _config_fingerprint
from ash_hawk.auto_research.types import LeverConfiguration


def _make_config(timeout_multiplier: float = 1.0) -> LeverConfiguration:
    return LeverConfiguration(
        agent="orchestrator",
        skills=("goal-tracking",),
        tools=("read", "grep"),
        context_strategy="file-based",
        prompt_preset="balanced",
        timeout_multiplier=timeout_multiplier,
    )


def test_config_fingerprint_is_stable() -> None:
    config = _make_config()

    first = _config_fingerprint(config)
    second = _config_fingerprint(config)

    assert first == second
    assert len(first) == 12


@pytest.mark.asyncio
async def test_evaluate_applies_timeout_multiplier(monkeypatch, tmp_path: Path) -> None:
    search = LeverMatrixSearch()
    config = _make_config(timeout_multiplier=1.5)
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("id: test\n", encoding="utf-8")

    captured: dict[str, object] = {}

    async def _fake_run_scenarios_async(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(metrics=SimpleNamespace(mean_score=0.73))

    monkeypatch.setattr(
        "ash_hawk.auto_research.lever_matrix.run_scenarios_async",
        _fake_run_scenarios_async,
    )
    monkeypatch.setattr(
        "ash_hawk.auto_research.lever_matrix.get_config",
        lambda: SimpleNamespace(default_timeout_seconds=100),
    )

    score = await search.evaluate(config=config, scenarios=[scenario], storage_path=tmp_path)

    assert score == 0.73
    assert captured["scenario_timeout_seconds"] == 150.0
    assert captured["grader_config_overrides"] == {"quiet": True}
    assert "injector" in captured
