# type-hygiene: skip-file  # test file — mock/factory types are intentionally loose
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest

import ash_hawk.auto_research.lever_matrix as lever_matrix
from ash_hawk.auto_research.lever_matrix import (
    LeverMatrixSearch,
    validate_combination,
)
from ash_hawk.auto_research.types import LeverConfiguration, LeverDimension


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

    fingerprint = cast(
        Callable[[LeverConfiguration], str],
        getattr(lever_matrix, "_config_fingerprint"),
    )
    first = fingerprint(config)
    second = fingerprint(config)

    assert first == second
    assert len(first) == 12


@pytest.mark.asyncio
async def test_evaluate_applies_timeout_multiplier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    search = LeverMatrixSearch()
    config = _make_config(timeout_multiplier=1.5)
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("id: test\n", encoding="utf-8")

    captured: dict[str, object] = {}

    async def _fake_run_scenarios_async(**kwargs: object) -> SimpleNamespace:
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


class TestValidateCombination:
    def test_valid_combination_returns_empty(self) -> None:
        warnings = validate_combination(
            {
                "context_strategy": "file-based",
                "prompt_preset": "balanced",
            }
        )
        assert warnings == []

    def test_invalid_dynamic_precision_returns_warning(self) -> None:
        warnings = validate_combination(
            {
                "context_strategy": "dynamic",
                "prompt_preset": "precision",
            }
        )
        assert len(warnings) == 1
        assert "dynamic" in warnings[0]
        assert "precision" in warnings[0]

    def test_invalid_composite_throughput_returns_warning(self) -> None:
        warnings = validate_combination(
            {
                "context_strategy": "composite",
                "prompt_preset": "throughput",
            }
        )
        assert len(warnings) == 1
        assert "composite" in warnings[0]

    def test_empty_dict_returns_empty(self) -> None:
        warnings = validate_combination({})
        assert warnings == []


class TestMutationStrategy:
    def test_gaussian_strategy_numeric_mutation(self) -> None:
        space = {
            "timeout_multiplier": LeverDimension(
                name="timeout_multiplier",
                values=[0.75, 1.0, 1.25],
                mutation_strategy="gaussian",
            ),
        }
        search = LeverMatrixSearch(lever_space=space)
        config = LeverConfiguration(
            agent="test",
            skills=(),
            tools=(),
            context_strategy="file-based",
            prompt_preset="balanced",
            timeout_multiplier=1.0,
        )
        results: set[float] = set()
        for _ in range(50):
            mutate = cast(
                Callable[[LeverConfiguration, str], LeverConfiguration],
                getattr(search, "_mutate_single_lever"),
            )
            mutated = mutate(config, "timeout_multiplier")
            results.add(mutated.timeout_multiplier)
        assert len(results) >= 2

    def test_categorical_strategy_uses_values_list(self) -> None:
        space = {
            "model_routing": LeverDimension(
                name="model_routing",
                values=["default", "fast", "reasoning"],
                mutation_strategy="categorical",
            ),
        }
        search = LeverMatrixSearch(lever_space=space)
        config = LeverConfiguration(
            agent="test",
            skills=(),
            tools=(),
            context_strategy="file-based",
            prompt_preset="balanced",
            model_routing="default",
        )
        for _ in range(20):
            mutate = cast(
                Callable[[LeverConfiguration, str], LeverConfiguration],
                getattr(search, "_mutate_single_lever"),
            )
            mutated = mutate(config, "model_routing")
            assert mutated.model_routing in ["default", "fast", "reasoning"]


class TestModelRoutingInLeverMatrix:
    def test_sample_random_includes_model_routing(self) -> None:
        search = LeverMatrixSearch()
        config = search.sample_random()
        assert config.model_routing in ["default", "fast", "reasoning", "creative"]

    def test_dict_to_config_includes_model_routing(self) -> None:
        search = LeverMatrixSearch()
        config_dict: dict[str, Any] = {"model_routing": "reasoning"}
        dict_to_config = cast(
            Callable[[dict[str, Any]], LeverConfiguration],
            getattr(search, "_dict_to_config"),
        )
        config = dict_to_config(config_dict)
        assert config.model_routing == "reasoning"
