from __future__ import annotations

from ash_hawk.auto_research.types import (
    DEFAULT_LEVER_SPACE,
    EvolvableConfig,
    EvolvableCycleResult,
    LeverConfiguration,
    LeverDimension,
)


class TestEvolvableConfig:
    def test_defaults(self) -> None:
        config = EvolvableConfig()
        assert config.enabled is False
        assert config.max_experiments == 100
        assert config.dimensions == []
        assert config.experiment_log_path == ".ash-hawk/evolvable-experiments.jsonl"
        assert config.improvement_threshold == 0.02
        assert config.safety_threshold == -0.05
        assert config.model_routing_enabled is True

    def test_custom_values(self) -> None:
        config = EvolvableConfig(
            enabled=True,
            max_experiments=50,
            dimensions=["context_strategy", "model_routing"],
            safety_threshold=-0.1,
        )
        assert config.enabled is True
        assert config.max_experiments == 50
        assert config.dimensions == ["context_strategy", "model_routing"]
        assert config.safety_threshold == -0.1


class TestEvolvableCycleResult:
    def test_defaults(self) -> None:
        result = EvolvableCycleResult()
        assert result.total_experiments == 0
        assert result.best_score == 0.0
        assert result.baseline_score == 0.0
        assert result.improvement == 0.0
        assert result.dimensions_explored == []
        assert result.reverted_experiments == 0
        assert result.best_configuration == {}
        assert result.started_at is None
        assert result.completed_at is None

    def test_improvement_calculation(self) -> None:
        result = EvolvableCycleResult(
            baseline_score=0.5,
            best_score=0.7,
            improvement=0.2,
        )
        assert result.improvement == 0.2


class TestLeverDimensionMutationStrategy:
    def test_default_mutation_strategy_is_random(self) -> None:
        dim = LeverDimension(name="test", values=[1, 2, 3])
        assert dim.mutation_strategy == "random"

    def test_custom_mutation_strategy(self) -> None:
        dim = LeverDimension(name="test", values=[1, 2, 3], mutation_strategy="gaussian")
        assert dim.mutation_strategy == "gaussian"

    def test_model_routing_dimension_has_categorical_strategy(self) -> None:
        mr = DEFAULT_LEVER_SPACE["model_routing"]
        assert mr.mutation_strategy == "categorical"
        assert mr.name == "model_routing"
        assert "default" in mr.values
        assert "fast" in mr.values


class TestLeverConfigurationModelRouting:
    def test_default_model_routing_is_empty(self) -> None:
        config = LeverConfiguration(
            agent="test",
            skills=(),
            tools=(),
            context_strategy="file-based",
            prompt_preset="balanced",
        )
        assert config.model_routing == ""

    def test_model_routing_in_config_dict(self) -> None:
        config = LeverConfiguration(
            agent="test",
            skills=(),
            tools=(),
            context_strategy="file-based",
            prompt_preset="balanced",
            model_routing="reasoning",
        )
        d = config.to_config_dict()
        assert d["model_routing"] == "reasoning"
