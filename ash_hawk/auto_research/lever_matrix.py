"""Lever matrix search for configuration space exploration.

# type-hygiene: skip-file  # pre-existing Any — lever values are intentionally heterogeneous

Explores the combinatorial space of agent/skills/tools/context_strategy/prompt
configurations using random sampling, neighbor mutation, and crossover.
"""

from __future__ import annotations

import logging
import random
from dataclasses import replace
from hashlib import sha1
from json import dumps
from pathlib import Path
from typing import Any

from ash_hawk.auto_research.types import (
    DEFAULT_LEVER_SPACE,
    LeverConfiguration,
    LeverDimension,
)
from ash_hawk.config import get_config
from ash_hawk.scenario import run_scenarios_async
from ash_hawk.services.dawn_kestrel_injector import DawnKestrelInjector

logger = logging.getLogger(__name__)

_LEVER_FIELD_MAP: dict[str, str] = {
    "agent": "agent",
    "skills": "skills",
    "tools": "tools",
    "context_strategy": "context_strategy",
    "prompt_preset": "prompt_preset",
    "timeout_multiplier": "timeout_multiplier",
    "model_routing": "model_routing",
}

# Dimension combination validation rules
_INVALID_COMBOS: frozenset[tuple[str, str]] = frozenset(
    {
        ("dynamic", "precision"),
        ("composite", "throughput"),
    }
)


def validate_combination(config_dict: dict[str, Any]) -> list[str]:
    """Validate a dimension combination and return list of warnings.

    Args:
        config_dict: A dict mapping dimension names to values (e.g., from
            ``LeverConfiguration.to_config_dict()``).

    Returns:
        List of warning strings. Empty list means the combination is valid.
    """
    warnings: list[str] = []
    context = config_dict.get("context_strategy", "")
    preset = config_dict.get("prompt_preset", "")

    pair = (context, preset)
    if pair in _INVALID_COMBOS:
        warnings.append(
            f"Combination {pair} may conflict: context_strategy={context!r} "
            f"with prompt_preset={preset!r}"
        )

    return warnings


class LeverMatrixSearch:
    """Search the configuration space via random sampling, mutation, and crossover.

    Each dimension in the lever space maps to a LeverConfiguration field.
    Dimensions have weights (for selection probability) and mutation rates
    (probability of mutating that dimension when chosen).
    """

    def __init__(
        self,
        lever_space: dict[str, LeverDimension] | None = None,
    ) -> None:
        self.lever_space = lever_space or DEFAULT_LEVER_SPACE
        self._rng = random.Random()  # nosec B311

    def sample_random(self) -> LeverConfiguration:
        """Sample a random configuration from the lever space."""
        config_dict: dict[str, Any] = {}
        for name, dimension in self.lever_space.items():
            value = self._rng.choice(dimension.values)
            config_dict[name] = value
        return self._dict_to_config(config_dict)

    def sample_neighbors(
        self,
        config: LeverConfiguration,
        n: int = 5,
    ) -> list[LeverConfiguration]:
        """Sample n neighbors by mutating 1-2 levers.

        Each neighbor differs from the source by 1 or 2 dimensions. Dimensions
        are selected weighted by their ``weight`` attribute, and only actually
        mutated when a random draw falls below their ``mutation_rate``.
        """
        neighbors: list[LeverConfiguration] = []
        lever_names = list(self.lever_space.keys())
        weights = [self.lever_space[name].weight for name in lever_names]

        for _ in range(n):
            num_mutations = self._rng.choice([1, 2])
            selected = self._weighted_sample(lever_names, weights, num_mutations)
            mutated = config
            for lever_name in selected:
                dimension = self.lever_space[lever_name]
                if self._rng.random() < dimension.mutation_rate:
                    mutated = self._mutate_single_lever(mutated, lever_name)
            neighbors.append(mutated)

        return neighbors

    def crossover(
        self,
        a: LeverConfiguration,
        b: LeverConfiguration,
    ) -> LeverConfiguration:
        """Combine two configurations via uniform crossover.

        For each lever dimension, randomly pick the value from parent *a* or *b*.
        """
        config_dict: dict[str, Any] = {}
        a_dict = a.to_config_dict()
        b_dict = b.to_config_dict()

        for name in self.lever_space:
            field = _LEVER_FIELD_MAP.get(name, name)
            source = a_dict if self._rng.random() < 0.5 else b_dict
            config_dict[name] = source.get(field, a_dict.get(field))

        return self._dict_to_config(config_dict)

    async def evaluate(
        self,
        config: LeverConfiguration,
        scenarios: list[Path],
        storage_path: Path,
    ) -> float:
        """Evaluate a configuration against scenarios.

        Runs the given scenario files via :func:`run_scenarios_async` and returns
        the ``mean_score`` from the summary metrics as fitness (0.0–1.0).

        Args:
            config: The lever configuration to evaluate.
            scenarios: Scenario YAML file paths.
            storage_path: Directory for evaluation artefacts.

        Returns:
            Fitness score between 0.0 and 1.0.
        """
        logger.info(
            "Evaluating configuration: agent=%s, skills=%s, context=%s, prompt=%s",
            config.agent,
            config.skills,
            config.context_strategy,
            config.prompt_preset,
        )

        eval_storage = storage_path / _config_fingerprint(config)
        eval_storage.mkdir(parents=True, exist_ok=True)

        injector = _build_injector_from_config(config)
        base_timeout = float(get_config().default_timeout_seconds)
        timeout_seconds = max(1.0, base_timeout * config.timeout_multiplier)

        summary = await run_scenarios_async(
            paths=[str(p) for p in scenarios],
            storage_path=eval_storage,
            injector=injector,
            scenario_timeout_seconds=timeout_seconds,
            grader_config_overrides={"quiet": True},
        )
        fitness = summary.metrics.mean_score
        logger.info("Configuration fitness: %.4f", fitness)
        return fitness

    def _mutate_single_lever(
        self,
        config: LeverConfiguration,
        lever_name: str,
    ) -> LeverConfiguration:
        """Mutate a single lever dimension.

        Picks a new value from the dimension's values list that differs from the
        current value. If only one value exists the config is returned unchanged.
        """
        dimension = self.lever_space[lever_name]
        field = _LEVER_FIELD_MAP.get(lever_name, lever_name)
        current_value = self._get_field(config, field)
        strategy = getattr(dimension, "mutation_strategy", "random")

        if strategy == "gaussian" and isinstance(current_value, int | float):
            deviation = max(0.01, abs(current_value) * 0.1)
            new_value = current_value + self._rng.gauss(0, deviation)
            new_value = type(current_value)(new_value)
            return self._set_field(config, field, new_value)

        candidates = [v for v in dimension.values if v != current_value]
        if not candidates:
            return config

        new_value = self._rng.choice(candidates)
        return self._set_field(config, field, new_value)

    def _weighted_sample(
        self,
        items: list[str],
        weights: list[float],
        k: int,
    ) -> list[str]:
        """Sample *k* unique items weighted by *weights*."""
        k = min(k, len(items))
        selected: list[str] = []
        remaining = list(zip(items, weights))

        for _ in range(k):
            total = sum(w for _, w in remaining)
            if total <= 0:
                break
            threshold = self._rng.random() * total
            cumulative = 0.0
            for idx, (item, weight) in enumerate(remaining):
                cumulative += weight
                if cumulative >= threshold:
                    selected.append(item)
                    remaining.pop(idx)
                    break

        return selected

    def _dict_to_config(self, d: dict[str, Any]) -> LeverConfiguration:
        """Convert a raw lever-name → value dict to a LeverConfiguration."""
        raw_skills: list[str] = d.get("skills", [])
        raw_tools: list[str] = d.get("tools", [])
        return LeverConfiguration(
            agent=str(d.get("agent", "default")),
            skills=tuple(raw_skills),
            tools=tuple(raw_tools),
            context_strategy=str(d.get("context_strategy", "file-based")),
            prompt_preset=str(d.get("prompt_preset", "balanced")),
            timeout_multiplier=float(d.get("timeout_multiplier", 1.0)),
            model_routing=str(d.get("model_routing", "")),
        )

    @staticmethod
    def _get_field(config: LeverConfiguration, field: str) -> Any:
        """Read a field value from a LeverConfiguration."""
        return getattr(config, field)

    @staticmethod
    def _set_field(config: LeverConfiguration, field: str, value: Any) -> LeverConfiguration:
        """Return a new LeverConfiguration with one field replaced.

        Handles list → tuple coercion for tuple-typed fields.
        """
        if field in ("skills", "tools"):
            value = _coerce_to_str_tuple(value)
        return replace(config, **{field: value})


def _coerce_to_str_tuple(value: Any) -> tuple[str, ...]:
    raw: list[Any] = list(value) if not isinstance(value, str) else [value]
    return tuple(str(v) for v in raw)


def _build_injector_from_config(config: LeverConfiguration) -> DawnKestrelInjector:
    strategy = _resolve_context_strategy(config.context_strategy)
    current_skill_name = config.skills[0] if config.skills else None
    if config.model_routing and config.model_routing != "default":
        # TODO: wire model_routing when DawnKestrelInjector supports it.
        pass
    return DawnKestrelInjector(
        project_root=Path.cwd(),
        strategy=strategy,
        current_skill_name=current_skill_name,
    )


def _resolve_context_strategy(strategy_name: str) -> Any | None:
    try:
        from dawn_kestrel.agents.context import (
            CompositeContextStrategy,
            DynamicContextStrategy,
            FileBasedContextStrategy,
        )
    except ImportError:
        return None

    normalized = strategy_name.strip().lower()
    project_root = Path.cwd()
    if normalized == "dynamic":
        return DynamicContextStrategy(project_root)
    if normalized == "composite":
        return CompositeContextStrategy(
            strategies=[
                FileBasedContextStrategy(project_root),
                DynamicContextStrategy(project_root),
            ],
            project_root=project_root,
        )

    return FileBasedContextStrategy(project_root)


def _config_fingerprint(config: LeverConfiguration) -> str:
    payload = dumps(config.to_config_dict(), sort_keys=True)
    return sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]  # nosec B324
