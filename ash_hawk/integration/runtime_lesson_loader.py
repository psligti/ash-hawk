"""Runtime lesson loader for dawn-kestrel integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ash_hawk.curation.experiment_store import ExperimentStore
from ash_hawk.experiments.registry import ExperimentRegistry
from ash_hawk.services.lesson_injector import LessonInjector
from ash_hawk.services.lesson_service import LessonService


@dataclass
class LessonContext:
    """Loaded lessons ready for runtime injection."""

    prompt_additions: str = ""
    tool_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    policy_rules: list[dict[str, Any]] = field(default_factory=list)
    harness_adjustments: dict[str, Any] = field(default_factory=dict)
    lesson_count: int = 0
    experiment_id: str | None = None


class RuntimeLessonLoader:
    """Load and inject lessons at dawn-kestrel runtime start.

    Usage in dawn-kestrel agents/runtime.py:

        from ash_hawk.integration.runtime_lesson_loader import RuntimeLessonLoader

        class AgentRuntime:
            def __init__(self, ...):
                ...
                self._lesson_loader = RuntimeLessonLoader(
                    experiment_id=options.get("experiment_id") if options else None,
                )

            async def execute_agent(self, agent_name, ...):
                lesson_ctx = self._lesson_loader.load(agent_name)

                # Inject into prompt
                system_prompt = lesson_ctx.prompt_additions + "\\n\\n" + base_prompt

                # Apply tool overrides
                for tool_name, overrides in lesson_ctx.tool_overrides.items():
                    self._apply_tool_overrides(tool_name, overrides)

                # Apply policy rules
                self._policy_engine.add_rules(lesson_ctx.policy_rules)
    """

    def __init__(
        self,
        experiment_id: str | None = None,
        strategy_filter: str | None = None,
        sub_strategy_filter: str | None = None,
    ) -> None:
        self._experiment_id = experiment_id
        self._strategy_filter = strategy_filter
        self._sub_strategy_filter = sub_strategy_filter
        self._service = LessonService()
        self._experiment_store = ExperimentStore()
        self._registry = ExperimentRegistry()

    def load(self, agent_id: str) -> LessonContext:
        """Load all applicable lessons for an agent.

        If experiment_id is set, loads from experiment-scoped storage.
        Otherwise loads from global lesson store.
        """
        injector = LessonInjector(
            strategy_filter=self._strategy_filter,
            sub_strategy_filter=self._sub_strategy_filter,
        )

        if self._experiment_id:
            lessons = self._experiment_store.get_for_agent(agent_id, self._experiment_id)
        else:
            lessons = injector.get_all_lessons(agent_id)

        if not lessons:
            return LessonContext(experiment_id=self._experiment_id)

        prompt = injector.inject_into_prompt(agent_id, "")
        tool_overrides = injector.get_tool_overrides(agent_id)
        policy_rules = injector.get_policy_rules(agent_id)
        harness_adjustments = injector.get_harness_adjustments(agent_id)

        return LessonContext(
            prompt_additions=prompt,
            tool_overrides=tool_overrides,
            policy_rules=policy_rules,
            harness_adjustments=harness_adjustments,
            lesson_count=len(lessons),
            experiment_id=self._experiment_id,
        )

    def load_for_experiment(
        self,
        agent_id: str,
        experiment_id: str,
    ) -> LessonContext:
        """Load lessons from a specific experiment."""
        loader = RuntimeLessonLoader(
            experiment_id=experiment_id,
            strategy_filter=self._strategy_filter,
            sub_strategy_filter=self._sub_strategy_filter,
        )
        return loader.load(agent_id)

    def get_active_experiments(self) -> list[dict[str, Any]]:
        """Get all active experiments."""
        experiments = self._registry.get_active()
        return [
            {
                "experiment_id": exp.experiment_id,
                "strategy": exp.strategy.value if exp.strategy else None,
                "target_agent": exp.target_agent,
                "trial_count": exp.trial_count,
                "lesson_count": exp.lesson_count,
            }
            for exp in experiments
        ]

    def create_experiment(
        self,
        experiment_id: str,
        target_agent: str,
        strategy: str | None = None,
        sub_strategies: list[str] | None = None,
    ) -> None:
        """Create a new experiment for tracking."""
        config: dict[str, Any] = {"target_agent": target_agent}
        if strategy:
            config["strategy"] = strategy
        if sub_strategies:
            config["sub_strategies"] = sub_strategies

        self._registry.create(experiment_id, config)
