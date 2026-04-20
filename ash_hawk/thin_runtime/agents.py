from __future__ import annotations

from ash_hawk.thin_runtime.models import AgentSpec, RegistrySummary, ThinRuntimeCatalog


class AgentRegistry:
    def __init__(self, agents: list[AgentSpec]) -> None:
        self._agents = {agent.name: agent for agent in agents}

    def get(self, name: str) -> AgentSpec:
        try:
            return self._agents[name]
        except KeyError as exc:
            raise ValueError(f"Unknown thin runtime agent: {name}") from exc

    def list_agents(self) -> list[AgentSpec]:
        return list(self._agents.values())

    def summary(self, catalog: ThinRuntimeCatalog) -> RegistrySummary:
        return RegistrySummary(
            agents=len(self._agents),
            skills=len(catalog.skills),
            tools=len(catalog.tools),
            hooks=len(catalog.hooks),
            memory_scopes=len(catalog.memory_scopes),
            context_fields=len(catalog.context_fields),
        )
