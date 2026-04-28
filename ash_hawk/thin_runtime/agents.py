from __future__ import annotations

from ash_hawk.thin_runtime.models import AgentSpec


class AgentRegistry:
    def __init__(self, agents: list[AgentSpec]) -> None:
        self._agents = {agent.name: agent for agent in agents}

    def get(self, name: str) -> AgentSpec:
        try:
            return self._agents[name]
        except KeyError as exc:
            raise ValueError(f"Unknown thin runtime agent: {name}") from exc
