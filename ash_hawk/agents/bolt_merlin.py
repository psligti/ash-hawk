from __future__ import annotations

from typing import Any

from ash_hawk.agents.dawn_kestrel import DawnKestrelAgentRunner
from ash_hawk.policy import PolicyEnforcer
from ash_hawk.types import EvalOutcome, EvalTask, EvalTranscript


class BoltMerlinAgentRunner(DawnKestrelAgentRunner):
    skill_agent_default_model = "claude-3-5-sonnet-20241022"
    skill_agent_default_tools = ["read", "write", "edit", "grep", "bash", "test"]

    def __init__(
        self, provider: str = "anthropic", model: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(
            provider=provider,
            model=model or self.skill_agent_default_model,
            **kwargs,
        )
        self._agent_type = "bolt-merlin"

    async def run(
        self,
        task: EvalTask,
        policy_enforcer: PolicyEnforcer,
        config: dict[str, Any],
    ) -> tuple[EvalTranscript, EvalOutcome]:
        config["agent_name"] = config.get("agent_name", "bolt-merlin")
        config["temperature"] = config.get("temperature", 0.7)

        skill_tools = self._get_skill_tools()
        if "allowed_tools_override" not in config:
            config["allowed_tools_override"] = skill_tools

        return await super().run(task, policy_enforcer, config)

    def _get_skill_tools(self) -> list[str]:
        return self.skill_agent_default_tools


__all__ = ["BoltMerlinAgentRunner"]
