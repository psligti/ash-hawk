from __future__ import annotations

from pathlib import Path
from typing import Any

from ash_hawk.agents.dawn_kestrel import DawnKestrelAgentRunner
from ash_hawk.policy import PolicyEnforcer
from ash_hawk.types import EvalOutcome, EvalTask, EvalTranscript

_IRON_ROOK_USER_DIR = Path.home() / ".iron-rook"
_IRON_ROOK_PACKAGE_DIR = Path(__file__).resolve().parents[3] / "iron-rook"


class IronRookAgentRunner(DawnKestrelAgentRunner):
    """Policy enforcement agent for rule validation and checking."""

    policy_agent_default_model = "claude-3-5-sonnet-20241022"
    policy_agent_default_tools = ["validate", "check_policy", "enforce_rules"]

    def __init__(
        self, provider: str = "anthropic", model: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(
            provider=provider,
            model=model or self.policy_agent_default_model,
            **kwargs,
        )
        self._agent_type = "iron-rook"

    async def run(
        self,
        task: EvalTask,
        policy_enforcer: PolicyEnforcer,
        config: dict[str, Any],
    ) -> tuple[EvalTranscript, EvalOutcome]:
        config["agent_name"] = config.get("agent_name", "iron-rook")

        policy_tools = self._get_policy_tools()
        if "allowed_tools_override" not in config:
            config["allowed_tools_override"] = policy_tools

        return await super().run(task, policy_enforcer, config)

    def _get_policy_tools(self) -> list[str]:
        return self.policy_agent_default_tools


__all__ = ["IronRookAgentRunner"]
