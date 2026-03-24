from __future__ import annotations

from pathlib import Path
from typing import Any

from ash_hawk.agents.dawn_kestrel import DawnKestrelAgentRunner
from ash_hawk.policy import PolicyEnforcer
from ash_hawk.types import EvalOutcome, EvalTask, EvalTranscript

_VOX_JAY_ASSETS_DIR = Path(__file__).resolve().parents[3] / "vox-jay" / "src" / "vox_jay" / "assets"
_VOX_JAY_USER_DIR = Path.home() / ".vox-jay"
_VOX_JAY_EVALS_DIR = Path(__file__).resolve().parents[3] / "vox-jay" / "evals"


class VoxJayAgentRunner(DawnKestrelAgentRunner):
    """Communication agent for content generation following persona guidelines."""

    communication_agent_default_model = "claude-3-5-sonnet-20241022"
    communication_agent_default_tools = ["read", "write", "format", "summarize"]

    def __init__(
        self, provider: str = "anthropic", model: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(
            provider=provider,
            model=model or self.communication_agent_default_model,
            **kwargs,
        )
        self._agent_type = "vox-jay"

    @classmethod
    def get_scenario_directory(cls) -> Path | None:
        if _VOX_JAY_EVALS_DIR.exists():
            return _VOX_JAY_EVALS_DIR
        return None

    async def run(
        self,
        task: EvalTask,
        policy_enforcer: PolicyEnforcer,
        config: dict[str, Any],
    ) -> tuple[EvalTranscript, EvalOutcome]:
        config["agent_name"] = config.get("agent_name", "vox-jay")
        config["temperature"] = config.get("temperature", 0.3)

        comm_tools = self._get_communication_tools()
        if "allowed_tools_override" not in config:
            config["allowed_tools_override"] = comm_tools

        return await super().run(task, policy_enforcer, config)

    def _get_communication_tools(self) -> list[str]:
        return self.communication_agent_default_tools


__all__ = ["VoxJayAgentRunner"]
