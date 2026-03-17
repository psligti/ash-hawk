from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pydantic as pd
import yaml

from ash_hawk.strategies.registry import Strategy, SubStrategy


class ExperimentConfig(pd.BaseModel):
    """Configuration for strategy-scoped experiments."""

    model_config = pd.ConfigDict(extra="forbid")

    experiment_id: str
    target_agent: str
    strategy: Strategy | None = None
    sub_strategies: list[SubStrategy] = pd.Field(default_factory=list)
    trial_count: int = 10
    baseline_experiment_id: str | None = None
    lesson_injection_enabled: bool = True
    created_at: datetime = pd.Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_yaml(cls, content: str) -> ExperimentConfig:
        """Load config from YAML string."""
        data = yaml.safe_load(content)
        return cls(**data)

    def to_yaml(self) -> str:
        """Export config to YAML string."""
        data = self.model_dump()
        # Convert enums to strings for YAML
        if self.strategy:
            data["strategy"] = self.strategy.value
        data["sub_strategies"] = [s.value for s in self.sub_strategies]
        data["created_at"] = self.created_at.isoformat()
        
        return str(yaml.dump(data, default_flow_style=False))
