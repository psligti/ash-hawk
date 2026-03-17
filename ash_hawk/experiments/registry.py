"""Experiment tracking registry."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pydantic as pd

from ash_hawk.strategies import Strategy, SubStrategy


class Experiment(pd.BaseModel):
    """Experiment model for tracking parallel trials."""

    experiment_id: str = pd.Field(description="Unique experiment identifier")
    strategy: Strategy | None = pd.Field(default=None, description="Top-level improvement strategy")
    sub_strategies: list[SubStrategy] = pd.Field(
        default_factory=list,
        description="Sub-strategies addressed",
    )
    target_agent: str | None = pd.Field(default=None, description="Target agent for the experiment")
    trial_count: int = pd.Field(default=0, ge=0, description="Number of trials run")
    status: Literal["active", "completed", "paused", "failed"] = pd.Field(
        default="active",
        description="Experiment status",
    )
    created_at: datetime = pd.Field(description="Creation timestamp")
    completed_at: datetime | None = pd.Field(default=None, description="Completion timestamp")
    lesson_count: int = pd.Field(default=0, ge=0, description="Number of lessons generated")
    baseline_experiment_id: str | None = pd.Field(
        default=None,
        description="ID of baseline experiment for comparison",
    )
    variant: str | None = pd.Field(default=None, description="A/B test variant identifier")
    metadata: dict[str, Any] = pd.Field(default_factory=dict, description="Additional metadata")

    model_config = pd.ConfigDict(extra="forbid")

    def is_active(self) -> bool:
        """Check if experiment is currently active."""
        return self.status == "active"

    def is_completed(self) -> bool:
        """Check if experiment has completed."""
        return self.status == "completed"


class ExperimentRegistry:
    """Registry for tracking experiments.

    Stores experiments in .ash-hawk/experiments/registry.json
    """

    def __init__(self, base_path: Path | None = None) -> None:
        self._base_path = base_path or Path(".ash-hawk")
        self._storage_path = self._base_path / "experiments"
        self._registry_file = self._storage_path / "registry.json"
        self._experiments: dict[str, Experiment] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load experiments from disk if not already loaded."""
        if self._loaded:
            return
        if self._registry_file.exists():
            try:
                with open(self._registry_file) as f:
                    data = json.load(f)
                for exp_data in data.get("experiments", []):
                    exp = Experiment(**exp_data)
                    self._experiments[exp.experiment_id] = exp
            except (json.JSONDecodeError, KeyError):
                pass
        self._loaded = True

    def _persist(self) -> None:
        """Persist experiments to disk."""
        self._storage_path.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "experiments": [],
            "last_updated": datetime.now(UTC).isoformat(),
        }
        for exp in self._experiments.values():
            exp_dict = exp.model_dump()
            exp_dict["created_at"] = exp.created_at.isoformat()
            exp_dict["completed_at"] = exp.completed_at.isoformat() if exp.completed_at else None
            data["experiments"].append(exp_dict)
        with open(self._registry_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def create(
        self,
        experiment_id: str,
        config: dict[str, Any],
    ) -> Experiment:
        """Create a new experiment."""
        self._ensure_loaded()

        if experiment_id in self._experiments:
            raise ValueError(f"Experiment '{experiment_id}' already exists")

        exp = Experiment(
            experiment_id=experiment_id,
            strategy=config.get("strategy"),
            sub_strategies=config.get("sub_strategies", []),
            target_agent=config.get("target_agent"),
            baseline_experiment_id=config.get("baseline_experiment_id"),
            variant=config.get("variant"),
            metadata=config.get("metadata", {}),
            created_at=datetime.now(UTC),
        )
        self._experiments[experiment_id] = exp
        self._persist()
        return exp

    def get(self, experiment_id: str) -> Experiment | None:
        """Get an experiment by ID."""
        self._ensure_loaded()
        return self._experiments.get(experiment_id)

    def get_or_create(
        self,
        experiment_id: str,
        config: dict[str, Any],
    ) -> Experiment:
        """Get an existing experiment or create if not exists."""
        self._ensure_loaded()
        if experiment_id in self._experiments:
            return self._experiments[experiment_id]
        return self.create(experiment_id, config)

    def list_all(self) -> list[Experiment]:
        """List all experiments."""
        self._ensure_loaded()
        return list(self._experiments.values())

    def update_status(
        self,
        experiment_id: str,
        status: Literal["active", "completed", "paused", "failed"],
    ) -> Experiment | None:
        """Update experiment status."""
        self._ensure_loaded()
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None

        updated = Experiment(
            **exp.model_dump(exclude={"status", "completed_at"}),
            status=status,
            completed_at=datetime.now(UTC) if status == "completed" else exp.completed_at,
        )
        self._experiments[experiment_id] = updated
        self._persist()
        return updated

    def increment_trial_count(self, experiment_id: str) -> Experiment | None:
        """Increment the trial count for an experiment."""
        self._ensure_loaded()
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None

        updated = Experiment(
            **exp.model_dump(exclude={"trial_count"}),
            trial_count=exp.trial_count + 1,
        )
        self._experiments[experiment_id] = updated
        self._persist()
        return updated

    def increment_lesson_count(self, experiment_id: str, count: int = 1) -> Experiment | None:
        """Increment the lesson count for an experiment."""
        self._ensure_loaded()
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None

        updated = Experiment(
            **exp.model_dump(exclude={"lesson_count"}),
            lesson_count=exp.lesson_count + count,
        )
        self._experiments[experiment_id] = updated
        self._persist()
        return updated

    def get_active(self) -> list[Experiment]:
        """Get all active experiments."""
        self._ensure_loaded()
        return [exp for exp in self._experiments.values() if exp.status == "active"]

    def get_by_strategy(self, strategy: Strategy) -> list[Experiment]:
        """Get all experiments for a specific strategy."""
        self._ensure_loaded()
        return [exp for exp in self._experiments.values() if exp.strategy == strategy]

    def get_by_agent(self, agent_id: str) -> list[Experiment]:
        """Get all experiments for a specific agent."""
        self._ensure_loaded()
        return [exp for exp in self._experiments.values() if exp.target_agent == agent_id]

    def delete(self, experiment_id: str) -> bool:
        """Delete an experiment from the registry."""
        self._ensure_loaded()
        if experiment_id not in self._experiments:
            return False
        del self._experiments[experiment_id]
        self._persist()
        return True
