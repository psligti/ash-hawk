from __future__ import annotations

from pathlib import Path

import pytest

from ash_hawk.curation.experiment_store import ExperimentStore
from ash_hawk.cycle.coordinator import IterationCoordinator
from ash_hawk.cycle.types import CycleConfig, CycleStatus
from ash_hawk.storage import FileStorage


@pytest.mark.asyncio
async def test_iteration_coordinator_uses_improve_cycle_orchestrator(tmp_path: Path) -> None:
    config = CycleConfig(
        cycle_id="cycle-test",
        experiment_id="exp-test",
        target_agent="bolt-merlin",
        max_iterations=1,
        metadata={
            "enable_competitor": True,
            "enable_triage": True,
            "enable_verifier": True,
            "enable_adversary": True,
        },
    )
    storage = FileStorage(str(tmp_path / "storage"))
    experiment_store = ExperimentStore(base_path=tmp_path)
    coordinator = IterationCoordinator(config, storage, experiment_store)

    result = await coordinator.run_iteration(iteration_num=1, lessons_to_apply=[])

    assert result.status == CycleStatus.COMPLETED
    assert result.lessons_generated >= 0
    role_summaries = result.metadata.get("role_summaries", {})
    assert isinstance(role_summaries, dict)
    assert "verifier" in role_summaries
    assert "promotion_manager" in role_summaries


@pytest.mark.asyncio
async def test_iteration_coordinator_respects_disabled_roles(tmp_path: Path) -> None:
    config = CycleConfig(
        cycle_id="cycle-test-disabled",
        experiment_id="exp-test-disabled",
        target_agent="bolt-merlin",
        max_iterations=1,
        metadata={
            "enable_competitor": False,
            "enable_triage": False,
            "enable_verifier": False,
            "enable_adversary": False,
        },
    )
    storage = FileStorage(str(tmp_path / "storage"))
    experiment_store = ExperimentStore(base_path=tmp_path)
    coordinator = IterationCoordinator(config, storage, experiment_store)

    result = await coordinator.run_iteration(iteration_num=1, lessons_to_apply=[])

    assert result.status == CycleStatus.COMPLETED
    role_summaries = result.metadata.get("role_summaries", {})
    assert isinstance(role_summaries, dict)
    assert "verifier" in role_summaries
