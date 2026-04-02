from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ash_hawk.research.target_registry import TargetEntry, TargetRegistry
from ash_hawk.research.types import TargetSurface


class TestTargetRegistry:
    def test_target_entry_success_rate(self) -> None:
        entry = TargetEntry(
            name="target",
            surface=TargetSurface.PROMPT,
            description="desc",
            mutation_count=4,
            success_count=2,
        )

        assert entry.success_rate == 0.5

    def test_target_entry_success_rate_zero_mutations(self) -> None:
        entry = TargetEntry(
            name="target",
            surface=TargetSurface.PROMPT,
            description="desc",
        )

        assert entry.success_rate == 0.0

    def test_register_adds_entry(self) -> None:
        registry = TargetRegistry(min_active=1)
        entry = TargetEntry(
            name="target",
            surface=TargetSurface.POLICY,
            description="policy",
        )

        registry.register(entry)

        assert registry.get_active_targets()[0].name == "target"

    def test_register_merges_existing_entry(self) -> None:
        registry = TargetRegistry(min_active=1)
        existing = TargetEntry(
            name="target",
            surface=TargetSurface.POLICY,
            description="",
            mutation_count=1,
            success_count=0,
            last_mutated="2024-01-01T00:00:00+00:00",
        )
        registry.register(existing)
        update = TargetEntry(
            name="target",
            surface=TargetSurface.POLICY,
            description="updated",
            mutation_count=2,
            success_count=1,
            last_mutated="2024-02-01T00:00:00+00:00",
        )

        registry.register(update)
        entry = registry.get_active_targets()[0]

        assert entry.description == "updated"
        assert entry.mutation_count == 3
        assert entry.success_count == 1
        assert entry.last_mutated == "2024-02-01T00:00:00+00:00"

    def test_get_active_targets_sorted_by_correlation(self) -> None:
        registry = TargetRegistry(min_active=1)
        low = TargetEntry(
            name="low",
            surface=TargetSurface.TOOL,
            description="",
            correlation_with_improvement=0.1,
        )
        high = TargetEntry(
            name="high",
            surface=TargetSurface.PROMPT,
            description="",
            correlation_with_improvement=0.9,
        )
        registry.register(low)
        registry.register(high)

        active = registry.get_active_targets()

        assert [entry.name for entry in active] == ["high", "low"]

    def test_update_correlation_increments_counts(self) -> None:
        registry = TargetRegistry(min_active=1)
        entry = TargetEntry(
            name="target",
            surface=TargetSurface.POLICY,
            description="",
        )
        registry.register(entry)

        registry.update_correlation("target", score_delta=0.1)
        updated = registry.get_active_targets()[0]

        assert updated.mutation_count == 1
        assert updated.success_count == 1
        assert updated.last_mutated is not None

    def test_prune_low_correlation_respects_min_active(self) -> None:
        registry = TargetRegistry(min_active=2)
        registry.register(
            TargetEntry(
                name="a",
                surface=TargetSurface.PROMPT,
                description="",
                correlation_with_improvement=0.1,
            )
        )
        registry.register(
            TargetEntry(
                name="b",
                surface=TargetSurface.POLICY,
                description="",
                correlation_with_improvement=0.2,
            )
        )

        pruned = registry.prune_low_correlation(threshold=0.5)

        assert pruned == []
        assert all(entry.status == "active" for entry in registry.get_active_targets())

    def test_prune_low_correlation_returns_pruned_names(self) -> None:
        registry = TargetRegistry(min_active=1)
        registry.register(
            TargetEntry(
                name="low",
                surface=TargetSurface.PROMPT,
                description="",
                correlation_with_improvement=0.1,
            )
        )
        registry.register(
            TargetEntry(
                name="mid",
                surface=TargetSurface.POLICY,
                description="",
                correlation_with_improvement=0.2,
            )
        )
        registry.register(
            TargetEntry(
                name="high",
                surface=TargetSurface.TOOL,
                description="",
                correlation_with_improvement=0.9,
            )
        )

        pruned = registry.prune_low_correlation(threshold=0.5)

        assert set(pruned) == {"low", "mid"}
        active = registry.get_active_targets()
        assert [entry.name for entry in active] == ["high"]

    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        registry = TargetRegistry(storage_path=tmp_path, min_active=1)
        entry = TargetEntry(
            name="target",
            surface=TargetSurface.PROMPT,
            description="desc",
            mutation_count=2,
            success_count=1,
            correlation_with_improvement=0.4,
            last_mutated=datetime.now(UTC).isoformat(),
        )
        registry.register(entry)

        await registry.save()
        loaded = TargetRegistry.load(tmp_path, min_active=1)

        loaded_entry = loaded.get_active_targets()[0]
        assert loaded_entry.name == "target"
        assert loaded_entry.success_count == 1

    def test_load_from_missing_path_returns_empty(self, tmp_path: Path) -> None:
        registry = TargetRegistry.load(tmp_path / "missing")

        assert registry.get_active_targets() == []
