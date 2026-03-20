"""Registry for managing Evaluator Packs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ash_hawk.eval_packs.base import EvalPack

if TYPE_CHECKING:
    pass


class PackRegistry:
    """Registry for discovering and retrieving Evaluator Packs.

    Provides centralized management of eval packs with:
    - Registration by pack_id
    - Lookup by agent name
    - Global registry instance
    """

    def __init__(self) -> None:
        """Initialize an empty pack registry."""
        self._packs: dict[str, EvalPack] = {}

    def register(self, pack: EvalPack) -> None:
        """Register an eval pack.

        Args:
            pack: The eval pack to register.

        Raises:
            ValueError: If a pack with the same ID is already registered.
        """
        if pack.pack_id in self._packs:
            raise ValueError(f"Pack with ID '{pack.pack_id}' already registered")
        self._packs[pack.pack_id] = pack

    def unregister(self, pack_id: str) -> bool:
        """Remove a pack from the registry.

        Args:
            pack_id: ID of the pack to remove.

        Returns:
            True if the pack was removed, False if not found.
        """
        if pack_id in self._packs:
            del self._packs[pack_id]
            return True
        return False

    def get(self, pack_id: str) -> EvalPack | None:
        """Get a pack by its ID.

        Args:
            pack_id: The pack ID to look up.

        Returns:
            The eval pack if found, None otherwise.
        """
        return self._packs.get(pack_id)

    def get_for_agent(self, agent_name: str) -> list[EvalPack]:
        """Get all packs that apply to a given agent.

        Args:
            agent_name: Name of the agent to find packs for.

        Returns:
            List of eval packs targeting this agent.
        """
        return [pack for pack in self._packs.values() if pack.is_for_agent(agent_name)]

    def list_all(self) -> list[EvalPack]:
        """Get all registered packs.

        Returns:
            List of all eval packs in the registry.
        """
        return list(self._packs.values())

    def list_ids(self) -> list[str]:
        """Get all registered pack IDs.

        Returns:
            List of pack IDs in the registry.
        """
        return list(self._packs.keys())

    def clear(self) -> None:
        """Remove all packs from the registry."""
        self._packs.clear()


# Global registry instance
_global_registry: PackRegistry | None = None


def get_pack_registry() -> PackRegistry:
    """Get the global pack registry, initializing if needed.

    Returns:
        The global PackRegistry instance.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PackRegistry()
        _register_default_packs(_global_registry)
    return _global_registry


def _register_default_packs(registry: PackRegistry) -> None:
    """Register the default built-in eval packs.

    Args:
        registry: The registry to populate.
    """
    from ash_hawk.eval_packs.bolt_merlin_pack import BoltMerlinEvalPack
    from ash_hawk.eval_packs.iron_rook_pack import IronRookEvalPack
    from ash_hawk.eval_packs.packs import (
        ComprehensiveEvalPack,
        HarnessEvalPack,
        PolicyEvalPack,
        SkillEvalPack,
        ToolEvalPack,
    )
    from ash_hawk.eval_packs.vox_jay_pack import VoxJayEvalPack

    registry.register(PolicyEvalPack)
    registry.register(SkillEvalPack)
    registry.register(ToolEvalPack)
    registry.register(HarnessEvalPack)
    registry.register(ComprehensiveEvalPack)
    registry.register(IronRookEvalPack)
    registry.register(BoltMerlinEvalPack)
    registry.register(VoxJayEvalPack)
