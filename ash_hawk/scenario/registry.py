"""Scenario adapter registry for managing and discovering scenario adapters.

This module provides a registry for scenario adapters that supports:
- Manual registration of adapter instances
- Lookup by name
- Discovery via Python entry points
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ash_hawk.scenario.adapters import ScenarioAdapter

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "ash_hawk.scenario_adapters"


class ScenarioAdapterRegistry:
    """Registry for managing scenario adapter instances.

    The registry supports:
    - Manual registration via register()
    - Lookup via get()
    - Listing all registered adapters via list_adapters()
    - Loading from entry points via load_from_entry_points()

    Adapters are stored by name and must be unique. Attempting to register
    an adapter with a name that already exists will overwrite the previous one.
    """

    def __init__(self) -> None:
        """Initialize an empty scenario adapter registry."""
        self._adapters: dict[str, ScenarioAdapter] = {}

    def register(self, adapter: ScenarioAdapter) -> None:
        """Register a scenario adapter instance.

        If an adapter with the same name already exists, it will be replaced.

        Args:
            adapter: The scenario adapter instance to register.
        """
        name = adapter.name
        if name in self._adapters:
            logger.warning(
                "Overwriting existing scenario adapter '%s' with %s",
                name,
                adapter.__class__.__name__,
            )
        self._adapters[name] = adapter
        logger.debug("Registered scenario adapter: %s", name)

    def get(self, name: str) -> ScenarioAdapter | None:
        """Get a scenario adapter by name.

        Args:
            name: The name of the adapter to retrieve.

        Returns:
            The adapter instance if found, None otherwise.
        """
        return self._adapters.get(name)

    def list_adapters(self) -> list[str]:
        """List all registered adapter names.

        Returns:
            A sorted list of registered adapter names.
        """
        return sorted(self._adapters.keys())

    def load_from_entry_points(self) -> None:
        """Load scenario adapters from Python entry points.

        Entry points should be defined in the 'ash_hawk.scenario_adapters' group.
        Each entry point should return a ScenarioAdapter instance when called.

        Example pyproject.toml entry:
            [project.entry-points."ash_hawk.scenario_adapters"]
            my_adapter = "my_package.adapters:MyAdapter()"

        Adapters loaded from entry points will be registered automatically.
        If an adapter with the same name already exists, a warning will be logged
        but the existing adapter will NOT be overwritten.
        """
        try:
            eps = entry_points(group=ENTRY_POINT_GROUP)
        except TypeError:
            # Python < 3.10 compatibility
            eps = entry_points().get(ENTRY_POINT_GROUP, [])

        for ep in eps:
            try:
                # Entry point can be a ScenarioAdapter class or instance
                loaded = cast(object, ep.load())

                # If it's a class, instantiate it
                from ash_hawk.scenario.adapters import ScenarioAdapter

                if isinstance(loaded, type) and hasattr(loaded, "name"):
                    # Check if it's a class that implements ScenarioAdapter
                    adapter = loaded()
                    if not isinstance(adapter, ScenarioAdapter):
                        logger.warning(
                            "Entry point '%s' must be a ScenarioAdapter class or instance",
                            ep.name,
                        )
                        continue
                elif isinstance(loaded, ScenarioAdapter):
                    adapter = loaded
                else:
                    logger.warning(
                        "Entry point '%s' must be a ScenarioAdapter class or instance",
                        ep.name,
                    )
                    continue

                # Only register if not already present
                if adapter.name in self._adapters:
                    logger.warning(
                        "Scenario adapter '%s' from entry point '%s' already registered, skipping",
                        adapter.name,
                        ep.name,
                    )
                else:
                    self.register(adapter)

            except Exception as e:
                logger.error(
                    "Failed to load scenario adapter from entry point '%s': %s",
                    ep.name,
                    e,
                )

    def __len__(self) -> int:
        """Return the number of registered adapters."""
        return len(self._adapters)

    def __contains__(self, name: str) -> bool:
        """Check if an adapter with the given name is registered."""
        return name in self._adapters


# Global default registry instance
_default_registry: ScenarioAdapterRegistry | None = None


def get_default_adapter_registry() -> ScenarioAdapterRegistry:
    """Get the default global scenario adapter registry.

    Creates the registry on first access and loads entry points.

    Returns:
        The default ScenarioAdapterRegistry instance.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ScenarioAdapterRegistry()
        _register_builtin_adapters(_default_registry)
        _default_registry.load_from_entry_points()
    return _default_registry


def _register_builtin_adapters(registry: ScenarioAdapterRegistry) -> None:
    """Register built-in scenario adapters.

    This function registers adapters that are included with ash-hawk.

    Args:
        registry: The registry to register adapters with.
    """
    from ash_hawk.scenario.adapters.mock_adapter import MockAdapter

    registry.register(MockAdapter())


__all__ = ["ScenarioAdapterRegistry", "get_default_adapter_registry", "ENTRY_POINT_GROUP"]
