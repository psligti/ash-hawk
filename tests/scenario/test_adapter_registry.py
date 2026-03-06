"""Tests for ash_hawk.scenario.registry module."""

from pathlib import Path
from typing import Any

import pytest

from ash_hawk.scenario.adapters import ScenarioAdapter
from ash_hawk.scenario.registry import (
    ENTRY_POINT_GROUP,
    ScenarioAdapterRegistry,
    get_default_adapter_registry,
)


class MockScenarioAdapter:
    """Mock scenario adapter for testing."""

    def __init__(self, name: str = "mock_adapter"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def run_scenario(
        self,
        scenario: dict[str, Any],
        workdir: Path,
        tooling_harness: dict[str, Any],
        budgets: dict[str, Any],
    ) -> tuple[Any, list[Any], dict[str, Any]]:
        """Mock implementation of run_scenario."""
        final_output = f"output for {scenario.get('id', 'unknown')}"
        trace_events = [{"event": "start"}, {"event": "end"}]
        artifacts = {"result": "success"}
        return final_output, trace_events, artifacts


class TestScenarioAdapterProtocol:
    """Test ScenarioAdapter protocol."""

    def test_adapter_protocol_is_runtime_checkable(self):
        """ScenarioAdapter can be checked with isinstance."""
        adapter = MockScenarioAdapter()
        assert isinstance(adapter, ScenarioAdapter)

    def test_adapter_requires_name_property(self):
        """Adapter must have name property."""

        class IncompleteAdapter:
            def run_scenario(self, scenario, workdir, tooling_harness, budgets):
                return None, [], {}

        adapter = IncompleteAdapter()
        # Should not be recognized as ScenarioAdapter
        assert not isinstance(adapter, ScenarioAdapter)

    def test_adapter_requires_run_scenario_method(self):
        """Adapter must have run_scenario method."""

        class IncompleteAdapter:
            @property
            def name(self):
                return "incomplete"

        adapter = IncompleteAdapter()
        # Should not be recognized as ScenarioAdapter
        assert not isinstance(adapter, ScenarioAdapter)


class TestScenarioAdapterRegistry:
    """Test ScenarioAdapterRegistry class."""

    def test_empty_registry(self):
        """New registry is empty."""
        registry = ScenarioAdapterRegistry()
        assert len(registry) == 0
        assert registry.list_adapters() == []

    def test_register_adapter(self):
        """Can register an adapter."""
        registry = ScenarioAdapterRegistry()
        adapter = MockScenarioAdapter()

        registry.register(adapter)

        assert len(registry) == 1
        assert "mock_adapter" in registry
        assert registry.get("mock_adapter") is adapter

    def test_register_multiple_adapters(self):
        """Can register multiple adapters."""
        registry = ScenarioAdapterRegistry()
        adapter1 = MockScenarioAdapter(name="adapter1")
        adapter2 = MockScenarioAdapter(name="adapter2")
        adapter3 = MockScenarioAdapter(name="adapter3")

        registry.register(adapter1)
        registry.register(adapter2)
        registry.register(adapter3)

        assert len(registry) == 3
        assert registry.list_adapters() == ["adapter1", "adapter2", "adapter3"]

    def test_register_overwrites(self):
        """Registering with same name overwrites previous adapter."""
        registry = ScenarioAdapterRegistry()
        adapter1 = MockScenarioAdapter(name="test")
        adapter2 = MockScenarioAdapter(name="test")

        registry.register(adapter1)
        registry.register(adapter2)

        assert len(registry) == 1
        assert registry.get("test") is adapter2

    def test_get_nonexistent_adapter(self):
        """Getting nonexistent adapter returns None."""
        registry = ScenarioAdapterRegistry()
        assert registry.get("nonexistent") is None

    def test_list_adapters_sorted(self):
        """list_adapters returns sorted list."""
        registry = ScenarioAdapterRegistry()
        registry.register(MockScenarioAdapter(name="zebra"))
        registry.register(MockScenarioAdapter(name="alpha"))
        registry.register(MockScenarioAdapter(name="middle"))

        assert registry.list_adapters() == ["alpha", "middle", "zebra"]

    def test_contains(self):
        """Can check if adapter is registered with 'in'."""
        registry = ScenarioAdapterRegistry()
        registry.register(MockScenarioAdapter())

        assert "mock_adapter" in registry
        assert "nonexistent" not in registry

    def test_load_from_entry_points_empty(self, monkeypatch):
        """load_from_entry_points handles no entry points gracefully."""
        registry = ScenarioAdapterRegistry()

        def mock_entry_points(*args, **kwargs):
            return []

        monkeypatch.setattr(
            "ash_hawk.scenario.registry.entry_points",
            mock_entry_points,
        )

        registry.load_from_entry_points()
        assert len(registry) == 0


class TestGetDefaultAdapterRegistry:
    """Test get_default_adapter_registry function."""

    def test_returns_singleton(self, monkeypatch):
        """get_default_adapter_registry returns same instance each time."""
        import ash_hawk.scenario.registry as reg_module

        reg_module._default_registry = None

        def mock_entry_points(*args, **kwargs):
            return []

        monkeypatch.setattr(
            "ash_hawk.scenario.registry.entry_points",
            mock_entry_points,
        )

        registry1 = get_default_adapter_registry()
        registry2 = get_default_adapter_registry()

        assert registry1 is registry2


class TestEntryPointGroup:
    """Test entry point group constant."""

    def test_entry_point_group_value(self):
        """ENTRY_POINT_GROUP has correct value."""
        assert ENTRY_POINT_GROUP == "ash_hawk.scenario_adapters"


class TestScenarioAdapterExecution:
    """Test adapter execution with real scenario data."""

    def test_adapter_run_scenario(self):
        """Adapter can run a scenario and return results."""
        adapter = MockScenarioAdapter(name="test_adapter")

        scenario = {"id": "scenario-1", "type": "test"}
        workdir = Path("/tmp/test")
        tooling_harness = {"allowed_tools": ["read", "write"]}
        budgets = {"max_tokens": 1000, "timeout_seconds": 60}

        final_output, trace_events, artifacts = adapter.run_scenario(
            scenario, workdir, tooling_harness, budgets
        )

        assert final_output == "output for scenario-1"
        assert len(trace_events) == 2
        assert artifacts["result"] == "success"

    def test_registry_with_executed_adapter(self):
        """Registry can store and retrieve an adapter that was executed."""
        registry = ScenarioAdapterRegistry()
        adapter = MockScenarioAdapter(name="executable_adapter")

        registry.register(adapter)

        retrieved = registry.get("executable_adapter")
        assert retrieved is not None
        assert isinstance(retrieved, ScenarioAdapter)

        # Execute the retrieved adapter
        final_output, trace_events, artifacts = retrieved.run_scenario(
            {"id": "test"}, Path("/tmp"), {}, {}
        )

        assert final_output == "output for test"
        assert len(trace_events) == 2
        assert "result" in artifacts
