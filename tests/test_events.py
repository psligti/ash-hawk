"""Tests for Ash-Hawk event system."""

import asyncio

import pytest

from ash_hawk.events import AHEvents, Event, bus, get_bus


class TestAHEvents:
    """Tests for Ash-Hawk event constants."""

    def test_event_names_are_defined(self):
        """All required event names should be defined."""
        assert hasattr(AHEvents, "TRIAL_STARTED")
        assert hasattr(AHEvents, "TRIAL_COMPLETED")
        assert hasattr(AHEvents, "TRIAL_FAILED")
        assert hasattr(AHEvents, "GRADER_RUN")
        assert hasattr(AHEvents, "GRADER_COMPLETED")
        assert hasattr(AHEvents, "SUITE_STARTED")
        assert hasattr(AHEvents, "SUITE_COMPLETED")
        assert hasattr(AHEvents, "RUN_ENVELOPE_CREATED")

    def test_event_names_have_correct_prefix(self):
        """All event names should have 'ah.' prefix."""
        assert AHEvents.TRIAL_STARTED.startswith("ah.")
        assert AHEvents.TRIAL_COMPLETED.startswith("ah.")
        assert AHEvents.TRIAL_FAILED.startswith("ah.")
        assert AHEvents.GRADER_RUN.startswith("ah.")
        assert AHEvents.GRADER_COMPLETED.startswith("ah.")
        assert AHEvents.SUITE_STARTED.startswith("ah.")
        assert AHEvents.SUITE_COMPLETED.startswith("ah.")
        assert AHEvents.RUN_ENVELOPE_CREATED.startswith("ah.")

    def test_event_names_are_unique(self):
        """All event names should be unique."""
        event_names = [
            AHEvents.TRIAL_STARTED,
            AHEvents.TRIAL_COMPLETED,
            AHEvents.TRIAL_FAILED,
            AHEvents.GRADER_RUN,
            AHEvents.GRADER_COMPLETED,
            AHEvents.SUITE_STARTED,
            AHEvents.SUITE_COMPLETED,
            AHEvents.RUN_ENVELOPE_CREATED,
        ]
        assert len(event_names) == len(set(event_names))


class TestGetBus:
    """Tests for get_bus function."""

    def test_get_bus_returns_same_instance(self):
        """get_bus should return the global bus instance."""
        bus1 = get_bus()
        bus2 = get_bus()
        assert bus1 is bus2

    def test_get_bus_returns_global_bus(self):
        """get_bus should return the same bus imported directly."""
        assert get_bus() is bus


class TestEventBusIntegration:
    """Tests for event bus publish/subscribe functionality."""

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self):
        """Should be able to subscribe to and publish events."""
        received_events = []

        async def handler(event: Event):
            received_events.append(event)

        unsubscribe = await bus.subscribe(AHEvents.TRIAL_STARTED, handler)
        await bus.publish(AHEvents.TRIAL_STARTED, {"trial_id": "test-123"})

        assert len(received_events) == 1
        assert received_events[0].name == AHEvents.TRIAL_STARTED
        assert received_events[0].data["trial_id"] == "test-123"

        await unsubscribe()
        await bus.clear_subscriptions(AHEvents.TRIAL_STARTED)

    @pytest.mark.asyncio
    async def test_event_data_received_correctly(self):
        """Event data should be received correctly by handlers."""
        received_data = None

        async def handler(event: Event):
            nonlocal received_data
            received_data = event.data

        unsubscribe = await bus.subscribe(AHEvents.GRADER_RUN, handler)

        test_data = {
            "grader_id": "grader-001",
            "trial_id": "trial-abc",
            "config": {"timeout": 30},
        }
        await bus.publish(AHEvents.GRADER_RUN, test_data)

        assert received_data == test_data

        await unsubscribe()
        await bus.clear_subscriptions(AHEvents.GRADER_RUN)

    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        """Multiple handlers should all receive events."""
        handler1_calls = []
        handler2_calls = []

        async def handler1(event: Event):
            handler1_calls.append(event)

        async def handler2(event: Event):
            handler2_calls.append(event)

        unsub1 = await bus.subscribe(AHEvents.SUITE_STARTED, handler1)
        unsub2 = await bus.subscribe(AHEvents.SUITE_STARTED, handler2)

        await bus.publish(AHEvents.SUITE_STARTED, {"suite_id": "suite-1"})

        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1

        await unsub1()
        await unsub2()
        await bus.clear_subscriptions(AHEvents.SUITE_STARTED)

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Unsubscribe should stop receiving events."""
        calls = []

        async def handler(event: Event):
            calls.append(event)

        unsubscribe = await bus.subscribe(AHEvents.TRIAL_COMPLETED, handler)

        await bus.publish(AHEvents.TRIAL_COMPLETED, {"status": "first"})
        assert len(calls) == 1

        await unsubscribe()

        await bus.publish(AHEvents.TRIAL_COMPLETED, {"status": "second"})
        assert len(calls) == 1

        await bus.clear_subscriptions(AHEvents.TRIAL_COMPLETED)

    @pytest.mark.asyncio
    async def test_once_subscription(self):
        """Once subscriptions should fire only once."""
        calls = []

        async def handler(event: Event):
            calls.append(event)

        unsubscribe = await bus.subscribe(AHEvents.TRIAL_FAILED, handler, once=True)

        await bus.publish(AHEvents.TRIAL_FAILED, {"error": "first"})
        await bus.publish(AHEvents.TRIAL_FAILED, {"error": "second"})

        assert len(calls) == 1
        assert calls[0].data["error"] == "first"

        await unsubscribe()
        await bus.clear_subscriptions(AHEvents.TRIAL_FAILED)

    @pytest.mark.asyncio
    async def test_sync_handler(self):
        """Sync handlers should also work."""
        calls = []

        def sync_handler(event: Event):
            calls.append(event)

        unsubscribe = await bus.subscribe(AHEvents.RUN_ENVELOPE_CREATED, sync_handler)

        await bus.publish(AHEvents.RUN_ENVELOPE_CREATED, {"envelope_id": "env-1"})

        assert len(calls) == 1

        await unsubscribe()
        await bus.clear_subscriptions(AHEvents.RUN_ENVELOPE_CREATED)
