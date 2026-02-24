"""Ash-Hawk event system using dawn-kestrel event bus."""

from dawn_kestrel.core.event_bus import Event, EventBus, bus

__all__ = ["AHEvents", "get_bus", "bus", "Event", "EventBus"]


class AHEvents:
    """Ash-Hawk specific event names.

    These events are published during trial execution and grading.
    All event names are prefixed with 'ah.' to avoid collisions.
    """

    # Trial lifecycle events
    TRIAL_STARTED = "ah.trial.started"
    TRIAL_COMPLETED = "ah.trial.completed"
    TRIAL_FAILED = "ah.trial.failed"

    # Grader events
    GRADER_RUN = "ah.grader.run"
    GRADER_COMPLETED = "ah.grader.completed"

    # Suite events
    SUITE_STARTED = "ah.suite.started"
    SUITE_COMPLETED = "ah.suite.completed"

    # Run envelope events
    RUN_ENVELOPE_CREATED = "ah.run.envelope_created"


def get_bus() -> EventBus:
    """Get the global event bus instance.

    Returns the dawn-kestrel global event bus for use in Ash-Hawk.
    All components should use this bus to ensure unified event handling.

    Returns:
        EventBus: The global event bus instance.
    """
    return bus
