"""Execution module for ash-hawk.

This module provides the TrialExecutor class that runs a single evaluation trial
with policy enforcement and envelope recording.

Key components:
- TrialExecutor: Main class for executing a single evaluation trial
- AgentRunner: Protocol for agent execution integration
- EvalRunner: Parallel suite runner with timeout, budget, and cancellation support
"""

from ash_hawk.execution.fixtures import FixtureError, FixtureResolver
from ash_hawk.execution.runner import EvalRunner, ResourceTracker
from ash_hawk.execution.trial import AgentRunner, TrialExecutor

__all__ = [
    "AgentRunner",
    "EvalRunner",
    "FixtureError",
    "FixtureResolver",
    "ResourceTracker",
    "TrialExecutor",
]
