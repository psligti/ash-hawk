"""Execution module for ash-hawk.

This module provides the TrialExecutor class that runs a single evaluation trial
with policy enforcement and envelope recording.

Key components:
- TrialExecutor: Main class for executing a single evaluation trial
- AgentRunner: Protocol for agent execution integration
- EvalRunner: Parallel suite runner with timeout, budget, and cancellation support
- LLMRequestQueue: Queue for throttling LLM API requests
- TrialExecutionQueue: Queue for throttling trial executions
"""

from ash_hawk.execution.fast_eval import FastEvalRunner
from ash_hawk.execution.fixtures import FixtureError, FixtureResolver
from ash_hawk.execution.queue import (
    LLMRequest,
    LLMRequestQueue,
    LLMResponse,
    TrialExecutionQueue,
    TrialJob,
    TrialJobResult,
    get_llm_queue,
    get_llm_queue_sync,
    register_llm_queue,
    reset_llm_queue,
)
from ash_hawk.execution.runner import EvalRunner, ResourceTracker
from ash_hawk.execution.trial import AgentRunner, TrialExecutor

__all__ = [
    "AgentRunner",
    "EvalRunner",
    "FastEvalRunner",
    "FixtureError",
    "FixtureResolver",
    "LLMRequest",
    "LLMRequestQueue",
    "LLMResponse",
    "ResourceTracker",
    "TrialExecutionQueue",
    "TrialExecutor",
    "TrialJob",
    "TrialJobResult",
    "get_llm_queue",
    "get_llm_queue_sync",
    "register_llm_queue",
    "reset_llm_queue",
]
