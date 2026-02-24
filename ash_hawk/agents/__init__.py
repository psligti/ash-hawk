"""Agent runners for ash-hawk evaluation harness.

This module provides agent runner implementations that integrate with
different LLM frameworks. Each runner implements the AgentRunner protocol.

Available runners:
- DawnKestrelAgentRunner: Integration with dawn-kestrel framework
"""

from ash_hawk.agents.dawn_kestrel import DawnKestrelAgentRunner

__all__ = ["DawnKestrelAgentRunner"]
